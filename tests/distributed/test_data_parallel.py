"""Tests for :mod:`rapid_llm.engine.data_parallel`.

Argument validation and the routing table run CPU-only through a
``_RouteHarness``; only the end-to-end two-GPU test needs real
devices.

Usage:
    pytest tests/distributed/test_data_parallel.py
"""

from __future__ import annotations

import inspect
import itertools
import queue
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from rapid_llm.engine.data_parallel import _SHUTDOWN, DataParallelEngine, _dp_worker, _ReplicaLoop
from rapid_llm.engine.outputs import RequestOutput
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Request, RequestStatus

# Enough KV for the short generations here, small enough that two replicas plus the
# single-GPU reference engine coexist on one card.
_KV_TOKENS = 4096

_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "The sun rises in the",
    "Water boils at",
    "Machine learning is",
    "The largest planet is",
]

#: Greedy and with every early-exit heuristic off, so the only thing that can make two
#: runs differ is the arithmetic itself.
_GREEDY = SamplingParams(
    temperature=0.0, max_gen_len=24, repetition_penalty=1.0, stop_on_repeat=False
)

requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason=f"data parallelism needs 2 GPUs, found {torch.cuda.device_count()}",
)


# --------------------------------------------------------------------------- #
# Construction contract (no GPU work)
# --------------------------------------------------------------------------- #
def test_device_kwarg_is_rejected():
    """A caller-supplied device would silently put every replica on one GPU."""
    with pytest.raises(ValueError, match="device is derived"):
        DataParallelEngine(model="unused", data_parallel_size=2, device="cuda:0")


def test_non_positive_replica_count_is_rejected():
    with pytest.raises(ValueError, match="data_parallel_size must be >= 1"):
        DataParallelEngine(model="unused", data_parallel_size=0)


def test_non_positive_tensor_parallel_size_is_rejected_before_spawning_workers():
    with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
        DataParallelEngine(model="unused", tensor_parallel_size=0)


def test_unknown_load_balancer_is_rejected():
    with pytest.raises(ValueError, match="unknown load_balancer"):
        DataParallelEngine(model="unused", data_parallel_size=1, load_balancer="magic")


def test_dp_attention_requires_multiple_replicas():
    with pytest.raises(ValueError, match="data_parallel_size > 1"):
        DataParallelEngine(model="unused", enable_dp_attention=True)


def test_dp_attention_requires_expert_parallelism():
    with pytest.raises(ValueError, match="enable_expert_parallel=True"):
        DataParallelEngine(
            model="unused",
            data_parallel_size=2,
            enable_dp_attention=True,
            use_cuda_graph=False,
        )


def test_dp_attention_requires_eager_mode():
    with pytest.raises(ValueError, match="does not support CUDA Graph"):
        DataParallelEngine(
            model="unused",
            data_parallel_size=2,
            enable_dp_attention=True,
            enable_expert_parallel=True,
        )


def test_requesting_more_gpus_than_exist_is_rejected(monkeypatch: pytest.MonkeyPatch):
    """The count is checked up front, not discovered by a worker failing to start."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=r"needs 4 GPUs, but only 2 are visible"):
        DataParallelEngine(model="unused", data_parallel_size=4)

    # The same check covers the composed grid: 2 replicas x TP 2 also wants 4.
    with pytest.raises(ValueError, match=r"needs 4 GPUs, but only 2 are visible"):
        DataParallelEngine(model="unused", data_parallel_size=2, tensor_parallel_size=2)


# --------------------------------------------------------------------------- #
# Routing (no GPU work): drive _route on a bare instance
# --------------------------------------------------------------------------- #
class _RouteHarness:
    """Minimal object carrying just what ``DataParallelEngine._route`` reads.

    ``_route`` only touches ``data_parallel_size`` and ``_balancer``, so a full engine
    (and its GPUs) is unnecessary to test how prompts are bucketed. It is handed the
    prompts' *token ids*, which is what the router tokenises up front; a policy that
    only wants a length takes it from ``len(ids)``.
    """

    def __init__(self, dp_size: int, policy: str = "round_robin"):
        from rapid_llm.engine.dp_load_balancer import make_load_balancer

        self.data_parallel_size = dp_size
        self._balancer = make_load_balancer(policy, dp_size)

    _select = DataParallelEngine._select
    route = DataParallelEngine._route


def _ids(*lengths: int) -> list[list[int]]:
    """One distinct token-id list per length, sharing no prefix with the others."""
    return [[tag * 100_000 + i for i in range(n)] for tag, n in enumerate(lengths, 1)]


def test_route_buckets_round_robin():
    buckets = _RouteHarness(dp_size=2).route(5)
    assert buckets == [[0, 2, 4], [1, 3]]


def test_route_covers_every_prompt_exactly_once():
    buckets = _RouteHarness(dp_size=3).route(10)
    assigned = sorted(i for bucket in buckets for i in bucket)
    assert assigned == list(range(10))


def test_route_leaves_idle_replicas_empty():
    """Fewer prompts than replicas is legal: the extra replicas get nothing."""
    buckets = _RouteHarness(dp_size=4).route(2)
    assert buckets == [[0], [1], [], []]


def test_route_sends_a_long_prompt_and_the_short_ones_apart():
    """With ``total_tokens``, prompt length — not arrival order — decides the split.

    This is the routing-level consequence of the balancer reading the estimate: the
    three short prompts must not be striped onto the replica already holding 1k tokens.
    """
    token_ids = _ids(1000, 10, 10, 10)
    buckets = _RouteHarness(dp_size=2, policy="total_tokens").route(4, token_ids)
    assert buckets == [[0], [1, 2, 3]]


def test_route_rejects_a_token_id_list_that_does_not_cover_the_batch():
    """A short ``token_ids`` would misalign every decision after the gap, silently."""
    harness = _RouteHarness(dp_size=2, policy="total_tokens")
    with pytest.raises(ValueError, match="expected 3 token id lists, got 2"):
        harness.route(3, _ids(10, 10))


def test_route_refuses_to_route_cache_aware_without_the_ids():
    """``cache_aware`` on zero ids is ``total_tokens`` wearing its name — so it raises.

    Nothing about the output would look wrong: every request would simply be routed as a
    cache miss. That is exactly the kind of silent downgrade a stated contract is for.
    """
    harness = _RouteHarness(dp_size=2, policy="cache_aware")
    with pytest.raises(ValueError, match="routes on prompt tokens"):
        harness.route(2)


def test_route_instantiates_fewer_copies_of_each_shared_prefix():
    """End of the wiring: ids reach the balancer, so affinity survives the router.

    The quantity that matters is how many replicas each distinct prefix ends up prefilled
    on. Every extra one is a preamble computed twice and a second copy taking cache
    capacity. Four prefixes over two replicas, arriving shuffled so that no policy is
    helped by the order lining up with the stripe.

    Not asserted as "one replica per prefix": while the pool is still cold an idle replica
    is genuinely the better place for a request, even at the price of warming a second
    copy, and the policy is right to say so. What it must not do is keep paying that price
    once the prefix is hot somewhere — which is what the strict inequality catches, and
    what would vanish if ``_route`` dropped the ids.
    """
    groups = [[group * 10_000 + i for i in range(512)] for group in range(4)]
    arrivals = [0, 2, 1, 3, 0, 0, 3, 1, 2, 1, 3, 2, 1, 0, 2, 3]
    token_ids = [[*groups[group], 900_000 + i] for i, group in enumerate(arrivals)]

    def copies(policy: str) -> int:
        buckets = _RouteHarness(dp_size=2, policy=policy).route(len(arrivals), token_ids)
        return len({(arrivals[i], replica) for replica, b in enumerate(buckets) for i in b})

    assert copies("total_tokens") == 8  # every prefix on both replicas
    assert copies("cache_aware") < 8


# --------------------------------------------------------------------------- #
# The process grid (no GPU work)
# --------------------------------------------------------------------------- #
def test_token_estimates_are_skipped_when_no_policy_reads_them():
    """``round_robin`` must never load a tokenizer — the pass would be dead cost.

    Reaching ``self.tokenizer`` on the harness would raise ``AttributeError``, so this
    asserts the short circuit rather than the ids.
    """
    harness = _RouteHarness(dp_size=2)
    assert DataParallelEngine._tokenize_for_routing(harness, ["a", "bb", "ccc"]) is None


@pytest.mark.parametrize("policy", ["total_tokens", "cache_aware"])
def test_a_token_aware_policy_gets_one_tokenizer_pass(policy):
    """Both token-aware policies are served by the same single batched call.

    Two flags, one pass: a length is ``len(ids)``, so tokenising twice would be the
    router paying for the same work under two names.
    """
    calls = []

    class _Tokenizing(_RouteHarness):
        def __call__(self, prompts, add_special_tokens):
            calls.append((tuple(prompts), add_special_tokens))
            return {"input_ids": [list(range(len(p))) for p in prompts]}

        tokenizer = property(lambda self: self)

    harness = _Tokenizing(dp_size=2, policy=policy)
    ids = DataParallelEngine._tokenize_for_routing(harness, ["a", "bb", "ccc"])

    assert calls == [(("a", "bb", "ccc"), True)]
    assert [len(x) for x in ids] == [1, 2, 3]


def test_a_replica_shares_one_queue_across_its_ranks():
    """A request goes to the replica, not to each of its ranks.

    The followers of a TP replica do not read requests at all: their leader owns the
    scheduler and broadcasts each step's plan to them. One queue per replica is that
    contract in the constructor — and the reason the old per-cell queues could not
    stay, since a follower reading a request would consume it from its leader.
    """
    from rapid_llm.engine import data_parallel as dp_module

    _FakeProcess.spawned = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "device_count", lambda: 4)
        patch.setattr(
            dp_module.mp,
            "get_context",
            lambda method: type("_Ctx", (), {"Queue": _FakeQueue, "Process": _FakeProcess})(),
        )
        engine = DataParallelEngine(model="unused", data_parallel_size=2, tensor_parallel_size=2)

    queues = [args[_REQUEST_QUEUE_ARG] for args in _FakeProcess.spawned]
    assert queues[0] is queues[1]  # replica 0's leader and follower
    assert queues[2] is queues[3]  # replica 1's
    assert queues[0] is not queues[2]

    engine.shutdown()


class _FakeQueue:
    """In-process stand-in for ``mp.Queue`` — enough for the coordinator's handshake."""

    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)

    def get(self, timeout: float | None = None):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


#: Position of the request queue in the arguments ``_dp_worker`` is spawned with, read
#: from the signature rather than written down: this test is about *which* queue each
#: rank gets, and a hard-coded index turns any new worker argument into a failure here.
_REQUEST_QUEUE_ARG = list(inspect.signature(_dp_worker).parameters).index("request_queue")
_ENABLE_DPA_ARG = list(inspect.signature(_dp_worker).parameters).index("enable_dp_attention")


class _FakeProcess:
    """Records the arguments a worker *would* have been spawned with, then reports ready."""

    spawned: ClassVar[list[tuple]] = []

    def __init__(self, target, args, daemon) -> None:
        self.args = args
        self._alive = True

    def start(self) -> None:
        _FakeProcess.spawned.append(self.args)
        self.args[-1].put(("ready", self.args[0], None))  # result_queue

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


def test_dp_attention_flag_reaches_every_grid_cell(monkeypatch: pytest.MonkeyPatch):
    """Every rank must build the widened EP/DP groups from the same DPA flag."""
    from rapid_llm.engine import data_parallel as dp_module

    _FakeProcess.spawned = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        dp_module.mp,
        "get_context",
        lambda method: type("_Ctx", (), {"Queue": _FakeQueue, "Process": _FakeProcess})(),
    )

    engine = DataParallelEngine(
        model="unused",
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_dp_attention=True,
        use_cuda_graph=False,
    )

    assert [args[_ENABLE_DPA_ARG] for args in _FakeProcess.spawned] == [True] * 4
    engine.shutdown()


def test_dpa_wake_reaches_idle_peer_queues():
    """One routed request wakes every peer that must join its collectives."""
    engine = object.__new__(DataParallelEngine)
    engine.enable_dp_attention = True
    engine._closed = True
    engine._request_queues = [_FakeQueue(), _FakeQueue(), _FakeQueue()]

    engine._send_with_dpa_wake(1, ("add", "r1", "prompt", _GREEDY))

    assert engine._request_queues[0].items == [("wake",)]
    assert engine._request_queues[1].items == [("add", "r1", "prompt", _GREEDY)]
    assert engine._request_queues[2].items == [("wake",)]


def test_dp_times_tp_spawns_one_process_per_grid_cell(monkeypatch: pytest.MonkeyPatch):
    """A 2x2 grid must be four processes, ranked ``dp_rank * tp_size + tp_rank``.

    This is the deadlock in assertion form. ``init_parallel(tp_size=2, dp_size=2)``
    rendezvouses a world of four, so spawning two processes hangs forever on ranks that
    were never started — a failure no timeout in the coordinator can explain. Asserting
    the grid needs no GPU at all, which is the point: the bug reproduces on a laptop.
    """
    from rapid_llm.engine import data_parallel as dp_module

    _FakeProcess.spawned = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        dp_module.mp,
        "get_context",
        lambda method: type("_Ctx", (), {"Queue": _FakeQueue, "Process": _FakeProcess})(),
    )

    engine = DataParallelEngine(model="unused", data_parallel_size=2, tensor_parallel_size=2)

    assert engine.world_size == 4
    assert len(_FakeProcess.spawned) == 4
    # (global_rank, dp_rank, tp_rank) per cell, in rank order.
    assert [args[:3] for args in _FakeProcess.spawned] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ]

    engine.shutdown()


# --------------------------------------------------------------------------- #
# The replica's engine loop (no GPU work)
# --------------------------------------------------------------------------- #
class _LoopQueue:
    """``mp.Queue``'s ``get(block=...)`` over a list, recording how it was called.

    Honouring ``block`` is the point: the loop must wait when it has nothing to run
    and must *not* wait when it is mid-decode, and a stub that returned immediately
    either way would let a busy-spinning replica pass its tests. Blocking on an
    empty queue is therefore an outright failure — in the real process it is a hang.
    """

    def __init__(self, items: Iterable = ()) -> None:
        self.items = list(items)
        self.gets: list[bool] = []

    def put(self, item) -> None:
        self.items.append(item)

    def get(self, block: bool = True):
        self.gets.append(block)
        if not self.items:
            if block:
                raise AssertionError("the loop blocked on an empty queue with no stop signal")
            raise queue.Empty
        return self.items.pop(0)


class _LoopEngine:
    """A :class:`Scheduler` stand-in whose requests finish on schedule.

    Hands out *real* ``Request`` objects, because the loop reads completion off the
    handles the engine updates in place; a renamed field has to fail here rather
    than in a two-GPU run.

    Args:
        tokens: Steps each request runs for, by prompt, defaulting to two.
        fail_on_step: Step number that raises, standing in for a kernel failure.
        reject: Prompts ``add_request`` refuses, as an over-long one would be.
        after_step: Called with the step number — how a test makes a request arrive
            *while* the replica is decoding.
    """

    def __init__(
        self,
        tokens: dict[str, int] | None = None,
        fail_on_step: int | None = None,
        reject: Iterable[str] = (),
        after_step=None,
    ) -> None:
        self._tokens = tokens or {}
        self._fail_on_step = fail_on_step
        self._reject = set(reject)
        self._after_step = after_step
        self._ids = itertools.count()
        self.running: list[Request] = []
        self.steps = 0
        self.aborted: list[str] = []
        self.admitted_at: list[int] = []

    def add_request(
        self, prompt: str, params: SamplingParams | None = None, request_id: str | None = None
    ) -> Request:
        if prompt in self._reject:
            raise ValueError("prompt of 9000 tokens exceeds the context window")
        request = Request(
            request_id=request_id or f"req-{next(self._ids)}",
            prompt=prompt,
            prompt_token_ids=[1],
            params=params or SamplingParams(),
        )
        self.running.append(request)
        self.admitted_at.append(self.steps)
        return request

    def has_unfinished_requests(self) -> bool:
        return bool(self.running)

    def step(self) -> list[Request]:
        self.steps += 1
        if self.steps == self._fail_on_step:
            raise RuntimeError("illegal memory access")
        advanced = []
        for request in list(self.running):
            request.output_token_ids.append(7)
            request.delta = "x"
            request.text += "x"
            if len(request.output_token_ids) >= self._tokens.get(request.prompt, 2):
                request.status = RequestStatus.FINISHED
                request.finish_reason = "eos"
                self.running.remove(request)
            else:
                advanced.append(request)
        if self._after_step is not None:
            self._after_step(self.steps)
        return advanced

    def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        self.running = [r for r in self.running if r.request_id != request_id]


def _loop(
    engine: _LoopEngine, *messages, enable_dp_attention: bool = False
) -> tuple[_ReplicaLoop, _LoopQueue, _LoopQueue]:
    """A loop wired to pre-queued messages plus the stop signal, and its two queues."""
    requests = _LoopQueue([*messages, _SHUTDOWN])
    results = _LoopQueue()
    return (
        _ReplicaLoop(
            engine,
            requests,
            results,
            enable_dp_attention=enable_dp_attention,
        ),
        requests,
        results,
    )


def test_a_finished_batch_is_reported_once_with_every_index():
    """The coordinator counts batches, so a batch must produce exactly one message.

    The indices matter as much as the text: they are the caller's positions, and
    losing one silently leaves that prompt unanswered forever.
    """
    engine = _LoopEngine()
    loop, _requests, results = _loop(engine, ("batch", 0, [3, 1], ["a", "b"], _GREEDY))

    loop.run()

    assert len(results.items) == 1
    kind, batch_id, payload = results.items[0]
    assert (kind, batch_id) == ("done", 0)
    completions, prompt_logprobs = payload
    assert {index: (text, reason) for index, text, reason, _ in completions} == {
        3: ("xx", "eos"),
        1: ("xx", "eos"),
    }
    assert all(logprobs is None for *_head, logprobs in completions)
    assert prompt_logprobs == {3: None, 1: None}


def test_a_batch_waits_for_its_slowest_request():
    """A batch is answered as a whole, but its members finish independently.

    The short request must leave the engine as soon as it stops — that freed slot is
    what continuous batching buys — while the answer stays held until the long one
    is done, because that is the unit the coordinator dispatched.
    """
    engine = _LoopEngine(tokens={"long": 5})
    requests = _LoopQueue([("batch", 0, [0, 1], ["short", "long"], _GREEDY)])
    results = _LoopQueue()
    loop = _ReplicaLoop(engine, requests, results)

    loop._take_arrivals(block=False)
    loop._step()
    loop._step()

    # "short" stopped at its second token and is out of the engine; its answer is
    # held, not sent, because its batch is not done.
    assert [request.prompt for request in engine.running] == ["long"]
    assert results.items == []

    requests.put(_SHUTDOWN)
    loop.run()
    assert len(results.items) == 1
    completions, prompt_logprobs = results.items[0][2]
    assert len(completions) == 2
    assert prompt_logprobs == {0: None, 1: None}


def test_the_loop_blocks_when_idle_and_polls_when_busy():
    """Waiting is conditional on having nothing to run.

    A replica between dispatches must cost no CPU, and a replica mid-decode must not
    stall its steps on a queue that may stay empty for minutes. Both are the same
    ``block`` flag, which is why it is asserted rather than assumed.
    """
    engine = _LoopEngine(tokens={"a": 3})
    loop, requests, _results = _loop(engine, ("batch", 0, [0], ["a"], _GREEDY))

    loop.run()

    assert requests.gets[0] is True  # idle: waited for the dispatch
    assert not any(requests.gets[1:])  # busy: never waited again


def test_idle_dpa_replica_keeps_stepping_while_a_peer_is_active(monkeypatch):
    """A locally drained replica must not block before the next global handshake."""
    from rapid_llm.engine import data_parallel as dp_module

    active = iter((1, 1, 0))
    monkeypatch.setattr(
        dp_module,
        "coordinate_forward_count_across_dp",
        lambda local: next(active),
    )
    requests = _LoopQueue([("batch", 0, [0], ["a"], _GREEDY)])
    results = _LoopQueue()
    engine = _LoopEngine(tokens={"a": 1})
    engine._after_step = lambda step: requests.put(_SHUTDOWN) if step == 2 else None
    loop = _ReplicaLoop(
        engine,
        requests,
        results,
        enable_dp_attention=True,
    )

    loop.run()

    assert engine.steps == 2, "the second step is the idle replica's lockstep pass"
    assert requests.gets[0] is True
    assert not any(requests.gets[1:]), "an active DPA epoch must only poll its queue"
    assert results.items[0][0] == "done"


def test_a_stop_signal_mid_batch_still_answers_it():
    """Shutdown stops admission, not the work already paid for.

    The coordinator is waiting on those answers, and the requests have already spent
    their prefill; dropping them would turn a clean shutdown into a hang.
    """
    engine = _LoopEngine(tokens={"a": 4})
    loop, _requests, results = _loop(engine, ("batch", 0, [0], ["a"], _GREEDY))

    loop.run()

    assert engine.steps == 4
    assert results.items[0][0] == "done"


def test_a_request_may_join_a_running_batch():
    """The whole point of a resident engine: work arriving mid-decode is admitted.

    ``admitted_at`` records the step count each request was added on, so the second
    batch being admitted at step 1 is the proof that it did not wait behind the first
    one — which is exactly what the one-shot ``generate()`` per dispatch could not do.
    """
    engine = _LoopEngine(tokens={"a": 4})
    requests = _LoopQueue([("batch", 0, [0], ["a"], _GREEDY)])
    results = _LoopQueue()
    loop = _ReplicaLoop(engine, requests, results)
    # The second batch is dispatched while the first is still decoding, and the stop
    # signal only after that — the order a coordinator would produce.
    arrivals = {1: ("batch", 1, [2], ["b"], _GREEDY), 2: _SHUTDOWN}
    engine._after_step = lambda step: (
        requests.items.append(arrivals[step]) if step in arrivals else None
    )

    loop.run()

    assert engine.admitted_at == [0, 1]
    assert sorted(batch_id for _kind, batch_id, _payload in results.items) == [0, 1]
    assert {kind for kind, _batch, _payload in results.items} == {"done"}


def test_a_step_failure_fails_the_batch_and_frees_its_slots():
    """A failed step must abort what it was running, or the slots leak.

    The replica keeps serving afterwards, so a request left in the scheduler would
    hold its cache slot for the process's lifetime.
    """
    engine = _LoopEngine(fail_on_step=1)
    loop, _requests, results = _loop(engine, ("batch", 0, [0, 1], ["a", "b"], _GREEDY))

    loop.run()

    kind, batch_id, detail = results.items[0]
    assert (kind, batch_id) == ("error", 0)
    assert "illegal memory access" in detail
    assert len(engine.aborted) == 2
    assert engine.running == []


def test_an_unservable_prompt_fails_its_batch_and_not_the_replica():
    """A prompt the engine refuses is the batch's failure, not silence.

    Answering it with an empty completion would be worse than an error: the caller
    asked for text and would get a plausible-looking blank. The sibling already
    admitted is aborted so the batch leaves nothing behind.
    """
    engine = _LoopEngine(reject=["too long"])
    loop, _requests, results = _loop(engine, ("batch", 0, [0, 1], ["a", "too long"], _GREEDY))

    loop.run()

    kind, batch_id, detail = results.items[0]
    assert (kind, batch_id) == ("error", 0)
    assert "exceeds the context window" in detail
    assert engine.aborted == ["req-0"]
    assert engine.steps == 0


# --------------------------------------------------------------------------- #
# The replica's engine loop, streaming front end (no GPU work)
# --------------------------------------------------------------------------- #
def test_a_streamed_request_reports_each_delta_and_then_finishes():
    """The streaming contract: progress every step, one final frame with the reason.

    A batch is answered once as a whole; a streamed request is answered *per
    step*, because its consumer is one HTTP connection and waiting for a batch
    boundary would be waiting for a grouping that does not exist.
    """
    engine = _LoopEngine(tokens={"a": 3})
    loop, _requests, results = _loop(engine, ("add", "r1", "a", _GREEDY))

    loop.run()

    assert [message[0] for message in results.items] == ["delta", "delta", "finished"]
    assert {message[1] for message in results.items} == {"r1"}
    # Delta frames carry the running text and counts the consumer reports as usage.
    assert [message[2] for message in results.items[:2]] == ["x", "x"]
    assert [message[3] for message in results.items[:2]] == ["x", "xx"]
    assert [message[5] for message in results.items[:2]] == [1, 2]
    # The finish frame carries the reason and the totals, with no delta of its own.
    assert results.items[-1][2:] == ("eos", "xxx", 1, 3)


def test_aborting_a_streamed_request_is_silent():
    """No acknowledgement comes back for an abort.

    The consumer that asked to cancel has stopped reading, so a reply would be
    dropped by id on the coordinator side anyway. What must hold instead: the
    request is aborted before any further step, and nothing is reported for it.
    """
    engine = _LoopEngine(tokens={"a": 4})
    loop, _requests, results = _loop(engine, ("add", "r1", "a", _GREEDY), ("abort", "r1"))

    loop.run()

    assert engine.aborted == ["r1"]
    assert engine.steps == 0, "aborted before the first step, not after burning one"
    assert results.items == []


def test_an_unservable_prompt_fails_only_its_own_request():
    """The failure unit of the streaming path is the request, not a batch.

    One caller waiting on one id must not take down the sibling that arrived
    next to it — that is the whole difference from ``_admit_batch``'s grouping.
    """
    engine = _LoopEngine(reject=["too long"])
    loop, _requests, results = _loop(
        engine, ("add", "bad", "too long", _GREEDY), ("add", "ok", "fine", _GREEDY)
    )

    loop.run()

    failed = [message for message in results.items if message[0] == "failed"]
    assert len(failed) == 1
    assert failed[0][1] == "bad"
    assert "exceeds the context window" in failed[0][2]
    assert [(m[1], m[2]) for m in results.items if m[0] == "finished"] == [("ok", "eos")]


def test_a_step_failure_fails_the_streamed_request_and_frees_its_slot():
    """A broken step must abort the streamed request too, or its slot leaks.

    Same requirement as the batch path, different failure unit: the message is
    ``("failed", request_id, ...)`` — and the deltas already sent stay sent,
    because the consumer saw them and the error supersedes only the future.
    """
    engine = _LoopEngine(tokens={"a": 4}, fail_on_step=2)
    loop, _requests, results = _loop(engine, ("add", "r1", "a", _GREEDY))

    loop.run()

    assert [message[0] for message in results.items] == ["delta", "failed"]
    assert results.items[-1][1] == "r1"
    assert "illegal memory access" in results.items[-1][2]
    assert engine.aborted == ["r1"]
    assert engine.running == []


def test_batched_and_streamed_work_share_one_loop_without_confusion():
    """Both front ends drive one replica at once; ids keep the answers apart.

    The batch must still be answered as one ``done`` holding only its own
    members, and the streamed request must still get its per-step deltas —
    neither bookkeeping may leak into the other's messages.
    """
    engine = _LoopEngine(tokens={"a": 2, "b": 2})
    loop, _requests, results = _loop(
        engine, ("batch", 0, [0], ["a"], _GREEDY), ("add", "solo", "b", _GREEDY)
    )

    loop.run()

    dones = [message for message in results.items if message[0] == "done"]
    assert len(dones) == 1
    assert dones[0][1] == 0
    completions, prompt_logprobs = dones[0][2]
    assert completions == [(0, "xx", "eos", None)], "the streamed prompt is not a batch member"
    assert prompt_logprobs == {0: None}
    assert [message[0] for message in results.items if message[1] == "solo"] == [
        "delta",
        "finished",
    ]


# --------------------------------------------------------------------------- #
# End to end across two GPUs
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
class TestTwoReplicas:
    """One engine, shared by every case here: startup is seconds, generation is not."""

    @pytest.fixture(scope="class")
    def engine(self, model_dir: Path):
        with DataParallelEngine(
            model=str(model_dir),
            data_parallel_size=2,
            max_seq_len=512,
            max_gpu_num_blocks=_KV_TOKENS,
        ) as engine:
            yield engine

    def test_generate_returns_one_output_per_prompt_in_order(self, engine: DataParallelEngine):
        outputs = engine.generate(_PROMPTS, _GREEDY)

        assert len(outputs) == len(_PROMPTS)
        for out, prompt in zip(outputs, _PROMPTS, strict=True):
            assert isinstance(out, RequestOutput)
            # The prompt echo is what proves reassembly did not rotate the results.
            assert out.prompt == prompt
            assert out.text
            assert out.outputs[0].finish_reason in ("eos", "length", "repeat")

    def test_generate_accepts_a_single_string(self, engine: DataParallelEngine):
        """One prompt leaves every replica but one idle, and must still work."""
        outputs = engine.generate("The capital of France is", _GREEDY)

        assert len(outputs) == 1
        assert outputs[0].prompt == "The capital of France is"
        assert outputs[0].text

    def test_generate_handles_an_empty_batch(self, engine: DataParallelEngine):
        assert engine.generate([]) == []

    def test_generate_is_repeatable(self, engine: DataParallelEngine):
        """Same prompts, same engine, greedy: the replicas must not drift apart."""
        first = [out.text for out in engine.generate(_PROMPTS, _GREEDY)]
        second = [out.text for out in engine.generate(_PROMPTS, _GREEDY)]
        assert first == second

    def test_matches_a_single_gpu_per_replica_batch(
        self, engine: DataParallelEngine, model_dir: Path
    ):
        """Each replica's batch must produce exactly what one GPU produces for it.

        The reference engine replays the *same* buckets, so batch shapes match and the
        comparison is byte-exact — see this module's docstring for why comparing
        against one run over all six prompts would not be.
        """
        from rapid_llm.engine.llm import LLM

        dp_texts = [out.text for out in engine.generate(_PROMPTS, _GREEDY)]

        reference = LLM(model=str(model_dir), max_seq_len=512, max_gpu_num_blocks=_KV_TOKENS)
        try:
            for bucket in engine._route(len(_PROMPTS), engine._tokenize_for_routing(_PROMPTS)):
                expected = reference.generate([_PROMPTS[i] for i in bucket], _GREEDY)
                for index, out in zip(bucket, expected, strict=True):
                    assert dp_texts[index] == out.text
        finally:
            del reference
            torch.cuda.empty_cache()

    def test_tokenizer_is_available_for_token_accounting(self, engine: DataParallelEngine):
        """Benchmarks and evals count tokens through the engine, not through a worker."""
        assert engine.tokenizer.encode("hello world")


@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
def test_total_tokens_engine_also_covers_every_prompt(model_dir: Path):
    """The token-aware policy must route just as completely as round-robin.

    It is the only policy that tokenises in the coordinator process, so this is also the
    end-to-end check that the estimate path works against a real tokenizer.
    """
    with DataParallelEngine(
        model=str(model_dir),
        data_parallel_size=2,
        load_balancer="total_tokens",
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
    ) as engine:
        outputs = engine.generate(_PROMPTS, _GREEDY)

    assert [out.prompt for out in outputs] == _PROMPTS
    assert all(out.text for out in outputs)


@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
def test_generate_after_shutdown_is_an_error(model_dir: Path):
    engine = DataParallelEngine(
        model=str(model_dir),
        data_parallel_size=2,
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
    )
    engine.shutdown()
    engine.shutdown()  # idempotent

    with pytest.raises(RuntimeError, match="has been shut down"):
        engine.generate(["hi"])
