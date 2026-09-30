"""CPU contracts and GPU integration tests for ``DataParallelController``."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from rapid_llm.engine.data_parallel import DataParallelController
from rapid_llm.engine.outputs import RequestOutput
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler_ipc import (
    PROTOCOL_VERSION,
    AbortRequest,
    AddRequest,
    AddRequestBatch,
    ReplicaReady,
    RequestEvent,
    SchedulerEvents,
    SchedulerReady,
    ShutdownScheduler,
    WakeScheduler,
)

_KV_TOKENS = 4096
_GREEDY = SamplingParams(
    temperature=0.0, max_gen_len=24, repetition_penalty=1.0, stop_on_repeat=False
)
_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "The sun rises in the",
    "Water boils at",
]
requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason=f"data parallelism needs 2 GPUs, found {torch.cuda.device_count()}",
)
requires_four_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason=f"DP x TP integration needs 4 GPUs, found {torch.cuda.device_count()}",
)


class _FakeQueue:
    def __init__(self) -> None:
        self._items: queue.Queue = queue.Queue()
        self.sent: list = []

    def put(self, item) -> None:
        self.sent.append(item)
        self._items.put(item)

    def get(self, block: bool = True, timeout: float | None = None):
        return self._items.get(block=block, timeout=timeout)

    def get_nowait(self):
        return self._items.get_nowait()

    def close(self) -> None:
        pass

    def join_thread(self) -> None:
        pass


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _FakeProcess:
    spawned: ClassVar[list[tuple]] = []

    def __init__(self, target, args, daemon) -> None:
        self.args = args
        self.pid = 1000 + args[0]
        self.exitcode = None
        self._alive = True

    def start(self) -> None:
        _FakeProcess.spawned.append(self.args)
        global_rank, _dp_rank, tp_rank = self.args[:3]
        ready = SchedulerReady(PROTOCOL_VERSION, "test", 128, 64, 8) if tp_rank == 0 else None
        self.args[-1].put(ReplicaReady(global_rank, ready))

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False
        self.exitcode = 0

    def terminate(self) -> None:
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -9


@pytest.fixture
def controller_factory(monkeypatch: pytest.MonkeyPatch):
    from rapid_llm.engine import data_parallel as module

    made = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        module.mp,
        "get_context",
        lambda method: type("Context", (), {"Queue": _FakeQueue, "Process": _FakeProcess})(),
    )

    def build(**kwargs):
        _FakeProcess.spawned = []
        controller = DataParallelController("unused", data_parallel_size=2, **kwargs)
        made.append(controller)
        return controller

    yield build
    for controller in made:
        controller.shutdown()


def test_argument_validation_happens_before_process_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="data_parallel_size must be >= 1"):
        DataParallelController("unused", data_parallel_size=0)
    with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
        DataParallelController("unused", tensor_parallel_size=0)
    with pytest.raises(ValueError, match="unknown load_balancer"):
        DataParallelController("unused", load_balancer="magic")
    with pytest.raises(ValueError, match="needs 4 GPUs, but only 2 are visible"):
        DataParallelController("unused", data_parallel_size=2, tensor_parallel_size=2)


def test_dp_attention_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="data_parallel_size > 1"):
        DataParallelController("unused", enable_dp_attention=True)
    with pytest.raises(ValueError, match="enable_expert_parallel=True"):
        DataParallelController(
            "unused", data_parallel_size=2, enable_dp_attention=True, use_cuda_graph=False
        )
    with pytest.raises(ValueError, match="does not support CUDA Graph"):
        DataParallelController(
            "unused",
            data_parallel_size=2,
            enable_dp_attention=True,
            enable_expert_parallel=True,
        )


def test_grid_spawns_every_rank_and_shares_one_queue_per_replica(controller_factory):
    controller = controller_factory(tensor_parallel_size=2, device="cpu")
    assert controller.world_size == 4
    assert [args[:3] for args in _FakeProcess.spawned] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ]
    queues = [args[-2] for args in _FakeProcess.spawned]
    assert queues[0] is queues[1]
    assert queues[2] is queues[3]
    assert queues[0] is not queues[2]


def test_round_robin_routes_typed_requests(controller_factory):
    controller = controller_factory(device="cpu")
    first = AddRequest("r1", (1, 2), _GREEDY, time.monotonic())
    second = AddRequest("r2", (3,), _GREEDY, time.monotonic())
    controller.send(first)
    controller.send(second)
    assert controller._command_queues[0].sent == [first]
    assert controller._command_queues[1].sent == [second]


def test_batched_requests_are_grouped_into_one_command_per_replica(controller_factory):
    controller = controller_factory(device="cpu")
    requests = tuple(
        AddRequest(f"r{index}", (index,), _GREEDY, time.monotonic()) for index in range(3)
    )

    controller.send(AddRequestBatch(requests))

    first = controller._command_queues[0].sent
    second = controller._command_queues[1].sent
    assert first == [AddRequestBatch((requests[0], requests[2]))]
    assert second == [AddRequestBatch((requests[1],))]


def test_blocking_generate_requests_only_the_final_scheduler_event(controller_factory):
    controller = controller_factory(device="cpu")
    controller._tokenizer = _FakeTokenizer()
    controller._event_queue.put(
        SchedulerEvents(outputs=(RequestEvent("batch-0", (111, 107), finish_reason="length"),))
    )

    outputs = controller.generate("hi", _GREEDY)

    batch = controller._command_queues[0].sent[0]
    assert isinstance(batch, AddRequestBatch)
    assert len(batch.requests) == 1
    assert batch.requests[0].stream is False
    assert outputs[0].text == "ok"


def test_logprob_generation_keeps_incremental_scheduler_events(controller_factory):
    controller = controller_factory(device="cpu")
    controller._tokenizer = _FakeTokenizer()
    controller._event_queue.put(
        SchedulerEvents(outputs=(RequestEvent("batch-0", (111, 107), finish_reason="length"),))
    )

    controller.generate("hi", SamplingParams(logprobs=0))

    batch = controller._command_queues[0].sent[0]
    assert isinstance(batch, AddRequestBatch)
    assert len(batch.requests) == 1
    assert batch.requests[0].stream is True


def test_dpa_request_wakes_every_idle_peer(controller_factory):
    controller = controller_factory(
        device="cpu",
        enable_dp_attention=True,
        enable_expert_parallel=True,
        use_cuda_graph=False,
    )
    command = AddRequest("r1", (1,), _GREEDY, time.monotonic())
    controller.send(command)
    assert controller._command_queues[0].sent == [command]
    assert controller._command_queues[1].sent == [WakeScheduler()]


def test_abort_returns_the_exact_request_charge(controller_factory):
    controller = controller_factory(device="cpu", load_balancer="total_tokens")
    controller.send(AddRequest("long", tuple(range(100)), _GREEDY, time.monotonic()))
    controller.send(AddRequest("short", (1,), _GREEDY, time.monotonic()))
    assert controller._policy.load == (100, 1)

    controller.send(AbortRequest("short"))
    assert controller._policy.load == (100, 0)
    controller.send(AbortRequest("long"))
    assert controller._policy.load == (0, 0)


def test_finished_event_releases_route_before_delivery(controller_factory):
    controller = controller_factory(device="cpu", load_balancer="total_requests")
    controller.send(AddRequest("r1", (1,), _GREEDY, time.monotonic()))
    controller._event_queue.put(
        SchedulerEvents(outputs=(RequestEvent("r1", (7,), finish_reason="eos"),))
    )
    event = controller.receive(timeout=1)
    assert isinstance(event, SchedulerEvents)
    assert controller._policy.load == (0, 0)
    assert "r1" not in controller._routes


def test_utility_queries_are_broadcast(controller_factory):
    controller = controller_factory(device="cpu")
    command = ShutdownScheduler()
    controller.send(command)
    assert all(channel.sent == [command] for channel in controller._command_queues)


def test_dead_rank_is_reported_instead_of_hanging(
    controller_factory, monkeypatch: pytest.MonkeyPatch
):
    from rapid_llm.engine import data_parallel as module

    monkeypatch.setattr(module, "_LIVENESS_POLL_S", 0.01)
    controller = controller_factory(device="cpu")
    controller._workers[0]._alive = False
    controller._workers[0].exitcode = -9
    with pytest.raises(RuntimeError, match="exited with codes"):
        controller.receive(timeout=1)


def test_shutdown_is_idempotent_and_broadcasts_typed_command(controller_factory):
    controller = controller_factory(device="cpu")
    controller.shutdown()
    controller.shutdown()
    assert all(
        any(isinstance(command, ShutdownScheduler) for command in channel.sent)
        for channel in controller._command_queues
    )


@pytest.fixture(scope="class")
def controller(model_dir: Path):
    with DataParallelController(
        model=str(model_dir),
        data_parallel_size=2,
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
    ) as controller:
        yield controller


@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
class TestTwoReplicas:
    def test_generate_returns_outputs_in_input_order(self, controller: DataParallelController):
        outputs = controller.generate(_PROMPTS, _GREEDY)
        assert [output.prompt for output in outputs] == _PROMPTS
        assert all(isinstance(output, RequestOutput) and output.text for output in outputs)

    def test_generate_is_repeatable(self, controller: DataParallelController):
        first = [output.text for output in controller.generate(_PROMPTS, _GREEDY)]
        second = [output.text for output in controller.generate(_PROMPTS, _GREEDY)]
        assert first == second

    def test_single_prompt_and_empty_batch(self, controller: DataParallelController):
        assert controller.generate([], _GREEDY) == []
        output = controller.generate(_PROMPTS[0], _GREEDY)
        assert len(output) == 1
        assert output[0].prompt == _PROMPTS[0]


@pytest.mark.gpu
@pytest.mark.weights
@requires_four_gpus
def test_two_replicas_with_two_tensor_parallel_ranks(model_dir: Path):
    with DataParallelController(
        model=str(model_dir),
        data_parallel_size=2,
        tensor_parallel_size=2,
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
    ) as controller:
        outputs = controller.generate(_PROMPTS[:2], _GREEDY)

    assert [output.prompt for output in outputs] == _PROMPTS[:2]
    assert all(output.text for output in outputs)
