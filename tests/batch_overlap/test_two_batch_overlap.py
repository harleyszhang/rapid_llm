"""L2 two-batch overlap: the split, the policy, and the overlap claim.

CPU tests pin the policy predicate and the row partition (both halves padded to
one shared length, metadata narrowed, KV write rows disjoint). The two-rank NCCL payloads pin
the two claims that justify L2 at all: a decode step run through
``forward_tbo`` produces the eager path's logits (parity), and its
all-reduces overlap the *other* half's compute on the device clock
(timeline evidence). A capture payload then pins the graph form: a runner
that recorded the interleave replays the eager TBO logits, and an
engine-level run with graphs and TBO both on keeps the baseline greedy
stream. Engine-level tests pin the whole worker integration — greedy token
streams identical with TBO on and off — on a dense (Qwen3-0.6B) and a MoE
(DeepSeek-V2-Lite) checkpoint.

Usage:
    pytest tests/batch_overlap/test_two_batch_overlap.py

The NCCL payloads and the engine-level tests need two CUDA devices. The dense
claims run on Qwen3-0.6B and the MoE seam claim on DeepSeek-V2-Lite, both under
``my_weight/`` and both overridable (``RAPID_LLM_TEST_QWEN3_DIR`` /
``RAPID_LLM_TEST_DSV2_DIR``); a machine without the checkpoint reports the claim
UNVERIFIED rather than passing it against a stand-in.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from functools import partial
from pathlib import Path

import pytest
import torch

from rapid_llm.batch_overlap.comm_overlap import CommStreamPool
from rapid_llm.batch_overlap.two_batch_overlap import (
    TBO_ENV,
    TBO_MIN_ROWS_ENV,
    TboPolicy,
    TboSplitter,
    reset_tbo_policy,
    tbo_policy,
)
from rapid_llm.executor.attention_metadata import AttentionMetadata
from rapid_llm.executor.cuda_graph import _GraphKey
from rapid_llm.executor.model_runner import ModelRunner
from rapid_llm.executor.slot_batch import SlotBatch
from tests.conftest import checkpoint_problem
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

ROOT = Path(__file__).resolve().parent.parent.parent

#: The dense checkpoint the payloads and the dense engine arms run, and the MoE
#: checkpoint the seam claim needs — named rather than the shared ``model_dir``:
#: the MoE claim would be vacuous against a dense stand-in, and the shared
#: default is whichever checkpoint a machine happens to hold. Each has an
#: override, so a machine that keeps weights outside ``my_weight/`` still runs
#: the claims instead of skipping them.
QWEN = "my_weight/Qwen3-0.6B"
QWEN_ENV = "RAPID_LLM_TEST_QWEN3_DIR"
DSV2L = "my_weight/DeepSeek-V2-Lite"
DSV2_ENV = "RAPID_LLM_TEST_DSV2_DIR"

ROWS = 8


def _checkpoint(reference: str, env_var: str) -> str:
    """The named checkpoint, or a visible xfail naming what this machine lacks.

    xfail, not skip: without its checkpoint a claim is UNVERIFIED, and that
    must stay visible rather than read as green. No ``weights`` mark either —
    that gate answers about one shared checkpoint, and these claims are
    checked per architecture.
    """
    path = Path(os.environ.get(env_var, reference))
    if not path.is_absolute():
        path = ROOT / path
    problem = checkpoint_problem(path)
    if problem:
        pytest.xfail(f"UNVERIFIED: {problem}")
    return str(path)


@pytest.fixture(autouse=True)
def _fresh_tbo_policy():
    """Every test reads the policy from the env it set, never the last test's cache."""
    reset_tbo_policy()
    yield
    reset_tbo_policy()


# --------------------------------------------------------------------------- #
# Policy parsing and activation predicate (CPU)
# --------------------------------------------------------------------------- #
def test_tbo_policy_is_off_by_default(monkeypatch):
    monkeypatch.delenv(TBO_ENV, raising=False)
    assert not TboPolicy.from_env().enabled
    for raw in ("0", "false", "off", "OFF"):
        monkeypatch.setenv(TBO_ENV, raw)
        assert not TboPolicy.from_env().enabled


def test_tbo_policy_accepts_the_on_spellings_and_parameters(monkeypatch):
    for raw in ("1", "tbo", "l2", "on"):
        monkeypatch.setenv(TBO_ENV, raw)
        assert TboPolicy.from_env().enabled
    monkeypatch.setenv(TBO_ENV, "1")
    monkeypatch.setenv(TBO_MIN_ROWS_ENV, "16")
    assert TboPolicy.from_env().min_rows == 16


def test_tbo_policy_min_rows_cannot_drop_below_two(monkeypatch):
    monkeypatch.setenv(TBO_ENV, "1")
    monkeypatch.setenv(TBO_MIN_ROWS_ENV, "1")
    assert TboPolicy.from_env().min_rows == 2, "a split needs two halves"


def test_tbo_policy_cache_is_read_once_per_process(monkeypatch):
    monkeypatch.setenv(TBO_ENV, "1")
    assert tbo_policy().enabled
    monkeypatch.setenv(TBO_ENV, "0")
    assert tbo_policy().enabled, "a cached policy must not re-read the env"
    reset_tbo_policy()
    assert not tbo_policy().enabled


def test_tbo_active_needs_peers_rows_and_no_graph():
    on = TboPolicy(enabled=True, min_rows=8)
    assert on.active(world_size=2, rows=8, graph_active=False)
    assert not on.active(world_size=1, rows=8, graph_active=False), "no peer to hide"
    assert not on.active(world_size=2, rows=7, graph_active=False), "below min_rows"
    assert not on.active(world_size=2, rows=8, graph_active=True), "graphs keep the step"
    assert not TboPolicy(enabled=False, min_rows=8).active(world_size=2, rows=8, graph_active=False)


def test_tbo_capture_eligible_needs_peers_and_the_floor():
    """The capture-time twin of :meth:`TboPolicy.active` — no ``graph_active``.

    A graph either is the interleave or is not, decided once per batch size,
    so the only inputs are the peers and the floor.
    """
    on = TboPolicy(enabled=True, min_rows=8)
    assert on.capture_eligible(world_size=2, batch=8)
    assert not on.capture_eligible(world_size=1, batch=8), "single rank: nothing to overlap"
    assert not on.capture_eligible(world_size=2, batch=7), "below min_rows: plain shape"
    assert not TboPolicy(enabled=False, min_rows=8).capture_eligible(world_size=2, batch=8)


# --------------------------------------------------------------------------- #
# The split: rows, views and disjoint KV write rows (CPU)
# --------------------------------------------------------------------------- #
def _cpu_metadata(rows: int) -> AttentionMetadata:
    """A metadata object shaped like one begin_decode installed."""
    return AttentionMetadata(
        kv_buffer=[torch.zeros(1024, 2, 128)],
        cur_select_index=torch.arange(rows) * 64,
        b_req_tokens_table=torch.zeros(64, 128, dtype=torch.int32),
        b_req_idx=torch.arange(rows),
        b_seq_len=torch.arange(2, rows + 2),
        max_actual_seq_len=rows + 1,
        is_prefill=False,
    )


def test_splitter_gives_both_halves_the_same_padded_length():
    """Equal-length micro-batches, sglang's ``tbo_padded_len`` discipline."""
    for rows in (2, 8, 9, 16):
        ids = torch.arange(rows).view(rows, 1)
        pos = torch.arange(100, 100 + rows).view(rows, 1)
        meta = _cpu_metadata(rows)
        a, b = TboSplitter().split(ids, pos, meta)

        padded = (rows + 1) // 2
        assert a.input_ids.shape[0] == b.input_ids.shape[0] == padded
        assert a.num_rows == rows // 2 and b.num_rows == rows - rows // 2
        assert a.num_rows + b.num_rows == rows, "padding must not invent or drop a request"

        # The real rows are the step's rows, in batch order.
        assert torch.equal(a.input_ids[: a.num_rows].flatten(), ids[: a.num_rows].flatten())
        assert torch.equal(b.input_ids[: b.num_rows].flatten(), ids[a.num_rows :].flatten())

        # Padding repeats the half's last real row: the duplicate attends to the
        # same request and writes the same slot, and its logits are dropped.
        if a.num_rows < padded:
            extra = padded - a.num_rows
            assert torch.equal(
                a.input_ids[a.num_rows :],
                a.input_ids[a.num_rows - 1 : a.num_rows].expand(extra, 1),
            )
            assert torch.equal(
                a.atten_info.cur_select_index[a.num_rows :],
                a.atten_info.cur_select_index[a.num_rows - 1 : a.num_rows].expand(extra),
            )

        # An even batch needs no padding, so the split stays a pure view.
        if rows % 2 == 0:
            for half in (a, b):
                assert half.input_ids.untyped_storage().data_ptr() == (
                    ids.untyped_storage().data_ptr()
                )

        # Metadata narrows to the same rows and shares the cache and the table.
        for half, start in ((a, 0), (b, a.num_rows)):
            stop = start + half.num_rows
            assert half.atten_info.kv_buffer is meta.kv_buffer
            assert half.atten_info.b_req_tokens_table is meta.b_req_tokens_table
            assert torch.equal(
                half.atten_info.b_req_idx[: half.num_rows], meta.b_req_idx[start:stop]
            )
            assert torch.equal(
                half.atten_info.b_seq_len[: half.num_rows], meta.b_seq_len[start:stop]
            )
            assert half.atten_info.max_actual_seq_len == meta.max_actual_seq_len
            assert not half.atten_info.is_prefill
            assert half.atten_info.b_start_loc is None


def test_splitter_write_rows_are_disjoint_and_cover_the_step():
    """The halves write disjoint KV rows — the split never aliases a cache row.

    Padding repeats a row the half already owns, so disjointness holds for an
    odd batch too: the duplicate lands on its own half's last slot.
    """
    for rows in (2, 8, 9):
        meta = _cpu_metadata(rows)
        a, b = TboSplitter().split(torch.zeros(rows, 1), torch.zeros(rows, 1), meta)
        rows_a = set(a.atten_info.cur_select_index.tolist())
        rows_b = set(b.atten_info.cur_select_index.tolist())
        assert not rows_a & rows_b, "a cache row written by both halves would corrupt K/V"
        assert rows_a | rows_b == set(meta.cur_select_index.tolist())


def test_splitter_refuses_single_row():
    with pytest.raises(ValueError, match=">= 2 rows"):
        TboSplitter().split(torch.zeros(1, 1), torch.zeros(1, 1), _cpu_metadata(1))


# --------------------------------------------------------------------------- #
# Two-rank NCCL payloads: parity and the overlap claim (Qwen3-0.6B)
# --------------------------------------------------------------------------- #
def _decode_step(runner: ModelRunner, slot_batch: SlotBatch, slots, seq_lens, token: int):
    """Install one decode step's metadata and return its prepared inputs."""
    slot_batch.begin_decode(slots, seq_lens)
    ids = torch.full((len(slots), 1), token, dtype=torch.long, device=runner.device)
    positions = slot_batch.seq_lens.view(-1, 1) - 1
    return ids, positions


def _payload_parity_and_overlap(rank: int, model: str) -> dict:
    """One rank of the parity + overlap payload over a real 2-shard checkpoint."""
    os.environ[TBO_ENV] = "1"
    os.environ[TBO_MIN_ROWS_ENV] = "2"
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "1"
    reset_tbo_policy()
    CommStreamPool.reset()
    device = torch.device("cuda", rank)

    runner = ModelRunner.build(model, max_seq_len=128, max_gpu_num_blocks=4096)
    slot_batch = SlotBatch(runner)

    # A stable fake running set: slot i's position p lives at cache row i*64+p,
    # every row of the cache carries a constant, and the table says so.
    slots = list(range(ROWS))
    seq_lens = [5, 9, 13, 7, 11, 15, 19, 23]
    table = runner.atten_info.b_req_tokens_table
    for slot in slots:
        table[slot, :64] = torch.arange(64, dtype=table.dtype, device=device) + slot * 64
    for layer_buf in runner.atten_info.kv_buffer:
        layer_buf.fill_(0.25)

    ids, positions = _decode_step(runner, slot_batch, slots, seq_lens, token=1000)
    with torch.no_grad():
        eager = runner.forward(ids, positions, None)
        overlapped = runner.forward_tbo(ids, positions)
    torch.cuda.synchronize()
    assert overlapped.shape == eager.shape, "TBO returns the decode step's logits shape"
    assert torch.equal(eager[:, -1, :].argmax(-1), overlapped[:, -1, :].argmax(-1)), (
        "row-wise argmax must survive the interleaving"
    )
    assert torch.allclose(eager.float(), overlapped.float(), rtol=5e-2, atol=5e-2), (
        "the halves' reductions are fenced before their results are read"
    )

    # Timeline evidence: A's deferred all-reduce must meet B's compute.
    CommStreamPool.reset()  # a fresh pool carries a fresh, empty timeline
    for step in range(3):  # a few steps so regions accumulate on the clock
        ids, positions = _decode_step(
            runner, slot_batch, slots, [length + 1 for length in seq_lens], token=2000 + step
        )
        with torch.no_grad():
            runner.forward_tbo(ids, positions)
        seq_lens = [length + 1 for length in seq_lens]
    torch.cuda.synchronize()

    records = CommStreamPool.for_device(device).timeline.collect()
    comm = [r for r in records if r.stream == "comm"]
    b_half = [r for r in records if r.name.endswith(".b")]
    assert comm and b_half, f"expected comm and half-B regions, saw {len(records)} records"
    overlaps = [
        min(reduce.end_ms, seg.end_ms) - max(reduce.start_ms, seg.start_ms)
        for reduce in comm
        for seg in b_half
    ]
    best = max(overlaps)
    assert best > 0.0, (
        f"no all-reduce overlapped the trailing half's compute "
        f"({len(comm)} comm regions, {len(b_half)} half-B segments)"
    )
    return {"argmax": True, "overlap_ms": round(best, 3), "comm_regions": len(comm)}


@needs_gpus(2)
@pytest.mark.slow
def test_forward_tbo_matches_eager_and_overlaps_on_two_ranks():
    model = _checkpoint(QWEN, QWEN_ENV)
    results = run_on_tp_ranks(partial(_payload_parity_and_overlap, model=model), tp_size=2)
    assert all(r["argmax"] for r in results)
    assert all(r["overlap_ms"] > 0.0 for r in results)


# --------------------------------------------------------------------------- #
# Two-rank NCCL payload: the TBO interleave captured inside a CUDA graph
# --------------------------------------------------------------------------- #
def _payload_graph_tbo_replay_parity(rank: int, model: str) -> dict:
    """One rank: capture the TBO interleave as a graph, replay it, compare.

    Capture runs before the eager pass, the way the engine orders them:
    startup captures on an empty cache, and the warmup forwards inside
    ``capture()`` write throwaway K/V at the persistent surface's rows (all
    zeros — see ``CUDAGraphRunner``). Capturing after the eager pass would
    let those warmup writes dirty history the eager side had read clean, and
    the comparison would pin the test's ordering, not the graph. With capture
    first, both the eager interleave and the replay read the same post-warmup
    KV state, and each rewrites the current token's rows before attending.
    """
    os.environ[TBO_ENV] = "1"
    os.environ[TBO_MIN_ROWS_ENV] = "2"
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "0"
    reset_tbo_policy()
    CommStreamPool.reset()

    runner = ModelRunner.build(model, max_seq_len=128, max_gpu_num_blocks=4096, use_cuda_graph=True)
    slot_batch = SlotBatch(runner)

    slots = list(range(ROWS))
    seq_lens = [5, 9, 13, 7, 11, 15, 19, 23]
    table = runner.atten_info.b_req_tokens_table
    for slot in slots:
        table[slot, :64] = torch.arange(64, dtype=table.dtype, device=f"cuda:{rank}") + slot * 64
    for layer_buf in runner.atten_info.kv_buffer:
        layer_buf.fill_(0.25)

    runner.enable_cuda_graph(batch_sizes=(ROWS,), seq_len_buckets=(64,), tbo=True)
    manager = runner._graph_manager
    key = _GraphKey(ROWS, 64)
    assert manager is not None and key in manager._runners, "the shape must have captured"
    assert manager._runners[key]._step is not runner.model, "the graph must hold the TBO step"

    ids, positions = _decode_step(runner, slot_batch, slots, seq_lens, token=1000)
    with torch.no_grad():
        eager_tbo = runner.forward_tbo(ids, positions)
    torch.cuda.synchronize()

    with torch.no_grad():
        replayed = runner.forward(ids, positions, None)
    torch.cuda.synchronize()
    assert replayed is not None

    assert torch.equal(eager_tbo[:, -1, :].argmax(-1), replayed[:, -1, :].argmax(-1)), (
        "row-wise argmax must survive the capture"
    )
    assert torch.allclose(eager_tbo.float(), replayed.float(), rtol=1e-4, atol=1e-4), (
        "replay must reproduce the interleave's logits, not a plain-forward's"
    )
    return {"replayed": True}


@needs_gpus(2)
@pytest.mark.slow
def test_tbo_graph_replay_matches_eager_tbo_on_two_ranks():
    """A captured TBO graph replays the interleave faithfully.

    This is the go/no-go of TBO-under-graphs: the deferred all-reduces ride
    the comm stream with event fences on both sides, so capture records them
    as ordinary cross-stream dependencies and replay keeps the overlap —
    and numerically the recorded kernel sequence is the eager interleave's.
    """
    model = _checkpoint(QWEN, QWEN_ENV)
    results = run_on_tp_ranks(partial(_payload_graph_tbo_replay_parity, model=model), tp_size=2)
    assert all(r["replayed"] for r in results)


# --------------------------------------------------------------------------- #
# Engine-level greedy parity: dense and MoE checkpoints
# --------------------------------------------------------------------------- #
#: The prompts every arm runs, so the arms cannot drift apart: the comparison
#: is only meaningful over identical inputs.
_PROMPTS = [
    "The capital of France is",
    "One two three four five",
    "Water boils at a temperature of",
    "The first president of the United States was",
]


def _boot(model_dir: str, *, use_cuda_graph: bool):
    """Boot a two-rank engine for one arm, inside the arm's own process.

    ``MASTER_PORT`` is cleared first so the fresh port ``launch_tensor_parallel``
    picks actually reaches the rendezvous: ``init_parallel`` resolves the port
    through that variable with ``setdefault``, so an inherited value hijacks the
    boot -- ``free_port()``'s result is ignored and the group binds a port this
    process has no business with.
    """
    os.environ.pop("MASTER_PORT", None)
    from rapid_llm.engine import ContinuousBatchingEngine

    return ContinuousBatchingEngine.from_pretrained(
        model_dir,
        tensor_parallel_size=2,
        max_seq_len=1024,
        max_num_seqs=8,
        use_cuda_graph=use_cuda_graph,
    )


def _greedy_stream(model_dir: str, *, tbo_on: bool, use_cuda_graph: bool) -> list[list[int]]:
    """One arm of the end-to-end comparison: the shared greedy stream.

    Same prompts and sampling on every arm by construction, differing only in
    the TBO switch and whether decode replays a captured interleave
    (``min_rows=2`` makes every captured batch size in the engine's grid,
    clamped to ``max_num_seqs=8``, record the interleave, while the batch-1
    graph keeps the plain shape -- both paths in one run). The eager arms are
    the comparison's point: the policy stands down when a graph is active, so
    only eager decode exercises the interleave.
    """
    os.environ[TBO_ENV] = "1" if tbo_on else "0"
    os.environ[TBO_MIN_ROWS_ENV] = "2"
    from rapid_llm import SamplingParams

    engine = _boot(model_dir, use_cuda_graph=use_cuda_graph)
    try:
        params = SamplingParams(max_gen_len=12, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        requests = [engine.add_request(prompt, params) for prompt in _PROMPTS]
        while engine.has_unfinished_requests():
            engine.step()
        return [list(request.output_token_ids) for request in requests]
    finally:
        engine.shutdown()


def _arm_in_process(model_dir: str, tbo_on: bool, use_cuda_graph: bool, results: mp.Queue) -> None:
    """Child entry of :func:`_run_arm`: run the arm, report streams or traceback."""
    try:
        streams = _greedy_stream(model_dir, tbo_on=tbo_on, use_cuda_graph=use_cuda_graph)
    except BaseException:  # travels as text: a child's exception is invisible to pytest
        results.put((None, traceback.format_exc()))
    else:
        results.put((streams, None))


def _run_arm(model_dir: str, *, tbo_on: bool, use_cuda_graph: bool) -> list[list[int]]:
    """Run one comparison arm in a fresh process and return its greedy streams.

    A process per arm, not two arms in the pytest process: the engine's second
    in-process boot is not safe to lean on -- ``init_parallel`` freezes the
    rendezvous port in ``MASTER_PORT`` with ``setdefault`` on the first boot,
    and the engine's teardown does not reliably release the group or its store,
    which has failed the second arm with EADDRINUSE and left a grid standing
    for later tests. A process that exits takes the group, the store, the port
    and the env with it, so the arms cannot leak into each other -- or into
    the rest of the suite.
    """
    context = mp.get_context("spawn")
    results: mp.Queue = context.Queue()
    arm = context.Process(
        target=_arm_in_process,
        args=(model_dir, tbo_on, use_cuda_graph, results),
        name="rapid-llm-tbo-arm",
        daemon=True,
    )
    arm.start()
    arm.join(timeout=600)
    if arm.is_alive():
        arm.terminate()
        arm.join(timeout=30)
        raise TimeoutError(f"tbo_on={tbo_on}, use_cuda_graph={use_cuda_graph}: no result in 600s")
    if arm.exitcode != 0:
        raise AssertionError(
            f"tbo_on={tbo_on}, use_cuda_graph={use_cuda_graph}: arm exited {arm.exitcode}"
        )
    streams, error = results.get(timeout=30)
    if error is not None:
        raise AssertionError(
            f"tbo_on={tbo_on}, use_cuda_graph={use_cuda_graph} arm failed:\n{error}"
        )
    return streams


@needs_gpus(2)
@pytest.mark.slow
def test_tbo_greedy_tokens_match_baseline_dense():
    model = _checkpoint(QWEN, QWEN_ENV)
    baseline = _run_arm(model, tbo_on=False, use_cuda_graph=False)
    overlapped = _run_arm(model, tbo_on=True, use_cuda_graph=False)
    assert baseline == overlapped, "TBO must not change the greedy token stream"


@needs_gpus(2)
@pytest.mark.slow
def test_tbo_greedy_tokens_match_baseline_moe():
    model = _checkpoint(DSV2L, DSV2_ENV)
    baseline = _run_arm(model, tbo_on=False, use_cuda_graph=False)
    overlapped = _run_arm(model, tbo_on=True, use_cuda_graph=False)
    assert baseline == overlapped, "the MoE stack interleaves through the same seam"


@needs_gpus(2)
@pytest.mark.slow
def test_tbo_graph_greedy_tokens_match_baseline_dense():
    """The whole pipeline with the interleave captured: graphs on, TBO on.

    The engine's capture asks the policy which batches record the interleave
    (see :meth:`ModelRunner.enable_cuda_graph`), the eager policy stands down
    under graphs, and every decode step replays a captured TBO sequence. The
    greedy stream must still equal the eager no-overlap baseline — the replay
    is bit-for-bit the eager interleave, which the eager parity test already
    pinned to the baseline.
    """
    model = _checkpoint(QWEN, QWEN_ENV)
    baseline = _run_arm(model, tbo_on=False, use_cuda_graph=False)
    graphed = _run_arm(model, tbo_on=True, use_cuda_graph=True)
    assert baseline == graphed, "a captured interleave must not change the greedy stream"
