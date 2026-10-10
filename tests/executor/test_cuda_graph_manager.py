"""Tests for the O13 lazy capture path in :class:`~rapid_llm.executor.cuda_graph`.

Pure CPU: ``CUDAGraphRunner`` is monkeypatched with a counter-equipped fake,
so these tests pin the *policy* — which shapes get captured when, what gets
blacklisted, what falls back to eager — without touching a GPU. The
correctness of an actually captured graph is covered by
``tests/compile/test_cuda_graph.py`` on CUDA machines.

Usage:
    pytest tests/executor/test_cuda_graph_manager.py
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from rapid_llm.executor import cuda_graph
from rapid_llm.executor.attention_metadata import AttentionMetadata
from rapid_llm.executor.cuda_graph import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQ_LEN_BUCKETS,
    LAZY_SEED_SHAPES,
    WORKSPACE_BYTES_PER_GRAPH,
    CUDAGraphManager,
    _GraphKey,
    estimate_capture_workspace,
)

_REPLAY_SENTINEL = object()


@pytest.fixture
def fake_runner_cls(monkeypatch):
    """Replace ``CUDAGraphRunner`` with a capture-counting fake.

    Fresh class per test, so ``instances`` / ``captures`` never leak across
    tests. ``fail_with`` lets a test arm an OOM for exactly one shape.
    """

    class FakeRunner:
        instances: ClassVar[list[FakeRunner]] = []
        captures = 0
        fail_with: BaseException | None = None

        def __init__(
            self,
            model,
            *,
            batch_size: int,
            seq_len_bucket: int,
            kv_buffer,
            b_req_tokens_table,
            device: str = "cuda",
            step=None,
        ) -> None:
            self.batch_size = batch_size
            self.seq_len_bucket = seq_len_bucket
            self.step = step
            self.instances.append(self)

        def capture(self, warmup_metadata=None) -> None:
            FakeRunner.captures += 1
            if FakeRunner.fail_with is not None:
                raise FakeRunner.fail_with

        def replay(self, *args, **kwargs):
            return _REPLAY_SENTINEL

    monkeypatch.setattr(cuda_graph, "CUDAGraphRunner", FakeRunner)
    # On-demand capture drains the stream first; irrelevant (and unavailable)
    # without a GPU.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return FakeRunner


@pytest.fixture
def manager() -> CUDAGraphManager:
    """Default-grid manager over dummy tensors; ``lazy`` set per test."""
    return CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
    )


def _decode_inputs(batch_size: int, max_seq_len: int = 100):
    """A decode-shaped step (seq_len == 1) landing in the smallest bucket."""
    ids = torch.zeros(batch_size, 1, dtype=torch.long)
    positions = torch.zeros(batch_size, 1, dtype=torch.long)
    attn = AttentionMetadata()
    attn.max_actual_seq_len = max_seq_len
    return ids, positions, attn


# --------------------------------------------------------------------------- #
# Workspace estimation
# --------------------------------------------------------------------------- #
def test_lazy_workspace_reserves_only_the_seed_pair():
    """Lazy mode must not pre-withhold the whole grid's memory."""
    eager = estimate_capture_workspace(4096)
    lazy = estimate_capture_workspace(4096, lazy=True)

    assert lazy == LAZY_SEED_SHAPES * WORKSPACE_BYTES_PER_GRAPH
    assert eager == (
        len(DEFAULT_BATCH_SIZES) * len(DEFAULT_SEQ_LEN_BUCKETS) * WORKSPACE_BYTES_PER_GRAPH
    )
    assert lazy < eager


def test_workspace_scales_with_buckets_below_max_seq_len():
    """Buckets beyond ``max_seq_len`` are unreachable and must not be counted."""
    assert estimate_capture_workspace(1024) == (
        len(DEFAULT_BATCH_SIZES) * 3 * WORKSPACE_BYTES_PER_GRAPH
    )  # buckets 256/512/1024


# --------------------------------------------------------------------------- #
# Grid membership
# --------------------------------------------------------------------------- #
def test_manager_starts_with_empty_state(manager):
    assert manager._runners == {}
    assert manager._failed == set()


@pytest.mark.parametrize(
    ("batch_sizes", "seq_len_buckets"),
    [
        ((), (256,)),
        ((0,), (256,)),
        ((1,), ()),
        ((1,), (-256,)),
    ],
)
def test_manager_rejects_invalid_capture_grids(batch_sizes, seq_len_buckets):
    with pytest.raises(ValueError, match="must be positive"):
        CUDAGraphManager(
            model=None,
            kv_buffer=[],
            b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
            device="cpu",
            batch_sizes=batch_sizes,
            seq_len_buckets=seq_len_buckets,
        )


def test_on_grid_classifies_keys(manager):
    assert manager._on_grid(_GraphKey(1, 256))
    assert manager._on_grid(_GraphKey(128, 4096))
    assert not manager._on_grid(_GraphKey(3, 256))  # batch off the grid
    assert not manager._on_grid(_GraphKey(1, 300))  # bucket off the grid


def test_capture_on_miss_skips_off_grid_shapes(manager, fake_runner_cls):
    """Shapes outside the grid stay eager — no runner, no blacklist entry."""
    assert manager._capture_on_miss(_GraphKey(3, 256), AttentionMetadata()) is None

    assert fake_runner_cls.instances == []
    assert manager._failed == set()


# --------------------------------------------------------------------------- #
# OOM blacklist
# --------------------------------------------------------------------------- #
def test_capture_on_miss_blacklists_oom_shapes(manager, fake_runner_cls):
    """An OOM'd shape is refused forever, not retried every step."""
    fake_runner_cls.fail_with = torch.cuda.OutOfMemoryError("simulated OOM")
    key = _GraphKey(2, 512)

    assert manager._capture_on_miss(key, AttentionMetadata()) is None
    assert key in manager._failed
    assert fake_runner_cls.captures == 1

    # Second attempt must short-circuit on the blacklist, not re-attempt.
    assert manager._capture_on_miss(key, AttentionMetadata()) is None
    assert fake_runner_cls.captures == 1


def test_blacklisted_shape_falls_back_to_eager_in_try_replay(fake_runner_cls):
    """The step that tripped the OOM still runs — try_replay returns None."""
    mgr = CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
        lazy=True,
    )
    fake_runner_cls.fail_with = torch.cuda.OutOfMemoryError("simulated OOM")
    ids, positions, attn = _decode_inputs(2)

    assert mgr.try_replay(ids, positions, attn) is None
    assert _GraphKey(2, 256) in mgr._failed


def test_lazy_capture_mismatch_stays_eager_on_every_tp_rank(monkeypatch, fake_runner_cls):
    """A peer OOM must retire this rank's successful graph before replay."""
    monkeypatch.setattr(cuda_graph, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cuda_graph, "tensor_model_parallel_ranks_agree", lambda _value: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    mgr = CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
        lazy=True,
    )
    key = _GraphKey(2, 512)

    assert mgr._capture_on_miss(key, AttentionMetadata()) is None
    assert key in mgr._failed
    assert key not in mgr._runners


# --------------------------------------------------------------------------- #
# Lazy vs eager fallback
# --------------------------------------------------------------------------- #
def test_try_replay_without_lazy_never_captures(manager, fake_runner_cls):
    """A non-lazy manager must return None on miss, not capture on demand."""
    ids, positions, attn = _decode_inputs(1)

    assert manager.try_replay(ids, positions, attn) is None
    assert fake_runner_cls.instances == []


def test_try_replay_lazy_captures_on_first_miss(fake_runner_cls):
    mgr = CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
        lazy=True,
    )
    ids, positions, attn = _decode_inputs(2)

    assert mgr.try_replay(ids, positions, attn) is _REPLAY_SENTINEL
    assert _GraphKey(2, 256) in mgr._runners

    # The cached runner serves the second call — no recapture.
    assert mgr.try_replay(ids, positions, attn) is _REPLAY_SENTINEL
    assert len(fake_runner_cls.instances) == 1


# --------------------------------------------------------------------------- #
# Seed capture
# --------------------------------------------------------------------------- #
def test_capture_seed_captures_exactly_two_shapes(manager, fake_runner_cls):
    """The seed pair brackets the grid: batch 1 short, max batch longest."""
    manager.capture_seed()

    assert set(manager._runners) == {_GraphKey(1, 256), _GraphKey(128, 4096)}
    assert len(fake_runner_cls.instances) == 2


def test_capture_seed_dedupes_a_degenerate_grid(fake_runner_cls):
    """A one-shape grid's seed pair collapses to a single capture."""
    mgr = CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
        batch_sizes=(2,),
        seq_len_buckets=(512,),
        lazy=True,
    )

    mgr.capture_seed()

    assert list(mgr._runners) == [_GraphKey(2, 512)]
    assert len(fake_runner_cls.instances) == 1


def test_seed_shapes_stay_replayable_after_capture(manager, fake_runner_cls):
    """Seeds are ordinary runners: try_replay must hit them without capturing."""
    manager.capture_seed()
    ids, positions, attn = _decode_inputs(1)  # (1, 256) — the small seed

    assert manager.try_replay(ids, positions, attn) is _REPLAY_SENTINEL
    assert len(fake_runner_cls.instances) == 2  # only the two seeds


# --------------------------------------------------------------------------- #
# Eligibility guards
# --------------------------------------------------------------------------- #
def test_try_replay_rejects_prefill_shapes(manager, fake_runner_cls):
    ids = torch.zeros(1, 7, dtype=torch.long)  # seq_len != 1
    positions = torch.zeros(1, 7, dtype=torch.long)

    assert manager.try_replay(ids, positions, AttentionMetadata()) is None
    assert fake_runner_cls.instances == []


def test_try_replay_rejects_contexts_beyond_the_largest_bucket(manager, fake_runner_cls):
    ids, positions, attn = _decode_inputs(1, max_seq_len=4097)

    assert manager.try_replay(ids, positions, attn) is None


# --------------------------------------------------------------------------- #
# Step factory: whose callable each runner records (TBO capture shape)
# --------------------------------------------------------------------------- #
def test_step_factory_shapes_each_runner_by_batch(fake_runner_cls):
    """The manager asks the factory per key; ``None`` means the plain forward.

    This is the TBO seam: the factory hands the interleave to batches that
    clear the policy floor and the plain forward to the rest, so both shapes
    can coexist in one captured grid.
    """
    tbo_step = object()

    def factory(batch_size: int):
        return tbo_step if batch_size >= 4 else None

    mgr = CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
        batch_sizes=(2, 4),
        seq_len_buckets=(256,),
        step_factory=factory,
    )
    mgr.capture_all()

    by_batch = {r.batch_size: r for r in fake_runner_cls.instances}
    assert by_batch[2].step is None, "below the floor: plain forward"
    assert by_batch[4].step is tbo_step, "clearing the floor: the TBO interleave"


def test_step_factory_not_consulted_when_absent(fake_runner_cls):
    """A plain manager records ``None`` steps — the pre-TBO behaviour."""
    mgr = CUDAGraphManager(
        model=None,
        kv_buffer=[],
        b_req_tokens_table=torch.zeros(4, 64, dtype=torch.int32),
        device="cpu",
        batch_sizes=(2,),
        seq_len_buckets=(256,),
    )
    mgr.capture_all()

    assert all(r.step is None for r in fake_runner_cls.instances)


# --------------------------------------------------------------------------- #
# Startup parity probe: the synthetic step the TP gate compares on
# --------------------------------------------------------------------------- #
def _probe_runner(batch_size: int, rows: int) -> cuda_graph.CUDAGraphRunner:
    """A real runner over a startup-state cache: the table maps nothing yet."""
    return cuda_graph.CUDAGraphRunner(
        model=None,
        batch_size=batch_size,
        seq_len_bucket=256,
        kv_buffer=[torch.zeros(rows, 2, 8)],
        b_req_tokens_table=torch.zeros(batch_size + 2, 300, dtype=torch.int32),
        device="cpu",
    )


@pytest.mark.parametrize(("batch_size", "rows"), [(8, 4096), (8, 9)], ids=["roomy", "one-row-each"])
def test_probe_rows_write_disjoint_cache_rows_on_an_empty_table(batch_size, rows):
    """Regression: an all-zero table aimed every probe row's write at row 0.

    That duplicate-index scatter has a run-dependent winner, so two eager passes
    disagreed by whole logits and the TP gate retired every graph with
    ``batch > 1`` on every model. Each row must now write its own row, at the end
    of its own read path, and never the null row.
    """
    runner = _probe_runner(batch_size, rows)
    runner._fill_probe_inputs(vocab_size=1000)

    info = runner.atten_info
    writes = info.cur_select_index.tolist()
    assert len(set(writes)) == batch_size, f"probe rows share a write slot: {writes}"
    assert 0 not in writes and max(writes) < rows
    length = int(info.b_seq_len[0])
    assert writes == info.b_req_tokens_table[info.b_req_idx, length - 1].tolist()


def test_probe_reads_no_unwritten_cache_memory():
    """Regression: the probe read rows the cache had never written.

    The KV buffer is ``torch.empty``; with the probe rows' earlier history left as
    found, a NaN bit pattern fed both forwards and the gate retired every graph
    (intermittent, nvfp4 TP=2). Every row a probe row reads must be zero.
    """
    runner = _probe_runner(batch_size=4, rows=4096)
    runner.atten_info.kv_buffer[0].fill_(float("nan"))
    runner._fill_probe_inputs(vocab_size=1000)

    info = runner.atten_info
    length = int(info.b_seq_len[0])
    read = info.b_req_tokens_table[info.b_req_idx, :length].flatten().long()
    assert torch.equal(info.kv_buffer[0][read], torch.zeros_like(info.kv_buffer[0][read]))


def test_parity_probe_leaves_the_token_table_as_it_found_it():
    """The probe borrows table rows; the live engine must get them back unchanged."""
    runner = _probe_runner(batch_size=4, rows=4096)
    table = runner.atten_info.b_req_tokens_table
    table[1, :3] = torch.tensor([7, 8, 9], dtype=table.dtype)
    before = table.clone()

    class _Replayed:
        def replay(self) -> None:
            pass

    runner._step = lambda ids, positions, info: torch.zeros(ids.shape[0], 1000)
    runner._graph, runner._output = _Replayed(), torch.zeros(4, 1000)

    assert runner.parity_error(vocab_size=1000) == 0.0
    assert torch.equal(table, before)
