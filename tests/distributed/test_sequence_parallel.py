"""Tests for the sequence-parallel region and the boundaries it is bounded by.

Three layers, in the order a failure is cheapest to read:

* No process group: what :class:`~rapid_llm.distributed.sequence_parallel.SequenceParallelPass`
  marks and refuses, and every gate that keeps a region shut.
* A real two-rank ``gloo`` grid (no device needed): the entry/exit round trip,
  and each boundary against the all-reduce path it replaces. This is where the
  arithmetic claim lives — a ``g-bar`` must equal the all-reduce a
  ``RowParallelLinear`` would have done, sliced to this rank's rows.
* A real TP=2 engine on a checkpoint: the same tokens with the region off and on.
  The unit tests above would not notice the region being wired into the wrong
  end of a block, because both ends type-check.

Usage:
    pytest tests/distributed/test_sequence_parallel.py
"""

from __future__ import annotations

import queue as queue_module
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from rapid_llm import SamplingParams
from rapid_llm.distributed import parallel_state as ps
from rapid_llm.distributed.sequence_parallel import (
    G_ATTR,
    G_BAR_ATTR,
    SequenceParallelPass,
    SequenceParallelRegion,
    current_region,
    sequence_parallel_enabled,
    sequence_parallel_min_tokens,
    sequence_parallel_region,
    sp_active,
)
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks


@pytest.fixture(autouse=True)
def _reset_grid():
    """Restore the world of one after each test (the grid is module-level state)."""
    yield
    ps.destroy_parallel()


# --------------------------------------------------------------------------- #
# Model shapes the pass is asked to recognise
# --------------------------------------------------------------------------- #
_HIDDEN, _HEADS, _HEAD_DIM, _INTER = 32, 4, 8, 64


def _block(*, fused_qkv: bool = True, moe: bool = False) -> nn.Module:
    """A decoder block of the shape the region shards, or a near miss of it.

    Structural only: the pass recognises a block by the two-stage split and
    resolves the four projections by attribute path, so nothing here needs the
    real :class:`~rapid_llm.models.base.DecoderLayer`'s config or weights. The
    two switches build two of the constructions the pass has to refuse — MLA's
    unfused QKV, and an MoE block with no gate/up-plus-down pair.
    """
    from rapid_llm.modules.linear import (
        ColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )

    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv_proj = (
                QKVParallelLinear(_HIDDEN, _HEADS, _HEADS, _HEAD_DIM)
                if fused_qkv
                else ReplicatedLinear(_HIDDEN, _HIDDEN)
            )
            self.o_proj = RowParallelLinear(_HEADS * _HEAD_DIM, _HIDDEN)

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = ColumnParallelLinear(_HIDDEN, 2 * _INTER)
            self.down_proj = RowParallelLinear(_INTER, _HIDDEN)
            # An MoE block's shared expert owns a down_proj of its own that is
            # *not* a boundary; the pass must resolve by path, not by tree walk.
            self.shared_experts = Mlp() if moe else None

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attn()
            self.mlp = Mlp() if not moe else nn.Identity()

        def forward_attn_stage(self):  # pragma: no cover - structural marker
            pass

        def forward_mlp_stage(self):  # pragma: no cover - structural marker
            pass

    return Block()


def _marks(model: nn.Module) -> dict[str, list[str]]:
    """Every marked module by boundary attribute, keyed by qualified name."""
    return {
        attr: sorted(name for name, module in model.named_modules() if getattr(module, attr, False))
        for attr in (G_ATTR, G_BAR_ATTR)
    }


# --------------------------------------------------------------------------- #
# The pass: what it marks, and what it refuses
# --------------------------------------------------------------------------- #
def test_pass_disabled_by_default(monkeypatch):
    """The region trades no bytes for activation memory, so it is opt-in."""
    monkeypatch.delenv("RAPID_LLM_SEQUENCE_PARALLEL", raising=False)
    assert not sequence_parallel_enabled()
    assert SequenceParallelPass().enabled is False


def test_pass_disabled_by_env(monkeypatch):
    """``RAPID_LLM_SEQUENCE_PARALLEL=0`` switches the pass off."""
    monkeypatch.setenv("RAPID_LLM_SEQUENCE_PARALLEL", "0")
    assert not sequence_parallel_enabled()
    assert SequenceParallelPass().enabled is False


def test_pass_is_noop_without_tp(monkeypatch):
    """A world of one has no peers to shard tokens across, so nothing is marked."""
    monkeypatch.setenv("RAPID_LLM_SEQUENCE_PARALLEL", "1")
    model = nn.ModuleList([_block()])
    passer = SequenceParallelPass()
    assert passer.apply(model) == 0
    assert passer.refusal == "tensor-parallel world size is 1"
    assert _marks(model) == {G_ATTR: [], G_BAR_ATTR: []}


def test_pass_clears_stale_marks_when_disabled():
    """A re-run leaves no mark ``matched`` does not report.

    ``apply`` is idempotent: marks from an earlier run (a different grid, or
    before the pass was switched off) are cleared first, so the module tree and
    the pass's own record cannot disagree about which boundaries are live. Both
    attributes are cleared, not just the one this run would have set.
    """
    model = nn.ModuleList([_block()])
    setattr(model[0].self_attn.qkv_proj, G_ATTR, True)  # as a TP>1 run would have left it
    setattr(model[0].self_attn.o_proj, G_BAR_ATTR, True)

    passer = SequenceParallelPass(enabled=False)
    assert passer.apply(model) == 0
    assert passer.matched == []
    assert _marks(model) == {G_ATTR: [], G_BAR_ATTR: []}


def _pass_marks_payload(rank: int) -> dict[str, Any]:
    """Two blocks under a real grid: which projections came out marked."""
    model = nn.ModuleList([_block() for _ in range(2)])
    count = SequenceParallelPass(enabled=True).apply(model)
    return {"count": count, "marks": _marks(model)}


def test_pass_marks_all_four_boundaries_of_every_block():
    """Both seams of both blocks, and nothing else -- notably no shared expert."""
    for result in run_on_tp_ranks(_pass_marks_payload, tp_size=2, backend="gloo"):
        assert result["count"] == 8, "four boundaries per block, two blocks"
        assert result["marks"] == {
            G_ATTR: [
                "0.mlp.gate_up_proj",
                "0.self_attn.qkv_proj",
                "1.mlp.gate_up_proj",
                "1.self_attn.qkv_proj",
            ],
            G_BAR_ATTR: [
                "0.mlp.down_proj",
                "0.self_attn.o_proj",
                "1.mlp.down_proj",
                "1.self_attn.o_proj",
            ],
        }


def _pass_refuses_payload(rank: int) -> list[dict[str, Any]]:
    """Each unsupported shape: the count marked and the reason given."""
    from rapid_llm.modules.linear import RowParallelLinear

    raw = _block()
    raw.self_attn.o_proj = RowParallelLinear(_HEADS * _HEAD_DIM, _HIDDEN, reduce_results=False)
    cases = [
        ("no block", nn.ModuleList([nn.Linear(4, 4)])),
        ("unfused qkv", nn.ModuleList([_block(fused_qkv=False)])),
        ("moe", nn.ModuleList([_block(moe=True)])),
        # One placeable block is not enough: the region spans the whole stack.
        ("mixed stack", nn.ModuleList([_block(), _block(fused_qkv=False)])),
        # A g-bar that hands its caller the raw partial cannot close the region.
        ("raw partial o_proj", nn.ModuleList([raw])),
    ]
    out = []
    for name, model in cases:
        passer = SequenceParallelPass(enabled=True)
        out.append({
            "case": name,
            "count": passer.apply(model),
            "refusal": passer.refusal,
            "marks": _marks(model),
        })
    return out


def test_pass_refuses_a_stack_it_cannot_place_entirely():
    """One block it cannot place leaves the *whole* stack on the all-reduce path.

    The region is a property of the stack: the rows the entry scatter took are
    only reassembled at the exit gather, so a block that handed back the full
    grid in the middle would feed its successor the wrong rows. Partial marking
    is therefore not a degraded mode, it is a wrong answer.
    """
    for result in run_on_tp_ranks(_pass_refuses_payload, tp_size=2, backend="gloo"):
        for case in result:
            assert case["count"] == 0, case["case"]
            assert case["refusal"], f"{case['case']}: refused without saying why"
            if case["case"] == "raw partial o_proj":
                assert "reduce_results" in case["refusal"], case["refusal"]
            assert case["marks"] == {G_ATTR: [], G_BAR_ATTR: []}, case["case"]


# --------------------------------------------------------------------------- #
# Region geometry and gating (no group needed: the gates are pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("num_tokens", "world_size", "padded", "local"),
    [(8, 2, 8, 4), (7, 2, 8, 4), (1, 4, 4, 1), (10, 4, 12, 3)],
)
def test_region_pads_to_a_whole_number_of_shards(num_tokens, world_size, padded, local):
    """An uneven token count pads rather than declining to shard.

    The padded rows are zeros that only per-row arithmetic ever touches, and the
    exit gather drops them -- which is why the region does not need the batch to
    have arrived at a multiple of the world size.
    """
    region = SequenceParallelRegion(num_tokens=num_tokens, world_size=world_size, rank=0)
    assert region.padded_tokens == padded
    assert region.local_tokens == local
    assert region.local_tokens * region.world_size == region.padded_tokens


def test_min_tokens_default_and_override(monkeypatch):
    """The floor is an env knob; its default keeps a decode step out."""
    monkeypatch.delenv("RAPID_LLM_SP_MIN_TOKENS", raising=False)
    assert sequence_parallel_min_tokens() == 1024
    monkeypatch.setenv("RAPID_LLM_SP_MIN_TOKENS", "16")
    assert sequence_parallel_min_tokens() == 16


def _open(num_tokens: int, *, marked: bool = True) -> bool:
    """Whether a region opens for these arguments, and closes again after."""
    with sequence_parallel_region(num_tokens, marked=marked) as region:
        opened = region is not None
        assert sp_active() is opened
        assert (current_region() is region) if opened else current_region() is None
    assert not sp_active(), "the region must not outlive its context"
    return opened


def _gates_payload(rank: int) -> dict[str, bool]:
    """Every gate, on a real two-rank grid so world size is not the one refusing."""
    import os

    from rapid_llm.batch_overlap.comm_overlap import (
        deferred_all_reduce,
        reset_comm_overlap_policy,
    )

    os.environ["RAPID_LLM_SP_MIN_TOKENS"] = "16"
    reset_comm_overlap_policy()
    out = {"at_floor": _open(16), "below_floor": _open(15), "unmarked": _open(64, marked=False)}

    # TBO owns the same collective and was asked for explicitly.
    with deferred_all_reduce(device="cpu"):
        out["under_tbo"] = _open(64)

    # So does L3, above its own row floor -- and below it, L3 is not active.
    os.environ["RAPID_LLM_COMM_OVERLAP"] = "1"
    os.environ["RAPID_LLM_L3_MIN_ROWS"] = "32"
    reset_comm_overlap_policy()
    out["under_l3"] = _open(64)
    out["below_l3_floor"] = _open(16)
    for name in ("RAPID_LLM_COMM_OVERLAP", "RAPID_LLM_L3_MIN_ROWS", "RAPID_LLM_SP_MIN_TOKENS"):
        os.environ.pop(name, None)
    reset_comm_overlap_policy()
    return out


def test_region_gates():
    """The region opens only when it is the one doing the communication."""
    for result in run_on_tp_ranks(_gates_payload, tp_size=2, backend="gloo"):
        assert result == {
            "at_floor": True,
            "below_floor": False,
            "unmarked": False,
            "under_tbo": False,
            "under_l3": False,
            "below_l3_floor": True,
        }


def test_region_stays_shut_without_a_grid(monkeypatch):
    """A world of one: no peers, so no region however many tokens the step has."""
    monkeypatch.setenv("RAPID_LLM_SP_MIN_TOKENS", "1")
    assert not _open(4096)


# --------------------------------------------------------------------------- #
# Entry and exit on a real grid: the round trip is the identity
# --------------------------------------------------------------------------- #
def _round_trip_payload(rank: int) -> list[dict[str, Any]]:
    """Scatter a known grid, then gather it back; report shapes and equality."""
    from rapid_llm.distributed.sequence_parallel import entry_scatter, exit_gather

    out = []
    for batch, seq in ((1, 8), (4, 3), (2, 5)):
        torch.manual_seed(11)  # the embedding output is replicated across ranks
        grid = torch.randn(batch, seq, _HIDDEN)
        region = SequenceParallelRegion(num_tokens=batch * seq, world_size=2, rank=rank)

        shard = entry_scatter(grid, region)
        back = exit_gather(shard, region, grid.shape)

        # The shard is this rank's contiguous slice of the flattened grid, up to
        # the padding the last rank may carry.
        flat = grid.reshape(-1, _HIDDEN)
        start = rank * region.local_tokens
        want = flat[start : min(start + region.local_tokens, flat.shape[0])]
        out.append({
            "shard_shape": tuple(shard.shape),
            "expected_shard": (1, region.local_tokens, _HIDDEN),
            "shard_is_slice": torch.equal(shard[0, : want.shape[0]], want),
            "padding_is_zero": bool((shard[0, want.shape[0] :] == 0).all()),
            "round_trip": torch.equal(back, grid),
        })
    return out


def test_entry_scatter_and_exit_gather_round_trip():
    """``exit_gather(entry_scatter(x))`` is ``x``, padding and all."""
    for rank, result in enumerate(run_on_tp_ranks(_round_trip_payload, tp_size=2, backend="gloo")):
        for case in result:
            assert case["shard_shape"] == case["expected_shard"], rank
            assert case["shard_is_slice"], f"rank {rank}: shard is not this rank's rows"
            assert case["padding_is_zero"], f"rank {rank}: padding is not zero"
            assert case["round_trip"], f"rank {rank}: round trip changed the grid"


# --------------------------------------------------------------------------- #
# The two boundaries against the all-reduce path they replace
# --------------------------------------------------------------------------- #
def _boundaries_payload(rank: int) -> list[dict[str, Any]]:
    """Each boundary vs its reference, for an even and an uneven token count."""
    import os

    from rapid_llm.distributed.parallel_state import tensor_model_parallel_all_reduce
    from rapid_llm.distributed.sequence_parallel import column_parallel_g, row_parallel_g_bar
    from rapid_llm.modules.linear import ColumnParallelLinear, RowParallelLinear

    # These shapes are the arithmetic, not a workload; the default floor exists
    # to keep small steps out of production, not out of a comparison.
    os.environ["RAPID_LLM_SP_MIN_TOKENS"] = "1"

    torch.manual_seed(3)  # the same seed on both ranks: each takes its own shard
    # ``torch.empty`` allocates the weights, so a comparison against an
    # uninitialised NaN is not a comparison at all -- fill them.
    g_layer = ColumnParallelLinear(_HIDDEN, 2 * _INTER)
    g_layer.weight.copy_(torch.randn(g_layer.weight.shape) * 0.05)
    g_bar_layer = RowParallelLinear(_INTER, _HIDDEN)
    g_bar_layer.weight.copy_(torch.randn(g_bar_layer.weight.shape) * 0.05)

    out = []
    for num_tokens in (8, 7):
        region = SequenceParallelRegion(num_tokens=num_tokens, world_size=2, rank=rank)
        lo = rank * region.local_tokens
        hi = lo + region.local_tokens

        torch.manual_seed(5)  # the residual stream is the same on both ranks
        stream = torch.randn(region.padded_tokens, _HIDDEN)
        stream[num_tokens:] = 0  # as the entry scatter left the padded rows
        shard = stream[lo:hi]

        torch.manual_seed(9 + rank)  # the row-parallel partial differs per rank
        full_rows = torch.randn(num_tokens, g_bar_layer.input_size)

        with sequence_parallel_region(num_tokens, marked=True) as opened:
            assert opened is not None
            # ``g``: gather to the full grid, then the GEMM -- which is what the
            # layer would have computed had it been handed the whole stream.
            got_g = column_parallel_g(g_layer, shard)
            # ``g-bar``: the all-reduce, sliced to this rank's rows.
            got_g_bar = row_parallel_g_bar(g_bar_layer, full_rows)

        want_g = g_layer.apply_linear(stream[:num_tokens])
        reduced = tensor_model_parallel_all_reduce(g_bar_layer.apply_linear(full_rows))
        padded = torch.cat((reduced, reduced.new_zeros(region.padded_tokens - num_tokens, _HIDDEN)))
        want_g_bar = padded[lo:hi]

        out.append({
            "num_tokens": num_tokens,
            "g_shape": tuple(got_g.shape),
            "g_expected_shape": (num_tokens, g_layer.output_size),
            "g_close": torch.allclose(got_g, want_g, rtol=1e-5, atol=1e-5),
            "g_max_diff": (got_g - want_g).abs().max().item(),
            "g_bar_shape": tuple(got_g_bar.shape),
            "g_bar_expected_shape": (region.local_tokens, _HIDDEN),
            "g_bar_close": torch.allclose(got_g_bar, want_g_bar, rtol=1e-5, atol=1e-5),
            "g_bar_max_diff": (got_g_bar - want_g_bar).abs().max().item(),
        })
    return out


def test_boundaries_match_the_all_reduce_path():
    """``g`` == gather-then-GEMM, ``g-bar`` == all-reduce sliced to this rank.

    The whole arithmetic claim, on a real grid: a region is only correct if a
    block computes what the all-reduce path would have, for its own rows. The
    uneven case is in here too, because the padding is what makes it work at all
    -- a reduce-scatter needs a row count that divides.
    """
    results = run_on_tp_ranks(_boundaries_payload, tp_size=2, backend="gloo")
    for rank, result in enumerate(results):
        for case in result:
            where = f"rank {rank}, {case['num_tokens']} tokens"
            assert case["g_shape"] == case["g_expected_shape"], where
            assert case["g_close"], f"{where}: g diverged by {case['g_max_diff']}"
            assert case["g_bar_shape"] == case["g_bar_expected_shape"], where
            assert case["g_bar_close"], f"{where}: g-bar diverged by {case['g_bar_max_diff']}"


def _uneven_last_rank_payload(rank: int) -> dict[str, Any]:
    """The last rank's shard is part padding; the gather still returns the grid."""
    from rapid_llm.distributed.sequence_parallel import entry_scatter, exit_gather

    torch.manual_seed(2)
    grid = torch.randn(1, 5, _HIDDEN)  # 5 tokens over 2 ranks: rank 1 holds a pad row
    region = SequenceParallelRegion(num_tokens=5, world_size=2, rank=rank)
    shard = entry_scatter(grid, region)
    return {
        "local_tokens": region.local_tokens,
        "real_rows": min(max(5 - rank * region.local_tokens, 0), region.local_tokens),
        "round_trip": torch.equal(exit_gather(shard, region, grid.shape), grid),
    }


def test_uneven_token_count_pads_the_last_rank():
    """5 tokens over 2 ranks: 3 each, one of rank 1's is padding."""
    results = run_on_tp_ranks(_uneven_last_rank_payload, tp_size=2, backend="gloo")
    assert [r["local_tokens"] for r in results] == [3, 3]
    assert [r["real_rows"] for r in results] == [3, 2]
    assert all(r["round_trip"] for r in results)


# --------------------------------------------------------------------------- #
# End to end on a real TP=2 engine: the region must not change the answer
# --------------------------------------------------------------------------- #
#: Prompt lengths chosen so some steps shard evenly over the two ranks and some
#: do not, so one run covers both the whole-shard and the padded geometry.
_E2E_PROMPTS = [
    "The capital of France is",
    "One two three four five",
    "Water boils at a temperature of",
    "The first president of the United States was",
]

_E2E_MAX_SEQ_LEN = 512
_E2E_MAX_GEN = 12

#: The token floor for this run. The default (1024) exists to keep small steps
#: out; a short-prompt checkpoint test would clear no region at all under it, so
#: the arm would compare the all-reduce path against itself.
_E2E_MIN_TOKENS = "1"

#: Loading a checkpoint, profiling a cache and rendezvousing two ranks. Generous
#: on purpose: it exists to turn a wedged rank into a failure instead of a hang.
_E2E_TIMEOUT_S = 600.0

#: Log-probability margin below which the runner-up is close enough that a
#: differently ordered sum may take the step either way -- the region
#: reassociates the reduction, so it is entitled to exactly those steps and no
#: others. Same bound and same reasoning as ``tests/distributed/test_tp_engine.py``,
#: which measures it between TP widths on this checkpoint class.
_TIE_GAP = 0.5

#: Fraction of tokens the two arms must agree on outright. Every divergence above
#: is licensed by a small margin; without a floor, a run that had become noise
#: would satisfy the parity check one coin flip at a time.
_MIN_AGREEING_FRACTION = 2 / 3


def _sp_engine_probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """One process: a TP=2 engine with the region as ``spec`` says, and its tokens.

    Spawned, so it takes picklable arguments only and imports the engine itself.
    The switches are set *here* rather than in the parent because this process
    goes on to spawn the follower rank, which inherits the environment: both
    ranks have to read the same values or they disagree about which collectives
    to call, which is a hang rather than a wrong answer.

    Reports the marked boundary count *and* how many regions actually opened.
    Without both a green test proves nothing -- a pass that marked nothing, or a
    step that never cleared the token floor, also leaves every token unchanged.
    """
    import os
    import traceback

    os.environ["RAPID_LLM_SEQUENCE_PARALLEL"] = spec["sp"]
    os.environ["RAPID_LLM_SP_MIN_TOKENS"] = spec["min_tokens"]
    try:
        from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
        from rapid_llm.models import base as model_base

        # Count the regions this rank actually opened, by wrapping the entry the
        # model calls. Patched on the module the call site resolves through, so
        # it counts rank 0's steps -- enough to prove the path was taken.
        opened = {"count": 0}
        real_scatter = model_base.entry_scatter

        def counting_scatter(hidden_states, region):
            opened["count"] += 1
            return real_scatter(hidden_states, region)

        model_base.entry_scatter = counting_scatter
        try:
            engine = ContinuousBatchingEngine.from_pretrained(
                model=spec["model"],
                device="cuda:0",
                max_seq_len=_E2E_MAX_SEQ_LEN,
                max_num_seqs=8,
                # Eager on both arms: a capture would fold a second variable into
                # every difference, and tests/distributed/test_tp_cuda_graph.py
                # already owns capture under TP.
                use_cuda_graph=False,
                tensor_parallel_size=2,
            )
            try:
                # ``logprobs=2`` is the instrument, not the subject: it reports the
                # runner-up at every step so a tie can be measured instead of assumed.
                params = SamplingParams(
                    temperature=0.0,
                    max_gen_len=_E2E_MAX_GEN,
                    repetition_penalty=1.0,
                    stop_on_repeat=False,
                    logprobs=2,
                )
                outputs = engine.generate(_E2E_PROMPTS, params)
                model = engine.engine.model_runner.model
                report: dict[str, Any] = {
                    "marked": len(model.sequence_parallel_pass.matched),
                    "layers": len(model.layers),
                    "regions": opened["count"],
                    "runs": [_steps(output.outputs[0]) for output in outputs],
                }
            finally:
                engine.shutdown()
        finally:
            model_base.entry_scatter = real_scatter
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _steps(completion) -> dict[str, list]:
    """One completion as picklable primitives: its tokens and their margins.

    The margins say which of these tokens the arithmetic was entitled to change
    its mind about. Nothing but ints and floats, because this crosses a process
    boundary on the results queue.
    """
    records = completion.logprobs or ()
    return {
        "tokens": [record.token_id for record in records],
        "gaps": [
            float(record.top_logprobs[0] - record.top_logprobs[1])
            if len(record.top_logprobs) >= 2
            else float("inf")
            for record in records
        ],
    }


def _run_sp_probe(model_dir: Path, sp: str) -> dict[str, Any]:
    """Run one arm to completion, surfacing its traceback as this test's failure."""
    context = mp.get_context("spawn")
    results = context.Queue()
    # Not a daemon: the probe spawns the follower rank, and a daemonic process is
    # not allowed children.
    process = context.Process(
        target=_sp_engine_probe,
        args=({"model": str(model_dir), "sp": sp, "min_tokens": _E2E_MIN_TOKENS}, results),
        daemon=False,
    )
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_E2E_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(f"SP={sp} probe produced nothing in {_E2E_TIMEOUT_S:.0f}s")
        if status == "error":
            pytest.fail(f"SP={sp} probe failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()


@needs_gpus(2)
@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.slow
def test_sequence_parallel_keeps_greedy_tokens_under_tp(model_dir: Path):
    """A real TP=2 engine says the same thing with the region off and on.

    The unit tests above pin each boundary against its reference and the pass
    against a synthetic tree; neither would notice the region being wired into
    the wrong end of a block, because both ends type-check. This runs the whole
    engine twice on the same checkpoint with nothing but
    ``RAPID_LLM_SEQUENCE_PARALLEL`` between the arms, and holds the generated
    tokens against each other.

    Exact equality is not asserted: reduce-scatter/all-gather sums a row in a
    different order than all-reduce does, so a step whose top two logits are
    within :data:`_TIE_GAP` may legitimately go either way. Every divergence has
    to be one of those, and :data:`_MIN_AGREEING_FRACTION` keeps a run that had
    dissolved into ties from passing on that licence alone.
    """
    off = _run_sp_probe(model_dir, "0")
    on = _run_sp_probe(model_dir, "1")

    assert off["marked"] == 0, "switched off, the pass must leave no mark behind"
    assert off["regions"] == 0, "switched off, no region may open"
    assert on["marked"] == 4 * on["layers"], (
        f"marked {on['marked']} boundaries across {on['layers']} blocks; four per block "
        "is the whole population, and a short count means this run measured the "
        "all-reduce path it was supposed to be replacing"
    )
    assert on["regions"] > 0, (
        "no region opened, so this arm ran the all-reduce path under a different name"
    )

    for prompt, want, got in zip(_E2E_PROMPTS, off["runs"], on["runs"], strict=True):
        assert len(got["tokens"]) == len(want["tokens"]), (
            f"{prompt!r}: {len(want['tokens'])} tokens without the region, "
            f"{len(got['tokens'])} with it"
        )
        for step, (a, b, gap) in enumerate(
            zip(want["tokens"], got["tokens"], want["gaps"], strict=True)
        ):
            assert a == b or gap <= _TIE_GAP, (
                f"{prompt!r} step {step}: the region chose {b} where the all-reduce path "
                f"chose {a}, which led its runner-up by {gap:.3f} nats -- too far for "
                "a reassociated sum to account for"
            )
        agreed = sum(a == b for a, b in zip(want["tokens"], got["tokens"], strict=True))
        assert agreed >= _MIN_AGREEING_FRACTION * len(want["tokens"]), (
            f"{prompt!r}: only {agreed}/{len(want['tokens'])} tokens agreed; every "
            "difference was inside a tie, but that many ties is its own failure"
        )
