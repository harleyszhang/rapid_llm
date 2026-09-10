"""L3 chunked all-reduce: the overlap claim, as timeline evidence.

The parity of the chunked path is pinned in ``test_comm_overlap.py``; this
file pins the *reason it exists*: with L3 on, chunk ``k+1``'s GEMM starts
before chunk ``k``'s all-reduce finishes — the two regions overlap on the
device clock, which a blocking all-reduce can never show (it never records a
comm region at all).

Usage:
    pytest tests/batch_overlap/test_l3_chunked_ar.py
"""

from __future__ import annotations

import os

import torch

from rapid_llm.batch_overlap.comm_overlap import (
    COMM_OVERLAP_ENV,
    L3_CHUNKS_ENV,
    CommStreamPool,
    reset_comm_overlap_policy,
)
from rapid_llm.modules import RowParallelLinear
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

#: A size whose GEMM and all-reduce each run ~1-2 ms on an A10: big enough
#: that the regions are resolvable on the device clock, small enough that the
#: payload finishes in a blink.
TOKENS = 2048
HIDDEN = 2048

#: Forwards issued back to back before the overlap assertion.
_ITERATIONS = 5


def _payload_overlap_evidence(rank: int) -> dict:
    """Run chunked row-parallel forwards; return the last iteration's overlap."""
    os.environ[COMM_OVERLAP_ENV] = "1"
    os.environ[L3_CHUNKS_ENV] = "2"
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "1"
    CommStreamPool.reset()
    reset_comm_overlap_policy()
    device = torch.device("cuda", rank)
    layer = RowParallelLinear(HIDDEN, HIDDEN, params_dtype=torch.float32).to(device)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(HIDDEN, HIDDEN // 2, device=device))
    x = torch.randn(TOKENS, HIDDEN // 2, device=device)

    for _ in range(3):  # warm every first-call cost out of the measurement
        layer.forward(x)
    torch.cuda.synchronize()
    CommStreamPool.reset()  # a fresh pool carries a fresh, empty timeline

    # Back-to-back forwards saturate the queue: the overlap window is set by
    # the device, not by host enqueue jitter around a single launch.
    for _ in range(_ITERATIONS):
        layer.forward(x)
    torch.cuda.synchronize()

    pool = CommStreamPool.for_device(device)
    records = pool.timeline.collect()
    reduces = {r.name: r for r in records if r.name.startswith("l3.all_reduce")}
    gemms = {r.name: r for r in records if r.name.startswith("l3.gemm")}
    assert len(reduces) == 2 and len(gemms) == 2, (
        f"two chunks must produce two reduces and two GEMMs, saw {sorted(records)}"
    )
    ar0, gemm1 = reduces["l3.all_reduce.0"], gemms["l3.gemm.1"]
    overlap = min(ar0.end_ms, gemm1.end_ms) - max(ar0.start_ms, gemm1.start_ms)
    # The claim itself: the second chunk's GEMM started before the first
    # chunk's reduction finished — compute did not wait the bubble out.
    assert overlap > 0.0, (
        f"l3.all_reduce.0 [{ar0.start_ms:.3f}, {ar0.end_ms:.3f}] ms did not overlap "
        f"l3.gemm.1 [{gemm1.start_ms:.3f}, {gemm1.end_ms:.3f}] ms"
    )
    return {"overlap_ms": round(overlap, 3)}


def _payload_blocking_records_no_comm_region(rank: int) -> str:
    """The control arm: with L3 off there is no comm region to overlap with."""
    os.environ[COMM_OVERLAP_ENV] = "0"
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "1"
    CommStreamPool.reset()
    reset_comm_overlap_policy()
    device = torch.device("cuda", rank)
    layer = RowParallelLinear(HIDDEN, HIDDEN, params_dtype=torch.float32).to(device)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(HIDDEN, HIDDEN // 2, device=device))
    x = torch.randn(TOKENS, HIDDEN // 2, device=device)
    for _ in range(3):
        layer.forward(x)
    torch.cuda.synchronize()
    CommStreamPool.reset()
    layer.forward(x)
    torch.cuda.synchronize()
    records = CommStreamPool.for_device(device).timeline.collect()
    assert not [r for r in records if r.stream == "comm"], (
        "the blocking all-reduce runs on the compute stream; a comm region here "
        "would mean the dispatch silently changed"
    )
    return "ok"


@needs_gpus(2)
def test_chunk_k_plus_one_gemm_overlaps_chunk_k_reduce():
    results = run_on_tp_ranks(_payload_overlap_evidence, tp_size=2)
    assert all(r["overlap_ms"] > 0.0 for r in results)


@needs_gpus(2)
def test_blocking_all_reduce_never_records_a_comm_region():
    assert run_on_tp_ranks(_payload_blocking_records_no_comm_region, tp_size=2) == ["ok", "ok"]
