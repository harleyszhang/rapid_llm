"""P8: data parallel replicas each capture and replay their own CUDA graphs.

Two claims to pin down. First, routing identical greedy work through a
two-replica engine gives the same completions whether the replicas decode
from captured graphs or eagerly — replay must be numerically the decode it
replaced. Second, every replica's process actually holds captured graphs:
DP replicas share no collectives (tp=1 per replica), so each captures on its
own device with nothing to lock-step over — the property that makes DP+CUDA
Graph safe where TP had to give graphs up.

Usage:
    pytest tests/engine/test_dp_cuda_graph.py
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path

import pytest

from rapid_llm import DataParallelController, SamplingParams

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = str(ROOT / "my_weight" / "Qwen3-0.6B")

PROMPTS = [
    "The capital of France is",
    "Write a haiku about autumn rain:",
    "Count from one to five:",
    "A for loop in Python looks like",
    "The three primary colors are",
    "Paris is the capital of",
]


def _greedy(gen_len: int = 32) -> SamplingParams:
    return SamplingParams(max_gen_len=gen_len, temperature=0.0, top_p=1.0)


def _run_dp(use_cuda_graph: bool, prompts: list[str], gen_len: int) -> list[str]:
    """One arm: identical routing, identical requests, only the graph differs."""
    with DataParallelController(
        model=CKPT,
        data_parallel_size=2,
        tensor_parallel_size=1,
        load_balancer="round_robin",
        max_num_seqs=8,
        max_seq_len=512,
        use_cuda_graph=use_cuda_graph,
    ) as engine:
        outputs = engine.generate(prompts, _greedy(gen_len))
    return [out.outputs[0].text for out in outputs]


@pytest.mark.gpu
def test_dp2_graph_replay_matches_eager_greedy():
    """Graph on/off through the full DP coordinator: identical completions."""
    eager = _run_dp(False, PROMPTS, gen_len=32)
    replayed = _run_dp(True, PROMPTS, gen_len=32)

    assert len(eager) == len(PROMPTS) and len(replayed) == len(PROMPTS)
    assert all(text.strip() for text in eager), "eager arm produced empty completions"
    assert eager == replayed, "graph replay changed greedy outputs under DP routing"


def _graph_probe(rank: int, queue) -> None:
    """Replica-side: build one LLM per device, generate, report graph state.

    This is the DP worker's own shape — one process, one device, tp=1, no
    collectives anywhere in the graph — so what it reports is exactly what
    each DataParallelController replica holds after startup.
    """
    try:
        import torch

        from rapid_llm import LLM

        # Same discipline as the DP worker: Triton launches resolve pointers
        # in the current device's context, so the probe must move there before
        # building anything (left on cuda:0, a cuda:1 tensor reads as "cpu"
        # to the launcher and every kernel rejects it).
        torch.cuda.set_device(rank)
        llm = LLM(
            model=CKPT,
            device=f"cuda:{rank}",
            max_seq_len=512,
            use_cuda_graph=True,
        )
        outputs = llm.generate(["The capital of France is"], _greedy(8))
        manager = llm.model_runner._graph_manager
        captured = len(manager._runners) if manager is not None else 0
        queue.put(
            {
                "rank": rank,
                "captured": captured,
                "uses_cuda_graph": llm.model_runner.uses_cuda_graph,
                "text": outputs[0].outputs[0].text,
            }
        )
        del llm
    except Exception:
        queue.put({"rank": rank, "error": traceback.format_exc()})


@pytest.mark.gpu
def test_each_replica_captures_its_own_graphs():
    """Both replica processes hold captured graphs and decode through them."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    workers = [ctx.Process(target=_graph_probe, args=(rank, queue)) for rank in range(2)]
    for worker in workers:
        worker.start()
    reports = [queue.get(timeout=600) for _ in workers]
    for worker in workers:
        worker.join(timeout=120)

    for report in reports:
        assert "error" not in report, f"replica {report.get('rank')} failed: {report.get('error')}"
        # Non-lazy capture takes the full grid at startup: many graphs, one
        # per (batch_size, seq_len_bucket) pair that fits max_seq_len.
        assert report["captured"] > 0, "replica built no CUDA graphs"
        assert report["uses_cuda_graph"], "replica's decode path ignores its graphs"
        assert report["text"].strip(), "replica produced no completion"

    # Both replicas are the same model on the same-shaped grid: their capture
    # counts must match, which is also evidence they captured independently
    # (a shared or skipped capture would show up as an asymmetry or zero).
    counts = {report["captured"] for report in reports}
    assert len(counts) == 1, f"replicas captured different graph grids: {reports}"
