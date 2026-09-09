"""Verify TP=2 sampling consistency: rank0 and rank1 must produce identical tokens.

This script spawns 2 processes (TP=2), runs non-greedy sampling (temperature=0.7),
and verifies that both ranks produce the same output tokens.

To demonstrate the FIX, we run twice:
  1. With broadcast enabled (fixed) - both ranks agree
  2. With broadcast disabled (simulated bug) - ranks diverge
"""

import json
import sys
from multiprocessing import Process, Queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_rank(
    rank: int,
    world_size: int,
    model_dir: str,
    temperature: float,
    seed: int,
    disable_broadcast: bool,
    q: Queue,
):
    """Run generation on one TP rank and return output token ids."""
    import torch

    torch.cuda.set_device(rank)
    torch.manual_seed(seed + rank)  # deliberately different seeds per rank
    torch.cuda.manual_seed(seed + rank)

    from rapid_llm.distributed.parallel_state import init_tensor_parallel

    init_tensor_parallel(rank=rank, world_size=world_size)

    if disable_broadcast:
        # Monkey-patch to simulate the pre-fix behavior
        import rapid_llm.engine.continuous_engine as ceng
        import rapid_llm.engine.llm_engine as eng

        # Override the module-level reference so the decode loop's broadcast is a no-op
        eng.tensor_model_parallel_broadcast = lambda t, src=0: t
        ceng.tensor_model_parallel_broadcast = lambda t, src=0: t

    from rapid_llm import SamplingParams, TextGenerator

    gen = TextGenerator(
        checkpoints_dir=model_dir,
        max_seq_len=512,
        use_cuda_graph=False,
        tensor_parallel_size=world_size,
    )
    params = SamplingParams(temperature=temperature, max_gen_len=32, top_p=0.9)
    outputs = gen.generate(["The meaning of life is"], params)

    from rapid_llm.distributed.parallel_state import destroy_tensor_parallel

    destroy_tensor_parallel()

    q.put({"rank": rank, "output": outputs[0]})


def run_tp2_test(model_dir: str, temperature: float, seed: int, disable_broadcast: bool) -> dict:
    """Run TP=2 and return outputs from both ranks."""
    q = Queue()
    procs = []
    for rank in range(2):
        p = Process(
            target=_run_rank, args=(rank, 2, model_dir, temperature, seed, disable_broadcast, q)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=120)

    results = {}
    while not q.empty():
        item = q.get()
        results[item["rank"]] = item["output"]
    return results


def main():
    model_dir = "/data/shared/llm_weights/Qwen3-0.6B"
    temperature = 0.7
    seed = 42

    print("=" * 70)
    print("TEST: TP=2 sampling consistency (temperature=0.7, seed differs per rank)")
    print("=" * 70)

    # Test 1: With broadcast (FIXED behavior)
    print("\n[AFTER FIX] broadcast enabled:")
    results_fixed = run_tp2_test(model_dir, temperature, seed, disable_broadcast=False)
    if 0 in results_fixed and 1 in results_fixed:
        match = results_fixed[0] == results_fixed[1]
        print(f"  Rank 0: {results_fixed[0][:80]!r}")
        print(f"  Rank 1: {results_fixed[1][:80]!r}")
        print(f"  Match: {'YES ✓' if match else 'NO ✗'}")
    else:
        print("  ERROR: could not get results from both ranks")
        match = False

    # Test 2: Without broadcast (simulated PRE-FIX bug)
    print("\n[BEFORE FIX] broadcast disabled (simulated):")
    results_bug = run_tp2_test(model_dir, temperature, seed, disable_broadcast=True)
    if 0 in results_bug and 1 in results_bug:
        diverge = results_bug[0] != results_bug[1]
        print(f"  Rank 0: {results_bug[0][:80]!r}")
        print(f"  Rank 1: {results_bug[1][:80]!r}")
        print(f"  Diverge: {'YES (bug reproduced) ✓' if diverge else 'NO (unexpectedly same)'}")
    else:
        print("  ERROR: could not get results from both ranks")
        diverge = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Fixed (broadcast ON):  ranks {'AGREE' if match else 'DIVERGE'}")
    print(f"  Bug (broadcast OFF):   ranks {'DIVERGE' if diverge else 'AGREE'}")
    print(f"  Verdict: {'PASS' if match and diverge else 'INCONCLUSIVE'}")
    print("=" * 70)

    # Save log
    log = {
        "test": "tp2_sampling_rng_consistency",
        "model": model_dir,
        "temperature": temperature,
        "seed": seed,
        "fixed": {"rank0": results_fixed.get(0), "rank1": results_fixed.get(1), "match": match},
        "bug_simulated": {
            "rank0": results_bug.get(0),
            "rank1": results_bug.get(1),
            "diverge": diverge,
        },
        "verdict": "PASS" if match and diverge else "INCONCLUSIVE",
    }
    out_path = Path("docs/benchmark_logs/engine/tp2_sampling_fix_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    print(f"\nLog saved: {out_path}")

    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
