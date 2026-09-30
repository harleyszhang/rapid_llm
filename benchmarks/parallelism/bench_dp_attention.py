#!/usr/bin/env python
"""Data-parallel attention scaling, lockstep, backend, and parity benchmark.

Every row uses eager execution because DPA does not support CUDA Graph yet.
``local1`` is the no-communication one-GPU baseline, ``ep2`` isolates pure EP,
``dpa2`` measures attention-DP over two ranks, and ``dpa2x2`` exercises the
full DP=2 x TP=2 grid. The reported overhead is the end-to-end gap from ideal
linear scaling; it includes DPA collectives, lockstep waiting, and process IPC.
Exact completions use local1 replays with the same round-robin replica sub-batch
shapes, so batch-width numerical changes are not counted as routing failures.

Usage:
    python benchmarks/parallelism/bench_dp_attention.py --model <moe-ckpt>
    python benchmarks/parallelism/bench_dp_attention.py --model <moe-ckpt> \
        --arms local1,dpa2 --backends agrs,a2a --scenarios balanced,idle,uneven
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm import DataParallelEngine, SamplingParams
from rapid_llm.benchmark import (
    count_gen_tokens,
    print_row_table,
    print_run_header,
    require_gpus,
    run_in_spawned_process,
    timed_rounds,
    timestamped_log_path,
    write_json_log,
)

ARMS: dict[str, dict[str, int | bool]] = {
    "local1": {"dp": 1, "tp": 1, "dpa": False, "ep": False},
    "ep2": {"dp": 1, "tp": 2, "dpa": False, "ep": True},
    "dpa2": {"dp": 2, "tp": 1, "dpa": True, "ep": True},
    "dpa2x2": {"dp": 2, "tp": 2, "dpa": True, "ep": True},
}

_SHORT = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at",
    "The largest planet in our solar system is",
    "Python is a language that",
    "Machine learning is",
    "The sun rises in the",
    "A compiler transforms source code into",
]
_LONG = "Explain distributed inference carefully. " + "Attention and experts cooperate. " * 80
SCENARIOS = {
    "balanced": _SHORT * 2,
    "idle": [_SHORT[0]],
    "uneven": [_SHORT[1], _LONG],
}

_PROBE_TIMEOUT_S = 1800.0


@contextlib.contextmanager
def _pooling_backend(name: str):
    old = os.environ.get("RAPID_MOE_A2A_BACKEND")
    if name == "a2a":
        os.environ["RAPID_MOE_A2A_BACKEND"] = "all_to_all"
    else:
        os.environ.pop("RAPID_MOE_A2A_BACKEND", None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RAPID_MOE_A2A_BACKEND", None)
        else:
            os.environ["RAPID_MOE_A2A_BACKEND"] = old


def _shape_matched_reference(
    engine: DataParallelEngine,
    prompts: list[str],
    params: SamplingParams,
    replicas: int,
) -> list[str]:
    """Replay round-robin replica buckets and restore original request order."""
    texts: list[str | None] = [None] * len(prompts)
    for replica in range(replicas):
        indices = list(range(replica, len(prompts), replicas))
        if not indices:
            continue
        outputs = engine.generate([prompts[index] for index in indices], params)
        for index, output in zip(indices, outputs, strict=True):
            texts[index] = output.text
    if any(text is None for text in texts):
        raise RuntimeError("shape-matched reference did not return every prompt")
    return [text for text in texts if text is not None]


def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    try:
        flags = spec["flags"]
        with _pooling_backend(spec["backend"]):
            started = time.perf_counter()
            engine = DataParallelEngine(
                model=spec["model"],
                data_parallel_size=flags["dp"],
                tensor_parallel_size=flags["tp"],
                enable_dp_attention=flags["dpa"],
                enable_expert_parallel=flags["ep"],
                use_cuda_graph=False,
                max_seq_len=spec["max_seq_len"],
                max_gpu_num_blocks=spec["kv_tokens"],
                max_num_seqs=spec["max_num_seqs"],
                max_num_batched_tokens=spec["max_num_batched_tokens"],
                max_chunk_size=spec["max_chunk_size"],
            )
            build_s = time.perf_counter() - started
            try:
                tokenizer = engine.tokenizer
                scenario_reports = []
                params = SamplingParams(
                    temperature=0.0,
                    max_gen_len=spec["gen_len"],
                    repetition_penalty=1.0,
                    stop_on_repeat=False,
                )
                warmup = SamplingParams(
                    temperature=0.0,
                    max_gen_len=min(4, spec["gen_len"]),
                    repetition_penalty=1.0,
                    stop_on_repeat=False,
                )
                for name, prompts in spec["scenarios"].items():
                    engine.generate(prompts, warmup)
                    latency_s, outputs = timed_rounds(
                        lambda prompts=prompts: engine.generate(prompts, params),
                        spec["iters"],
                    )
                    texts = [output.text for output in outputs]
                    tokens = count_gen_tokens(texts, tokenizer)
                    scenario_reports.append(
                        {
                            "scenario": name,
                            "batch": len(prompts),
                            "latency_s": round(latency_s, 4),
                            "gen_tokens": tokens,
                            "tps": round(tokens / latency_s, 2) if latency_s else 0.0,
                            "texts": texts,
                        }
                    )
                report = {
                    "arm": spec["arm"],
                    "backend": spec["backend"],
                    "dp": flags["dp"],
                    "tp": flags["tp"],
                    "parallel_degree": flags["dp"] * flags["tp"],
                    "build_s": round(build_s, 2),
                    "scenarios": scenario_reports,
                }
                if spec["arm"] == "local1":
                    measured = {item["scenario"]: item["texts"] for item in scenario_reports}
                    report["parity_references"] = {}
                    for replicas in spec["reference_dp_sizes"]:
                        report["parity_references"][str(replicas)] = {
                            name: measured[name]
                            if replicas == 1
                            else _shape_matched_reference(engine, prompts, params, replicas)
                            for name, prompts in spec["scenarios"].items()
                        }
            finally:
                engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    status, payload = run_in_spawned_process(_probe, spec, _PROBE_TIMEOUT_S)
    if status == "timeout":
        raise SystemExit(f"{spec['arm']}/{spec['backend']} timed out after {_PROBE_TIMEOUT_S:.0f}s")
    if status == "error":
        raise SystemExit(f"{spec['arm']}/{spec['backend']} failed:\n{payload}")
    return payload


def _entry(report: dict[str, Any], scenario: str) -> dict[str, Any]:
    return next(item for item in report["scenarios"] if item["scenario"] == scenario)


def _attach_comparisons(reports: list[dict[str, Any]], scenarios: dict[str, list[str]]) -> None:
    baseline = next(report for report in reports if report["arm"] == "local1")
    for report in reports:
        degree = report["parallel_degree"]
        for name in scenarios:
            entry = _entry(report, name)
            base = _entry(baseline, name)
            ideal = base["tps"] * degree
            entry["tgs"] = round(entry["tps"] / degree, 2)
            entry["scaling_efficiency"] = round(entry["tps"] / ideal, 4) if ideal else None
            entry["sync_communication_overhead"] = (
                round(max(0.0, 1.0 - entry["tps"] / ideal), 4) if ideal else None
            )
            reference = baseline["parity_references"][str(report["dp"])][name]
            entry["exact_completions"] = sum(
                want == got for want, got in zip(reference, entry["texts"], strict=True)
            )


def _print(reports: list[dict[str, Any]], scenarios: dict[str, list[str]]) -> None:
    for name, prompts in scenarios.items():
        rows = []
        for report in reports:
            entry = _entry(report, name)
            rows.append(
                [
                    f"{report['arm']}/{report['backend']}",
                    str(report["parallel_degree"]),
                    f"{entry['latency_s']:.3f}",
                    f"{entry['tps']:.1f}",
                    f"{entry['tgs']:.1f}",
                    f"{entry['scaling_efficiency']:.1%}",
                    f"{entry['sync_communication_overhead']:.1%}",
                    f"{entry['exact_completions']}/{len(prompts)}",
                ]
            )
        print(f"\nscenario {name}: batch={len(prompts)}, offline")
        print_row_table(
            ["config", "ranks", "latency", "TPS", "TGS", "efficiency", "sync+comm", "exact"],
            [18, 7, 10, 10, 10, 12, 12, 10],
            rows,
        )
    print(
        "\nsync+comm is the end-to-end gap from ideal linear scaling; it includes "
        "collectives, lockstep waiting, and coordinator IPC, not wire time alone."
    )
    print(
        "exact compares against local1 replays of the same round-robin replica "
        "sub-batches, restored to original request order."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="MoE checkpoint directory")
    parser.add_argument("--arms", default="local1,ep2,dpa2,dpa2x2")
    parser.add_argument("--backends", default="agrs", help="agrs,a2a (DPA arms only)")
    parser.add_argument("--scenarios", default="balanced,idle,uneven")
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--kv-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-chunk-size", type=int, default=64)
    parser.add_argument("--log-dir", default="docs/benchmark_logs/parallel")
    args = parser.parse_args()

    arm_names = [item for item in args.arms.split(",") if item]
    backends = [item for item in args.backends.split(",") if item]
    scenario_names = [item for item in args.scenarios.split(",") if item]
    if unknown := set(arm_names) - ARMS.keys():
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    if unknown := set(backends) - {"agrs", "a2a"}:
        raise SystemExit(f"unknown backends: {sorted(unknown)}")
    if unknown := set(scenario_names) - SCENARIOS.keys():
        raise SystemExit(f"unknown scenarios: {sorted(unknown)}")
    if "local1" not in arm_names:
        raise SystemExit("local1 must be selected because it defines scaling efficiency")

    scenarios = {name: SCENARIOS[name] for name in scenario_names}
    required = max(ARMS[name]["dp"] * ARMS[name]["tp"] for name in arm_names)
    require_gpus(int(required))
    print_run_header(
        args.model,
        {
            "mode": "DPA offline",
            "arms": ",".join(arm_names),
            "backends": ",".join(backends),
            "scenarios": ",".join(scenario_names),
            "gen_len": args.gen_len,
            "cuda_graph": "off (required by DPA)",
        },
    )

    reports = []
    reference_dp_sizes = sorted({int(ARMS[name]["dp"]) for name in arm_names})
    for arm in arm_names:
        selected_backends = backends if ARMS[arm]["dpa"] else ["n/a"]
        for backend in selected_backends:
            spec = {
                "arm": arm,
                "backend": backend,
                "flags": ARMS[arm],
                "model": args.model,
                "scenarios": scenarios,
                "gen_len": args.gen_len,
                "iters": args.iters,
                "max_seq_len": args.max_seq_len,
                "kv_tokens": args.kv_tokens,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_chunk_size": args.max_chunk_size,
                "reference_dp_sizes": reference_dp_sizes,
            }
            print(f"[{arm}/{backend}] building", flush=True)
            reports.append(_run(spec))

    _attach_comparisons(reports, scenarios)
    _print(reports, scenarios)
    if args.log_dir:
        path = timestamped_log_path(args.log_dir, f"dp_attention_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "arms": {name: ARMS[name] for name in arm_names},
                "backends": backends,
                "scenarios": {name: len(prompts) for name, prompts in scenarios.items()},
                "gen_len": args.gen_len,
                "iters": args.iters,
                "cuda_graph": False,
                "overhead_definition": "1 - TPS / (local1 TPS * parallel_degree); "
                "includes collectives, lockstep waiting, and coordinator IPC",
                "exact_completion_reference": "local1 replay of the same round-robin "
                "replica sub-batches, restored to request order",
            },
            reports,
        )


if __name__ == "__main__":
    main()
