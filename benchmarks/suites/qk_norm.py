"""QK-RMSNorm fusion A/B runner and archived-result summarizer."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

SINGLE_MODELS = (
    ("qwen3-4b", "Qwen3/Qwen3-4B-Thinking-2507", (1, 8, 32)),
    ("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-Instruct-2507", (1, 8)),
    ("qwen2.5-0.5b-control", "Qwen/Qwen2.5-0.5B-Instruct", (1, 8, 32)),
)
ONLINE_MODELS = (
    ("qwen3-4b", "Qwen3/Qwen3-4B-Thinking-2507"),
    ("qwen2.5-0.5b-control", "Qwen/Qwen2.5-0.5B-Instruct"),
)
SCENARIOS = ("offline_static", "offline_continuous", "online_static", "online_continuous")


def _single(args: Any, runner: Any) -> None:
    print(f"===== variant={args.variant} 离线矩阵 (one_batch eager+graph --verify) =====")
    env = {"CUDA_VISIBLE_DEVICES": "1", "RAPID_LLM_AUTOTUNE": "0"}
    for tag, model_path, batches in SINGLE_MODELS:
        csv = ",".join(map(str, batches))
        print(f"--- {tag} batches={csv} ---")
        runner.tail(
            [
                runner.python,
                "-m",
                "rapid_llm.benchmark.one_batch",
                "--model",
                str(args.weight_root / model_path),
                "--batch-sizes",
                csv,
                "--input-len",
                "64",
                "--output-len",
                "64",
                "--iters",
                "2",
                "--verify",
                "--tag",
                f"offline_{tag}_{args.variant}",
                "--log-dir",
                str(args.out),
            ],
            lines=8,
            extra_env=env,
        )

    print(f"\n===== variant={args.variant} 在线矩阵 (scheduler continuous --scenario both) =====")
    for tag, model_path in ONLINE_MODELS:
        print(f"--- {tag} online ---")
        runner.tail(
            [
                runner.python,
                "benchmarks/engine/run.py",
                "scheduler",
                "continuous",
                "--model-dir",
                str(args.weight_root / model_path),
                "--scenario",
                "both",
                "--batch",
                "8",
                "--max-gen-len",
                "64",
                "--max-seq-len",
                "1024",
                "--json",
                str(args.out / f"online_{tag}_{args.variant}.json"),
            ],
            lines=8,
            extra_env=env,
        )


def _parallel(args: Any, runner: Any) -> None:
    print(f"===== variant={args.variant} TP2 (optimizations --tp 2) =====")
    env = {"RAPID_LLM_AUTOTUNE": "0"}
    for tag, model_path in ONLINE_MODELS:
        print(f"--- {tag} tp2 ---")
        runner.tail(
            [
                runner.python,
                "benchmarks/engine/run.py",
                "optimizations",
                "--model-dir",
                str(args.weight_root / model_path),
                "--tp",
                "2",
                "--mode",
                "single",
                "--features",
                "cuda_graph",
                "--greedy",
                "--verify",
                "--batch",
                "8",
                "--max-gen-len",
                "64",
                "--json",
                str(args.out / f"tp2_{tag}_{args.variant}.json"),
            ],
            lines=7,
            extra_env=env,
        )

    print(f"\n===== variant={args.variant} DP2 (bench_data_parallel --mode scaling --dp 2) =====")
    for tag, model_path in ONLINE_MODELS:
        print(f"--- {tag} dp2 ---")
        runner.tail(
            [
                runner.python,
                "benchmarks/parallelism/bench_data_parallel.py",
                "--mode",
                "scaling",
                "--model",
                str(args.weight_root / model_path),
                "--dp",
                "2",
                "--batch-size",
                "8",
                "--gen-len",
                "64",
                "--iters",
                "2",
                "--log-dir",
                str(args.out / f"dp2_{tag}_{args.variant}"),
            ],
            lines=8,
            extra_env=env,
        )


def run_matrix(args: Any, runner: Any) -> int:
    if not runner.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    if args.scope in ("single", "all"):
        _single(args, runner)
    if args.scope == "all":
        print()
    if args.scope in ("parallel", "all"):
        _parallel(args, runner)
    print(f"\nvariant={args.variant} scope={args.scope} 完成，JSON 归档于 {args.out}")
    return 0


def _load(directory: Path, name: str) -> dict:
    with (directory / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_latest(directory: Path, pattern: str) -> dict:
    hits = sorted(directory.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no archive under {directory} matches {pattern!r}")
    with hits[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def _offline_metrics(directory: Path, tag: str, variant: str, batch: int, mode: str) -> dict:
    aggregated = sorted(directory.glob(f"offline_{tag}_{variant}_*.json"))
    if aggregated:
        with aggregated[-1].open(encoding="utf-8") as handle:
            return json.load(handle)["results"][mode][str(batch)]
    return _load(directory, f"offline_{tag}_b{batch}_{variant}.json")["results"][mode]


def _geo(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else float("nan")


def _ttft(metrics: dict) -> float:
    if "ttfts_ms" in metrics:
        return statistics.mean(metrics["ttfts_ms"]) if metrics["ttfts_ms"] else float("nan")
    return metrics.get("ttft_p50_ms", metrics.get("ttft_ms", float("nan")))


def _print_offline(directory: Path) -> list[tuple[str, str, float, float, float]]:
    print("=" * 100)
    print("offline one_batch eager+graph --verify   ratio = fused/baseline, <1 = faster")
    print("=" * 100)
    print(
        f"{'model':<24s}{'b':>4s}{'mode':<7s}{'TTFT base':>11s}{'TTFT fused':>11s}"
        f"{'TPOT base':>11s}{'TPOT fused':>11s}{'TPOT r':>8s}{'TPS r':>8s}"
    )
    print("-" * 100)
    rows = []
    for tag, _, batches in SINGLE_MODELS:
        for batch in batches:
            for mode in ("eager", "graph"):
                before = _offline_metrics(directory, tag, "baseline", batch, mode)
                after = _offline_metrics(directory, tag, "fused", batch, mode)
                r_tpot = after["tpot_p50_ms"] / before["tpot_p50_ms"]
                r_tps = after["tps"] / before["tps"]
                r_ttft = _ttft(after) / _ttft(before)
                rows.append((tag, mode, r_tpot, r_tps, r_ttft))
                print(
                    f"{tag:<24s}{batch:>4d}{mode:<7s}{_ttft(before):>11.2f}"
                    f"{_ttft(after):>11.2f}{before['tpot_p50_ms']:>11.2f}"
                    f"{after['tpot_p50_ms']:>11.2f}{r_tpot:>8.3f}{r_tps:>8.3f}"
                )
    return rows


def _print_online(directory: Path) -> list[tuple[str, str, float, float]]:
    print("\n" + "=" * 100)
    print("online scheduler continuous --scenario both batch=8")
    print("=" * 100)
    print(
        f"{'model':<24s}{'scenario':<20s}{'TTFT base':>11s}{'TTFT fused':>11s}"
        f"{'TPS base':>10s}{'TPS fused':>10s}{'TPS r':>8s}{'lat r':>8s}"
    )
    print("-" * 100)
    rows = []
    for tag, _ in ONLINE_MODELS:
        base = _load(directory, f"online_{tag}_baseline.json")
        fused = _load(directory, f"online_{tag}_fused.json")
        for scenario in SCENARIOS:
            before, after = base["results"][scenario], fused["results"][scenario]
            r_tps = after["tps"] / before["tps"]
            r_lat = statistics.mean(after["latencies_ms"]) / statistics.mean(before["latencies_ms"])
            rows.append((tag, scenario, r_tps, r_lat))
            print(
                f"{tag:<24s}{scenario:<20s}{_ttft(before):>11.1f}{_ttft(after):>11.1f}"
                f"{before['tps']:>10.1f}{after['tps']:>10.1f}{r_tps:>8.3f}{r_lat:>8.3f}"
            )
    return rows


def _print_parallel(directory: Path) -> tuple[list[tuple], list[tuple]]:
    print("\n" + "=" * 100)
    print("TP2 optimizations --tp 2 (baseline cell = eager, cuda_graph cell = graph)")
    print("=" * 100)
    print(
        f"{'model':<24s}{'cell':<14s}{'TPOT base':>11s}{'TPOT fused':>11s}"
        f"{'TPOT r':>8s}{'TPS/GPU base':>13s}{'TPS/GPU fused':>14s}{'r':>8s}"
    )
    print("-" * 100)
    tp_rows = []
    for tag, _ in ONLINE_MODELS:
        base = _load(directory, f"tp2_{tag}_baseline.json")
        fused = _load(directory, f"tp2_{tag}_fused.json")
        for before, after in zip(base["results"], fused["results"], strict=True):
            r_tpot = after["tpot_ms"] / before["tpot_ms"]
            r_gpu = after["tps_per_gpu"] / before["tps_per_gpu"]
            tp_rows.append((tag, before["label"], r_tpot, r_gpu))
            print(
                f"{tag:<24s}{before['label']:<14s}{before['tpot_ms']:>11.2f}{after['tpot_ms']:>11.2f}"
                f"{r_tpot:>8.3f}{before['tps_per_gpu']:>13.1f}{after['tps_per_gpu']:>14.1f}{r_gpu:>8.3f}"
            )

    print("\n" + "=" * 100)
    print("DP2 bench_data_parallel --mode scaling --dp 2 (weak scaling)")
    print("=" * 100)
    print(
        f"{'model':<24s}{'row':<26s}{'TPS base':>10s}{'TPS fused':>11s}{'r':>8s}"
        f"{'TPS/GPU base':>13s}{'TPS/GPU fused':>14s}{'r':>8s}"
    )
    print("-" * 100)
    dp_rows = []
    for tag, _ in ONLINE_MODELS:
        base = _load_latest(directory / f"dp2_{tag}_baseline", "*.json")
        fused = _load_latest(directory / f"dp2_{tag}_fused", "*.json")
        for before, after in zip(base["results"], fused["results"], strict=True):
            r_tps = after["tps"] / before["tps"]
            r_gpu = after["tps_per_gpu"] / before["tps_per_gpu"]
            dp_rows.append((tag, before["label"], r_tps, r_gpu))
            print(
                f"{tag:<24s}{before['label']:<26s}{before['tps']:>10.1f}{after['tps']:>11.1f}"
                f"{r_tps:>8.3f}{before['tps_per_gpu']:>13.1f}{after['tps_per_gpu']:>14.1f}{r_gpu:>8.3f}"
            )
    return tp_rows, dp_rows


def _print_geomeans(offline: list[tuple], online: list[tuple], tp_rows: list[tuple], dp_rows: list[tuple]) -> None:
    print("\n" + "=" * 100)
    print("geometric means (fused/baseline)")
    print("=" * 100)
    groups = (
        ("qk_norm models - eager", lambda row: row[0].startswith("qwen3") and row[1] == "eager"),
        ("qk_norm models - graph", lambda row: row[0].startswith("qwen3") and row[1] == "graph"),
        ("control qwen2 - eager", lambda row: row[0] == "qwen2.5-0.5b-control" and row[1] == "eager"),
        ("control qwen2 - graph", lambda row: row[0] == "qwen2.5-0.5b-control" and row[1] == "graph"),
    )
    for name, predicate in groups:
        selected = [row for row in offline if predicate(row)]
        print(
            f"  {name:<24s} n={len(selected):2d}  TPOT geo={_geo([r[2] for r in selected]):.4f}  "
            f"TPS geo={_geo([r[3] for r in selected]):.4f}  TTFT geo={_geo([r[4] for r in selected]):.4f}"
        )
    for name, tag in (("online - qk_norm model", "qwen3-4b"), ("online - control", "qwen2.5-0.5b-control")):
        selected = [row for row in online if row[0] == tag]
        print(
            f"  {name:<24s} n={len(selected):2d}  TPS geo={_geo([r[2] for r in selected]):.4f}  "
            f"latency geo={_geo([r[3] for r in selected]):.4f}"
        )
    print("\n" + "=" * 100)
    print("parallel geometric means (fused/baseline)")
    print("=" * 100)
    for prefix, metric, rows in (("TP2", "TPOT", tp_rows), ("DP2", "TPS", dp_rows)):
        for name, tag in (("qk_norm model", "qwen3-4b"), ("control", "qwen2.5-0.5b-control")):
            selected = [row for row in rows if row[0] == tag]
            print(
                f"  {prefix + ' - ' + name:<24s} n={len(selected):2d}  "
                f"{metric} geo={_geo([r[2] for r in selected]):.4f}  "
                f"TPS/GPU geo={_geo([r[3] for r in selected]):.4f}"
            )


def summarize(directory: Path) -> int:
    offline = _print_offline(directory)
    online = _print_online(directory)
    tp_rows, dp_rows = _print_parallel(directory)
    _print_geomeans(offline, online, tp_rows, dp_rows)
    return 0
