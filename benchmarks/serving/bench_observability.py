"""What observability costs: logprobs (GPU, costly) vs metrics/trace (host, cheap).

Each :class:`Variant` enables one observability surface and ``measure``
times the same workload, so the per-token overhead of every knob is a
row in one table.

Usage:
    python benchmarks/serving/bench_observability.py --model-dir <ckpt>
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.benchmark import (
    GREEDY_PARAMS,
    PROMPTS,
    BenchResult,
    expand_prompts,
    free_gpu,
    print_table,
    require_gpus,
    run_requests,
    write_json_log,
)
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.tools.observability import METRICS_ENV, EngineMetrics, Tracer

CKPT = "my_weight/Qwen3-0.6B"

_TOP_K = 5  # how many alternatives to report; the width matters far less than reporting at all


@dataclass(frozen=True)
class Variant:
    """One configuration under test: a sampling-parameter switch plus engine-side observers."""

    label: str
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    metrics: bool = False
    trace: bool = False

    def params(self, max_gen_len: int) -> SamplingParams:
        return SamplingParams(
            max_gen_len=max_gen_len,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs,
            **GREEDY_PARAMS,
        )


#: The order is deliberate: a baseline opens and closes the list, and every row in
#: between turns on exactly one thing relative to it.
VARIANTS = (
    Variant("baseline"),
    Variant("metrics", metrics=True),
    Variant("metrics+trace", metrics=True, trace=True),
    Variant(f"logprobs={_TOP_K}", logprobs=_TOP_K),
    Variant(f"prompt_logprobs={_TOP_K}", prompt_logprobs=_TOP_K),
    Variant("both", logprobs=_TOP_K, prompt_logprobs=_TOP_K),
    Variant("baseline (again)"),
)


def build_metrics(enabled: bool) -> EngineMetrics:
    """Build a metrics object, on or off.

    It must go through ``from_env``: ``EngineMetrics(enabled=False)`` only clears the
    flag, while ``from_env`` is what substitutes the no-op instruments (one attribute
    lookup plus an empty method). A directly constructed "off" object still records
    into buckets, so measuring it would not measure the off state.
    """
    previous = os.environ.get(METRICS_ENV)
    os.environ[METRICS_ENV] = "1" if enabled else "0"
    try:
        return EngineMetrics.from_env()
    finally:
        if previous is None:
            os.environ.pop(METRICS_ENV, None)
        else:
            os.environ[METRICS_ENV] = previous


def build_tracer(enabled: bool) -> Tracer:
    """Enabled state exports in memory, not over the network.

    What is measured is the cost span creation and attribute writes put on the step
    loop; in a real deployment ``BatchSpanProcessor``'s background thread does the
    export, so a collector round trip must not be charged to TTFT. With an in-memory
    exporter the path is structurally identical to production, only the sink differs.
    """
    if not enabled:
        return Tracer()
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ModuleNotFoundError:
        print("opentelemetry SDK not installed; the trace row degrades to a no-op (= baseline)")
        return Tracer()

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(InMemorySpanExporter()))
    return Tracer(provider.get_tracer("bench_observability"))


def measure(engine, prompts: list[str], params: SamplingParams) -> BenchResult:
    """Submit the whole batch then time each step, the same discipline as EngineBackend."""
    return run_requests(engine, prompts, params).result(len(prompts))


def run_variant(
    engine, variant: Variant, prompts: list[str], max_gen_len: int, iters: int
) -> BenchResult:
    """Install this configuration, warm up to steady state, take the median TPS of ``iters``.

    The warm-up must use the variant's own parameters: the first step carrying logprobs
    compiles the topk kernel, and counting that in the measurement would report compile
    time as logprobs overhead.
    """
    engine.metrics = build_metrics(variant.metrics)
    engine.tracer = build_tracer(variant.trace)
    params = variant.params(max_gen_len)

    engine.generate(prompts[:2], variant.params(8))
    rounds = [measure(engine, prompts, params) for _ in range(iters)]
    return sorted(rounds, key=lambda r: r.tps)[len(rounds) // 2]


def report(results: dict[str, BenchResult]) -> dict[str, float]:
    """Print each variant's cost against the baseline, with the noise floor — sub-noise
    differences do not count."""
    baselines = [r.tps for label, r in results.items() if label.startswith("baseline")]
    reference = statistics.mean(baselines)
    noise = (max(baselines) - min(baselines)) / reference if len(baselines) > 1 else 0.0

    print(f"\nbaseline TPS {reference:.1f} tok/s, self-difference {noise * 100:.1f}% (noise floor)")
    costs: dict[str, float] = {}
    for label, result in results.items():
        cost = 1.0 - result.tps / reference
        costs[label] = cost
        if label.startswith("baseline"):
            continue
        verdict = "within noise" if abs(cost) <= noise else f"{cost * 100:+.1f}%"
        print(f"{label:22s} throughput cost {cost * 100:+6.1f}% -> {verdict}")
    return {"baseline_tps": reference, "noise": noise, **costs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--kv-blocks", type=int, default=40960)
    ap.add_argument(
        "--iters", type=int, default=3, help="Rounds per configuration; the median TPS is reported"
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    require_gpus(1)
    prompts = expand_prompts(PROMPTS, args.batch)

    engine = ContinuousBatchingEngine.from_pretrained(
        args.model_dir,
        max_seq_len=args.max_seq_len,
        max_num_seqs=args.max_num_seqs,
        max_gpu_num_blocks=args.kv_blocks,
        use_cuda_graph=True,
    )
    try:
        print(f"=== {args.batch} requests x {args.max_gen_len} tokens, {args.iters} iters ===")
        results = {
            variant.label: run_variant(engine, variant, prompts, args.max_gen_len, args.iters)
            for variant in VARIANTS
        }
    finally:
        engine.shutdown()
        del engine
        free_gpu()

    print_table(results)
    summary = report(results)

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {"runs": {k: v.as_dict() for k, v in results.items()}, "summary": summary},
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
