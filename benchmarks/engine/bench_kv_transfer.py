"""KV-transfer acceptance: accuracy parity, TPOT tax, and overflow reuse.

The ROADMAP v0.12.0 acceptance for the CPU tier, two arms of the three (the
DP-skew arm lives with its peers in ``benchmarks/parallelism`` because it
needs the DP coordinator, not this engine):

* ``accuracy`` — tier off/on, GPU prefix cache on in both, greedy sampling:
  the output token ids must match request for request, and the decode TPOT
  must not regress beyond ``--tpot-tolerance`` (ROADMAP: < 2%). Only the
  ``offloading`` argument differs between the two builds.
* ``overflow`` — a constrained GPU pool (``--overflow-gpu-blocks``) forces
  blocks out of the GPU index between the waves; with the CPU tier the same
  workload must reuse at least as many prompt tokens, measured as
  ``sum(num_cached_tokens) / sum(prompt_len)`` — the effective-reuse basis,
  which counts CPU promotions the GPU-only ``prefix_cache_hit_rate`` cannot.

Evidence JSON lands in ``docs/benchmark_logs/kv_transfer/``.

Usage:
    python benchmarks/engine/run.py kv-transfer --model-dir my_weight/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

from rapid_llm.benchmark import (
    PROMPTS,
    expand_prompts,
    free_gpu,
    print_run_header,
    require_gpus,
    run_requests,
    sampling_params,
    timestamped_log_path,
    write_json_log,
)
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.llm_engine import LLMEngine
from rapid_llm.engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE
from rapid_llm.engine.scheduler import SchedulerConfig
from rapid_llm.executor.executor import UniProcExecutor
from rapid_llm.executor.kv_offload import build_cpu_tier

CKPT = "my_weight/Qwen3-0.6B"
LOG_DIR = "docs/benchmark_logs/kv_transfer"
_FILLER = "Follow every instruction carefully and answer as precisely as you can. "

#: Warm-up prompts, deliberately outside every workload: a warm-up on the
#: prompts under test would pre-seed the prefix cache and hand the first wave
#: hits it did not earn (see ``measure_generate``'s note).
_WARMUP = [
    "Name three primary colors and their complements.",
    "What is the capital of France? Answer in one word.",
]


def shared_prefix_prompts(groups: int, per_group: int, sentences: int) -> list[str]:
    """Requests sharing a long prefix per group; groups arrive interleaved."""
    prefixes = [f"You are assistant number {g}. " + _FILLER * sentences for g in range(groups)]
    arrivals = [g for g in range(groups) for _ in range(per_group)]
    return [f"{prefixes[g]}Question {i}: what is {i} plus {i + 1}?" for i, g in enumerate(arrivals)]


@dataclass
class ParityArm:
    """One side of the accuracy arm: timed numbers plus the output token ids."""

    label: str
    tier: bool
    total_s: float
    steps: int
    gen_tokens: int
    ttft_ms: float
    tpot_ms: float
    token_ids: list[list[int]] = field(default_factory=list)
    tier_stats: dict | None = None

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    def row(self) -> str:
        return (
            f"{self.label:9s} {self.total_s:6.2f}s | TTFT {self.ttft_ms:7.1f} ms | "
            f"TPOT {self.tpot_ms:6.2f} ms | TPS {self.tps:8.1f} tok/s"
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("token_ids")  # the parity check's input, not a report number
        return {**data, "tps": self.tps}


@dataclass
class OverflowArm:
    """One side of the overflow arm: per-wave reuse accounting."""

    label: str
    tier: bool
    total_s: float
    waves: int
    requests: int
    cached_tokens: int
    prompt_tokens: int
    gpu_hit_rate: float
    tier_stats: dict | None = None

    @property
    def effective_reuse(self) -> float:
        """Reused prompt tokens over offered ones — GPU hits plus CPU promotions."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def row(self) -> str:
        return (
            f"{self.label:9s} {self.total_s:6.2f}s | reuse {self.effective_reuse:6.1%} "
            f"({self.cached_tokens}/{self.prompt_tokens} tok) | "
            f"gpu hit {self.gpu_hit_rate:6.1%} | {self.requests} requests x {self.waves} waves"
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "effective_reuse": self.effective_reuse}


def _stats_dict(manager) -> dict | None:
    """The tier's counters as plain data; ``None`` for a GPU-only arm."""
    if manager is None:
        return None
    stats = manager.get_stats()
    return {
        **asdict(stats),
        "misses": stats.misses,
        "hit_rate": stats.hit_rate,
        "medium": manager.medium,
    }


def build_engine(
    model_dir: str,
    *,
    max_seq_len: int,
    max_num_seqs: int,
    kv_blocks: int | None,
    tier_blocks: int,
    tier: bool,
    use_cuda_graph: bool,
):
    """The acceptance harness' single construction path (plan section 6b).

    Order is load-bearing: the executor exists first, because it is what calls
    ``enable_slot_kv_cache`` — the device buffer the host region mirrors — and
    the tier exists before the engine, whose scheduler takes it.

    Returns:
        ``(engine, manager)`` — the manager is ``None`` on a GPU-only arm.
    """
    config = SchedulerConfig(
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        enable_prefix_cache=True,
    )
    # kv_blocks is in scheduler blocks (16 tok each); LLMEngine takes tokens.
    token_blocks = kv_blocks * PREFIX_CACHE_BLOCK_SIZE if kv_blocks is not None else None
    llm = LLMEngine(
        model_dir,
        max_seq_len=max_seq_len,
        max_gpu_num_blocks=token_blocks,
        use_cuda_graph=use_cuda_graph,
    )
    executor = UniProcExecutor(llm, config.max_num_seqs, config.max_seq_len)
    manager = build_cpu_tier(llm.model_runner.kv_cache_manager, tier_blocks) if tier else None
    engine = ContinuousBatchingEngine(llm, config, executor, offloading=manager)
    return engine, manager


def measure_parity_arm(
    model_dir: str,
    prompts: list[str],
    params,
    *,
    tier: bool,
    kv_blocks: int | None,
    tier_blocks: int,
    max_seq_len: int,
    max_num_seqs: int,
    use_cuda_graph: bool,
) -> ParityArm:
    """Build one side of the accuracy arm and time one greedy batch."""
    engine, manager = build_engine(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        kv_blocks=kv_blocks,
        tier_blocks=tier_blocks,
        tier=tier,
        use_cuda_graph=use_cuda_graph,
    )
    engine.generate(_WARMUP, sampling_params(8))
    run = run_requests(engine, prompts, params)
    result = run.result(len(prompts))
    arm = ParityArm(
        label=f"tier {'on' if tier else 'off'}",
        tier=tier,
        total_s=run.total_s,
        steps=result.steps,
        gen_tokens=run.gen_tokens,
        ttft_ms=result.ttft_ms,
        tpot_ms=result.tpot_ms,
        token_ids=[list(request.output_token_ids) for request in run.requests],
        tier_stats=_stats_dict(manager),
    )
    del engine
    free_gpu()
    return arm


def measure_overflow_arm(
    model_dir: str,
    prompts: list[str],
    params,
    waves: int,
    *,
    tier: bool,
    kv_blocks: int,
    tier_blocks: int,
    max_seq_len: int,
    max_num_seqs: int,
    use_cuda_graph: bool,
) -> OverflowArm:
    """Build one side of the overflow arm and replay the workload ``waves`` times."""
    engine, manager = build_engine(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        kv_blocks=kv_blocks,
        tier_blocks=tier_blocks,
        tier=tier,
        use_cuda_graph=use_cuda_graph,
    )
    engine.generate(_WARMUP, sampling_params(4))
    cached = prompt_total = 0
    total_s = 0.0
    for _ in range(waves):
        run = run_requests(engine, prompts, params)
        total_s += run.total_s
        cached += sum(request.num_cached_tokens for request in run.requests)
        prompt_total += sum(request.prompt_len for request in run.requests)
    gpu_hit_rate = engine.scheduler.prefix_cache_hit_rate
    arm = OverflowArm(
        label=f"tier {'on' if tier else 'off'}",
        tier=tier,
        total_s=total_s,
        waves=waves,
        requests=len(prompts),
        cached_tokens=cached,
        prompt_tokens=prompt_total,
        gpu_hit_rate=gpu_hit_rate,
        tier_stats=_stats_dict(manager),
    )
    del engine
    free_gpu()
    return arm


def compare_parity(off: ParityArm, on: ParityArm, tolerance: float) -> dict:
    """The accuracy criteria: token parity, then the TPOT regression budget."""
    equal = off.token_ids == on.token_ids
    delta = (on.tpot_ms - off.tpot_ms) / off.tpot_ms if off.tpot_ms else 0.0
    accepted = equal and delta <= tolerance
    print(
        f"\n-> tokens {'match' if equal else 'DIFFER'}; TPOT {off.tpot_ms:.2f} -> "
        f"{on.tpot_ms:.2f} ms ({delta:+.2%}, tolerance {tolerance:.0%}): "
        f"{'ACCEPTED' if accepted else 'REJECTED'}"
    )
    return {
        "criterion": f"tier-on token ids match AND TPOT regression <= {tolerance:.0%}",
        "tokens_equal": equal,
        "tpot_delta": delta,
        "accepted": accepted,
    }


def compare_overflow(off: OverflowArm, on: OverflowArm) -> dict:
    """The overflow criterion: tier-on effective reuse must not drop."""
    accepted = on.effective_reuse >= off.effective_reuse
    loads = (on.tier_stats or {}).get("loads", 0)
    note = "happened" if loads else "DID NOT HAPPEN — raise the GPU-pool pressure"
    print(
        f"\n-> effective reuse {off.effective_reuse:.1%} -> {on.effective_reuse:.1%}; "
        f"tier loads {loads} (CPU promotions {note}): "
        f"{'ACCEPTED' if accepted else 'REJECTED'}"
    )
    return {
        "criterion": "tier-on effective reuse >= tier-off",
        "reuse_tier_off": off.effective_reuse,
        "reuse_tier_on": on.effective_reuse,
        "cpu_loads": loads,
        "accepted": accepted,
    }


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=CKPT)
    parser.add_argument("--arm", choices=["accuracy", "overflow", "both"], default="both")
    parser.add_argument("--parity-batch", type=int, default=16, help="[accuracy] request count")
    parser.add_argument("--parity-gen-len", type=int, default=256, help="[accuracy] tokens each")
    parser.add_argument(
        "--tpot-tolerance",
        type=float,
        default=0.02,
        help="[accuracy] decode TPOT regression budget with the tier on",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--tier-blocks", type=int, default=1024, help="CPU pool capacity in blocks (16 tok each)"
    )
    parser.add_argument("--waves", type=int, default=2, help="[overflow] workload repeats")
    parser.add_argument("--groups", type=int, default=4, help="[overflow] distinct prefixes")
    parser.add_argument("--per-group", type=int, default=2, help="[overflow] requests per prefix")
    parser.add_argument(
        "--prefix-sentences", type=int, default=32, help="[overflow] filler sentences per prefix"
    )
    parser.add_argument("--overflow-gen-len", type=int, default=16, help="[overflow] tokens each")
    parser.add_argument(
        "--overflow-gpu-blocks",
        type=int,
        default=128,
        help="[overflow] constrained GPU pool in scheduler blocks (each 16 tokens); "
        "internally multiplied by 16 for LLMEngine. Default 128 scheduler blocks "
        "(=2048 tokens) forces eviction at ~G2 with 4 groups x 42-block prefixes",
    )
    parser.add_argument(
        "--eager", action="store_true", help="Disable CUDA graphs (capture costs build time)"
    )
    parser.add_argument(
        "--log-dir", default=LOG_DIR, help="Evidence JSON directory; empty string disables"
    )


def run(args: argparse.Namespace) -> int:
    require_gpus(1)
    use_cuda_graph = not args.eager
    results: dict[str, dict] = {}

    if args.arm in ("accuracy", "both"):
        prompts = expand_prompts(PROMPTS, args.parity_batch)
        params = sampling_params(args.parity_gen_len)
        print_run_header(
            f"kv-transfer accuracy | {args.model_dir}",
            {
                "batch": args.parity_batch,
                "gen_len": args.parity_gen_len,
                "max_seq_len": args.max_seq_len,
                "max_num_seqs": args.max_num_seqs,
                "tier_blocks": args.tier_blocks,
                "cuda_graph": use_cuda_graph,
            },
        )
        print(f"\n=== accuracy: {len(prompts)} greedy requests, tier off vs on ===")
        off = measure_parity_arm(
            args.model_dir,
            prompts,
            params,
            tier=False,
            kv_blocks=None,
            tier_blocks=args.tier_blocks,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            use_cuda_graph=use_cuda_graph,
        )
        print(off.row())
        on = measure_parity_arm(
            args.model_dir,
            prompts,
            params,
            tier=True,
            kv_blocks=None,
            tier_blocks=args.tier_blocks,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            use_cuda_graph=use_cuda_graph,
        )
        print(on.row())
        results["accuracy"] = {
            "arms": {"tier_off": off.as_dict(), "tier_on": on.as_dict()},
            "criteria": compare_parity(off, on, args.tpot_tolerance),
        }

    if args.arm in ("overflow", "both"):
        prompts = shared_prefix_prompts(args.groups, args.per_group, args.prefix_sentences)
        params = sampling_params(args.overflow_gen_len)
        print_run_header(
            f"kv-transfer overflow | {args.model_dir}",
            {
                "requests": len(prompts),
                "groups": args.groups,
                "waves": args.waves,
                "gen_len": args.overflow_gen_len,
                "gpu_blocks": args.overflow_gpu_blocks,
                "tier_blocks": args.tier_blocks,
                "cuda_graph": use_cuda_graph,
            },
        )
        print(
            f"\n=== overflow: {len(prompts)} requests x {args.waves} waves, "
            f"gpu pool {args.overflow_gpu_blocks} blocks ==="
        )
        off = measure_overflow_arm(
            args.model_dir,
            prompts,
            params,
            args.waves,
            tier=False,
            kv_blocks=args.overflow_gpu_blocks,
            tier_blocks=args.tier_blocks,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            use_cuda_graph=use_cuda_graph,
        )
        print(off.row())
        on = measure_overflow_arm(
            args.model_dir,
            prompts,
            params,
            args.waves,
            tier=True,
            kv_blocks=args.overflow_gpu_blocks,
            tier_blocks=args.tier_blocks,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            use_cuda_graph=use_cuda_graph,
        )
        print(on.row())
        results["overflow"] = {
            "arms": {"tier_off": off.as_dict(), "tier_on": on.as_dict()},
            "criteria": compare_overflow(off, on),
        }

    if args.log_dir:
        import rapid_llm

        path = timestamped_log_path(args.log_dir, "kv_transfer")
        write_json_log(
            path,
            {
                "model": args.model_dir,
                "version": rapid_llm.__version__,
                "matter": "CPU tier acceptance — accuracy parity, TPOT tax, overflow reuse",
                "arm": args.arm,
                "parity_batch": args.parity_batch,
                "parity_gen_len": args.parity_gen_len,
                "tpot_tolerance": args.tpot_tolerance,
                "waves": args.waves,
                "groups": args.groups,
                "per_group": args.per_group,
                "prefix_sentences": args.prefix_sentences,
                "overflow_gen_len": args.overflow_gen_len,
                "overflow_gpu_blocks": args.overflow_gpu_blocks,
                "max_seq_len": args.max_seq_len,
                "max_num_seqs": args.max_num_seqs,
                "tier_blocks": args.tier_blocks,
                "use_cuda_graph": use_cuda_graph,
                "sampling": "greedy",
                "offline": True,
                "reproduce": (
                    "python benchmarks/engine/run.py kv-transfer "
                    f"--model-dir {args.model_dir} --arm {args.arm} "
                    f"--parity-batch {args.parity_batch} --parity-gen-len {args.parity_gen_len} "
                    f"--waves {args.waves} --groups {args.groups} --per-group {args.per_group} "
                    f"--prefix-sentences {args.prefix_sentences} "
                    f"--overflow-gpu-blocks {args.overflow_gpu_blocks} "
                    f"--tier-blocks {args.tier_blocks} --max-num-seqs {args.max_num_seqs}"
                ),
            },
            results,
        )
    return 0
