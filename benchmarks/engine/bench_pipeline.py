"""Pipeline depth A/B: the launch/harvest overlap, depth 1 versus depth 2.

ROADMAP v0.12.0 accepts the N-batch pipeline on one number: at depth 2 the
host's wait on readback events must fall under 50% of depth 1's. Both arms run
the same saturated batch through the same engine configuration, differing only
in the depth; the wait itself is the engine's
``rapid_llm:kv_pipeline_sync_wait_seconds_total`` counter — host seconds spent
blocked in ``event.synchronize()`` — read as a delta around the timed run so
the warm-up's drain is excluded. The share is that wait over the run's wall
clock (``run_requests``'s ``time.monotonic`` basis, the clock every offline
benchmark's TTFT/TPOT already use).

The default workload is saturation inside the CUDA-graph capture grid
(batch 128): measured on the A10 + Qwen3-0.6B, a smaller batch leaves the loop
host-bound (N=1 pays no wait at all, nothing to compare), and a batch *past*
the captured grid is non-deterministic with or without the pipeline — depth 1
against depth 1 differs by ~10/256 requests — so token agreement is only
asserted inside the grid.

Usage:
    python benchmarks/engine/run.py pipeline --model-dir my_weight/Qwen3-0.6B
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

CKPT = "my_weight/Qwen3-0.6B"
LOG_DIR = "docs/benchmark_logs/kv_transfer"


@dataclass
class Arm:
    """One depth's timed run plus the host-sync wait it paid."""

    label: str
    depth: int
    total_s: float
    steps: int
    gen_tokens: int
    ttft_ms: float
    tpot_ms: float
    sync_wait_s: float
    texts: list[str] = field(default_factory=list)

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    @property
    def wait_share(self) -> float:
        """Host readback wait as a fraction of the measured wall clock."""
        return self.sync_wait_s / self.total_s if self.total_s else 0.0

    def row(self) -> str:
        return (
            f"depth {self.depth}  {self.total_s:7.2f}s | "
            f"TTFT {self.ttft_ms:7.1f} ms | TPOT {self.tpot_ms:6.2f} ms | "
            f"TPS {self.tps:7.1f} tok/s | sync wait {self.sync_wait_s:7.3f}s "
            f"({self.wait_share:6.1%})"
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("texts")  # the agreement check's input, not a report number
        return {**data, "tps": self.tps, "wait_share": self.wait_share}


def measure_depth(
    model_dir: str,
    prompts: list[str],
    params,
    depth: int,
    *,
    kv_blocks: int | None,
    max_seq_len: int,
    max_num_seqs: int,
    use_cuda_graph: bool,
) -> Arm:
    """Build one depth, warm it up, and time one saturated batch."""
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        max_gpu_num_blocks=kv_blocks,
        use_cuda_graph=use_cuda_graph,
        pipeline=True,
        pipeline_depth=depth,
    )
    # Warm-up covers kernel autotune and the pipeline's own first drain; its
    # sync waits must not be charged to the arm, hence the counter delta.
    engine.generate(prompts[:2], sampling_params(8))
    counter = engine.metrics.kv_pipeline_sync_wait
    before = counter.value()
    run = run_requests(engine, prompts, params)
    waited = counter.value() - before
    result = run.result(len(prompts))
    arm = Arm(
        label=f"depth {depth}",
        depth=depth,
        total_s=run.total_s,
        steps=result.steps,
        gen_tokens=run.gen_tokens,
        ttft_ms=result.ttft_ms,
        tpot_ms=result.tpot_ms,
        sync_wait_s=waited,
        texts=run.texts,
    )
    del engine
    free_gpu()
    return arm


def compare(arms: dict[str, Arm]) -> dict | None:
    """The ROADMAP acceptance: depth 2's wait share under half of depth 1's."""
    if "depth_1" not in arms or "depth_2" not in arms:
        return None
    d1, d2 = arms["depth_1"], arms["depth_2"]
    # Greedy parity modulo the depth-late stop: one output is the other's
    # prefix (depth 2 retires an EOS request up to two steps later).
    agree = len(d1.texts) == len(d2.texts) and all(
        a == b or a.startswith(b) or b.startswith(a)
        for a, b in zip(d1.texts, d2.texts, strict=True)
    )
    ratio = d2.wait_share / d1.wait_share if d1.wait_share else float("inf")
    accepted = d2.wait_share < 0.5 * d1.wait_share
    print(
        f"\n-> host sync wait share {d1.wait_share:.1%} -> {d2.wait_share:.1%} "
        f"(depth2/depth1 = {ratio:.3f}, threshold 0.500): "
        f"{'ACCEPTED' if accepted else 'REJECTED'}"
    )
    print(f"-> greedy texts {'agree' if agree else 'DISAGREE'} across depths")
    return {
        "criterion": "depth-2 host sync wait share < 50% of depth-1",
        "wait_share_depth_1": d1.wait_share,
        "wait_share_depth_2": d2.wait_share,
        "ratio": ratio,
        "tokens_agree": agree,
        "accepted": accepted,
    }


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=CKPT)
    parser.add_argument("--batch", type=int, default=128, help="Concurrent requests (saturation)")
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Keep at or under the CUDA-graph capture grid (128) for token agreement",
    )
    parser.add_argument(
        "--kv-blocks",
        type=int,
        default=None,
        help="KV pool in blocks (16 tokens each); profiled when omitted",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--eager", action="store_true", help="Disable CUDA graphs (capture costs build time)"
    )
    parser.add_argument(
        "--log-dir", default=LOG_DIR, help="Evidence JSON directory; empty string disables"
    )


def run(args: argparse.Namespace) -> int:
    require_gpus(1)
    prompts = expand_prompts(PROMPTS, args.batch)
    params = sampling_params(args.max_gen_len)
    print_run_header(
        f"pipeline depth A/B | {args.model_dir}",
        {
            "batch": args.batch,
            "gen_len": args.max_gen_len,
            "max_seq_len": args.max_seq_len,
            "max_num_seqs": args.max_num_seqs,
            "kv_blocks": args.kv_blocks if args.kv_blocks else "profiled",
            "depths": " ".join(map(str, args.depths)),
            "cuda_graph": not args.eager,
        },
    )

    arms: dict[str, Arm] = {}
    for depth in args.depths:
        print(f"\n=== depth {depth}: {len(prompts)} requests x {args.max_gen_len} tokens ===")
        arm = measure_depth(
            args.model_dir,
            prompts,
            params,
            depth,
            kv_blocks=args.kv_blocks,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            use_cuda_graph=not args.eager,
        )
        arms[f"depth_{depth}"] = arm
        print(arm.row())

    acceptance = compare(arms)

    if args.log_dir:
        import rapid_llm

        path = timestamped_log_path(args.log_dir, "pipeline_depth")
        write_json_log(
            path,
            {
                "model": args.model_dir,
                "version": rapid_llm.__version__,
                "matter": "pipeline depth A/B — host sync wait share",
                "batch": args.batch,
                "max_gen_len": args.max_gen_len,
                "max_seq_len": args.max_seq_len,
                "max_num_seqs": args.max_num_seqs,
                "kv_blocks": args.kv_blocks,
                "depths": args.depths,
                "pipeline": True,
                "use_cuda_graph": not args.eager,
                "sampling": "greedy",
                "offline": True,
                "reproduce": (
                    "python benchmarks/engine/run.py pipeline "
                    f"--model-dir {args.model_dir} --batch {args.batch} "
                    f"--max-gen-len {args.max_gen_len} --max-seq-len {args.max_seq_len} "
                    f"--max-num-seqs {args.max_num_seqs} "
                    f"--depths {' '.join(map(str, args.depths))}"
                ),
            },
            {
                "arms": {key: arm.as_dict() for key, arm in arms.items()},
                "acceptance": acceptance,
            },
        )
    return 0
