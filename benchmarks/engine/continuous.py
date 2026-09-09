"""Continuous vs static batching, across offline and skewed-arrival scenarios.

The same prompts run through a one-shot engine and a continuous engine;
``Measurement`` captures TTFT / TPOT / throughput per scenario so the
gain from continuous batching is one comparable table.

Usage:
    python benchmarks/engine/run.py scheduler continuous --model-dir <ckpt>
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import torch

from rapid_llm.benchmark import (
    PROMPTS,
    count_gen_tokens,
    expand_prompts,
    free_gpu,
    pctl,
    run_requests,
    sampling_params,
    write_json_log,
)
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.llm_engine import LLMEngine

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"


@dataclass
class Measurement:
    """One measurement; ``gen_tokens`` counts only tokens actually delivered."""

    label: str
    total_s: float
    gen_tokens: int
    requests: int
    ttfts_ms: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    wasted_tokens: int = 0  # tokens a single batch-wide cap generated past a request's own

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    def row(self) -> str:
        ttft = statistics.mean(self.ttfts_ms) if self.ttfts_ms else 0.0
        latency = statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
        return (
            f"{self.label:22s} {self.total_s:7.2f}s | TPS {self.tps:8.1f} tok/s | "
            f"TTFT mean {ttft:8.1f} ms p95 {pctl(self.ttfts_ms, 95):8.1f} ms | "
            f"latency mean {latency:8.1f} ms | {self.gen_tokens} tok"
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "tps": self.tps}


@contextmanager
def static_engine(model_dir: str, kv_blocks: int, max_seq_len: int, warm_prompts: list[str]):
    """LLMEngine with CUDA graph; the KV budget must be stated (as in one_batch)."""
    engine = LLMEngine(
        model_dir, max_seq_len=max_seq_len, max_gpu_num_blocks=kv_blocks, use_cuda_graph=True
    )
    engine.model_runner.enable_cuda_graph()
    engine.generate_text(
        [engine.tokenizer.encode(p, add_special_tokens=True) for p in warm_prompts],
        sampling_params(8),
    )
    try:
        yield engine
    finally:
        del engine
        free_gpu()


@contextmanager
def continuous_engine(
    model_dir: str, kv_blocks: int, max_seq_len: int, max_num_seqs: int, warm_prompts: list[str]
):
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        max_gpu_num_blocks=kv_blocks,
        use_cuda_graph=True,
    )
    engine.generate(warm_prompts, sampling_params(8))  # autotune
    try:
        yield engine
    finally:
        del engine
        free_gpu()


def skewed_caps(count: int, short: int, long: int, every: int = 4) -> list[int]:
    """One long-output request per ``every``, the rest short."""
    return [long if index % every == 0 else short for index in range(count)]


def measure_static_offline(
    model_dir: str, prompts: list[str], params, kv_blocks: int, max_seq_len: int
) -> Measurement:
    with static_engine(model_dir, kv_blocks, max_seq_len, warm_prompts=prompts) as engine:
        token_ids = [engine.tokenizer.encode(p, add_special_tokens=True) for p in prompts]

        # The streaming interface yields TTFT and the full text in one pass, so the
        # two numbers do not need two runs.
        torch.cuda.synchronize()
        started = time.perf_counter()
        first_token_at = 0.0
        completions = [""] * len(prompts)
        for deltas in engine.generate(token_ids, params):
            if not first_token_at and any(deltas):
                first_token_at = time.perf_counter()
            for index, delta in enumerate(deltas):
                completions[index] += delta
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        ttft = (first_token_at - started) * 1000 if first_token_at else 0.0
        tokenizer = engine.tokenizer
    # Lockstep: the batch starts together and is held back by its longest sequence,
    # so TTFT and latency are shared across all requests.
    return Measurement(
        label="static (one-shot)",
        total_s=elapsed,
        gen_tokens=count_gen_tokens(completions, tokenizer),
        requests=len(prompts),
        ttfts_ms=[ttft] * len(prompts),
        latencies_ms=[elapsed * 1000] * len(prompts),
    )


def measure_static_skewed(
    model_dir: str, prompts: list[str], caps: list[int], kv_blocks: int, max_seq_len: int
) -> Measurement:
    """One-shot batching runs to the longest cap; a short request's excess is waste."""
    with static_engine(model_dir, kv_blocks, max_seq_len, warm_prompts=prompts) as engine:
        token_ids = [engine.tokenizer.encode(p, add_special_tokens=True) for p in prompts]

        params = sampling_params(max(caps))
        torch.cuda.synchronize()
        started = time.perf_counter()
        completions = engine.generate_text(token_ids, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokenizer = engine.tokenizer

    produced = [len(tokenizer.encode(text, add_special_tokens=False)) for text in completions]
    useful = sum(min(count, cap) for count, cap in zip(produced, caps, strict=True))
    return Measurement(
        label="static (one cap)",
        total_s=elapsed,
        gen_tokens=useful,
        requests=len(prompts),
        latencies_ms=[elapsed * 1000] * len(prompts),
        wasted_tokens=sum(produced) - useful,
    )


def measure_continuous_batch(
    model_dir: str,
    prompts: list[str],
    params_list: list,
    kv_blocks: int,
    max_seq_len: int,
    max_num_seqs: int,
    label: str,
) -> Measurement:
    """Continuous batching over a batch submitted at once; ``params_list`` is per request."""
    with continuous_engine(
        model_dir, kv_blocks, max_seq_len, max_num_seqs, warm_prompts=prompts[:2]
    ) as engine:
        run = run_requests(engine, prompts, params_list)
    return Measurement(
        label=label,
        total_s=run.total_s,
        gen_tokens=run.gen_tokens,
        requests=len(prompts),
        ttfts_ms=run.ttfts_ms(),
        latencies_ms=run.latencies_ms(),
    )


def measure_static_online(
    model_dir: str, prompts: list[str], params, kv_blocks: int, max_seq_len: int, interval: float
) -> Measurement:
    """One-shot batching cannot join a running batch, so requests are served serially."""
    with static_engine(model_dir, kv_blocks, max_seq_len, warm_prompts=prompts[:1]) as engine:
        started = time.perf_counter()
        completions, latencies = [], []
        for index, prompt in enumerate(prompts):
            arrival = started + index * interval
            # Wait for the arrival; if the engine is busy, queueing time counts as latency.
            now = time.perf_counter()
            if now < arrival:
                time.sleep(arrival - now)
            encoded = [engine.tokenizer.encode(prompt, add_special_tokens=True)]
            completions.append(engine.generate_text(encoded, params)[0])
            latencies.append((time.perf_counter() - arrival) * 1000)
        elapsed = time.perf_counter() - started
        tokenizer = engine.tokenizer
    return Measurement(
        label="static (serial)",
        total_s=elapsed,
        gen_tokens=count_gen_tokens(completions, tokenizer),
        requests=len(prompts),
        latencies_ms=latencies,
    )


def measure_continuous_online(
    model_dir: str,
    prompts: list[str],
    params,
    kv_blocks: int,
    max_seq_len: int,
    max_num_seqs: int,
    interval: float,
) -> Measurement:
    with continuous_engine(
        model_dir, kv_blocks, max_seq_len, max_num_seqs, warm_prompts=prompts[:2]
    ) as engine:
        started = time.perf_counter()
        arrivals: dict = {}
        pending = list(enumerate(prompts))
        live = []

        while pending or engine.has_unfinished_requests():
            now = time.perf_counter()
            # A request whose arrival time has passed joins the running batch at once.
            while pending and now - started >= pending[0][0] * interval:
                index, prompt = pending.pop(0)
                request = engine.add_request(prompt, params)
                arrivals[request.request_id] = started + index * interval
                live.append(request)
            if not engine.has_unfinished_requests():
                if pending:
                    time.sleep(0.001)
                    continue
                break
            engine.step()
        elapsed = time.perf_counter() - started
    return Measurement(
        label="continuous",
        total_s=elapsed,
        gen_tokens=sum(len(r.output_token_ids) for r in live),
        requests=len(prompts),
        ttfts_ms=[
            (r.first_token_time - arrivals[r.request_id]) * 1000 for r in live if r.first_token_time
        ],
        latencies_ms=[
            (r.finish_time - arrivals[r.request_id]) * 1000 for r in live if r.finish_time
        ],
    )


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=CKPT)
    parser.add_argument(
        "--scenario",
        choices=["offline", "offline-skew", "online", "both", "all"],
        default="all",
    )
    parser.add_argument("--batch", type=int, default=16, help="Request count")
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--kv-blocks", type=int, default=40960)
    parser.add_argument(
        "--interval", type=float, default=0.25, help="Arrival interval for the online scenario (s)"
    )
    parser.add_argument(
        "--skew-short", type=int, default=32, help="Cap for the short requests in offline-skew"
    )
    parser.add_argument(
        "--skew-long", type=int, default=256, help="Cap for the long requests in offline-skew"
    )
    parser.add_argument("--json", default=None)


def run(args: argparse.Namespace) -> int:
    prompts = expand_prompts(PROMPTS, args.batch)
    params = sampling_params(args.max_gen_len)
    results: dict[str, Measurement] = {}

    if args.scenario in ("offline", "both", "all"):
        print(f"=== offline: {args.batch} requests submitted together ===")
        results["offline_static"] = measure_static_offline(
            args.model_dir, prompts, params, args.kv_blocks, args.max_seq_len
        )
        results["offline_continuous"] = measure_continuous_batch(
            args.model_dir,
            prompts,
            [params] * len(prompts),
            args.kv_blocks,
            args.max_seq_len,
            args.max_num_seqs,
            label="continuous",
        )
        for key in ("offline_static", "offline_continuous"):
            print(results[key].row())
        speedup = results["offline_continuous"].tps / results["offline_static"].tps
        print(f"-> continuous batching throughput x{speedup:.2f}\n")

    if args.scenario in ("offline-skew", "all"):
        caps = skewed_caps(len(prompts), args.skew_short, args.skew_long)
        print(
            f"=== offline-skew: {caps.count(args.skew_long)} long ({args.skew_long}) + "
            f"{caps.count(args.skew_short)} short ({args.skew_short}) requests ==="
        )
        results["skew_static"] = measure_static_skewed(
            args.model_dir, prompts, caps, args.kv_blocks, args.max_seq_len
        )
        results["skew_continuous"] = measure_continuous_batch(
            args.model_dir,
            prompts,
            [sampling_params(cap) for cap in caps],
            args.kv_blocks,
            args.max_seq_len,
            args.max_num_seqs,
            label="continuous (per-req)",
        )
        for key in ("skew_static", "skew_continuous"):
            print(results[key].row())
        print(
            f"-> static wasted {results['skew_static'].wasted_tokens} tokens nobody asked for; "
            f"useful throughput x{results['skew_continuous'].tps / results['skew_static'].tps:.2f}, "
            f"wall time x{results['skew_static'].total_s / results['skew_continuous'].total_s:.2f} faster\n"
        )

    if args.scenario in ("online", "both", "all"):
        print(f"=== online: {args.batch} requests, {args.interval * 1000:.0f} ms apart ===")
        results["online_static"] = measure_static_online(
            args.model_dir, prompts, params, args.kv_blocks, args.max_seq_len, args.interval
        )
        results["online_continuous"] = measure_continuous_online(
            args.model_dir,
            prompts,
            params,
            args.kv_blocks,
            args.max_seq_len,
            args.max_num_seqs,
            args.interval,
        )
        for key in ("online_static", "online_continuous"):
            print(results[key].row())
        static_latency = statistics.mean(results["online_static"].latencies_ms)
        cb_latency = statistics.mean(results["online_continuous"].latencies_ms)
        print(
            f"-> mean latency {static_latency:.0f} ms -> {cb_latency:.0f} ms "
            f"(x{static_latency / cb_latency:.2f} better)"
        )
        print(
            f"-> throughput x{results['online_continuous'].tps / results['online_static'].tps:.2f}\n"
        )

    if args.json:
        write_json_log(args.json, vars(args), {k: v.as_dict() for k, v in results.items()})
    return 0
