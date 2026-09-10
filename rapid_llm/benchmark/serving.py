"""Online serving benchmark — sglang ``serving`` mirror, one endpoint type.

OpenAI-compatible ``/v1/chat/completions`` over SSE. Timing is client-side
black-box: the first chunk stamps TTFT, chunk gaps stamp ITL,
TPOT = (E2EL - TTFT) / (tokens - 1) — the client never trusts the server with
the clock. Traffic (vLLM's convention): ``--burstiness 1`` is Poisson, other
values draw Gamma-gapped arrivals, normalized to ``num_prompts / rate`` so a
finite sample cannot bias the offered load; ``--max-concurrency`` caps
in-flight requests via a semaphore, sglang-style.

Usage:
    # 1. serve:  .venv/bin/rapid-llm serve --model my_weight/Qwen2.5-0.5B-Instruct
    .venv/bin/python -m rapid_llm.benchmark.serving \
        --host 127.0.0.1 --port 30000 --model <served-name> \
        --num-prompts 32 --random-input-len 1024 --random-output-len 128 \
        --request-rate 4 --max-concurrency 8 --ttft-slo-ms 500 \
        --log-dir docs/benchmark_logs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from .datasets import add_dataset_args, get_dataset
from .metrics import pctl
from .utils import (
    count_gen_tokens,
    get_tokenizer,
    gpu_tag,
    timestamped_log_path,
    wait_for_endpoint,
    write_json_log,
)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    output_len: int
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list = field(default_factory=list)
    output_tokens: int = 0


async def request_chat_completions(
    request: RequestFuncInput, session: httpx.AsyncClient, output: RequestFuncOutput
) -> RequestFuncOutput:
    """Stream one chat completion, stamping TTFT/ITL on every SSE chunk."""
    payload: dict = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.ignore_eos:
        payload["ignore_eos"] = True  # honoured by vLLM; ignored by strict servers

    st = time.perf_counter()
    last = st
    try:
        # stream(): SSE needs unbuffered reads; a plain post() would await the
        # whole body before the first chunk could be stamped.
        async with session.stream("POST", request.api_url, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(f"HTTP {response.status_code}: {body[:200]!r}")
            async for raw in response.aiter_lines():
                if not raw.startswith("data:"):
                    continue
                chunk = raw[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                body = json.loads(chunk)
                if output.ttft == 0.0:
                    output.ttft = time.perf_counter() - st
                else:
                    output.itl.append(time.perf_counter() - last)
                last = time.perf_counter()
                choices = body.get("choices") or []
                if choices:
                    output.generated_text += choices[0].get("delta", {}).get("content") or ""
                if body.get("usage"):
                    output.output_tokens = body["usage"].get("completion_tokens") or 0
        output.latency = time.perf_counter() - st
        output.success = True
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
        output.success = False
        output.generated_text = f"{type(exc).__name__}: {exc}"
    return output


def arrival_times(num: int, rate: float, burstiness: float, rng) -> list[float]:
    """vLLM's traffic model: Poisson at burstiness=1, Gamma gaps otherwise,
    normalized to the nominal duration so sampling noise never biases the load."""
    if burstiness == 1.0:
        intervals = rng.exponential(1.0 / rate, num)
    else:
        intervals = rng.gamma(shape=burstiness, scale=1.0 / (rate * burstiness), size=num)
    total = num / rate
    intervals = intervals * (total / intervals.sum())
    return np.cumsum(intervals).tolist()


def summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {name: "n/a"}
    return {
        name: {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p50": pctl(values, 50),
            "p90": pctl(values, 90),
            "p99": pctl(values, 99),
        }
    }


async def run_benchmark(rows, args, api_url: str, input_tokens: int, tokenizer=None) -> dict:
    async def one(row, semaphore, done) -> RequestFuncOutput:
        request = RequestFuncInput(
            prompt=row.prompt,
            api_url=api_url,
            model=args.model,
            output_len=row.output_len,
            ignore_eos=args.ignore_eos,
        )
        async with semaphore:
            out = await request_chat_completions(request, session, RequestFuncOutput())
        if out.success and out.output_tokens == 0 and out.generated_text:
            # Not every server honours stream_options.include_usage; count the
            # streamed text locally so TPOT/throughput stay defined.
            out.output_tokens = count_gen_tokens([out.generated_text], tokenizer)
        done.add(1)
        if done.mark % max(1, len(rows) // 10) == 0:
            print(f"  {done.mark}/{len(rows)} requests completed", flush=True)
        return out

    limits = httpx.Limits(max_connections=args.max_concurrency or 512)
    timeout = httpx.Timeout(timeout=args.timeout_s, connect=10.0)
    rng = np.random.default_rng(args.seed)
    starts = arrival_times(len(rows), args.request_rate, args.burstiness, rng)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as session:
        # Warm up: fill decode kernels' caches with short requests outside the load.
        warmups = [
            request_chat_completions(
                RequestFuncInput(rows[0].prompt, api_url, args.model, 8),
                session,
                RequestFuncOutput(),
            )
            for _ in range(args.warmup_requests)
        ]
        await asyncio.gather(*warmups)

        semaphore = asyncio.Semaphore(args.max_concurrency or 512)
        done = _Counter()
        started = time.perf_counter()

        async def launch(index: int, row) -> RequestFuncOutput:
            # Hold the request until its scheduled arrival time (the Poisson/
            # Gamma schedule), then queue it through the concurrency cap.
            delay = starts[index]
            if delay > 0:
                await asyncio.sleep(delay)
            return await one(row, semaphore, done)

        outputs = await asyncio.gather(*(launch(i, row) for i, row in enumerate(rows)))
        duration = time.perf_counter() - started

    return collect_metrics(outputs, duration, args, input_tokens=input_tokens)


class _Counter:
    """Mutable completion counter the per-request tasks can bump."""

    def __init__(self):
        self.mark = 0

    def add(self, _):
        self.mark += 1


def collect_metrics(outputs, duration: float, args, input_tokens: int = 0) -> dict:
    completed = [o for o in outputs if o.success]
    failed = len(outputs) - len(completed)
    ttfts = [o.ttft * 1000 for o in completed]
    tpots = [
        (o.latency - o.ttft) / (o.output_tokens - 1) * 1000
        for o in completed
        if o.output_tokens > 1 and o.latency > o.ttft
    ]
    itls = [g * 1000 for o in completed for g in o.itl]
    output_tokens = sum(o.output_tokens for o in completed)

    metrics: dict = {
        "duration_s": duration,
        "completed": len(completed),
        "failed": failed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_throughput": len(outputs) / duration,
        "output_token_throughput": output_tokens / duration,
        "goodput": None,
    }
    metrics.update(summarize("ttft_ms", ttfts))
    metrics.update(summarize("tpot_ms", tpots))
    metrics.update(summarize("itl_ms", itls))
    if args.ttft_slo_ms or args.tpot_slo_ms:
        good = sum(
            1
            for o in completed
            if (not args.ttft_slo_ms or o.ttft * 1000 <= args.ttft_slo_ms)
            and (
                not args.tpot_slo_ms
                or o.output_tokens <= 1
                or (o.latency - o.ttft) / (o.output_tokens - 1) * 1000 <= args.tpot_slo_ms
            )
        )
        metrics["goodput"] = good / duration if duration else 0.0
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", required=True, help="The served model name")
    parser.add_argument("--request-rate", type=float, default=4.0, help="Requests/s")
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Poisson at 1; Gamma gaps otherwise (vLLM convention)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=0, help="In-flight cap; 0 = unbounded"
    )
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Send ignore_eos in the payload (vLLM servers honour it)",
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=None, help="Goodput SLO: TTFT")
    parser.add_argument("--tpot-slo-ms", type=float, default=None, help="Goodput SLO: TPOT")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--tag", default="bench_serving")
    add_dataset_args(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = f"http://{args.host}:{args.port}"
    api_url = f"{base}/v1/chat/completions"

    heartbeat_s = wait_for_endpoint(f"{base}/health")
    print(f"endpoint ready (heartbeat {heartbeat_s * 1000:.1f} ms)")

    tokenizer = get_tokenizer(args.model)
    rows = get_dataset(args, tokenizer)
    if not rows:
        print("dataset produced no rows", file=sys.stderr)
        return 1

    input_tokens = sum(row.prompt_len for row in rows)
    metrics = asyncio.run(run_benchmark(rows, args, api_url, input_tokens, tokenizer))
    metrics["heartbeat_latency_s"] = heartbeat_s

    print(f"\n{'─' * 66}")
    print(
        f"Successful: {metrics['completed']} / {len(rows)}   duration {metrics['duration_s']:.2f} s"
    )
    print(f"Request throughput:  {metrics['request_throughput']:9.2f} req/s")
    print(f"Output throughput:   {metrics['output_token_throughput']:9.1f} tok/s")
    if metrics.get("goodput") is not None:
        print(
            f"Goodput (SLO {args.ttft_slo_ms}ms/{args.tpot_slo_ms}ms): "
            f"{metrics['goodput']:9.2f} req/s"
        )
    for key in ("ttft_ms", "tpot_ms", "itl_ms"):
        stats = metrics.get(key)
        if isinstance(stats, dict):
            print(
                f"{key.upper():>8}: mean {stats['mean']:8.2f} | median {stats['median']:8.2f}"
                f" | p99 {stats['p99']:8.2f} ms"
            )
    print(f"{'─' * 66}")

    if args.log_dir:
        config = {
            "host": args.host,
            "port": args.port,
            "model": args.model,
            "request_rate": args.request_rate,
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
            "num_prompts": args.num_prompts,
            "dataset_name": args.dataset_name,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "seed": args.seed,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "gpu": gpu_tag(),
        }
        write_json_log(
            timestamped_log_path(args.log_dir, f"{args.tag}_{gpu_tag()}"), config, metrics
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
