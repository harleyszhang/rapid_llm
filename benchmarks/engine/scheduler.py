"""Scheduler benchmarks and diagnostics behind one CLI.

Four subcommands share the workload builders, the offline harness
(:func:`run_workload`) and the JSON archiving convention:

* ``matrix`` — tp x cuda-graph x prefix-cache x chunk-tokens against two
  workloads: shared-prefix waves (measures hit rate and the TTFT gain) and
  long prompts beside running decodes (measures the chunked-prefill step mix
  and the graph replay coverage of resumed chunks).
* ``serving`` — the same engine over the HTTP API (``rapid-llm serve``):
  TTFT/TPOT/aggregate throughput per concurrency level, plus three
  prefix-parity checks that separate batch-dependent arithmetic from leaked
  request state.
* ``diag-prefix`` — the shared-prefix workload split by admission wave, with
  KV-copy timings and the stream timeline.
* ``diag-preempt`` — an oversubscribed slot pool with preemption on, verified
  against a plain run for greedy-text agreement.

Usage:
    python benchmarks/engine/run.py scheduler matrix --model-dir CKPT --graph --prefix-cache
    python benchmarks/engine/run.py scheduler serving --model-dir CKPT --schemes fp8
    python benchmarks/engine/run.py scheduler diag-prefix --model-dir CKPT
    python benchmarks/engine/run.py scheduler diag-preempt --model-dir CKPT
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from rapid_llm.benchmark import (
    PROMPTS,
    expand_prompts,
    free_gpu,
    gpu_tag,
    pctl,
    require_gpus,
    run_in_spawned_process,
    run_requests,
    sampling_params,
    write_json_log,
)

#: The modelzoo root when it is set, else the ``my_weight`` convention every other
#: bench script uses — a machine-specific path here would only work on one host.
_MODELZOO = os.environ.get("RAPID_LLM_MODELZOO")
_DEFAULT_MODEL = (
    f"{_MODELZOO}/Qwen/Qwen2___5-0___5B-Instruct" if _MODELZOO else "my_weight/Qwen2.5-0.5B"
)
_FILLER = "Follow every instruction carefully and answer as precisely as you can. "


# --------------------------------------------------------------------------- #
# Shared workload builders and the offline harness
# --------------------------------------------------------------------------- #
def build_prefix_workload(groups: int, per_group: int, sentences: int) -> list[str]:
    """Requests sharing a long prefix per group; groups arrive interleaved."""
    prefixes = [f"You are assistant number {g}. " + _FILLER * sentences for g in range(groups)]
    arrivals = [g for g in range(groups) for _ in range(per_group)]
    return [f"{prefixes[g]}Question {i}: what is {i} plus {i + 1}?" for i, g in enumerate(arrivals)]


def build_chunk_workload(long_prompts: int, short_prompts: int, sentences: int) -> list[str]:
    """A few very long prompts plus shorts, submitted together."""
    prompts = [
        f"Context number {i}: " + _FILLER * sentences + f" Now answer: what is {i} times 2?"
        for i in range(long_prompts)
    ]
    prompts += [f"Question {i}: what is {i} plus {i + 1}?" for i in range(short_prompts)]
    return prompts


@dataclass
class WorkloadStats:
    """One workload run under one configuration."""

    workload: str
    requests: int
    total_s: float
    gen_tokens: int
    ttfts_ms: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    steps: int = 0
    steps_with_prefill: int = 0
    steps_with_decode: int = 0
    chunk_tokens: int = 0
    graph_replays: int = 0
    prefix_hit_rate: float = 0.0
    prefix_hit_tokens: int = 0
    prefix_queried_tokens: int = 0
    texts: list[str] = field(default_factory=list)

    @property
    def ttft_p50_ms(self) -> float:
        return statistics.median(self.ttfts_ms) if self.ttfts_ms else 0.0

    @property
    def ttft_p95_ms(self) -> float:
        return pctl(self.ttfts_ms, 95)

    def row(self) -> str:
        replay_ratio = self.graph_replays / self.steps if self.steps else 0.0
        return (
            f"{self.workload:8s} {self.total_s:6.2f}s | TTFT p50 {self.ttft_p50_ms:7.1f} "
            f"p95 {self.ttft_p95_ms:7.1f} ms | steps {self.steps:4d} "
            f"(prefill {self.steps_with_prefill:4d} decode {self.steps_with_decode:4d}) | "
            f"graph replays {self.graph_replays:4d} ({replay_ratio:4.0%} of steps) | "
            f"prefix hit {self.prefix_hit_rate:4.0%} | {self.gen_tokens} tok"
        )

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "texts"}
        d["ttft_p50_ms"] = self.ttft_p50_ms
        d["ttft_p95_ms"] = self.ttft_p95_ms
        return d


def run_workload(engine, label: str, prompts: list[str], gen_len: int) -> WorkloadStats:
    """Submit all prompts, drive step() to exhaustion, count the schedule mix."""
    stats = WorkloadStats(workload=label, requests=len(prompts), total_s=0.0, gen_tokens=0)

    schedule = engine.scheduler.schedule

    def counted_schedule():
        out = schedule()
        stats.steps += 1
        if out.prefill:
            stats.steps_with_prefill += 1
        if out.decode:
            stats.steps_with_decode += 1
        stats.chunk_tokens += sum(out.prefill_chunk_lens)
        return out

    engine.scheduler.schedule = counted_schedule

    cache = engine.scheduler._prefix_cache
    queried_before = getattr(getattr(cache, "stats", None), "queried_tokens", 0)
    hit_before = getattr(getattr(cache, "stats", None), "hit_tokens", 0)

    mgr = getattr(engine, "_graph_manager", lambda: None)()
    replays_before = mgr.replays if mgr else 0

    run = run_requests(engine, prompts, sampling_params(gen_len))

    engine.scheduler.schedule = schedule  # restore

    stats.total_s = run.total_s
    stats.gen_tokens = run.gen_tokens
    stats.ttfts_ms = run.ttfts_ms()
    stats.latencies_ms = run.latencies_ms()
    stats.texts = run.texts

    if mgr is not None:
        stats.graph_replays = mgr.replays - replays_before
    cache_stats = getattr(cache, "stats", None)
    if cache_stats is not None:
        stats.prefix_hit_tokens = cache_stats.hit_tokens - hit_before
        stats.prefix_queried_tokens = cache_stats.queried_tokens - queried_before
        stats.prefix_hit_rate = (
            stats.prefix_hit_tokens / stats.prefix_queried_tokens
            if stats.prefix_queried_tokens
            else 0.0
        )
    return stats


def _write_texts(json_path: str, results: dict[str, WorkloadStats]) -> None:
    """A greedy-text sidecar next to the JSON log, for cross-run auditing."""
    texts_path = Path(json_path).with_suffix(".texts.json")
    texts_path.write_text(
        json.dumps({name: stats.texts for name, stats in results.items()}, indent=1)
    )
    print(f"-> {texts_path}")


# --------------------------------------------------------------------------- #
# matrix — the feature matrix, in-process
# --------------------------------------------------------------------------- #
def _matrix_main(args: argparse.Namespace) -> int:
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

    require_gpus(args.tp)

    config = {
        "model": args.model_dir,
        "tp": args.tp,
        "cuda_graph": args.graph,
        "prefix_cache": args.prefix_cache,
        "chunk_tokens": args.chunk_tokens or 8192,
        "max_seq_len": args.max_seq_len,
        "max_num_seqs": args.max_num_seqs,
        "kv_blocks": args.kv_blocks,
        "gpu": torch.cuda.get_device_name(0),
    }
    label = (
        f"tp{args.tp} graph={'on' if args.graph else 'off'} "
        f"cache={'on' if args.prefix_cache else 'off'} "
        f"chunk={'on' if args.chunk_tokens else 'off'}"
    )
    print(f"=== {label} ===")

    engine = ContinuousBatchingEngine.from_pretrained(
        args.model_dir,
        max_seq_len=args.max_seq_len,
        max_num_seqs=args.max_num_seqs,
        # A nonzero --chunk-tokens rides max_num_batched_tokens: the scheduler
        # caps every chunk at this budget, which is chunked prefill.
        max_num_batched_tokens=args.chunk_tokens or 8192,
        max_gpu_num_blocks=args.kv_blocks,
        use_cuda_graph=args.graph,
        tensor_parallel_size=args.tp,
        enable_prefix_cache=args.prefix_cache,
    )
    # Warm-up outside both workloads so autotune never lands in the numbers;
    # the prompt shares no prefix with either workload.
    engine.generate(["Warm up the kernels, please."], sampling_params(8))

    workloads = {
        "prefix": (
            build_prefix_workload(args.prefix_groups, args.per_group, args.prefix_sentences),
            args.gen_len,
        ),
        "chunk": (
            build_chunk_workload(args.long_prompts, args.short_prompts, args.chunk_sentences),
            args.gen_len,
        ),
    }
    results = {
        name: run_workload(engine, name, prompts, gen_len)
        for name, (prompts, gen_len) in workloads.items()
    }
    for stats in results.values():
        print(stats.row())

    engine.shutdown()
    del engine
    free_gpu()

    if args.json:
        write_json_log(
            args.json, config, {name: stats.as_dict() for name, stats in results.items()}
        )
        _write_texts(args.json, results)
    return 0


def _configure_matrix(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=0,
        help="max_num_batched_tokens; nonzero caps every prefill chunk",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-blocks", type=int, default=65536)
    parser.add_argument("--prefix-groups", type=int, default=4)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--prefix-sentences", type=int, default=32)
    parser.add_argument("--long-prompts", type=int, default=4)
    parser.add_argument("--short-prompts", type=int, default=4)
    parser.add_argument("--chunk-sentences", type=int, default=100)
    parser.add_argument("--gen-len", type=int, default=48)
    parser.add_argument("--json", default=None)


# --------------------------------------------------------------------------- #
# serving — the HTTP API: TTFT/TPOT/TPS per concurrency, with parity checks
# --------------------------------------------------------------------------- #
_MAX_TOKENS = 64
_CONCURRENCY = (1, 8, 32)
#: A checkpoint load, a KV profile and a graph capture, on every rank.
_READY_TIMEOUT_S = 600.0
#: One wave of requests. Generous: at concurrency 32 the scheduler admits in waves.
_WAVE_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class ServeSpec:
    """One served configuration. Every field appears in :attr:`label`."""

    scheme: str | None = None
    kv_cache_dtype: str = "auto"
    tp: int = 1
    dp: int = 1
    graph: bool = True

    @property
    def label(self) -> str:
        parts = [self.scheme or "bf16"]
        if self.kv_cache_dtype != "auto":
            parts.append("kv" + self.kv_cache_dtype.replace("_", "").replace("e4m3", "fp8"))
        parts.append(f"tp{self.tp}")
        if self.dp > 1:
            parts.append(f"dp{self.dp}")
        parts.append("graph" if self.graph else "eager")
        return "+".join(parts)

    @property
    def gpus_needed(self) -> int:
        return self.tp * self.dp

    def serve_argv(
        self, model_dir: str, port: int, max_seq_len: int, max_num_seqs: int
    ) -> list[str]:
        """The ``rapid-llm serve`` command line for this configuration."""
        argv = [
            sys.executable,
            "-m",
            "rapid_llm.cli",
            "serve",
            "--model-dir",
            model_dir,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--max-seq-len",
            str(max_seq_len),
            "--max-num-seqs",
            str(max_num_seqs),
            "--tensor-parallel-size",
            str(self.tp),
            "--data-parallel-size",
            str(self.dp),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--cuda-graph" if self.graph else "--no-cuda-graph",
        ]
        if self.scheme:
            argv += ["--quantization", self.scheme]
        return argv


@dataclass
class ServeRow:
    """One (configuration, concurrency) measurement, or the record of its failure."""

    config: str
    model: str
    concurrency: int = 0
    ttft_mean_ms: float | None = None
    ttft_p99_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tps: float = 0.0
    wave_s: float = 0.0
    completed: int = 0
    issued: int = 0
    #: Prefix agreement with this server answering the same prompts one at a time.
    batch_prefix: float | None = None
    #: Prefix agreement between repeated copies of one prompt within this wave.
    dup_prefix: float | None = None
    #: Prefix agreement with the same scheme run offline, both sides at batch one.
    offline_prefix: float | None = None
    #: Agreement between copies of one prompt submitted offline in a single batch,
    #: so they share a batch trajectory. Below 1.000 means state is shared between
    #: concurrent requests; at 1.000 a low ``batch``/``dup`` is arithmetic.
    dup_batch_prefix: float | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.completed > 0

    @property
    def success_rate(self) -> float:
        return self.completed / self.issued if self.issued else 0.0


def _free_port() -> int:
    """A port the OS says is free, so a crashed run's socket cannot break the next."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(port: int, process: subprocess.Popen, timeout_s: float) -> None:
    """Poll ``/health`` until the server answers, or explain why it never will.

    The process is checked on every pass: a server that dies during startup
    would otherwise be indistinguishable from a slow one until the timeout.
    """
    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode} during startup")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0).status_code == 200:
                return
        except Exception:
            pass  # not listening yet
        time.sleep(1.0)
    raise TimeoutError(f"server was not ready within {timeout_s:.0f}s")


def _stop(process: subprocess.Popen) -> None:
    """Terminate the server and everything it spawned.

    ``killpg`` rather than ``terminate``: TP followers and DP replicas are
    separate processes holding GPU memory, and signalling only the parent
    leaves them resident, which the next configuration discovers as an OOM.
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), 15)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=60.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=30.0)


@dataclass
class _Result:
    """One request's timeline, as its own frames reported it."""

    ok: bool = False
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    tokens: int = 0
    text: str = ""
    error: str = ""


async def _one_request(client, url: str, model: str, prompt: str, max_tokens: int) -> _Result:
    """Stream one completion, timed from the caller's side of the socket.

    TTFT includes queueing — what the client experiences and what rises with
    concurrency. TPOT comes from the gaps between *this request's own* frames,
    so the queue wait is charged to TTFT alone rather than smeared across both.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "stream": True,
    }
    result = _Result()
    frame_times: list[float] = []
    pieces: list[str] = []
    start = time.perf_counter()
    try:
        async with client.stream("POST", url, json=body, timeout=_WAVE_TIMEOUT_S) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ").strip()
                if payload == "[DONE]":
                    break
                frame_times.append(time.perf_counter())
                chunk = json.loads(payload)
                pieces.append(chunk["choices"][0].get("text") or "")
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if not frame_times:
        result.error = "no frames"
        return result
    result.ok = True
    result.ttft_s = frame_times[0] - start
    # One frame per decode step, so the gaps between frames are per-token latency.
    gaps = [b - a for a, b in itertools.pairwise(frame_times)]
    result.tpot_s = statistics.mean(gaps) if gaps else 0.0
    result.tokens = len(frame_times)
    result.text = "".join(pieces)
    return result


async def _wave(
    port: int, model: str, prompts: list[str], max_tokens: int
) -> tuple[list[_Result], float]:
    """Fire every prompt at once and wait for all of them."""
    import httpx

    url = f"http://127.0.0.1:{port}/v1/completions"
    limits = httpx.Limits(max_connections=len(prompts) + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *(_one_request(client, url, model, p, max_tokens) for p in prompts)
        )
        return list(results), time.perf_counter() - start


async def _serial(port: int, model: str, prompts: list[str], max_tokens: int) -> list[str]:
    """Answer each prompt with nothing else in flight — the batch-of-one reference.

    Same socket, same server, so the only difference from a wave is who else
    was decoding: that is what makes it a reference for batch invariance.
    """
    import httpx

    url = f"http://127.0.0.1:{port}/v1/completions"
    texts: list[str] = []
    async with httpx.AsyncClient() as client:
        for prompt in prompts:
            result = await _one_request(client, url, model, prompt, max_tokens)
            texts.append(result.text if result.ok else "")
    return texts


def _offline_child(payload: dict[str, Any], out: mp.Queue) -> None:
    """The in-process reference: same scheme, batch of one, plus one duplicate batch."""
    try:
        from rapid_llm import SamplingParams
        from rapid_llm.engine import ContinuousBatchingEngine
        from rapid_llm.engine.scheduler import DEFAULT_MAX_NUM_SEQS

        spec: ServeSpec = payload["spec"]
        kwargs: dict[str, Any] = {
            "max_seq_len": payload["max_seq_len"],
            "use_cuda_graph": spec.graph,
            "tensor_parallel_size": spec.tp,
        }
        if spec.scheme:
            kwargs["quantization"] = spec.scheme
        if spec.kv_cache_dtype != "auto":
            kwargs["kv_cache_dtype"] = spec.kv_cache_dtype
        engine = ContinuousBatchingEngine.from_pretrained(payload["model"], **kwargs)
        try:
            # Every sampling field spelled out: the CLI's defaults and the wire
            # protocol's are *not* the same, and an implicit field would make the
            # offline column measure that disagreement instead of the serving path.
            params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_gen_len=payload["max_tokens"],
                repetition_penalty=1.0,
            )
            # One prompt per call: this reference is compared against the
            # server's batch-of-one pass, so batch size must be held equal.
            texts = [
                engine.generate([prompt], params)[0].outputs[0].text
                for prompt in payload["prompts"]
            ]
            # The control for "can requests see each other": one prompt duplicated
            # in a single call shares a batch trajectory by construction, which
            # HTTP arrivals cannot guarantee. Capped at the scheduler's default
            # concurrency ceiling, past which the engine admits in waves.
            copies = min(payload["dup_copies"], DEFAULT_MAX_NUM_SEQS)
            dup_texts = (
                [
                    out.outputs[0].text
                    for out in engine.generate([payload["prompts"][0]] * copies, params)
                ]
                if copies > 1
                else []
            )
        finally:
            engine.shutdown()
    except Exception:
        out.put(("error", traceback.format_exc()))
    else:
        out.put(("ok", {"serial": texts, "dup": dup_texts}))


def _offline_texts(payload: dict[str, Any], timeout_s: float) -> tuple[str, Any]:
    """Run the same scheme in-process, before any server holds the GPUs.

    A separate — and *finished* — process: the reference engine must release
    its KV cache before the server profiles for its own.
    """
    status, result = run_in_spawned_process(_offline_child, payload, timeout_s)
    if status == "timeout":
        return "error", f"offline reference produced nothing in {timeout_s:.0f}s"
    return status, result


def _prefix_rate(tokenizer, want: list[str], got: list[str]) -> float | None:
    """Fraction of each completion reproduced before its first differing token.

    Position-wise over token ids rather than characters: two schemes can spell
    the same prose with different tokens, and it is the tokens that were
    sampled. Greedy decoding is chaotic, so once one token differs the suffixes
    are unrelated; the prefix length is the informative quantity.
    """
    leading = total = 0
    for want_text, got_text in zip(want, got, strict=False):
        want_ids = tokenizer(want_text, add_special_tokens=False).input_ids
        got_ids = tokenizer(got_text, add_special_tokens=False).input_ids
        length = min(len(want_ids), len(got_ids))
        pairs = list(zip(want_ids[:length], got_ids[:length], strict=True))
        leading += next((i for i, (a, b) in enumerate(pairs) if a != b), length)
        total += length
    return leading / total if total else None


def _copies_rate(tokenizer, copies: list[str]) -> float | None:
    """Agreement between answers to one prompt that ran together."""
    if len(copies) < 2:
        return None
    return _prefix_rate(tokenizer, [copies[0]] * (len(copies) - 1), copies[1:])


def _dup_rate(tokenizer, prompts: list[str], texts: list[str]) -> float | None:
    """Agreement between copies of one prompt that shared a wave, worst prompt.

    A wave wider than the prompt list contains repeats, but the scheduler
    admits as requests arrive, so those copies need not have shared a batch at
    every step — this bounds how far batch-dependent arithmetic moved an
    answer. The offline duplicate batch is what separates arithmetic from
    leaked state. The worst prompt's rate, not the mean: one prompt whose
    copies diverged is the finding.
    """
    groups: dict[str, list[str]] = {}
    for prompt, text in zip(prompts, texts, strict=True):
        groups.setdefault(prompt, []).append(text)
    rates = [
        rate
        for copies in groups.values()
        for rate in [_copies_rate(tokenizer, copies)]
        if rate is not None
    ]
    return min(rates) if rates else None


def _render_serving(rows: list[ServeRow]) -> str:
    header = (
        "| Config | Conc | TTFT mean (ms) | TTFT p99 (ms) | TPOT (ms) | TPS "
        "| ok | batch | dup | offline |"
    )
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        if not r.ok:
            lines.append(f"| {r.config} | {r.concurrency} | FAILED: {r.note} |" + " |" * 7)
            continue

        def num(value, spec="{:.2f}"):
            return "—" if value is None else spec.format(value)

        lines.append(
            f"| {r.config} | {r.concurrency} | {num(r.ttft_mean_ms, '{:.1f}')} "
            f"| {num(r.ttft_p99_ms, '{:.1f}')} | {num(r.tpot_ms)} "
            f"| {r.throughput_tps:.1f} | {r.completed}/{r.issued} "
            f"| {num(r.batch_prefix, '{:.3f}')} | {num(r.dup_prefix, '{:.3f}')} "
            f"| {num(r.offline_prefix, '{:.3f}')} |"
        )
    return "\n".join(lines)


def _check_parity(rows: list[ServeRow]) -> None:
    """Say what the parity columns mean instead of leaving three numbers side by side.

    The offline duplicate batch decides the reading, because it is the only
    comparison in which the copies provably shared a batch trajectory:

    * duplicates agree — concurrent requests cannot see each other. A ``batch``
      or ``dup`` below 1.000 then says the answer depends on how many sequences
      shared a step, which is arithmetic (a GEMM tile chosen per M, a padded
      graph bucket, a different reduction order). bf16 argmax ties are decided
      by the last bit, so a 1e-3 logit shift rewrites a completion.
    * duplicates disagree — identical prompts, queued together, same length,
      and the answers still differ. Nothing about arithmetic can do that:
      something is shared between concurrent requests, and the row's
      throughput describes an engine answering the wrong question.
    """
    leaked = [
        r for r in rows if r.ok and r.dup_batch_prefix is not None and r.dup_batch_prefix < 1.0
    ]
    if leaked:
        print("\nWARNING: copies sharing one offline batch disagreed - state is shared:")
        for row in dict.fromkeys(r.config for r in leaked):
            rate = next(r.dup_batch_prefix for r in leaked if r.config == row)
            print(f"  {row}: offline duplicate batch {rate:.3f}")
    varying = [
        r
        for r in rows
        if r.ok
        and r.batch_prefix is not None
        and r.batch_prefix < 1.0
        and (r.dup_batch_prefix is None or r.dup_batch_prefix >= 1.0)
    ]
    if varying:
        print("\nNote: greedy answers vary with batch size (arithmetic, not shared state):")
        for row in varying:
            dup = "—" if row.dup_prefix is None else f"{row.dup_prefix:.3f}"
            print(
                f"  {row.config} @ concurrency {row.concurrency}: "
                f"batch {row.batch_prefix:.3f}, in-wave dup {dup}"
            )


def _first_line(text: str) -> str:
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1][:200] if lines else "unknown failure"


@dataclass
class _ServePlan:
    """Everything one serving measurement needs, so the loop stays readable."""

    model_dir: str
    max_tokens: int = _MAX_TOKENS
    max_seq_len: int = 1024
    concurrency: tuple[int, ...] = _CONCURRENCY
    offline_check: bool = True
    #: Answer the prompt set one request at a time before the waves, as the
    #: reference for batch invariance. Costs one serial pass per configuration.
    batch_check: bool = True
    ready_timeout_s: float = _READY_TIMEOUT_S
    server_log_dir: Path | None = None


def _measure_spec(spec: ServeSpec, plan: _ServePlan, tokenizer) -> list[ServeRow]:
    """Bring one configuration up, run every concurrency level, tear it down."""
    model_name = Path(plan.model_dir).name
    rows: list[ServeRow] = []

    def failed(note: str, concurrency: int = 0) -> list[ServeRow]:
        print(f"  FAILED: {note}")
        return [ServeRow(config=spec.label, model=model_name, concurrency=concurrency, note=note)]

    offline: list[str] | None = None
    dup_batch: float | None = None
    if plan.offline_check:
        print("  offline reference ...", flush=True)
        status, result = _offline_texts(
            {
                "spec": spec,
                "model": plan.model_dir,
                "prompts": expand_prompts(PROMPTS, max(plan.concurrency)),
                "dup_copies": max(plan.concurrency),
                "max_tokens": plan.max_tokens,
                "max_seq_len": plan.max_seq_len,
            },
            plan.ready_timeout_s,
        )
        if status == "ok":
            offline = result["serial"]
            dup_batch = _copies_rate(tokenizer, result["dup"])
            if dup_batch is not None:
                print(f"  offline duplicate batch (shared trajectory): {dup_batch:.3f}")
        else:
            print(f"  offline reference unavailable: {_first_line(result)}")

    port = _free_port()
    argv = spec.serve_argv(plan.model_dir, port, plan.max_seq_len, max(plan.concurrency))
    log_handle = None
    log_path = None
    if plan.server_log_dir is not None:
        plan.server_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = plan.server_log_dir / f"serve_{spec.label.replace('+', '_')}.log"
        log_handle = log_path.open("w")
    # Its own process group, so tearing the server down takes its followers and
    # replicas with it rather than orphaning them on the GPUs.
    process = subprocess.Popen(
        argv,
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_ready(port, process, plan.ready_timeout_s)
    except Exception as exc:
        _stop(process)
        if log_handle:
            log_handle.close()
        hint = f" (see {log_path})" if log_path else ""
        return failed(f"{type(exc).__name__}: {exc}{hint}")

    # The batch-of-one reference, taken from this server before any wave loads
    # it: same weights, same socket, one request in flight.
    serial: list[str] | None = None
    offline_prefix: float | None = None
    try:
        if plan.batch_check:
            print("  batch-of-one reference ...", flush=True)
            serial = asyncio.run(
                _serial(
                    port,
                    model_name,
                    expand_prompts(PROMPTS, max(plan.concurrency)),
                    plan.max_tokens,
                )
            )
            if offline is not None:
                offline_prefix = _prefix_rate(tokenizer, offline, serial)
                print(f"  offline parity (batch-of-one both sides): {offline_prefix:.3f}")
        for concurrency in plan.concurrency:
            prompts = expand_prompts(PROMPTS, concurrency)
            try:
                results, wall = asyncio.run(_wave(port, model_name, prompts, plan.max_tokens))
            except Exception as exc:
                rows += failed(f"{type(exc).__name__}: {exc}", concurrency)
                continue

            good = [r for r in results if r.ok]
            row = ServeRow(
                config=spec.label,
                model=model_name,
                concurrency=concurrency,
                issued=len(results),
                completed=len(good),
                wave_s=wall,
            )
            if not good:
                row.note = _first_line(results[0].error if results else "no results")
                rows.append(row)
                print(f"  conc {concurrency}: FAILED: {row.note}")
                continue

            ttfts = sorted(r.ttft_s * 1000 for r in good)
            row.ttft_mean_ms = statistics.mean(ttfts)
            # Highest observed rather than an interpolated quantile: at
            # concurrency 8 an interpolated p99 is a fiction.
            row.ttft_p99_ms = ttfts[-1]
            row.tpot_ms = statistics.mean(r.tpot_s * 1000 for r in good)
            row.throughput_tps = sum(r.tokens for r in good) / wall if wall else 0.0

            # Parity needs prompt-to-completion alignment, which a dropped
            # request breaks. A partial wave gets its timings and no parity claim.
            if len(good) == len(results):
                texts = [r.text for r in results]
                row.dup_prefix = _dup_rate(tokenizer, prompts, texts)
                if serial is not None:
                    row.batch_prefix = _prefix_rate(tokenizer, serial[: len(texts)], texts)
            # Per configuration, not per wave: both sides are batch-of-one,
            # which is what makes it a statement about the serving path.
            row.offline_prefix = offline_prefix
            row.dup_batch_prefix = dup_batch
            rows.append(row)

            def parity(name: str, value: float | None) -> str:
                return "" if value is None else f" | {name} {value:.3f}"

            print(
                f"  conc {concurrency}: TPS {row.throughput_tps:.1f} "
                f"| TTFT {row.ttft_mean_ms:.1f}/{row.ttft_p99_ms:.1f} ms "
                f"| TPOT {row.tpot_ms:.2f} ms | ok {row.completed}/{row.issued}"
                + parity("batch", row.batch_prefix)
                + parity("dup", row.dup_prefix)
                + parity("offline", row.offline_prefix)
            )
    finally:
        _stop(process)
        if log_handle:
            log_handle.close()
    return rows


def _specs_from_args(args: argparse.Namespace) -> list[ServeSpec]:
    schemes = [None if s in ("None", "none", "fp16", "bf16") else s for s in args.schemes]
    graphs: list[bool] = []
    if args.cuda_graph or not args.no_cuda_graph:
        graphs.append(True)
    if args.no_cuda_graph:
        graphs.append(False)
    return [
        ServeSpec(scheme=scheme, kv_cache_dtype=kv, tp=tp, dp=dp, graph=graph)
        for scheme in schemes
        for kv in args.kv_cache_dtype
        for tp in args.tp
        for dp in args.dp
        for graph in graphs
    ]


def _serving_main(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1
    try:
        import httpx  # noqa: F401
    except ImportError:
        print("serving needs httpx: pip install 'rapid_llm[serve]'", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer

    plan = _ServePlan(
        model_dir=args.model_dir,
        max_tokens=args.max_tokens,
        max_seq_len=args.max_seq_len,
        concurrency=tuple(sorted(set(args.concurrency))),
        offline_check=not args.no_offline_check,
        batch_check=not args.no_batch_check,
        ready_timeout_s=args.ready_timeout,
        server_log_dir=Path(args.server_log_dir) if args.server_log_dir else None,
    )
    specs = _specs_from_args(args)
    print(f"{len(specs)} configurations x {len(plan.concurrency)} concurrency levels")

    tokenizer = AutoTokenizer.from_pretrained(plan.model_dir)
    visible = torch.cuda.device_count()
    model_name = Path(plan.model_dir).name
    rows: list[ServeRow] = []
    for spec in specs:
        print(f"\n=== {model_name} — {spec.label} ===", flush=True)
        if spec.gpus_needed > visible:
            note = f"needs {spec.gpus_needed} GPUs, {visible} visible"
            rows.append(ServeRow(config=spec.label, model=model_name, note=note))
            print(f"  SKIPPED: {note}")
            continue
        rows += _measure_spec(spec, plan, tokenizer)
    _check_parity(rows)
    print("\n" + _render_serving(rows))

    gpu = gpu_tag()
    out_path = Path(
        args.json
        or Path(__file__).resolve().parents[2]
        / "docs"
        / "benchmark_logs"
        / f"bench_serving_{model_name}_{gpu}_{date.today():%Y%m%d}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "model_dir": args.model_dir,
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_count": visible,
                    "torch": torch.__version__,
                    "command": " ".join(sys.argv),
                    "max_tokens": args.max_tokens,
                    "date": date.today().isoformat(),
                },
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
    )
    print(f"\nJSON saved to {out_path}")
    return 0 if any(row.ok for row in rows) else 1


def _configure_serving(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--schemes", nargs="+", default=["fp16", "fp8", "int4"])
    parser.add_argument(
        "--kv-cache-dtype", nargs="+", default=["auto"], choices=["auto", "fp8", "fp8_e4m3"]
    )
    parser.add_argument("--tp", nargs="+", type=int, default=[1])
    parser.add_argument("--dp", nargs="+", type=int, default=[1])
    parser.add_argument("--cuda-graph", action="store_true", help="Capture decode graphs (default)")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Serve with eager decode")
    parser.add_argument("--concurrency", nargs="+", type=int, default=list(_CONCURRENCY))
    parser.add_argument("--max-tokens", type=int, default=_MAX_TOKENS)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--no-offline-check",
        action="store_true",
        help="Skip the in-process reference run (saves one model load per configuration)",
    )
    parser.add_argument(
        "--no-batch-check",
        action="store_true",
        help="Skip the serial batch-of-one pass (leaves the batch-invariance column empty)",
    )
    parser.add_argument("--ready-timeout", type=float, default=_READY_TIMEOUT_S)
    parser.add_argument(
        "--server-log-dir",
        default="/tmp/rapid_llm_serving",
        help="Where each server's stdout goes; a failed startup is diagnosed from it",
    )
    parser.add_argument("--json", help="Output path; a default under docs/benchmark_logs is used")


# --------------------------------------------------------------------------- #
# diag-prefix — where the cache hits, and what a hit costs
# --------------------------------------------------------------------------- #
def _diag_prefix_main(args: argparse.Namespace) -> int:
    """Run the shared-prefix workload with the cache on, then split TTFT by wave.

    The first admitted wave is a guaranteed miss, later waves hit. Also patches
    ``SlotBatch.copy_prefix`` to time KV copies and prints the stream timeline
    so forward.prefill / forward.extend / forward.decode are each accounted for.
    """
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.executor.slot_batch import SlotBatch

    copy_wall_s = 0.0
    copy_calls = 0
    orig_copy = SlotBatch.copy_prefix

    def timed_copy(self, segments):
        nonlocal copy_wall_s, copy_calls
        if segments:
            copy_calls += 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            orig_copy(self, segments)
            torch.cuda.synchronize()
            copy_wall_s += time.perf_counter() - start
        else:
            orig_copy(self, segments)

    SlotBatch.copy_prefix = timed_copy
    try:
        engine = ContinuousBatchingEngine.from_pretrained(
            args.model_dir,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            max_gpu_num_blocks=args.kv_blocks,
            use_cuda_graph=True,
            enable_prefix_cache=True,
        )
        engine.generate(["Warm up the kernels, please."], sampling_params(8))

        prompts = build_prefix_workload(args.prefix_groups, args.per_group, args.prefix_sentences)
        params = sampling_params(args.gen_len)

        # Admission bookkeeping: which step admitted each request.
        admitted_at: dict[str, int] = {}
        step_no = 0
        schedule = engine.scheduler.schedule

        def counted():
            nonlocal step_no
            out = schedule()
            step_no += 1
            for request in out.prefill:
                admitted_at.setdefault(request.request_id, step_no)
            return out

        engine.scheduler.schedule = counted

        run = run_requests(engine, prompts, params)
        requests, started, total = run.requests, run.started, run.total_s

        engine.scheduler.schedule = schedule

        waves = sorted(set(admitted_at.values()))
        print(
            f"total {total:.3f}s | steps {step_no} | prefix copies: {copy_calls} calls, "
            f"{copy_wall_s * 1000:.1f} ms wall (sync-bounded)"
        )
        print(
            "admission waves (step -> #requests): "
            f"{[(w, sum(1 for s in admitted_at.values() if s == w)) for w in waves]}"
        )
        for wave in waves:
            group = [r for r in requests if admitted_at.get(r.request_id) == wave]
            cached = [r.num_cached_tokens for r in group]
            ttfts = [(r.first_token_time - started) * 1000 for r in group if r.first_token_time]
            if ttfts:
                print(
                    f"wave @step {wave}: {len(group)} reqs | cached tokens/req "
                    f"min {min(cached)} max {max(cached)} | "
                    f"TTFT mean {sum(ttfts) / len(ttfts):7.1f} ms (n={len(ttfts)})"
                )
        print("\n--- stream timeline ---")
        print(engine.timeline_summary())
        engine.shutdown()
    finally:
        SlotBatch.copy_prefix = orig_copy
    return 0


def _configure_diag_prefix(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=_DEFAULT_MODEL)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-blocks", type=int, default=65536)
    parser.add_argument("--prefix-groups", type=int, default=4)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--prefix-sentences", type=int, default=32)
    parser.add_argument("--gen-len", type=int, default=24)


# --------------------------------------------------------------------------- #
# diag-preempt — oversubscription, preemption and CUDA graphs together
# --------------------------------------------------------------------------- #
def _diag_preempt_main(args: argparse.Namespace) -> int:
    """Oversubscribe the slot pool so the scheduler must preempt, with graphs on.

    ``slots_tokens=16384`` gives 8 slots while ``max_num_seqs=16`` oversubscribes
    2x, so the scheduler time-shares slots by recomputing the youngest request.
    Verifies the combination with CUDA graphs stays correct: every request
    finishes, preemptions actually happened, and the greedy texts match an
    un-preempted run.
    """
    import gc

    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.engine.llm_engine import LLMEngine
    from rapid_llm.engine.scheduler import SchedulerConfig
    from rapid_llm.executor.executor import UniProcExecutor

    def build(preempt: bool, max_num_seqs: int, slots_tokens: int):
        engine_llm = LLMEngine(
            args.model_dir,
            max_seq_len=2048,
            max_gpu_num_blocks=slots_tokens,
            use_cuda_graph=True,
        )
        config = SchedulerConfig(
            max_seq_len=engine_llm.max_seq_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
            enable_prefix_cache=True,
            enable_preemption=preempt,
        )
        executor = UniProcExecutor(engine_llm, config.max_num_seqs, config.max_seq_len)
        return ContinuousBatchingEngine(engine_llm, config, executor)

    prompts = build_prefix_workload(groups=4, per_group=6, sentences=32)

    engine = build(preempt=True, max_num_seqs=16, slots_tokens=16384)
    engine.generate(["Warm up."], sampling_params(8))
    preempted_stats = run_workload(engine, "preempted", prompts, 24)
    n_preempt = engine.scheduler.num_preemptions
    engine.shutdown()
    del engine

    gc.collect()
    torch.cuda.empty_cache()

    engine = build(preempt=False, max_num_seqs=8, slots_tokens=65536)
    engine.generate(["Warm up."], sampling_params(8))
    plain_stats = run_workload(engine, "plain", prompts, 24)
    engine.shutdown()

    print(preempted_stats.row())
    print(f"preemptions: {n_preempt}")
    print(plain_stats.row())
    same = sum(a == b for a, b in zip(preempted_stats.texts, plain_stats.texts, strict=True))
    print(f"text agreement preempted vs plain: {same}/{len(prompts)}")
    return 0


def _configure_diag_preempt(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=_DEFAULT_MODEL)


# --------------------------------------------------------------------------- #
# Scheduler command registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Command:
    """One subcommand: an argparse configurator bound to its entry point."""

    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


COMMANDS = (
    Command(
        name="matrix",
        help="feature matrix (tp x graph x prefix-cache x chunk-tokens), in-process",
        configure=_configure_matrix,
        run=_matrix_main,
    ),
    Command(
        name="serving",
        help="online benchmark through the HTTP API: TTFT/TPOT/TPS + parity checks",
        configure=_configure_serving,
        run=_serving_main,
    ),
    Command(
        name="diag-prefix",
        help="prefix-cache diagnosis: TTFT by admission wave, KV-copy timings, timeline",
        configure=_configure_diag_prefix,
        run=_diag_prefix_main,
    ),
    Command(
        name="diag-preempt",
        help="oversubscription + preemption + CUDA graphs: correctness and agreement",
        configure=_configure_diag_preempt,
        run=_diag_preempt_main,
    ),
)

