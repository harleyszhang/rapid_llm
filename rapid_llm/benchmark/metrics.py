"""The metrics vocabulary: TTFT / TPOT / TPS folded from request and step timestamps.

``BenchResult`` plus ``steps_to_result`` define every script's numbers;
``run_requests`` is the one submit-and-drain loop every step-driven backend
shares — which is what makes their numbers comparable. Two bases live here:
**step** (``steps_to_result``), intervals between engine steps, available to
every backend; **request** (``result_from_requests``), each request's own
first-token/finish timestamps with percentiles — the sglang serving convention.

Both use ``time.monotonic()`` — the engine's clock. Measuring on any other
clock (``perf_counter`` mixes two epochs) yields silently wrong deltas.

Usage:
    from rapid_llm.benchmark import BenchResult, steps_to_result, run_requests
"""

from __future__ import annotations

import itertools
import statistics
import time
from dataclasses import asdict, dataclass

import torch


def pctl(values: list[float], q: float) -> float:
    """The ``q`` percentile (0-100) of ``values``; 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


@dataclass
class BenchResult:
    """One benchmark measurement. ``gen_tokens`` is the throughput denominator.

    The ``*_p50_ms`` / ``*_p99_ms`` fields carry the per-request distribution
    (sglang serving's basis) and are 0.0 when only a batch-level view exists
    (HF / vLLM one-shot paths) — a zero percentile means "not measured", never
    "instantaneous".
    """

    ttft_ms: float
    tpot_ms: float
    total_s: float
    steps: int
    batch: int
    gen_tokens: int
    tpot_p50_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_p99_ms: float = 0.0

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "tps": self.tps}

    def row(self, label: str) -> str:
        return (
            f"{label:18s} TTFT {self.ttft_ms:7.1f} ms | "
            f"TPOT {self.tpot_ms:6.2f} ms | "
            f"TPS {self.tps:7.1f} tok/s | "
            f"{self.gen_tokens} tok in {self.total_s:.2f}s"
        )


def steps_to_result(
    step_ends: list[float],
    *,
    t_start: float,
    total_s: float,
    batch: int,
    gen_tokens: int | None = None,
) -> BenchResult:
    """Fold per-step completion timestamps into a :class:`BenchResult`.

    TTFT is the first step's end minus submission time; TPOT is the mean interval
    of the steps after it. Every step-driven backend goes through this function,
    which is what makes their numbers comparable.

    Args:
        step_ends: ``time.monotonic()`` at the end of each step.
        t_start: Submission time (taken after ``torch.cuda.synchronize()``).
        total_s: Whole-run wall clock, computed by the caller after the final sync.
        batch: Concurrent request count.
        gen_tokens: Tokens actually produced; omitted means lockstep advance
            (``batch`` per step).
    """
    deltas = [b - a for a, b in itertools.pairwise(step_ends)]
    return BenchResult(
        ttft_ms=(step_ends[0] - t_start) * 1000 if step_ends else 0.0,
        tpot_ms=(statistics.mean(deltas) * 1000) if deltas else 0.0,
        tpot_p50_ms=(statistics.median(deltas) * 1000) if deltas else 0.0,
        total_s=total_s,
        steps=len(step_ends),
        batch=batch,
        gen_tokens=len(step_ends) * batch if gen_tokens is None else gen_tokens,
    )


@dataclass
class RequestRun:
    """What one :func:`run_requests` call produced.

    Both metric bases live here because benchmarks need different ones: ``step_ends``
    gives lockstep step intervals (:meth:`result` falls back to them), each request's
    own timestamps the per-request latency distribution — anchored on ``started``, in
    the engine's monotonic clock.
    """

    requests: list
    started: float
    total_s: float
    step_ends: list[float]

    @property
    def gen_tokens(self) -> int:
        """Tokens produced. Requests leave on their own EOS, so ``steps * batch`` overcounts."""
        return sum(len(r.output_token_ids) for r in self.requests)

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.requests]

    def ttfts_ms(self) -> list[float]:
        """Per-request first-token latency in ms, from submission."""
        return [
            (r.first_token_time - self.started) * 1000 for r in self.requests if r.first_token_time
        ]

    def latencies_ms(self) -> list[float]:
        """Per-request completion latency in ms, from submission."""
        return [(r.finish_time - self.started) * 1000 for r in self.requests if r.finish_time]

    def per_request_tpot_ms(self) -> list[float]:
        """Per-request decode interval, from first token to finish (sglang's TPOT).

        ``TTFT`` covers the first token, so the decode rate divides the remaining
        tokens by the remaining time: ``(finish - first) / (n - 1)``.
        """
        tpots = []
        for r in self.requests:
            n = len(r.output_token_ids)
            if (
                r.first_token_time
                and r.finish_time
                and n > 1
                and r.finish_time > r.first_token_time
            ):
                tpots.append((r.finish_time - r.first_token_time) / (n - 1) * 1000)
        return tpots

    def steady_state(self) -> tuple[float, float, float] | None:
        """``(start, end, output_tokens)`` of the full-batch decode window, or None.

        The window runs from the *last* request's first token to the *first*
        request's finish — the only span in which the whole batch is decoding
        (sglang's ``BatchStreamRecorder`` steady state, computed from the
        engine's per-request timestamps instead of streamed chunks). Tokens
        inside the window are interpolated per request from its own decode rate,
        so same-step completion order cannot skew the window.
        """
        firsts = [r.first_token_time for r in self.requests if r.first_token_time]
        finishes = [
            (r.finish_time, len(r.output_token_ids)) for r in self.requests if r.finish_time
        ]
        if len(firsts) < len(self.requests) or len(finishes) < len(self.requests):
            return None
        t0, t1 = max(firsts), min(finish for finish, _ in finishes)
        if t1 <= t0:
            return None
        tokens = 0.0
        for r in self.requests:
            n = len(r.output_token_ids)
            if not (r.first_token_time and r.finish_time and n > 1):
                continue
            rate = (n - 1) / (r.finish_time - r.first_token_time)
            lo, hi = max(t0, r.first_token_time), min(t1, r.finish_time)
            if hi > lo:
                tokens += rate * (hi - lo)
        return t0, t1, tokens

    def result(self, batch: int) -> BenchResult:
        """The metrics of this run: per-request basis when timestamps exist.

        TTFT/TPOT means come from the requests' own timestamps (the engine
        clock), with p50/p99 percentiles filled; the step-interval view is the
        fallback for runs that lack request timestamps.
        """
        ttfts = self.ttfts_ms()
        tpots = self.per_request_tpot_ms()
        if ttfts and tpots:
            base = steps_to_result(
                self.step_ends,
                t_start=self.started,
                total_s=self.total_s,
                batch=batch,
                gen_tokens=self.gen_tokens,
            )
            base.ttft_ms = statistics.mean(ttfts)
            base.tpot_ms = statistics.mean(tpots)
            base.tpot_p50_ms = pctl(tpots, 50)
            base.ttft_p50_ms = pctl(ttfts, 50)
            base.ttft_p99_ms = pctl(ttfts, 99)
            base.tpot_p99_ms = pctl(tpots, 99)
            return base
        return steps_to_result(
            self.step_ends,
            t_start=self.started,
            total_s=self.total_s,
            batch=batch,
            gen_tokens=self.gen_tokens,
        )


def result_from_requests(run: RequestRun, batch: int) -> BenchResult:
    """The per-request metrics of a run — :meth:`RequestRun.result` as a free function.

    Kept as a module-level name so scripts can express which basis they asked for:
    this is the sglang serving basis (per-request TTFT/TPOT with percentiles),
    versus :func:`steps_to_result`'s lockstep basis.
    """
    return run.result(batch)


def run_requests(engine, prompts: list[str], params) -> RequestRun:
    """Submit a batch to a continuous-batching engine and step until it drains.

    The one loop every offline benchmark runs; its copies differed only in which
    numbers they derived afterwards, so the derivation moved here too. The engine is
    not warmed up — callers do that with their own parameters. ``params`` is one
    ``SamplingParams`` for the batch, or a sequence aligned with ``prompts`` when each
    request needs its own.

    All timestamps are ``time.monotonic()`` — the same clock the engine stamps
    ``Request.first_token_time`` / ``Request.finish_time`` with, so per-request
    deltas are differences within one epoch.
    """
    torch.cuda.synchronize()
    started = time.monotonic()
    per_request = isinstance(params, (list, tuple))
    requests = [
        engine.add_request(prompt, params[i] if per_request else params)
        for i, prompt in enumerate(prompts)
    ]
    step_ends: list[float] = []
    while engine.has_unfinished_requests():
        engine.step()
        step_ends.append(time.monotonic())
    torch.cuda.synchronize()
    return RequestRun(requests, started, time.monotonic() - started, step_ends)


def print_table(results: dict[str, BenchResult]) -> None:
    for label, r in results.items():
        print(r.row(label))
