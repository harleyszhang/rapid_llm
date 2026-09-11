"""Request-level metrics: a tiny in-process registry with a Prometheus rendering.

:class:`Counter` / :class:`Gauge` / :class:`Histogram` are the primitives
and :class:`EngineMetrics` wires them to the request lifecycle; every
``render`` emits Prometheus text format with no server dependency.

Usage:
    metrics = EngineMetrics.from_env()
    print(metrics.render_prometheus())
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...engine.scheduler import Request

from ...utils.env_compat import getenv

#: Environment variable switching collection off entirely (default on).
METRICS_ENV = "RAPID_LLM_METRICS"

_NAMESPACE = "rapid_llm"

# vLLM-shaped bucket grids: queue/TTFT span interactive to batch-scale waits,
# TPOT spans fast kernels to long-context attention.
_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
_TOKEN_BUCKETS = (1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


class Counter:
    """A monotonically increasing value, labelled or not."""

    def __init__(self, name: str, documentation: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.label_names = label_names
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(labels[name] for name in self.label_names)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        """The current total, ``0.0`` when never incremented.

        For callers that act on a counter rather than only render it — a
        benchmark comparing two arms reads one arm's total directly.
        """
        key = tuple(labels[name] for name in self.label_names)
        with self._lock:
            return self._values.get(key, 0.0)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} counter"]
        for key, value in sorted(self._values.items()):
            lines.append(f"{self.name}{_format_labels(self.label_names, key)} {_format(value)}")
        return "\n".join(lines)


class Gauge:
    """A value that goes up and down; set, never accumulated."""

    def __init__(self, name: str, documentation: str) -> None:
        self.name = name
        self.documentation = documentation
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} gauge"]
        lines.append(f"{self.name} {_format(self._value)}")
        return "\n".join(lines)


class Histogram:
    """Count of observations per bucket, plus the sum Prometheus wants.

    Buckets are cumulative (``le`` semantics): observing ``x`` increments every
    bucket whose bound is ``>= x``, so the rendered ``le="0.5"`` count answers
    "how many observations were at most 0.5" directly.
    """

    def __init__(
        self, name: str, documentation: str, buckets: Iterable[float] = _LATENCY_BUCKETS
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.buckets = tuple(buckets)
        self._counts = [0] * (len(self.buckets) + 1)  # last entry is +Inf
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[index] += 1
            self._counts[-1] += 1
            self._sum += value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} histogram"]
        for bound, count in zip(self.buckets, self._counts, strict=False):  # +Inf trails
            lines.append(f'{self.name}_bucket{{le="{_format(bound)}"}} {count}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._counts[-1]}')
        lines.append(f"{self.name}_sum {_format(self._sum)}")
        lines.append(f"{self.name}_count {self._counts[-1]}")
        return "\n".join(lines)


def _format_labels(names: tuple[str, ...], values: tuple[str, ...]) -> str:
    if not names:
        return ""
    pairs = ",".join(f'{name}="{value}"' for name, value in zip(names, values, strict=True))
    return "{" + pairs + "}"


def _format(value: float) -> str:
    """Prometheus accepts what Python prints; trim the trailing ``.0`` for ints."""
    return str(int(value)) if value == int(value) else repr(value)


class _NullInstrument:
    """Metric stand-in used when collection is disabled; every call is a no-op."""

    name = ""
    documentation = ""

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        pass

    def value(self, **labels: str) -> float:
        return 0.0

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def render(self) -> str:
        return ""


_NULL = _NullInstrument()


@dataclass
class EngineMetrics:
    """The metric set one continuous-batching engine reports.

    Attributes are the instruments themselves, so the engine's hot paths read
    as one method call (``metrics.ttft.observe(x)``) with no lookup. When
    ``enabled`` is ``False`` every attribute is the null instrument and the
    whole object costs the attribute reads alone.
    """

    enabled: bool = True
    running: Gauge | _NullInstrument = field(
        default_factory=lambda: Gauge(
            f"{_NAMESPACE}:num_requests_running", "Requests currently decoding."
        )
    )
    waiting: Gauge | _NullInstrument = field(
        default_factory=lambda: Gauge(
            f"{_NAMESPACE}:num_requests_waiting", "Requests queued for admission."
        )
    )
    prompt_tokens: Histogram | _NullInstrument = field(
        default_factory=lambda: Histogram(
            f"{_NAMESPACE}:request_prompt_tokens",
            "Prompt length per finished request.",
            buckets=_TOKEN_BUCKETS,
        )
    )
    generation_tokens: Histogram | _NullInstrument = field(
        default_factory=lambda: Histogram(
            f"{_NAMESPACE}:request_generation_tokens",
            "Generated tokens per finished request.",
            buckets=_TOKEN_BUCKETS,
        )
    )
    queue_time: Histogram | _NullInstrument = field(
        default_factory=lambda: Histogram(
            f"{_NAMESPACE}:request_queue_time_seconds",
            "Arrival to first schedule — the queue wait.",
        )
    )
    ttft: Histogram | _NullInstrument = field(
        default_factory=lambda: Histogram(
            f"{_NAMESPACE}:time_to_first_token_seconds",
            "Arrival to first generated token.",
        )
    )
    tpot: Histogram | _NullInstrument = field(
        default_factory=lambda: Histogram(
            f"{_NAMESPACE}:time_per_output_token_seconds",
            "Mean decode gap per request: (finish - first token) / (tokens - 1).",
        )
    )
    finished: Counter | _NullInstrument = field(
        default_factory=lambda: Counter(
            f"{_NAMESPACE}:request_success_total",
            "Finished requests by finish reason.",
            label_names=("finish_reason",),
        )
    )
    prompt_tokens_total: Counter | _NullInstrument = field(
        default_factory=lambda: Counter(
            f"{_NAMESPACE}:prompt_tokens_total", "Prompt tokens processed."
        )
    )
    generation_tokens_total: Counter | _NullInstrument = field(
        default_factory=lambda: Counter(
            f"{_NAMESPACE}:generation_tokens_total", "Tokens generated."
        )
    )
    kv_pipeline_sync_wait: Counter | _NullInstrument = field(
        default_factory=lambda: Counter(
            f"{_NAMESPACE}:kv_pipeline_sync_wait_seconds_total",
            "Host seconds blocked on the launch/harvest pipeline's readback events.",
        )
    )

    @classmethod
    def from_env(cls) -> EngineMetrics:
        """Read ``RAPID_LLM_METRICS``; ``0``/``false``/``off`` disable collection."""
        raw = getenv(METRICS_ENV, "1").strip().lower()
        metrics = cls(enabled=raw not in ("0", "false", "off"))
        if not metrics.enabled:
            for name in (
                "running",
                "waiting",
                "prompt_tokens",
                "generation_tokens",
                "queue_time",
                "ttft",
                "tpot",
                "finished",
                "prompt_tokens_total",
                "generation_tokens_total",
                "kv_pipeline_sync_wait",
            ):
                setattr(metrics, name, _NULL)
        return metrics

    # ------------------------------------------------------------ instrument #
    def observe_load(self, running: int, waiting: int) -> None:
        """Refresh the occupancy gauges; called once per engine step."""
        self.running.set(running)
        self.waiting.set(waiting)

    def observe_queue_time(self, request: Request) -> None:
        """Record the queue wait the first time a request is scheduled."""
        if request.scheduled_time is not None:
            self.queue_time.observe(request.scheduled_time - request.arrival_time)

    def observe_finish(self, request: Request) -> None:
        """Record everything a finished request answers for.

        This is where the per-request latency decomposition lands: queue wait
        (arrival → scheduled), prefill stretch (scheduled → first token) and
        the decode rate (first token → finish over the tokens it produced).
        """
        now = request.finish_time or time.monotonic()
        self.finished.inc(finish_reason=request.finish_reason or "unknown")
        self.prompt_tokens.observe(request.prompt_len)
        self.prompt_tokens_total.inc(request.prompt_len)
        n_output = len(request.output_token_ids)
        self.generation_tokens.observe(n_output)
        self.generation_tokens_total.inc(n_output)
        if request.first_token_time is not None:
            self.ttft.observe(request.first_token_time - request.arrival_time)
            # TPOT needs at least one gap: a one-token completion has none.
            if n_output > 1:
                self.tpot.observe((now - request.first_token_time) / (n_output - 1))

    # ---------------------------------------------------------------- export #
    def render_prometheus(self) -> str:
        """The whole registry in Prometheus text exposition format."""
        blocks = [
            instrument.render()
            for instrument in (
                self.running,
                self.waiting,
                self.prompt_tokens,
                self.generation_tokens,
                self.queue_time,
                self.ttft,
                self.tpot,
                self.finished,
                self.prompt_tokens_total,
                self.generation_tokens_total,
                self.kv_pipeline_sync_wait,
            )
        ]
        return "\n".join(block for block in blocks if block) + "\n"
