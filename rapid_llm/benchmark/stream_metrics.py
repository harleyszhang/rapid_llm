"""Early-stop-aware accounting for one-batch streaming benchmarks.

Steady-state window = last request's first token -> first request's finish.
Token counts come from each request's cumulative completion tokens; boundary
counts are interpolated per request so same-step chunk delivery order cannot
skew the window.

sglang ``stream_metrics`` mirror, on plain dataclasses (no msgspec dependency).
"""

from dataclasses import dataclass

# (arrival time, cumulative completion tokens) of one delivered chunk.
_Obs = tuple[float, int]


@dataclass(frozen=True)
class SteadyStateWindow:
    """Full-batch decode window: every request started, none finished yet."""

    start: float
    end: float
    output_tokens: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def output_throughput(self) -> float:
        return self.output_tokens / self.duration


class _Boundary:
    """Per-request observations bracketing one boundary instant."""

    def __init__(self, time: float, before: list[_Obs | None]):
        self.time = time
        self.before = before
        self.after: list[_Obs | None] = [None] * len(before)

    def observe(self, index: int, obs: _Obs) -> None:
        if self.after[index] is None:
            self.after[index] = obs

    def tokens_at_boundary(self, index: int) -> float:
        """Tokens of `index` at `time`, interpolated between its chunks."""
        before = self.before[index]
        if before is None:
            return 0.0
        t_before, c_before = before
        after = self.after[index]
        if after is None or self.time <= t_before:
            return float(c_before)
        t_after, c_after = after
        if self.time >= t_after or t_after <= t_before:
            return float(c_after)
        return c_before + (c_after - c_before) * (self.time - t_before) / (
            t_after - t_before
        )

    def total_tokens(self) -> float:
        return sum(self.tokens_at_boundary(i) for i in range(len(self.before)))


class BatchStreamRecorder:
    """Tracks per-request progress of one batched streaming /generate call."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.all_started_time: float | None = None
        self._last_obs: list[_Obs | None] = [None] * batch_size
        self._first_token_time: list[float | None] = [None] * batch_size
        self._completion_tokens: list[int] = [0] * batch_size
        self._num_started = 0
        self._t0: _Boundary | None = None
        self._t1: _Boundary | None = None

    def record_chunk(
        self,
        *,
        index: int,
        completion_tokens: int,
        now: float,
    ) -> None:
        obs = (now, completion_tokens)
        for boundary in (self._t0, self._t1):
            if boundary is not None:
                boundary.observe(index, obs)
        self._last_obs[index] = obs
        self._completion_tokens[index] = completion_tokens
        if self._first_token_time[index] is None and completion_tokens > 0:
            self._first_token_time[index] = now
            self._num_started += 1
            if self._num_started == self.batch_size:
                self.all_started_time = now
                self._t0 = _Boundary(time=now, before=list(self._last_obs))

    def finish(self, index: int, *, now: float) -> None:
        """Mark request `index` complete at `now`, closing the window on the first."""
        obs = (now, self._completion_tokens[index])
        for boundary in (self._t0, self._t1):
            if boundary is not None:
                boundary.observe(index, obs)
        self._last_obs[index] = obs
        if self._t1 is None:
            self._t1 = _Boundary(time=now, before=list(self._last_obs))

    def missing_indices(self) -> list[int]:
        return [i for i, t in enumerate(self._first_token_time) if t is None]

    @property
    def total_output_tokens(self) -> int:
        return sum(self._completion_tokens)

    @property
    def tokens_before_all_started(self) -> float | None:
        """Batch tokens at the last request's first token; None before then."""
        if self._t0 is None:
            return None
        return self._t0.total_tokens()

    def steady_state_window(self) -> SteadyStateWindow | None:
        """None when the batch never decoded at full size for a nonzero span."""
        if self._t0 is None or self._t1 is None:
            return None
        if self._t1.time <= self._t0.time:
            return None
        output_tokens = self._t1.total_tokens() - self._t0.total_tokens()
        if output_tokens <= 0:
            return None
        return SteadyStateWindow(
            start=self._t0.time,
            end=self._t1.time,
            output_tokens=output_tokens,
        )
