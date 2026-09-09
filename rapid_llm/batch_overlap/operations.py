"""The two-batch-overlap executor primitives: op streams, stages, and state.

sglang's ``operations.py`` discipline, ported one-for-one: ops separated by
:class:`YieldOperation` markers form indivisible *stages*; every op takes the
micro-batch's :class:`StateDict`, pops what it consumes and writes under a
*new* key; :func:`execute_overlapped_operations` walks two streams in
lockstep, the lead ``delta_stages`` ahead, so A's GEMMs occupy the SMs while
B sits in a comm op.

:class:`StateDict` is the part worth porting: a key may be written once until
popped, so clobbering a predecessor's result raises instead of silently
feeding stale data -- the bug the TBO closure snapshot produced.

Usage:
    state = StateDict({"hidden_states": h, "residual": None})
    execute_overlapped_operations([state_a, state_b], [ops, ops], delta_stages=[0, 2])
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class YieldOperation:
    """Stage boundary in an op stream: the only place the executor may switch.

    Ops between two yields form one stage — indivisible, run to completion
    before the other micro-batch resumes. The yields are where micro-batch A's
    GEMMs overlap micro-batch B sitting inside a communication op.
    """


@dataclass(frozen=True)
class ExecutionOperation:
    """One op plus the name it is logged under.

    Attributes:
        debug_name: The op's function name without the ``op_`` prefix, so a
            stage reads as ``attn`` / ``dispatch_a`` in a trace.
        fn: ``fn(state)`` — mutates the micro-batch state, returns nothing.
    """

    debug_name: str
    fn: Callable[[StateDict], Any]


# A module-level alias is evaluated eagerly (``from __future__ import
# annotations`` only covers annotations), so the not-yet-defined StateDict
# must ride as a forward reference.
Operation = YieldOperation | ExecutionOperation | Callable[["StateDict"], Any]
Stage = list[ExecutionOperation]


class StateDict:
    """Explicit-key state bag with overwrite and lifetime checks.

    Write a key once, pop it when consumed; ``clear(expect_keys)`` at a layer
    boundary then proves nothing was held longer than it should be. Attribute
    access is the interface (``state.hidden_states``), matching sglang's ops.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", dict(initial or {}))

    def __setattr__(self, key: str, value: Any) -> None:
        data = object.__getattribute__(self, "_data")
        if key in data:
            raise AssertionError(
                f"`{key}` already exists — pop it before rewriting; an op that "
                "overwrites a live key silently replaces a predecessor's result"
            )
        data[key] = value

    def __getattr__(self, item: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if item not in data:
            raise AttributeError(f"state has no `{item}` (keys: {sorted(data)})")
        return data[item]

    def pop(self, item: str) -> Any:
        """Consume a key: the value leaves the state, freeing the name."""
        return object.__getattribute__(self, "_data").pop(item)

    def clear(self, expect_keys: Sequence[str]) -> None:
        """Empty the state, asserting it held exactly ``expect_keys``.

        A mismatch means an intermediate was neither consumed nor released —
        either a leak or an op that skipped its pop. The overwrite check above
        cannot catch that: a forgotten pop leaves the key alone, so nothing
        ever tries to rewrite it.
        """
        data = object.__getattribute__(self, "_data")
        if set(data) != set(expect_keys):
            raise AssertionError(
                f"unexpected keys when clearing: held {sorted(data)}, "
                f"expected {sorted(expect_keys)}"
            )
        data.clear()


class _StageExecutor:
    """Walks one op stream stage by stage, threading one state through it."""

    def __init__(self, debug_name: str, stages: list[Stage], state: StateDict) -> None:
        self._debug_name = debug_name
        self._stages = stages
        self._state = state
        self._index = 0

    def next(self) -> None:
        """Run the next stage's ops in order; a stage never interleaves."""
        if self.done:
            raise AssertionError(f"{self._debug_name}: all {self.num_stages} stages ran")
        for op in self._stages[self._index]:
            op.fn(self._state)
        self._index += 1

    @property
    def state(self) -> StateDict:
        return self._state

    @property
    def done(self) -> bool:
        return self._index >= self.num_stages

    @property
    def num_stages(self) -> int:
        return len(self._stages)


def execute_operations(state: StateDict, operations: Sequence[Operation]) -> StateDict:
    """Run one op stream to completion, with nothing interleaved.

    The serial counterpart of :func:`execute_overlapped_operations`: same ops,
    same state discipline, one micro-batch. It is what the interleaved schedule
    is checked against — if the ping-pong changed the math, the two disagree.

    Args:
        state: The micro-batch's :class:`StateDict`, preloaded with its inputs.
        operations: The op stream, yields included. They only mark stage
            boundaries, which mean nothing when no other stream is running.

    Returns:
        The same state, after every op has run.
    """
    executor = _StageExecutor("serial", _convert_operations_to_stages(operations), state)
    for _ in range(executor.num_stages):
        executor.next()
    if not executor.done:
        raise AssertionError("the serial stream did not run to completion")
    return executor.state


def execute_overlapped_operations(
    states: Sequence[StateDict],
    operations_arr: Sequence[Sequence[Operation]],
    delta_stages: Sequence[int],
) -> list[StateDict]:
    """Run two micro-batch streams interleaved, stage by stage.

    Args:
        states: One :class:`StateDict` per micro-batch, preloaded with the
            step's inputs (hidden states, attention metadata, the deferred
            all-reduce context).
        operations_arr: One op stream per micro-batch; both are usually the
            same stream built from the same layers.
        delta_stages: ``[lead_delta, trail_delta]`` — the lead runs this many
            stages before alternation starts, and the trail drains the same
            number after it ends. The lead's delta must be 0 when the trail
            carries the lead width (sglang's convention), which keeps the
            schedule a single parameter.

    Returns:
        The two states, in order.
    """
    state_a, state_b = states
    operations_a, operations_b = operations_arr
    delta_a, delta_b = delta_stages
    if delta_a != 0:
        raise ValueError(f"the lead stream's delta must be 0, got {delta_a}")
    if delta_b < 0:
        raise ValueError(f"delta_stages must be >= 0, got {delta_b}")

    executor_a = _StageExecutor("a", _convert_operations_to_stages(operations_a), state_a)
    executor_b = _StageExecutor("b", _convert_operations_to_stages(operations_b), state_b)

    for _ in range(delta_b):
        executor_a.next()
    for _ in range(executor_a.num_stages - delta_b):
        executor_a.next()
        executor_b.next()
    for _ in range(delta_b):
        executor_b.next()

    if not (executor_a.done and executor_b.done):
        raise AssertionError("one stream ran out of stages before the other")
    return [executor_a.state, executor_b.state]


def _convert_operations_to_stages(operations: Sequence[Operation]) -> list[Stage]:
    """Decorate the ops, then cut the stream at its yields."""
    decorated = [_decorate_operation(op) for op in operations]
    stages: list[Stage] = [[]]
    for op in decorated:
        if isinstance(op, YieldOperation):
            stages.append([])
        else:
            stages[-1].append(op)
    if any(not stage for stage in stages):
        raise ValueError(
            "a yield must separate two non-empty stages; a leading or trailing "
            "yield makes the stage count — and so delta_stages — ambiguous"
        )
    return stages


def _decorate_operation(op: Operation) -> Operation:
    """Wrap a bare callable, taking its debug name from the function it wraps."""
    if isinstance(op, YieldOperation):
        return op
    name = getattr(op, "__name__", None) or getattr(
        getattr(op, "func", None), "__name__", "unknown"
    )
    return ExecutionOperation(name.removeprefix("op_"), op)
