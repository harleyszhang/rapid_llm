"""Declarative KernelSpec: one implementation's dispatch contract, torch-free.

A :class:`KernelSpec` names its op and backend target, states its shape /
layout requirements, and carries golden records plus perf tags; it is
pure data, so the registry can filter rows without importing torch.

Usage:
    spec = KernelSpec(...); register(spec)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ...platform.spec import CapabilityRequirement

#: Constraint kinds a :class:`ShapeConstraint` can express.
ConstraintKind = Literal["min", "max", "mod"]

#: ``"module.path:attribute"`` — resolved lazily so registration stays import-free.
_TARGET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")

#: Logical op ids are dotted lowercase words: ``linear``, ``attention.decode``.
_OP_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


@dataclass(frozen=True)
class ShapeConstraint:
    """One predicate over a named dimension of the dispatch shape key.

    Dimensions are symbolic (``"m"``, ``"n"``, ``"k"``, ``"num_heads"`` …) so
    the same structure serves linear GEMM buckets and attention geometry. A
    constraint naming a dimension the caller did not provide does not match:
    declaring a constraint means the implementation cannot run without knowing
    that dimension.

    Attributes:
        dim: Symbolic dimension name the caller must supply.
        kind: ``"min"`` (dim >= value), ``"max"`` (dim <= value) or
            ``"mod"`` (dim % value == 0).
        value: Right-hand side of the predicate.
    """

    dim: str
    kind: ConstraintKind
    value: int

    def satisfied_by(self, dims: Mapping[str, int]) -> bool:
        """Evaluate this constraint against a symbolic-shape mapping."""
        if self.dim not in dims:
            return False
        d = dims[self.dim]
        if self.kind == "min":
            return d >= self.value
        if self.kind == "max":
            return d <= self.value
        return d % self.value == 0

    def __str__(self) -> str:  # explain lines read like the math they encode
        sym = {"min": ">=", "max": "<=", "mod": "%"}[self.kind]
        return f"{self.dim}{sym}{self.value}"


@dataclass(frozen=True)
class ShapeRequirement:
    """Hard feasibility gates plus soft preferences over the dispatch shape.

    ``hard`` decides membership in the candidate set; ``prefer`` only breaks
    ties between feasible candidates (a kernel may love M=128 tiles yet still
    be correct — just slower — elsewhere). Keeping the two apart is what lets
    the dispatcher prefer without excluding.

    Attributes:
        hard: All must hold, else the implementation is filtered out.
        prefer: Each satisfied constraint adds one point to the rank.
    """

    hard: tuple[ShapeConstraint, ...] = ()
    prefer: tuple[ShapeConstraint, ...] = ()

    def is_feasible(self, dims: Mapping[str, int]) -> bool:
        """Whether every hard constraint holds for ``dims``."""
        return all(c.satisfied_by(dims) for c in self.hard)

    def preference_score(self, dims: Mapping[str, int]) -> int:
        """Number of prefer constraints that hold (higher ranks first)."""
        return sum(c.satisfied_by(dims) for c in self.prefer)


@dataclass(frozen=True)
class LayoutRequirement:
    """Input layout tags an implementation insists on, as data.

    Tags are ``domain:property`` strings (``"weight:nt"``, ``"kv:paged"``,
    ``"scale:block_128"``); their concrete meaning is defined by each op's
    interface, the dispatcher only does set algebra. An implementation gets
    filtered out when the call site cannot provide a required tag — and the
    missing tag shows up verbatim in ``explain``, so a layout mismatch is a
    diagnosable event instead of a wrong-number bug inside the kernel.

    Attributes:
        required: Tags the call site (or a registered layout conversion) must
            supply before this implementation may be dispatched.
    """

    required: tuple[str, ...] = ()

    def missing_from(self, available: frozenset[str]) -> frozenset[str]:
        """Required tags absent from ``available`` (empty = satisfied)."""
        return frozenset(self.required) - available

    def satisfied_by(self, available: frozenset[str]) -> bool:
        return not self.missing_from(available)


@dataclass(frozen=True)
class GoldenRecord:
    """Accuracy-gate status of an implementation against its reference.

    An unverified implementation is excluded from default dispatch — the acc
    gate is the only line of defence against a fast-but-wrong kernel, so
    "no evidence" must read as "no", not "yes until proven otherwise". Only an
    explicit ``backend=`` override (user asked for it by name) may bypass it.

    Attributes:
        verified: Set by the acc.align gate tool after the max-abs-diff run;
            native golden baselines ship verified with ``max_abs_diff=0.0``.
        max_abs_diff: Largest elementwise difference observed vs the baseline.
        baseline: What it was compared against (impl name or reference path).
    """

    verified: bool = False
    max_abs_diff: float | None = None
    baseline: str = ""


@dataclass(frozen=True)
class KernelSpec:
    """Full dispatch contract of one implementation of a logical op.

    Attributes:
        name: Globally unique implementation id, ``"<backend>/<impl>"``.
        op: Logical op id this implements (``"linear"``, ``"attention.decode"``).
        backend: Backend family (``"native"``, ``"flashinfer"``, …); exactly
            one ``native`` row per op acts as the never-failing floor.
        target: ``"module.path:attribute"`` of the callable, imported lazily
            at first dispatch, never at registration.
        available: ``"module.path:attribute"`` of a ``() -> bool``
            availability check (library present, driver usable); ``None`` means
            always available and is only legal for ``native`` rows.
        capability: Hardware windows, OR semantics, empty = anywhere
            (see :class:`~rapid_llm.platform.spec.CapabilityRequirement`).
        dtypes: Activation dtype labels (``"bf16"``, ``"fp16"``); empty = any.
        schemes: Quantisation scheme labels this impl implements; the scheme
            is one dimension of the dispatch key, not a separate registry.
        shape: Hard gates + preferences over the symbolic shape key.
        layout: Layout tags the call site must be able to provide.
        golden: Accuracy gate; unverified impls are excluded from default
            dispatch (native baselines register verified).
        priority: Static tie-breaker, higher first, used after perf_key and
            shape preferences; native floors should sit at 0.
        graph_safe: Whether the implementation may run inside a FULL CUDA graph
            capture. False for kernels that assemble per-step inputs on the
            host (Python-side index slicing, ``plan()``-style scheduling): a
            capture bakes those host values in, and replay would then attend
            the capture-time rows forever. The runner refuses to capture while
            a selected decode row is unsafe (see :func:`unsafe_for_graph`);
            the native data-driven kernels are safe by construction.
        step_prepare: ``"module.path:attr"`` of a per-step preparation hook,
            run once per forward step *outside* any graph (vLLM's
            ``build_metadata``). It receives ``(atten_info, runner)`` and may
            assemble per-step plan payloads on the host — the runner calls it
            on every eager decode step when this row wins dispatch, so a
            backend can hoist per-layer host work (index assembly, wrapper
            planning) to once per step. ``None`` = the implementation needs
            no per-step preparation.
    """

    name: str
    op: str
    backend: str
    target: str
    available: str | None = None
    capability: tuple[CapabilityRequirement, ...] = ()
    dtypes: tuple[str, ...] = ()
    schemes: tuple[str, ...] = ("unquantized",)
    shape: ShapeRequirement = field(default=ShapeRequirement())
    layout: LayoutRequirement = field(default=LayoutRequirement())
    golden: GoldenRecord = field(default=GoldenRecord())
    priority: int = 0
    graph_safe: bool = True
    step_prepare: str | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` on a malformed spec — at registration time."""
        if not self.name or "/" not in self.name:
            raise ValueError(f"KernelSpec.name must be '<backend>/<impl>', got {self.name!r}")
        head, _, _tail = self.name.partition("/")
        if head != self.backend:
            raise ValueError(f"name prefix {head!r} != backend {self.backend!r} in {self.name!r}")
        if not _OP_RE.match(self.op):
            raise ValueError(f"KernelSpec.op must be a lowercase dotted id, got {self.op!r}")
        for role, ref in (
            ("target", self.target),
            ("available", self.available),
            ("step_prepare", self.step_prepare),
        ):
            if ref is not None and not _TARGET_RE.match(ref):
                raise ValueError(f"KernelSpec.{role} must be 'module:attr', got {ref!r}")
        if self.available is None and self.backend != "native":
            # Every external backend must check its library: "missing wheel can
            # never hard-fail" only holds when absence is observable.
            raise ValueError(
                f"{self.name!r}: non-native backend must declare an availability check"
            )

    # ------------------------------------------------------------------ #
    # Dimension-wise membership predicates (composed by dispatch)
    # ------------------------------------------------------------------ #
    def dtype_ok(self, dtype: str) -> bool:
        """Empty ``dtypes`` declares the implementation is dtype-agnostic."""
        return not self.dtypes or dtype in self.dtypes

    def scheme_ok(self, scheme: str) -> bool:
        return scheme in self.schemes

    def shape_ok(self, dims: Mapping[str, int]) -> bool:
        return self.shape.is_feasible(dims)

    def layout_missing(self, available: frozenset[str]) -> frozenset[str]:
        return self.layout.missing_from(available)


# --------------------------------------------------------------------------- #
# Constants shared by every registration site (ops groups + backend modules)
# --------------------------------------------------------------------------- #
#: The native rows define the golden baseline themselves: they *are* what the
#: golden gate measures everything else against.
NATIVE_BASELINE = GoldenRecord(verified=True, max_abs_diff=0.0, baseline="native")

#: Static priority for external rows that have no frozen measurement yet. An
#: installed library is not a measurement — ranking it above the native floor
#: before the benchmark tool has spoken would silently change default
#: behaviour on every machine that happens to have the wheel.
UNMEASURED = -1

#: The paged KV layout tag family. The repo's cache manager allocates
#: ``[max_tokens, 2 * num_kv_heads, head_dim]`` — the K heads first, then the V
#: heads, so one token's K and V are adjacent — and any kernel that consumes
#: that buffer (or an equivalent external pool, e.g. FlashInfer's own) must
#: require the tag so the two pools are never confused by dispatch.
PAGED_KV_TAGS = frozenset({"kv:paged"})

#: :class:`LayoutRequirement` form of :data:`PAGED_KV_TAGS`, for spec rows.
PAGED_KV = LayoutRequirement(required=("kv:paged",))

#: The MLA latent cache's layout tag family: one latent row per token, in its
#: own pool — never interchangeable with the per-head paged buffer above.
MLA_LATENT_TAGS = frozenset({"kv:mla_latent"})

#: :class:`LayoutRequirement` form of :data:`MLA_LATENT_TAGS`, for spec rows.
MLA_LATENT = LayoutRequirement(required=("kv:mla_latent",))
