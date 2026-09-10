"""Tests for the registry and deterministic dispatch (ROADMAP foundation 2).

Filtering by shape / dtype / platform, ranking with the native floor,
caching, forced backends, and the ``explain()`` reasons — a fresh fake
registry per test keeps cases independent.

Usage:
    pytest tests/ops/test_dispatch.py
"""

from __future__ import annotations

import json
import re

import pytest

from rapid_llm.kernels.dispatcher import (
    REGISTRY,
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    OpRegistry,
    ShapeConstraint,
    ShapeRequirement,
    dispatch,
    dtype_label,
    op_backend_env,
    resolve_target,
    set_perf_provider,
    step_prepare_for,
    unsafe_for_graph,
)
from rapid_llm.platform.spec import CapabilityRequirement, PlatformInfo

A10 = PlatformInfo("cuda", 8, 6, "NVIDIA A10")
H100 = PlatformInfo("cuda", 9, 0, "NVIDIA H100")

#: Coloured log output carries these; strip before parsing the JSON payload.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

VERIFIED = GoldenRecord(verified=True, max_abs_diff=0.0, baseline="self")


# Availability checks referenced by specs below; module-level so they resolve
# through this test module's own import path.
def _available_yes() -> bool:
    return True


def _available_no() -> bool:
    return False


def _fake_prepare(atten_info, runner) -> None:
    return None


def _fake_prepare_b(atten_info, runner) -> None:
    return None


try:
    import flashinfer  # noqa: F401

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False


def make_reg(*specs: KernelSpec) -> OpRegistry:
    reg = OpRegistry()
    for s in specs:
        reg.register(s)
    return reg


def native(**over) -> KernelSpec:
    """The always-eligible floor row every op must ship."""
    base = {
        "name": "native/floor",
        "op": "test.op",
        "backend": "native",
        "target": "math:sin",
        "golden": VERIFIED,
    }
    return KernelSpec(**{**base, **over})


def external(name: str, **over) -> KernelSpec:
    base = {
        "name": name,
        "op": "test.op",
        "backend": name.split("/")[0],
        "target": "math:cos",
        "available": "tests.ops.test_dispatch:_available_yes",
        "golden": VERIFIED,
    }
    return KernelSpec(**{**base, **over})


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RAPID_LLM_FORCE_BACKEND", raising=False)
    monkeypatch.delenv("RAPID_LLM_KERNEL_TRACE", raising=False)
    monkeypatch.delenv(op_backend_env("test.op"), raising=False)


class TestRegistry:
    def test_register_and_lookup(self) -> None:
        reg = make_reg(native())
        assert [s.name for s in reg.implementations("test.op")] == ["native/floor"]
        assert reg.spec("native/floor") is reg.implementations("test.op")[0]

    def test_duplicate_identical_registration_is_idempotent(self) -> None:
        reg = make_reg(native())
        reg.register(native())
        assert len(reg.implementations("test.op")) == 1

    def test_conflicting_registration_raises(self) -> None:
        reg = make_reg(native())
        with pytest.raises(ValueError, match="different spec"):
            reg.register(native(priority=5))

    def test_native_floor_helper(self) -> None:
        reg = make_reg(native(), external("flashinfer/fast"))
        assert reg.native_floor("test.op").name == "native/floor"

    def test_registration_invalidates_cached_decisions(self) -> None:
        reg = make_reg(native(priority=0))
        first = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert first.spec.name == "native/floor"
        # A faster external row lands; the cached decision must not survive it.
        reg.register(external("flashinfer/fast", priority=10))
        second = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert second.spec.name == "flashinfer/fast"


class TestFiltering:
    def test_unknown_op_lists_alternatives(self) -> None:
        reg = make_reg(native())
        with pytest.raises(LookupError, match="registered"):
            dispatch("nope.op", dtype="bf16", platform_info=A10, registry=reg)

    def test_dtype_and_scheme_gates(self) -> None:
        reg = make_reg(native(), external("x/a", dtypes=("fp16",), schemes=("fp8",)))
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"
        assert "dtype" in sel.rejections["x/a"]

        sel = dispatch("test.op", dtype="fp16", scheme="fp8", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/a"

    def test_capability_window_filters_and_explains(self) -> None:
        reg = make_reg(
            native(),
            external("d/hopper", capability=(CapabilityRequirement("cuda", min_cc=(9, 0)),)),
        )
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"
        assert "capability" in sel.rejections["d/hopper"]

        sel = dispatch("test.op", dtype="bf16", platform_info=H100, registry=reg)
        assert sel.spec.name == "d/hopper"  # registration order keeps it above the floor

    def test_availability_false_excludes_with_reason(self) -> None:
        reg = make_reg(
            native(), external("x/broken", available="tests.ops.test_dispatch:_available_no")
        )
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"
        assert "library unavailable" in sel.rejections["x/broken"]

    def test_shape_hard_gate(self) -> None:
        reg = make_reg(
            native(),
            external(
                "x/tiled",
                priority=1,
                shape=ShapeRequirement(hard=(ShapeConstraint("k", "mod", 16),)),
            ),
        )
        sel = dispatch("test.op", dtype="bf16", shape={"k": 40}, platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"
        assert "shape" in sel.rejections["x/tiled"]

        sel = dispatch("test.op", dtype="bf16", shape={"k": 64}, platform_info=A10, registry=reg)
        assert sel.spec.name == "x/tiled"

    def test_layout_tag_gate(self) -> None:
        reg = make_reg(
            native(),
            external("x/paged", priority=1, layout=LayoutRequirement(required=("kv:paged",))),
        )
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"
        assert "kv:paged" in sel.rejections["x/paged"]

        sel = dispatch(
            "test.op", dtype="bf16", layout=frozenset({"kv:paged"}), platform_info=A10, registry=reg
        )
        assert sel.spec.name == "x/paged"

    def test_unverified_golden_excluded_from_default_dispatch(self) -> None:
        reg = make_reg(native(), external("x/unverified", golden=GoldenRecord()))
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"
        assert "golden" in sel.rejections["x/unverified"]


class TestRanking:
    def test_priority_orders_unmeasured_rows(self) -> None:
        reg = make_reg(native(), external("x/low", priority=1), external("x/high", priority=9))
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/high"

    def test_shape_preference_beats_priority(self) -> None:
        reg = make_reg(
            native(),
            external(
                "x/loves_128",
                priority=1,
                shape=ShapeRequirement(prefer=(ShapeConstraint("m", "mod", 128),)),
            ),
            external("x/blank", priority=9),
        )
        sel = dispatch("test.op", dtype="bf16", shape={"m": 256}, platform_info=A10, registry=reg)
        assert sel.spec.name == "x/loves_128"

    def test_frozen_measurement_beats_everything(self) -> None:
        reg = make_reg(native(), external("x/slow", priority=9), external("x/measured", priority=0))

        def measured_wins(spec: KernelSpec, key: object) -> float | None:
            return 1.5 if spec.name == "x/measured" else None

        set_perf_provider(measured_wins)
        try:
            sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
            assert sel.spec.name == "x/measured"
            assert "perf=1.500ms" in sel.explain()
        finally:
            set_perf_provider(lambda spec, key: None)

    def test_name_is_the_final_tie_break(self) -> None:
        reg = make_reg(native(), external("b/row"), external("a/row"))
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "a/row"  # equal specs -> lexicographic, not import order


class TestCachingAndDeterminism:
    def test_same_key_returns_the_same_decision_object(self) -> None:
        reg = make_reg(native(), external("x/a"))
        a = dispatch("test.op", dtype="bf16", shape={"k": 7}, platform_info=A10, registry=reg)
        b = dispatch("test.op", dtype="bf16", shape={"k": 7}, platform_info=A10, registry=reg)
        assert a is b  # cached

    def test_different_shape_is_a_different_key(self) -> None:
        reg = make_reg(
            native(),
            external(
                "x/shape",
                priority=1,
                shape=ShapeRequirement(hard=(ShapeConstraint("k", "min", 8),)),
            ),
        )
        small = dispatch("test.op", dtype="bf16", shape={"k": 4}, platform_info=A10, registry=reg)
        big = dispatch("test.op", dtype="bf16", shape={"k": 64}, platform_info=A10, registry=reg)
        assert small.spec.name == "native/floor"
        assert big.spec.name == "x/shape"

    def test_platform_is_part_of_the_key(self) -> None:
        reg = make_reg(
            native(),
            external("d/hopper", capability=(CapabilityRequirement("cuda", min_cc=(9, 0)),)),
        )
        assert (
            dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg).spec.name
            == "native/floor"
        )
        assert (
            dispatch("test.op", dtype="bf16", platform_info=H100, registry=reg).spec.name
            == "d/hopper"
        )


class TestForcedBackend:
    def test_explicit_backend_pins_the_family(self) -> None:
        reg = make_reg(native(), external("x/a", priority=99))
        sel = dispatch("test.op", dtype="bf16", backend="native", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"

    def test_forced_backend_bypasses_only_the_golden_gate(self) -> None:
        reg = make_reg(native(), external("x/unverified", golden=GoldenRecord()))
        sel = dispatch("test.op", dtype="bf16", backend="x", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/unverified"
        # ...but physical gates still hold: a dtype it cannot run excludes it.
        reg2 = make_reg(native(), external("x/fp16only", dtypes=("fp16",), golden=GoldenRecord()))
        with pytest.raises(LookupError, match="dtype"):
            dispatch("test.op", dtype="bf16", backend="x", platform_info=A10, registry=reg2)

    def test_env_variable_forces_globally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reg = make_reg(native(), external("x/a"))
        monkeypatch.setenv("RAPID_LLM_FORCE_BACKEND", "x")
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/a"

    def test_op_id_becomes_an_env_key(self) -> None:
        # Dots are not legal in a shell variable name, so they become '_'.
        assert op_backend_env("attention.decode") == "RAPID_LLM_ATTENTION_DECODE_BACKEND"
        assert op_backend_env("linear") == "RAPID_LLM_LINEAR_BACKEND"

    def test_per_op_env_pins_just_that_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reg = make_reg(native(), external("x/a"))
        reg.register(native(name="native/other", op="other.op"))
        monkeypatch.setenv(op_backend_env("test.op"), "x")
        assert dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg).spec.name == "x/a"
        # The neighbouring op is untouched — that separation is the whole point
        # of having per-op keys next to the global one.
        other = dispatch("other.op", dtype="bf16", platform_info=A10, registry=reg)
        assert other.spec.name == "native/other"

    def test_per_op_env_beats_the_global_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Narrower wins: the run says "x everywhere", the op says "native here".
        reg = make_reg(native(), external("x/a", priority=99))
        monkeypatch.setenv("RAPID_LLM_FORCE_BACKEND", "x")
        monkeypatch.setenv(op_backend_env("test.op"), "native")
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"

    def test_backend_argument_beats_every_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reg = make_reg(native(), external("x/a", priority=99))
        monkeypatch.setenv("RAPID_LLM_FORCE_BACKEND", "x")
        monkeypatch.setenv(op_backend_env("test.op"), "x")
        sel = dispatch("test.op", dtype="bf16", backend="native", platform_info=A10, registry=reg)
        assert sel.spec.name == "native/floor"

    def test_forcing_a_missing_backend_fails_loud(self) -> None:
        reg = make_reg(native())
        with pytest.raises(LookupError, match="forced"):
            dispatch("test.op", dtype="bf16", backend="flashinfer", platform_info=A10, registry=reg)


class TestExplainAndTrace:
    def test_explain_names_the_winner_and_every_loser(self) -> None:
        reg = make_reg(
            native(),
            external("x/lo", priority=1),
            external("d/hopper", capability=(CapabilityRequirement("cuda", min_cc=(9, 0)),)),
        )
        text = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg).explain()
        assert "[x/lo] dispatched" in text
        assert "[d/hopper] excluded: capability" in text
        # The floor was feasible but outranked: explain must say so, not hide it.
        assert "[native/floor] feasible, ranked below" in text

    def test_trace_emits_one_json_line(
        self, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("RAPID_LLM_KERNEL_TRACE", "1")
        reg = make_reg(native())
        dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        # The project logger propagates to a coloured stderr handler only, so
        # assert on the emitted line instead of caplog.
        out = capfd.readouterr().err
        line = next((l for l in out.splitlines() if '"op":' in l), "")
        assert line, f"no trace JSON in stderr:\n{out}"
        # The coloured stderr handler appends ANSI resets around the payload.
        payload = json.loads(_ANSI_RE.sub("", line[line.index("{") :]))
        assert payload["op"] == "test.op"
        assert payload["kernel"] == "native/floor"
        assert payload["dtype"] == "bf16"


class TestTargetResolution:
    def test_load_resolves_the_target_lazily(self) -> None:
        import math

        reg = make_reg(native())  # target is math:sin
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.load() is math.sin

    def test_resolve_failure_names_the_target(self) -> None:
        with pytest.raises(ImportError, match="cannot resolve kernel target"):
            resolve_target("no.such.module:attr")

    def test_dtype_label_maps_torch_types(self) -> None:
        import torch

        assert dtype_label(torch.bfloat16) == "bf16"
        assert dtype_label(torch.float16) == "fp16"
        assert dtype_label("bf16") == "bf16"  # labels pass through


class TestGraphGate:
    """graph_safe marking + the runner's pre-capture query — the rapid_llm
    analogue of vLLM's AttentionCGSupport check before its first capture."""

    def test_ranked_unsafe_backend_is_reported(self) -> None:
        # Perf/priority ranking may put an unsafe row first (flashinfer's
        # measured win over native is exactly the real-world trigger).
        reg = make_reg(
            native(),
            external("ext/decode", op="attention.decode", priority=10, graph_safe=False),
        )
        assert dispatch("attention.decode", dtype="bf16", registry=reg).spec.name == "ext/decode"
        assert unsafe_for_graph("attention.decode", reg) == ("ext/decode",)

    def test_safe_backend_reports_nothing(self) -> None:
        reg = make_reg(native(), external("ext/decode", op="attention.decode", priority=10))
        dispatch("attention.decode", dtype="bf16", registry=reg)
        assert unsafe_for_graph("attention.decode", reg) == ()

    def test_unsafe_rows_of_other_ops_do_not_leak(self) -> None:
        reg = make_reg(external("ext/other", op="other.op", priority=10, graph_safe=False))
        dispatch("other.op", dtype="bf16", registry=reg)
        assert unsafe_for_graph("attention.decode", reg) == ()

    def test_one_name_per_spec_across_dispatch_keys(self) -> None:
        # The same spec selected under two quantisation schemes is reported
        # once, not once per dispatch key.
        reg = make_reg(
            native(),
            external(
                "ext/decode",
                op="attention.decode",
                priority=10,
                schemes=("unquantized", "fp8_kv"),
                graph_safe=False,
            ),
        )
        dispatch("attention.decode", dtype="bf16", scheme="unquantized", registry=reg)
        dispatch("attention.decode", dtype="bf16", scheme="fp8_kv", registry=reg)
        assert unsafe_for_graph("attention.decode", reg) == ("ext/decode",)


class TestStepPrepare:
    """step_prepare_for: the runner's per-step hook lookup — returns the hook
    only when the dispatch winner declares one, never for a loser."""

    HOOK = "tests.ops.test_dispatch:_fake_prepare"
    HOOK_B = "tests.ops.test_dispatch:_fake_prepare_b"

    def test_native_winner_without_hook_gives_none(self) -> None:
        # The floor outranks the hook row: dispatch would run native, so the
        # hook of the losing row must not fire.
        reg = make_reg(
            native(op="attention.decode", priority=10),
            external("ext/decode", op="attention.decode", priority=5, step_prepare=self.HOOK),
        )
        assert step_prepare_for("attention.decode", registry=reg) is None

    def test_winning_row_with_hook_is_returned(self) -> None:
        reg = make_reg(
            native(op="attention.decode"),
            external("ext/decode", op="attention.decode", priority=10, step_prepare=self.HOOK),
        )
        assert step_prepare_for("attention.decode", registry=reg) is _fake_prepare

    def test_pinned_backend_without_hook_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(op_backend_env("attention.decode"), "native")
        reg = make_reg(
            native(op="attention.decode"),
            external("ext/decode", op="attention.decode", priority=10, step_prepare=self.HOOK),
        )
        assert step_prepare_for("attention.decode", registry=reg) is None

    def test_pinned_backend_with_hook_wins_over_priority(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pinned to the losing-priority backend: only its rows are ranked, so
        # its hook is the one that matches what dispatch will actually run.
        monkeypatch.setenv(op_backend_env("attention.decode"), "flashinfer")
        reg = make_reg(
            native(op="attention.decode", priority=10),
            external("flashinfer/dec", op="attention.decode", priority=0, step_prepare=self.HOOK),
        )
        assert step_prepare_for("attention.decode", registry=reg) is _fake_prepare

    def test_unavailable_hook_row_is_skipped(self) -> None:
        reg = make_reg(
            native(op="attention.decode", priority=10),
            external(
                "ext/decode",
                op="attention.decode",
                priority=5,
                step_prepare=self.HOOK,
                available="tests.ops.test_dispatch:_available_no",
            ),
        )
        assert step_prepare_for("attention.decode", registry=reg) is None

    def test_highest_priority_hook_wins_between_two(self) -> None:
        reg = make_reg(
            external("a/dec", op="attention.decode", priority=5, step_prepare=self.HOOK),
            external("b/dec", op="attention.decode", priority=9, step_prepare=self.HOOK_B),
        )
        assert step_prepare_for("attention.decode", registry=reg) is _fake_prepare_b

    @pytest.mark.skipif(_HAS_FLASHINFER, reason="flashinfer installed: the row is available")
    def test_real_registry_without_flashinfer_gives_none(self) -> None:
        # The shipped attention.decode rows: flashinfer declares a hook but is
        # unavailable without the wheel, native needs no hook.
        assert step_prepare_for("attention.decode") is None
        assert REGISTRY.implementations("attention.decode")  # rows exist at all
