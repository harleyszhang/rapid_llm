"""Tests for the registered spec rows: catalogue, routing and the floor impl.

Every op group has its rows, scheme-to-row routing is pinned, the
native row is always feasible where promised, and rows stay torch-free
— the registry's structural invariants.

Usage:
    pytest tests/ops/test_native_specs.py
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F

import rapid_llm.kernels  # noqa: F401 — import side effect: spec registration
import rapid_llm.kernels.ops as ops_pkg
from rapid_llm.kernels.dispatcher import REGISTRY
from rapid_llm.kernels.dispatcher import dispatch as _dispatch
from rapid_llm.kernels.dispatcher.dispatch import resolve_target
from rapid_llm.kernels.ops import LOGICAL_OPS
from rapid_llm.kernels.ops.gemm.linear import linear_torch
from rapid_llm.modules.quantization.unquant import UnquantizedLinearMethod
from rapid_llm.platform import PlatformInfo
from tests.registry import import_needs_cuda

import_needs_cuda()

#: resolve_target imports the native Triton modules to prove they are callable;
#: on a machine without Triton there is nothing to resolve.
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None


def dispatch(op, **kwargs):
    """These catalogue checks describe GPU kernels, regardless of the test host."""
    kwargs.setdefault("platform_info", PlatformInfo("cuda", 8, 6))
    return _dispatch(op, **kwargs)


#: scheme key -> spec name, the routing every quant method relies on.
SCHEME_TO_ROW = {
    "unquantized": "native/linear_torch",
    "fp8": "native/linear_w8a16",
    "blockwise_int8": "native/linear_w8a16",
    "awq": "native/linear_w4a16",
    "gptq": "native/linear_w4a16",
    "w8a8_int8": "native/linear_w8a8_int8",
    "w8a8_fp8": "native/linear_w8a8_fp8",
    "nvfp4": "native/linear_nvfp4",
}

#: Schemes with a linear row but deliberately no MoE row. ``fused_moe`` infers
#: the expert format from ``w1.dtype``, which cannot distinguish nvfp4's packed
#: uint8 from fp8's uint8 — the two-level unpacking would need its own kernel,
#: and NVFP4 MoE is out of scope. Named here so the gap is an assertion rather
#: than a silently skipped scheme.
SCHEMES_WITHOUT_MOE = {"nvfp4"}

#: native rows only — the floor every op falls back to. External rows sit at
#: priority ``UNMEASURED`` (below the native floor) until a golden run on the
#: right hardware freezes a measured number, so routing tests land here too.
LINEAR_NATIVE_ROWS = {
    "native/linear_torch",
    "native/linear_w8a16",
    "native/linear_w4a16",
    "native/linear_w8a8_int8",
    "native/linear_w8a8_fp8",
    "native/linear_nvfp4",
}

#: the external linear contender, gated on sm90 and on a golden run.
LINEAR_EXTERNAL_ROWS = {"deepgemm/fp8_gemm_nt"}

#: ``op -> native row``, the attention domain. The MLA pair landed in v0.11:
#: the latent decode kernel and the chunked-upsample prefill.
ATTENTION_NATIVE_ROWS = {
    "attention.prefill": "native/flash_attention2_no_pad",
    "attention.decode": "native/flash_decoding",
    "attention.mla_decode": "native/mla_decode",
    "attention.mla_prefill": "native/mla_prefill",
    "kv_write": "native/update_kv_buffer",
}

#: ``op -> external rows``: the attention contenders. ``attention.mla_prefill``
#: has no external row — FlashMLA ships decode only.
ATTENTION_EXTERNAL_ROWS = {
    "attention.prefill": {"flashinfer/prefill"},
    "attention.decode": {"flashinfer/decode"},
    "attention.mla_decode": {"flashmla/mla_decode"},
    "attention.mla_prefill": set(),
}

#: Layout tags the paged KV buffer of this repo satisfies.
PAGED = frozenset({"kv:paged"})

#: The MLA latent cache's layout tag — its own pool, not the per-head one.
MLA_LATENT = frozenset({"kv:mla_latent"})

#: ``op -> native row`` for the per-layer glue. ``elementwise.*`` is two rows
#: because the two arities are two contracts: the fused projection hands over
#: one packed tensor, the split path two halves.
GLUE_NATIVE_ROWS = {
    "moe": "native/fused_moe",
    "rmsnorm": "native/skip_rmsnorm",
    "rope": "native/rope_emb_forward",
    "elementwise.swiglu": "native/swiglu_forward_fused",
    "elementwise.swiglu_split": "native/swiglu_forward",
}

#: Native rows beyond the floor. Only ``moe`` has them, one per W8A8 scheme:
#: ``fused_moe`` infers the expert format from ``w1.dtype``, which cannot tell
#: weight-only fp8 from W8A8 fp8 (both are ``uint8`` e4m3 experts) nor
#: weight-only int8 from W8A8 int8 (both ``int8``). The difference is whether
#: the *activation* is quantised, which no dtype records, so the choice has to
#: be the entry point and therefore the row.
GLUE_EXTRA_NATIVE_ROWS = {"moe": {"native/fused_moe_w8a8_fp8", "native/fused_moe_w8a8_int8"}}

#: ``op -> external rows`` for the same domains.
GLUE_EXTERNAL_ROWS = {
    "moe": {"deepgemm/grouped_fp8_moe"},
    "rmsnorm": {"flashinfer/rmsnorm"},
    "rope": {"flashinfer/rope"},
    "elementwise.swiglu": set(),
    "elementwise.swiglu_split": set(),
}

#: Ops whose contract is declared but has no row at all, with the reason.
#: Pinned so the gap stays a decision on record rather than something
#: forgotten: EP comms need an expert-parallel group this repo does not have
#: yet, and the elementwise root is the open namespace's parent, never a row.
OPS_WITHOUT_ROWS = {
    "comm.dispatch": "no EP group; MoE here is tensor parallel",
    "comm.combine": "pairs with comm.dispatch",
    "elementwise": "open namespace root; only its members register rows",
}

#: Ops served by external rows only — no native implementation exists or is
#: planned. The rows carry the availability/capability/golden gates; the op
#: itself is legal to dispatch only where a backend survives them.
EXTERNAL_ONLY_OPS = {
    "sample": "flashinfer serves it; engine.Sampler stays the default path",
}

#: The nine operator-domain groups whose ``__init__.py`` files hold the rows.
OPS_GROUPS = (
    "activation",
    "attention",
    "embeddings",
    "gemm",
    "kvcache",
    "layernorm",
    "moe",
    "rope",
    "sampling",
)

#: ``ops/quantization`` is a shared implementation helper of the gemm/moe
#: groups — it re-exports the quantisation kernels, so it does import torch —
#: not a registration group; the nine groups above are.
NON_GROUP_DIRS = {"quantization"}


class TestLinearCatalogue:
    def test_linear_has_the_six_native_rows_plus_deepgemm(self) -> None:
        names = {s.name for s in REGISTRY.implementations("linear")}
        assert names == LINEAR_NATIVE_ROWS | LINEAR_EXTERNAL_ROWS

    def test_native_rows_are_the_verified_floor(self) -> None:
        for spec in REGISTRY.implementations("linear"):
            if spec.backend == "native":
                assert spec.golden.verified, f"{spec.name} must be golden-verified"

    def test_deepgemm_row_is_golden_gated(self) -> None:
        # Untested on hardware (CI is sm86, DeepGEMM needs sm90+): the golden
        # gate keeps the row out of default dispatch until a Hopper run
        # produces a max-abs-diff, and UNMEASURED ranks it below the floor.
        spec = next(s for s in REGISTRY.implementations("linear") if s.backend == "deepgemm")
        assert not spec.golden.verified
        assert spec.priority < 0

    @pytest.mark.parametrize("scheme,row", sorted(SCHEME_TO_ROW.items()))
    def test_scheme_routes_to_its_row(self, scheme: str, row: str) -> None:
        sel = dispatch("linear", dtype="bf16", scheme=scheme, shape={"m": 8, "n": 8, "k": 8})
        assert sel.spec.name == row

    def test_quantised_rows_reject_fp32_activations(self) -> None:
        # The Triton rows declare bf16/fp16 only. fp8 weights with fp32
        # activations is physically invalid — no floor can serve it, so
        # dispatch fails loud naming the dtype gate.
        sel = dispatch("linear", dtype="fp32", scheme="unquantized", shape={"m": 8, "n": 8, "k": 8})
        assert sel.spec.name == "native/linear_torch"
        with pytest.raises(LookupError, match="dtype"):
            dispatch("linear", dtype="fp32", scheme="fp8", shape={"m": 8, "n": 8, "k": 8})

    def test_floor_row_is_the_native_floor(self) -> None:
        assert REGISTRY.native_floor("linear").name == "native/linear_torch"


class TestAttentionCatalogue:
    @pytest.mark.parametrize("op,row", sorted(ATTENTION_NATIVE_ROWS.items()))
    def test_each_phase_has_its_native_row(self, op: str, row: str) -> None:
        native = {s.name for s in REGISTRY.implementations(op) if s.backend == "native"}
        assert native == {row}
        assert REGISTRY.native_floor(op).name == row

    @pytest.mark.parametrize("op,rows", sorted(ATTENTION_EXTERNAL_ROWS.items()))
    def test_external_rows_are_registered(self, op: str, rows: set[str]) -> None:
        external = {
            s.name for s in REGISTRY.implementations(op) if s.backend not in {"native", "cpu"}
        }
        assert external == rows

    def test_flashinfer_rows_are_verified_with_golden_diffs(self) -> None:
        # The only externally-verified rows in the tree: golden records carry
        # the max-abs-diff that earned the flag, and priority stays UNMEASURED
        # until a perf number says otherwise.
        for op in ("attention.prefill", "attention.decode"):
            spec = next(s for s in REGISTRY.implementations(op) if s.backend == "flashinfer")
            assert spec.golden.verified
            assert spec.golden.max_abs_diff is not None
            assert spec.priority < 0

    def test_paged_rows_need_the_layout_tag(self) -> None:
        # The cache-facing rows read this repo's paged buffer. Without the tag
        # the call site is describing some other pool, and dispatch must say so
        # instead of handing over a kernel that would read the wrong strides.
        for op in ("attention.decode", "kv_write"):
            with pytest.raises(LookupError, match="layout"):
                dispatch(op, dtype="bf16")
            assert dispatch(op, dtype="bf16", layout=PAGED).spec.name == ATTENTION_NATIVE_ROWS[op]

    def test_prefill_reads_no_cache_so_needs_no_layout(self) -> None:
        # Prefill attends the freshly projected tensors, not the cache.
        sel = dispatch("attention.prefill", dtype="bf16")
        assert sel.spec.name == ATTENTION_NATIVE_ROWS["attention.prefill"]

    def test_kv_write_accepts_quantised_rows(self) -> None:
        # An fp8 cache is written as uint8 bytes: quantisation happens before
        # the write, so the row must not gate on the activation dtype.
        assert dispatch("kv_write", dtype="u8", layout=PAGED).spec.name == "native/update_kv_buffer"

    def test_mla_decode_never_dispatches_without_its_gates(self) -> None:
        # The latent cache is not interchangeable with the per-head paged
        # pool, so both rows demand the ``kv:mla_latent`` tag — and even with
        # it, default dispatch refuses until a golden run verifies a row.
        # Either gate refuses, and the failure names the row instead of
        # silently routing somewhere wrong.
        with pytest.raises(LookupError, match="no usable implementation"):
            dispatch("attention.mla_decode", dtype="bf16")
        with pytest.raises(LookupError, match="golden"):
            dispatch("attention.mla_decode", dtype="bf16", layout=MLA_LATENT)


class TestGlueCatalogue:
    """moe / rmsnorm / rope / elementwise: the domains between the two GEMMs."""

    @pytest.mark.parametrize("op,row", sorted(GLUE_NATIVE_ROWS.items()))
    def test_each_has_one_native_row(self, op: str, row: str) -> None:
        native = {s.name for s in REGISTRY.implementations(op) if s.backend == "native"}
        assert native == {row} | GLUE_EXTRA_NATIVE_ROWS.get(op, set())
        # Extra rows are scheme-gated, so the floor is still the unqualified one.
        assert REGISTRY.native_floor(op).name == row

    @pytest.mark.parametrize("op,rows", sorted(GLUE_EXTERNAL_ROWS.items()))
    def test_external_rows_are_registered(self, op: str, rows: set[str]) -> None:
        external = {
            s.name for s in REGISTRY.implementations(op) if s.backend not in {"native", "cpu"}
        }
        assert external == rows

    @pytest.mark.parametrize("op,row", sorted(GLUE_NATIVE_ROWS.items()))
    def test_dispatches_at_the_default_precision(self, op: str, row: str) -> None:
        assert dispatch(op, dtype="bf16").spec.name == row

    @pytest.mark.parametrize("op", sorted(GLUE_NATIVE_ROWS))
    def test_rows_declare_only_measured_dtypes(self, op: str) -> None:
        # bf16 (the default) and fp16 (what the kernel tests cover), native
        # and external rows alike. fp32 is not claimed: no path in the repo
        # runs these kernels at fp32, so declaring it would be an unbacked
        # promise dispatch would happily act on.
        for spec in REGISTRY.implementations(op):
            if spec.backend == "cpu":
                continue
            assert set(spec.dtypes) == {"bf16", "fp16"}, spec.name
        with pytest.raises(LookupError, match="dtype"):
            dispatch(op, dtype="fp32")

    def test_one_moe_row_serves_every_scheme_but_the_w8a8_pair(self) -> None:
        # fused_moe reads the expert format off ``w1.dtype`` (uint8 -> fp8,
        # int8, int32 -> int4), so for those the scheme is not a choice between
        # rows; splitting them would write several specs for one internal branch.
        # The W8A8 pair are the exceptions because their bytes are
        # indistinguishable from the weight-only formats' — int8 exactly as fp8
        # before it: see ``GLUE_EXTRA_NATIVE_ROWS``.
        row = REGISTRY.native_floor("moe")
        served_by_floor = SCHEME_TO_ROW.keys() - SCHEMES_WITHOUT_MOE - {"w8a8_fp8", "w8a8_int8"}
        for scheme in served_by_floor:
            assert dispatch("moe", dtype="bf16", scheme=scheme).spec.name == row.name
        sel = dispatch("moe", dtype="bf16", scheme="w8a8_fp8")
        assert sel.spec.name == "native/fused_moe_w8a8_fp8"
        sel = dispatch("moe", dtype="bf16", scheme="w8a8_int8")
        assert sel.spec.name == "native/fused_moe_w8a8_int8"

    @pytest.mark.parametrize("scheme", sorted(SCHEMES_WITHOUT_MOE))
    def test_schemes_without_a_moe_kernel_fail_loudly(self, scheme: str) -> None:
        # The complement of the test above, asserted rather than assumed: a
        # scheme the MoE kernel cannot run must be refused at dispatch, not
        # routed to a row that would misread its packed bytes.
        with pytest.raises(LookupError, match="scheme"):
            dispatch("moe", dtype="bf16", scheme=scheme)


class TestContractCoverage:
    def test_every_registered_op_is_a_declared_contract(self) -> None:
        from rapid_llm.kernels.ops import is_logical_op

        for op in REGISTRY.ops():
            assert is_logical_op(op), f"{op!r} has rows but no ABC"

    def test_contracts_without_rows_are_the_documented_ones(self) -> None:
        """A contract with no implementation is fine — silently is not."""
        registered = set(REGISTRY.ops())
        assert set(LOGICAL_OPS) - registered == set(OPS_WITHOUT_ROWS)

    def test_external_only_ops_have_no_native_row(self) -> None:
        """Rows without a native floor stay a decision on record."""
        for op in EXTERNAL_ONLY_OPS:
            rows = list(REGISTRY.implementations(op))
            assert rows, f"{op} must have its external row"
            assert all(s.backend != "native" for s in rows), f"{op} gained a native row"


class TestLinearTorchFloor:
    def test_matches_f_linear_on_cpu(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        b = torch.randn(8, dtype=torch.bfloat16)
        assert torch.equal(linear_torch(x, w, bias=b), F.linear(x, w, b))

    def test_rejects_quantised_inputs_loudly(self) -> None:
        x = torch.randn(2, 8)
        w = torch.randn(4, 8)
        with pytest.raises(ValueError, match="unquantised floor"):
            linear_torch(x, w, weight_scale=torch.ones(4))


class TestUnquantMethodRoutesThroughDispatch:
    def test_apply_is_dispatched_not_hardwired(self) -> None:
        class _Layer:
            weight = torch.randn(8, 16)

        out = UnquantizedLinearMethod().apply(_Layer(), torch.randn(2, 16))  # type: ignore[arg-type]
        assert out.shape == (2, 8)


class TestRegistryStaysTorchFree:
    def test_group_inits_declare_no_torch_import(self) -> None:
        """Registration must not pay the torch import — targets are strings."""
        ops_root = Path(ops_pkg.__file__).parent
        group_inits = sorted(
            p.parent.name
            for p in ops_root.glob("*/__init__.py")
            if p.parent.name not in NON_GROUP_DIRS
        )
        assert group_inits == list(OPS_GROUPS)
        for path in ops_root.glob("*/__init__.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names += [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
            offenders = [n for n in names if n.split(".")[0] in {"torch", "triton"}]
            assert not offenders, f"{path} must stay torch-free, found {offenders}"

    def test_rows_point_at_the_right_tier(self) -> None:
        """Native rows live under ``ops/<group>/``, external under ``backend/<lib>/``.

        No wrapper tier either way: every target names a module that holds the
        callable itself.
        """
        for op in REGISTRY.ops():
            for spec in REGISTRY.implementations(op):
                module, attr = spec.target.split(":")
                assert module.startswith("rapid_llm.kernels."), spec.name
                if spec.backend == "native":
                    assert module.startswith("rapid_llm.kernels.ops."), spec.name
                else:
                    assert module.startswith("rapid_llm.kernels.backend."), spec.name
                assert attr.isidentifier()

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="resolving GPU targets requires Triton")
    def test_every_row_resolves_to_a_callable(self) -> None:
        for op in REGISTRY.ops():
            for spec in REGISTRY.implementations(op):
                assert callable(resolve_target(spec.target)), spec.name


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="resolving GPU targets requires Triton")
class TestTargetsMatchTheirContract:
    """Parameter *names, kinds and defaults* are the contract, because nothing adapts them.

    Dispatch hands the caller the kernel function itself — the backend
    adapters translate library calling conventions, never argument names.
    So a kernel whose parameters merely happen to sit in the right order
    silently satisfies the ABC while breaking any caller that passes one by
    keyword, and breaking the next backend that reads the ABC to know what it
    must accept. This test is what caught ``update_kv_buffer(K_Values, ...)``
    and ``skip_rmsnorm(X, ...)``. Kinds and defaults joined the pin in
    v0.10: the ABC evaluation measured exactly one drift (a keyword-only
    marker on ``attention.decode`` that neither decode row honoured) and
    removed it — the ABC documents the kernels, not the other way round.
    """

    #: ``elementwise.*`` members declare their own arity under an ABC that is
    #: deliberately ``(x, *args)``, so beyond "takes at least one operand" there
    #: is nothing to compare; see :class:`ElementwiseOp`.
    OPEN_ARITY = ("elementwise.",)

    #: The arity each open-arity member promises in its own docstring: the fused
    #: row takes the packed gate/up tensor, the split row takes the two halves.
    ELEMENTWISE_ARITY: ClassVar[dict[str, int]] = {
        "native/swiglu_forward_fused": 1,
        "native/swiglu_forward": 2,
    }

    def _ops_to_check(self) -> list[str]:
        return [op for op in sorted(REGISTRY.ops()) if not op.startswith(self.OPEN_ARITY)]

    @staticmethod
    def _contract_params(fn) -> list[tuple[str, str, str]]:
        """(name, kind, default) per parameter — the full substitutability surface."""
        return [
            (
                p.name,
                p.kind.name,
                "<required>" if p.default is inspect.Parameter.empty else repr(p.default),
            )
            for p in inspect.signature(fn).parameters.values()
            if p.name != "self"
        ]

    def test_parameter_names_match_the_abc(self) -> None:
        for op in self._ops_to_check():
            expected = self._contract_params(LOGICAL_OPS[op].__call__)
            for spec in REGISTRY.implementations(op):
                got = self._contract_params(resolve_target(spec.target))
                assert got == expected, (
                    f"{spec.name} takes {got} but the {op!r} contract says {expected}"
                )

    def test_open_arity_members_keep_the_arity_they_advertise(self) -> None:
        # The two swiglu rows differ only in arity, which is exactly why they are
        # two ops rather than one: dispatch cannot guess how many tensors the
        # call site holds, so the op id has to say it.
        for op in sorted(REGISTRY.ops()):
            if not op.startswith(self.OPEN_ARITY):
                continue
            for spec in REGISTRY.implementations(op):
                params = inspect.signature(resolve_target(spec.target)).parameters
                assert len(params) == self.ELEMENTWISE_ARITY[spec.name], spec.name
