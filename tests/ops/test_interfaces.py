"""Tests for the logical-operator ABCs (ROADMAP foundation 2).

Catalogue completeness, ABC enforcement (unimplemented ops cannot
instantiate), signature pinning against the pinned table, and CPU
reference implementations honouring the contracts.

Usage:
    pytest tests/ops/test_interfaces.py
"""

from __future__ import annotations

import ast
from inspect import signature

import pytest
import torch

from rapid_llm.kernels.ops.interfaces import (
    LOGICAL_OPS,
    AttentionDecodeOp,
    AttentionPrefillOp,
    CombineOp,
    DispatchOp,
    ElementwiseOp,
    KvWriteOp,
    LinearOp,
    LogicalOp,
    MlaDecodeOp,
    MlaPrefillOp,
    MoeOp,
    RmsNormOp,
    RopeOp,
    SampleOp,
    is_logical_op,
)

# ---------------------------------------------------------------------------
# CPU reference implementations — the templates native adapters copy.
# ---------------------------------------------------------------------------


class CpuRmsNorm(RmsNormOp):
    def __call__(self, x, residual, weight, eps=1e-5):
        if residual is not None:
            residual = x + residual  # the Triton path stores the sum back into R
            x = residual
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        y = (x * torch.rsqrt(var + eps)).to(x.dtype) * weight
        # Always a pair: on the plain path the second element is the input, so
        # a decoder layer threads one pair through either path (see RmsNormOp).
        return y, (residual if residual is not None else x)


class CpuLinear(LinearOp):
    def __call__(
        self,
        x,
        weight,
        *,
        bias=None,
        weight_scale=None,
        weight_zeros=None,
        group_n=0,
        group_k=0,
    ):
        w = weight.to(torch.float32)
        if weight_scale is not None:
            # per-output-channel scales: [N] broadcasts along K, not across rows
            w = w * weight_scale.to(torch.float32)[:, None]
        y = x.to(torch.float32) @ w.T
        if bias is not None:
            y = y + bias.to(torch.float32)
        return y.to(x.dtype)


class CpuMoe(MoeOp):
    def __call__(
        self,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        *,
        w1_scale=None,
        w2_scale=None,
        w1_zeros=None,
        w2_zeros=None,
        group_n=0,
        group_k=0,
        swiglu_limit=float("inf"),
    ):
        inter = w1.shape[1] // 2
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        for t in range(hidden_states.shape[0]):
            for j in range(topk_ids.shape[1]):
                e, w = int(topk_ids[t, j]), float(topk_weights[t, j])
                gate, up = w1[e, :inter] @ hidden_states[t], w1[e, inter:] @ hidden_states[t]
                if swiglu_limit < float("inf"):
                    gate = min(gate, swiglu_limit)
                    up = torch.clamp(up, -swiglu_limit, swiglu_limit)
                out[t] += w * (w2[e] @ (torch.nn.functional.silu(gate) * up))
        return out.to(hidden_states.dtype)


class CpuKvWrite(KvWriteOp):
    def __call__(self, k, v, select_index, kv_buffer):
        half = kv_buffer.shape[0] // 2
        kv_buffer[select_index] = k
        kv_buffer[select_index + half] = v


class CpuSample(SampleOp):
    def __call__(self, logits, *, temperature=1.0, top_k=0, top_p=1.0, deterministic=False):
        if deterministic:
            return logits.argmax(-1)
        probs = torch.softmax(logits.to(torch.float32) / max(temperature, 1e-6), -1)
        if top_k > 0:
            kth = torch.topk(probs, top_k, -1).values[..., -1:]
            probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
        return torch.multinomial(probs, 1).squeeze(-1)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

PLANNED_OPS = {
    "attention.prefill",
    "attention.decode",
    "attention.mla_decode",
    "attention.mla_prefill",
    "attention.chunked_prefill",
    "linear",
    "moe",
    "comm.dispatch",
    "comm.combine",
    "rmsnorm",
    "rope",
    "kv_write",
    "sample",
    "elementwise",
}


class TestCatalog:
    def test_op_ids_match_the_plan_catalogue(self) -> None:
        assert set(LOGICAL_OPS) == PLANNED_OPS

    def test_catalogue_entries_are_logical_ops(self) -> None:
        for op_id, cls in LOGICAL_OPS.items():
            assert issubclass(cls, LogicalOp)
            assert cls.op_id == op_id  # key and class agree

    @pytest.mark.parametrize(
        ("op", "expected"),
        [
            ("linear", True),
            ("attention.decode", True),
            ("elementwise", True),
            ("elementwise.swiglu", True),
            ("elementwise.silu", True),
            ("linear.dense", False),  # only elementwise is open
            ("attention", False),
            ("", False),
        ],
    )
    def test_is_logical_op(self, op: str, expected: bool) -> None:
        assert is_logical_op(op) is expected


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestAbcEnforcement:
    def test_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            LogicalOp()  # type: ignore[abstract]

    @pytest.mark.parametrize("cls", sorted(LOGICAL_OPS.values(), key=lambda c: c.op_id))
    def test_interfaces_are_abstract(self, cls) -> None:
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]

    def test_a_full_implementation_instanciates_and_calls(self) -> None:
        norm = CpuRmsNorm()
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        y, _ = norm(x, None, torch.ones(8))
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Signature pinning — changing a contract must break a test, visibly.
# ---------------------------------------------------------------------------

SIGNATURES = {
    AttentionPrefillOp: (
        ["q", "k", "v", "sm_scale", "b_start_loc", "b_seq_len", "max_seq_len"],
        [],
    ),
    AttentionDecodeOp: (
        [
            "q",
            "k_cache",
            "v_cache",
            "qk_scale",
            "b_req_tokens_table",
            "b_req_idx",
            "b_seq_len",
            "max_actual_seq_len",
            # v0.10 ABC audit: k_scale/v_scale follow the native kernels
            # (positional-or-keyword, default 1.0); the keyword-only marker on
            # the ABC was the one signature drift in the catalogue.
            "k_scale",
            "v_scale",
        ],
        [],
    ),
    MlaDecodeOp: (["q", "kv_cache", "block_table", "cache_seqlens"], ["max_seq_len", "sm_scale"]),
    MlaPrefillOp: (
        [
            "q_nope",
            "q_pe",
            "c_kv",
            "k_pe",
            "w_uk",
            "w_uv",
            "sm_scale",
            "b_start_loc",
            "b_seq_len",
            "max_seq_len",
        ],
        [],
    ),
    LinearOp: (
        ["x", "weight"],
        [
            "bias",
            "weight_scale",
            "weight_zeros",
            "weight_global_scale",
            "group_n",
            "group_k",
        ],
    ),
    MoeOp: (
        ["hidden_states", "w1", "w2", "topk_weights", "topk_ids"],
        [
            "w1_scale",
            "w2_scale",
            "w1_zeros",
            "w2_zeros",
            "group_n",
            "group_k",
            "swiglu_limit",
            "mxfp4",
        ],
    ),
    DispatchOp: (["x", "topk_idx"], ["num_experts"]),
    CombineOp: (["x", "unsorted_src_idx", "unsorted_weights"], []),
    RmsNormOp: (["x", "residual", "weight", "eps"], []),
    RopeOp: (["q", "k", "cos", "sin"], []),
    KvWriteOp: (["k", "v", "select_index", "kv_buffer"], []),
    SampleOp: (["logits"], ["temperature", "top_k", "top_p", "deterministic"]),
    ElementwiseOp: (["x"], []),
}


class TestSignaturePin:
    @pytest.mark.parametrize(
        ("cls", "expected"), sorted(SIGNATURES.items(), key=lambda kv: kv[0].op_id)
    )
    def test_call_parameters_are_pinned(self, cls, expected) -> None:
        pos_names, kw_names = expected
        params = list(signature(cls.__call__).parameters.values())[1:]  # drop self
        assert [
            p.name
            for p in params
            if p.kind not in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.KEYWORD_ONLY, p.VAR_KEYWORD)
        ] == pos_names, f"{cls.__name__} positional/keyword-or-positional drifted"
        assert [p.name for p in params if p.kind is p.KEYWORD_ONLY] == kw_names, (
            f"{cls.__name__} keyword-only set drifted"
        )

    def test_decode_defaults_match_native_kernel(self) -> None:
        params = signature(AttentionDecodeOp.__call__).parameters
        assert params["k_scale"].default == 1.0
        assert params["v_scale"].default == 1.0


# ---------------------------------------------------------------------------
# CPU numerics — the contracts bind and compute correctly on CPU tensors.
# ---------------------------------------------------------------------------


class TestCpuImplementations:
    def test_rmsnorm_skip_path_matches_eager_reference(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16, dtype=torch.float32)
        residual = torch.randn(2, 4, 16, dtype=torch.float32)
        weight = torch.randn(16)

        y, new_residual = CpuRmsNorm()(x, residual, weight, eps=1e-6)

        fused = x + residual
        var = fused.pow(2).mean(-1, keepdim=True)
        ref_y = fused * torch.rsqrt(var + 1e-6) * weight
        assert torch.allclose(y, ref_y, atol=1e-5)
        assert torch.allclose(new_residual, fused, atol=1e-6)

    def test_rmsnorm_plain_path_passes_the_input_through_as_residual(self) -> None:
        # The pair shape is the same on both paths; what changes is what the
        # second element means. Here it is the untouched input, which is what
        # lets the caller keep one unpacking form.
        x = torch.randn(3, 8, dtype=torch.float32)
        weight = torch.ones(8)
        y, passthrough = CpuRmsNorm()(x, None, weight)
        assert y.shape == x.shape
        assert passthrough is x

    def test_linear_scales_before_matmul(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(6, 8, dtype=torch.bfloat16)
        weight = torch.randint(-32, 32, (4, 8), dtype=torch.int8)
        scale = torch.full((4,), 0.125, dtype=torch.float32)

        y = CpuLinear()(x, weight, weight_scale=scale)

        ref = x.to(torch.float32) @ (weight.to(torch.float32) * scale[:, None]).T
        # both paths compute in fp32 and round once to bf16 — expect bit equality
        assert torch.equal(y, ref.to(x.dtype))
        assert y.dtype == x.dtype

    def test_moe_binds_the_full_signature(self) -> None:
        torch.manual_seed(2)
        tokens, hidden, inter, experts, top_k = 3, 8, 4, 2, 2
        x = torch.randn(tokens, hidden)
        w1 = torch.randn(experts, 2 * inter, hidden)
        w2 = torch.randn(experts, hidden, inter)
        weights = torch.rand(tokens, top_k)
        ids = torch.tensor([[0, 1], [1, 0], [0, 1]])

        out = CpuMoe()(x, w1, w2, weights, ids)

        assert out.shape == (tokens, hidden)
        row0 = weights[0, 0] * (
            w2[0] @ (torch.nn.functional.silu(w1[0, :inter] @ x[0]) * (w1[0, inter:] @ x[0]))
        ) + weights[0, 1] * (
            w2[1] @ (torch.nn.functional.silu(w1[1, :inter] @ x[0]) * (w1[1, inter:] @ x[0]))
        )
        assert torch.allclose(out[0], row0, atol=1e-5)

    def test_kv_write_scatters_k_first_half_v_second(self) -> None:
        max_tokens, heads, dim = 16, 2, 4
        buffer = torch.zeros(2 * max_tokens, heads, dim)
        k = torch.randn(3, heads, dim)
        v = torch.randn(3, heads, dim)
        idx = torch.tensor([2, 7, 0])

        CpuKvWrite()(k, v, idx, buffer)

        assert torch.equal(buffer[idx], k)
        assert torch.equal(buffer[idx + max_tokens], v)

    def test_sample_deterministic_is_argmax(self) -> None:
        logits = torch.tensor([[1.0, 9.0, 3.0], [5.0, 1.0, 4.0]])
        tokens = CpuSample()(logits, deterministic=True)
        assert tokens.tolist() == [1, 0]

    def test_sample_top_k_zeroes_outside_the_window(self) -> None:
        torch.manual_seed(3)
        logits = torch.full((1, 6), -10.0)
        logits[0, 3] = 10.0
        tokens = [int(CpuSample()(logits, top_k=1)[0]) for _ in range(8)]
        assert set(tokens) == {3}


# ---------------------------------------------------------------------------
# torch-free at runtime
# ---------------------------------------------------------------------------


class TestTorchFreeContracts:
    def test_interfaces_module_never_imports_torch_at_runtime(self) -> None:
        import rapid_llm.kernels.ops.interfaces as iface

        tree = ast.parse(__import__("pathlib").Path(iface.__file__).read_text(encoding="utf-8"))

        def runtime_imports(node: ast.AST, guarded: bool) -> list[str]:
            found: list[str] = []
            for child in ast.iter_child_nodes(node):
                is_guard = (
                    isinstance(child, ast.If)
                    and isinstance(child.test, ast.Name)
                    and child.test.id == "TYPE_CHECKING"
                )
                if is_guard:
                    found += runtime_imports(child, True)
                elif isinstance(child, (ast.Import, ast.ImportFrom)) and not guarded:
                    names = (
                        [a.name for a in child.names]
                        if isinstance(child, ast.Import)
                        else [child.module or ""]
                    )
                    found += names
                else:
                    found += runtime_imports(child, guarded)
            return found

        offenders = [
            n for n in runtime_imports(tree, False) if n.split(".")[0] in {"torch", "triton"}
        ]
        assert not offenders, f"interfaces.py must stay runtime torch-free, found {offenders}"
