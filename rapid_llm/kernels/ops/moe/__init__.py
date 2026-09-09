"""MoE domain: the fused expert GEMMs, one row per implementation.

Registers the domain's spec rows and re-exports the grouped-GEMM entry points
``fused_moe`` / ``fused_moe_w8a8_fp8`` / ``fused_moe_w8a8_int8`` plus the
alignment helper. One weight-only row on purpose: ``fused_moe`` derives the
expert format from ``w1.dtype`` (uint8 fp8-e4m3 / int8 / packed int32), so a
spec row per scheme would just restate a dispatch the kernel already does
internally.

``w8a8_fp8``/``w8a8_int8`` are their own rows — they quantise the activation
too, and no dtype can say so: weight-only and W8A8 store identical expert
dtypes, so the scheme must pick the row, or a W8A8 scheme would quietly get
the weight-only kernel. DeepGEMM's grouped variant is the sm90+ contender —
same grouped-contiguous semantics, Hopper tensor cores, unverified until it
runs on real hardware.

Usage:
    from rapid_llm.kernels import fused_moe
"""

from rapid_llm.kernels.backend.deepgemm import CUDA_SM90
from rapid_llm.kernels.dispatcher import (
    NATIVE_BASELINE,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    register,
)

register(
    KernelSpec(
        name="native/fused_moe",
        op="moe",
        backend="native",
        target="rapid_llm.kernels.ops.moe.fused_moe:fused_moe",
        dtypes=("bf16", "fp16"),
        schemes=(
            "unquantized",
            "fp8",
            "blockwise_int8",
            "awq",
            "gptq",
        ),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/fused_moe_w8a8_fp8",
        op="moe",
        backend="native",
        target="rapid_llm.kernels.ops.moe.fused_moe:fused_moe_w8a8_fp8",
        # No capability floor, matching ``native/linear_w8a8_fp8``: sm89+ takes
        # the fp8 MMA and everything below widens both operands by bit trick, so
        # the row is correct everywhere and only its speed depends on the device.
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/fused_moe_w8a8_int8",
        op="moe",
        backend="native",
        target="rapid_llm.kernels.ops.moe.fused_moe:fused_moe_w8a8_int8",
        # int8 imma exists from Turing on, so there is no capability floor — the
        # same reasoning as ``native/linear_w8a8_int8``.
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_int8",),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="deepgemm/grouped_fp8_moe",
        op="moe",
        backend="deepgemm",
        target="rapid_llm.kernels.backend.deepgemm.moe:grouped_moe",
        available="rapid_llm.kernels.backend.deepgemm:available",
        capability=CUDA_SM90,
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        # Untested on hardware (needs sm90+; CI is sm86) — the golden gate
        # keeps this row out of default dispatch until a Hopper run.
        golden=GoldenRecord(verified=False, max_abs_diff=None, baseline="native/fused_moe"),
        priority=UNMEASURED,
    )
)
