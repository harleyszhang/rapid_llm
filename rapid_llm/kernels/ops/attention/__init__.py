"""Attention domain: the prefill/decode phases, KV write and MLA decode.

Registers the domain's spec rows and points at the implementations:
:mod:`flashattention2_nopad` for prefill, :mod:`flashdecoding` for paged
decode, :mod:`mla` for the latent-cache MLA pair, plus the KV-write kernels
in ``kvcache``.

Usage:
    from rapid_llm.kernels.ops.attention.flashdecoding import flash_decoding
"""

from rapid_llm.kernels.backend.flashinfer import CUDA_SM75
from rapid_llm.kernels.backend.flashmla import CUDA_SM90
from rapid_llm.kernels.dispatcher import (
    NATIVE_BASELINE,
    PAGED_KV,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    register,
)

# --------------------------------------------------------------------------- #
# native — the Triton kernels this repo ships
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/flash_attention2_no_pad",
        op="attention.prefill",
        backend="native",
        target="rapid_llm.kernels.ops.attention.flashattention2_nopad:flash_attention2_no_pad",
        # fp32 has no path through the kernel: it casts inputs to fp16.
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/flash_attention2_chunked",
        op="attention.chunked_prefill",
        backend="native",
        target="rapid_llm.kernels.ops.attention.flashattention2_nopad:flash_attention2_chunked",
        # fp8_kv rides the same kernel as the unquantised cache: uint8 rows are
        # e4m3 bytes dequantised on load, so no separate row is warranted.
        dtypes=("bf16", "fp16"),
        schemes=("unquantized", "fp8_kv"),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/flash_decoding",
        op="attention.decode",
        backend="native",
        target="rapid_llm.kernels.ops.attention.flashdecoding:flash_decoding",
        dtypes=("bf16", "fp16"),
        # fp8_kv rides the same kernel: K/V arrive as uint8 rows and are
        # dequantised inside, so no separate row is warranted.
        schemes=("unquantized", "fp8_kv"),
        layout=PAGED_KV,
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/mla_decode",
        op="attention.mla_decode",
        backend="native",
        target="rapid_llm.kernels.ops.attention.mla:mla_decode",
        dtypes=("bf16", "fp16"),
        layout=LayoutRequirement(required=("kv:mla_latent",)),
        # Unverified until the DeepSeek-V2 golden run turns the kernel-test
        # evidence into a measured diff against HF; the in-file PyTorch
        # reference is the baseline those tests already pin the row to.
        golden=GoldenRecord(
            verified=False,
            max_abs_diff=None,
            baseline="ops/attention/mla.py:mla_decode_reference",
        ),
    )
)
register(
    KernelSpec(
        name="native/mla_prefill",
        op="attention.mla_prefill",
        backend="native",
        target="rapid_llm.kernels.ops.attention.mla:mla_prefill",
        dtypes=("bf16", "fp16"),
        layout=LayoutRequirement(required=("kv:mla_latent",)),
        golden=GoldenRecord(
            verified=False,
            max_abs_diff=None,
            baseline="full-upsample oracle over native/flash_attention2_no_pad",
        ),
    )
)

# --------------------------------------------------------------------------- #
# flashinfer — both phases, wheel-installable, Ampere onward
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="flashinfer/prefill",
        op="attention.prefill",
        backend="flashinfer",
        target="rapid_llm.kernels.backend.flashinfer.attention:prefill_attention",
        available="rapid_llm.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16"),
        golden=GoldenRecord(
            verified=True, max_abs_diff=2.0e-2, baseline="native/flash_attention2_no_pad"
        ),
        priority=UNMEASURED,
    )
)
register(
    KernelSpec(
        name="flashinfer/decode",
        op="attention.decode",
        backend="flashinfer",
        target="rapid_llm.kernels.backend.flashinfer.attention:decode_attention",
        available="rapid_llm.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16"),
        layout=PAGED_KV,
        golden=GoldenRecord(verified=True, max_abs_diff=2.0e-2, baseline="native/flash_decoding"),
        priority=UNMEASURED,
        # decode_attention assembles the page indices with Python-side slicing
        # over the live lengths and plans the wrapper on every call: captured
        # into a CUDA graph, both bake the capture-time lengths in and replay
        # silently attends stale rows. graph_safe=False makes the runner refuse
        # to capture while this row is chosen (see unsafe_for_graph); the
        # backend module's paged_kv_indices_gpu + prepare_decode (fed by the
        # engine's CPU length ledger) cover the eager half of the vLLM-style
        # fix — fast_decode_plan is what remains for the capture half.
        graph_safe=False,
        step_prepare="rapid_llm.kernels.backend.flashinfer.attention:prepare_decode",
    )
)

# --------------------------------------------------------------------------- #
# flashmla — the external MLA decode contender, Hopper-only
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="flashmla/mla_decode",
        op="attention.mla_decode",
        backend="flashmla",
        target="rapid_llm.kernels.backend.flashmla.mla_decode:mla_decode",
        available="rapid_llm.kernels.backend.flashmla:available",
        capability=CUDA_SM90,
        # The latent cache (c_kv: [Skv, L_kv]) is not interchangeable with the
        # per-head paged pool the rows above share — its own tag says so.
        layout=LayoutRequirement(required=("kv:mla_latent",)),
        golden=GoldenRecord(
            verified=False, max_abs_diff=None, baseline="minimal MLA harness reference"
        ),
        priority=UNMEASURED,
        # Same wrapper family as flashinfer's decode (host-side plan, per-call
        # scheduling), so the same capture hazard applies: refuse it while
        # graphs are on.
        graph_safe=False,
    )
)
