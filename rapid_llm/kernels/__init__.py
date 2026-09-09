"""Kernel layer: implementations in ``ops/``, policy in ``dispatcher/``.

The public names re-export the dispatch tier (``dispatch``) and the
individual op entry points; importing the package loads no CUDA library
until a kernel is actually dispatched.

Usage:
    from rapid_llm.kernels import dispatch, fused_moe
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import backend as backend
from . import dispatcher as dispatcher
from . import ops as ops
from .backend import cpu_specs as cpu_specs

# Dispatch machinery: the ops with contenders go through these.
from .dispatcher import Selected, dispatch, explain, invalidate_cache, op_backend_env
from .dispatcher.autotune import install_frozen_perf_provider

# The kernels the model/engine layers call directly, resolved on first use:
# importing an implementation module pulls in Triton, while callers such as
# modules/attention only need ``dispatch`` from the torch-free machinery above.
_EXPORTS: dict[str, tuple[str, str]] = {
    "gelu": (".ops.activation.activations", "gelu"),
    "leaky_relu": (".ops.activation.activations", "leaky_relu"),
    "relu": (".ops.activation.activations", "relu"),
    "silu": (".ops.activation.activations", "silu"),
    "tanh": (".ops.activation.activations", "tanh"),
    "swiglu_forward": (".ops.activation.swiglu", "swiglu_forward"),
    "swiglu_forward_fused": (".ops.activation.swiglu", "swiglu_forward_fused"),
    "flash_attention2_no_pad": (".ops.attention.flashattention2_nopad", "flash_attention2_no_pad"),
    "flash_attention2_chunked": (
        ".ops.attention.flashattention2_nopad",
        "flash_attention2_chunked",
    ),
    "flash_decoding": (".ops.attention.flashdecoding", "flash_decoding"),
    "vocab_parallel_embedding": (".ops.embeddings.vocab_embedding", "vocab_parallel_embedding"),
    "linear_torch": (".ops.gemm.linear", "linear_torch"),
    "linear_w4a16": (".ops.gemm.linear", "linear_w4a16"),
    "linear_w8a8_fp8": (".ops.gemm.linear", "linear_w8a8_fp8"),
    "linear_w8a8_int8": (".ops.gemm.linear", "linear_w8a8_int8"),
    "linear_w8a16": (".ops.gemm.linear", "linear_w8a16"),
    "update_kv_buffer": (".ops.kvcache.update_kv_buffer", "update_kv_buffer"),
    "update_kv_index": (".ops.kvcache.update_kv_index", "update_kv_index"),
    "skip_rmsnorm": (".ops.layernorm.skip_rmsnorm", "skip_rmsnorm"),
    "fused_moe": (".ops.moe.fused_moe", "fused_moe"),
    "fused_moe_w8a8_fp8": (".ops.moe.fused_moe", "fused_moe_w8a8_fp8"),
    "fused_moe_w8a8_int8": (".ops.moe.fused_moe", "fused_moe_w8a8_int8"),
    "moe_align_block_size": (".ops.moe.fused_moe", "moe_align_block_size"),
    "grouped_topk": (".ops.moe.grouped_topk", "grouped_topk"),
    "grouped_topk_torch": (".ops.moe.grouped_topk", "grouped_topk_torch"),
    "repack_int4_experts": (".ops.quantization", "repack_int4_experts"),
    "smoothquant_matmul": (".ops.quantization", "smoothquant_matmul"),
    "unpack_int8_experts": (".ops.quantization", "unpack_int8_experts"),
    "w4a16_matmul": (".ops.quantization", "w4a16_matmul"),
    "w8a16_matmul": (".ops.quantization", "w8a16_matmul"),
    "qk_rmsnorm": (".ops.layernorm.skip_rmsnorm", "qk_rmsnorm"),
    "fused_add_rmsnorm": (".ops.layernorm.skip_rmsnorm", "fused_add_rmsnorm"),
    "fused_allreduce_rmsnorm": (".ops.layernorm.skip_rmsnorm", "fused_allreduce_rmsnorm"),
    "rope_emb_forward": (".ops.rope.rope_emb", "rope_emb_forward"),
}


_CPU_OPS = frozenset(
    {
        "skip_rmsnorm",
        "fused_add_rmsnorm",
        "fused_allreduce_rmsnorm",
        "qk_rmsnorm",
        "rope_emb_forward",
        "vocab_parallel_embedding",
        "update_kv_buffer",
        "update_kv_index",
        "flash_attention2_no_pad",
        "flash_attention2_chunked",
        "flash_decoding",
        "fused_moe",
        "repack_int4_experts",
        "unpack_int8_experts",
        "fused_moe_w8a8_fp8",
        "fused_moe_w8a8_int8",
    }
)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    if name in _CPU_OPS:
        implementations = {}

        def value(*args, **kwargs):
            if args:
                tensor = args[0]
            else:
                from inspect import signature

                cpu_impl = getattr(import_module(".backend.cpu", __name__), attribute)
                tensor = kwargs[next(iter(signature(cpu_impl).parameters))]
            device_type = tensor.device.type
            if device_type not in implementations:
                target = ".backend.cpu" if device_type == "cpu" else module_name
                implementations[device_type] = getattr(import_module(target, __name__), attribute)
            return implementations[device_type](*args, **kwargs)

        value.__name__ = name
    else:
        value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())


# Frozen measured ranking (ROADMAP v0.10): records under the autotune cache's
# frozen/ dir become the rank step's perf input. Nothing is read until the
# first dispatch asks, and RAPID_LLM_FROZEN_RANK=0 turns the lookup off.
install_frozen_perf_provider()

__all__ = [
    "Selected",
    "backend",
    "dispatch",
    "explain",
    "flash_attention2_chunked",
    "flash_attention2_no_pad",
    "flash_decoding",
    "fused_add_rmsnorm",
    "fused_allreduce_rmsnorm",
    "fused_moe",
    "fused_moe_w8a8_fp8",
    "fused_moe_w8a8_int8",
    "gelu",
    "grouped_topk",
    "grouped_topk_torch",
    "invalidate_cache",
    "leaky_relu",
    "linear_torch",
    "linear_w4a16",
    "linear_w8a8_fp8",
    "linear_w8a8_int8",
    "linear_w8a16",
    "moe_align_block_size",
    "op_backend_env",
    "ops",
    "qk_rmsnorm",
    "relu",
    "repack_int4_experts",
    "rope_emb_forward",
    "silu",
    "skip_rmsnorm",
    "smoothquant_matmul",
    "swiglu_forward",
    "swiglu_forward_fused",
    "tanh",
    "unpack_int8_experts",
    "update_kv_buffer",
    "update_kv_index",
    "vocab_parallel_embedding",
    "w4a16_matmul",
    "w8a16_matmul",
]
