"""DeepSeek-V4 attention stack: mHC hyper-connections, interleaved partial
RoPE, shared-KV MQA with sliding / compressed (CSA / HCA) attention, the
grouped low-rank output projection and the Lightning Indexer.

The module set mirrors transformers 5.8's ``modeling_deepseek_v4.py`` — V4
ships no public weights, so that implementation (over a randomly-initialised
trimmed checkpoint) is the parity reference. Each submodule documents its own
deviations from that reference; they are all deliberate.

Layout:
    rope.py              interleaved partial RoPE, main / compress tables
    norm.py              weighted and unweighted RMSNorm
    hyper_connection.py  mHC residual (Sinkhorn mixing + hc_head)
    cache.py             sliding K==V per-layer cache, bypassing the paged store
    compressor.py        HCA / CSA compressors and the Lightning Indexer
    grouped_linear.py    block-diagonal ``o_a_proj``
    attention.py         the attention that composes all of the above

Names resolve lazily (like :mod:`rapid_llm.modules`); ``rope`` stays free of
the Triton kernels ``norm`` imports.

Usage:
    attn = DeepseekV4Attention(config, layer_index)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attention import DeepseekV4Attention
    from .cache import V4LayerCache
    from .compressor import (
        DeepseekV4CSACompressor,
        DeepseekV4HCACompressor,
        DeepseekV4Indexer,
    )
    from .grouped_linear import DeepseekV4GroupedLinear
    from .hyper_connection import DeepseekV4HyperConnection, DeepseekV4HyperHead
    from .norm import DeepseekV4RMSNorm, DeepseekV4UnweightedRMSNorm
    from .rope import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb, rotate_half

_EXPORTS: dict[str, tuple[str, str]] = {
    "DeepseekV4Attention": (".attention", "DeepseekV4Attention"),
    "DeepseekV4CSACompressor": (".compressor", "DeepseekV4CSACompressor"),
    "DeepseekV4GroupedLinear": (".grouped_linear", "DeepseekV4GroupedLinear"),
    "DeepseekV4HCACompressor": (".compressor", "DeepseekV4HCACompressor"),
    "DeepseekV4HyperConnection": (".hyper_connection", "DeepseekV4HyperConnection"),
    "DeepseekV4HyperHead": (".hyper_connection", "DeepseekV4HyperHead"),
    "DeepseekV4Indexer": (".compressor", "DeepseekV4Indexer"),
    "DeepseekV4RMSNorm": (".norm", "DeepseekV4RMSNorm"),
    "DeepseekV4RotaryEmbedding": (".rope", "DeepseekV4RotaryEmbedding"),
    "DeepseekV4UnweightedRMSNorm": (".norm", "DeepseekV4UnweightedRMSNorm"),
    "V4LayerCache": (".cache", "V4LayerCache"),
    "apply_rotary_pos_emb": (".rope", "apply_rotary_pos_emb"),
    "rotate_half": (".rope", "rotate_half"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4CSACompressor",
    "DeepseekV4GroupedLinear",
    "DeepseekV4HCACompressor",
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4Indexer",
    "DeepseekV4RMSNorm",
    "DeepseekV4RotaryEmbedding",
    "DeepseekV4UnweightedRMSNorm",
    "V4LayerCache",
    "apply_rotary_pos_emb",
    "rotate_half",
]
