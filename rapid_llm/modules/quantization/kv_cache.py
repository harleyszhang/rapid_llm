"""KV cache quantisation methods (mirrors sglang ``kv_cache.py``).

``get_kv_cache_method`` maps a ``kv_cache_dtype`` string to its method;
:class:`Fp8KVCacheMethod` stores K/V rows as fp8-e4m3 with the scale
plumbing the attention kernels already accept.

Usage:
    method = get_kv_cache_method(kv_cache_dtype)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .base_config import QuantizeMethodBase
from .utils import quantize_fp8_per_tensor


class BaseKVCacheMethod(QuantizeMethodBase, ABC):
    """Abstract base for KV-cache quantisation strategies.

    Attributes:
        k_scale: Per-tensor scale the write side applied to keys, which the read
            side must divide back out. Both scales belong to the strategy rather
            than the attention layer: the two halves are only correct together,
            and a layer holding its own copy could silently disagree.
        v_scale: Same for values.
    """

    k_scale: float = 1.0
    v_scale: float = 1.0

    @abstractmethod
    def quantize_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise K and V before writing to the cache."""
        raise NotImplementedError()

    # Satisfy the QuantizeMethodBase interface (not used directly for KV).
    def apply(self, layer, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("BaseKVCacheMethod.apply should not be called directly")


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """fp8-e4m3 KV cache: per-tensor scale, uint8 storage.

    The bit-trick dequant in the flash-decoding and chunked prefill kernels
    under-estimates by 2**8; the caller-side wrappers pre-multiply that factor
    into the scale for zero host-side overhead in the hot path.

    Args:
        k_scale: Per-tensor scale for key quantisation (default 1.0).
        v_scale: Per-tensor scale for value quantisation (default 1.0).
    """

    def __init__(self, k_scale: float = 1.0, v_scale: float = 1.0) -> None:
        self.k_scale = k_scale
        self.v_scale = v_scale

    def quantize_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise K/V to e4m3 bytes in a uint8 container."""
        return (
            quantize_fp8_per_tensor(k, self.k_scale),
            quantize_fp8_per_tensor(v, self.v_scale),
        )


def get_kv_cache_method(kv_cache_dtype: torch.dtype) -> BaseKVCacheMethod | None:
    """Factory: return the appropriate KV cache method for the given dtype.

    Returns None for fp16 (no quantisation needed).
    """
    if kv_cache_dtype == torch.uint8:
        return Fp8KVCacheMethod()
    return None
