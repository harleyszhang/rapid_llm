"""Paged attention: the KV-cache write plus the phase-appropriate kernel call.

:class:`PagedAttention` owns no math: it scatters fresh K/V rows into the
paged buffer, then dispatches prefill (varlen) or decode (paged) to the
kernel layer using the step's :class:`AttentionMetadata`.

Usage:
    attn = PagedAttention(num_kv_heads, head_dim, kv_cache_dtype, params_dtype)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..kernels import dispatch
from ..kernels.dispatcher import PAGED_KV_TAGS
from .quantization.kv_cache import get_kv_cache_method


class PagedAttention(nn.Module):
    """Writes K/V into the paged cache and runs the phase-appropriate kernel.

    Both phases share the cache layout ``[2 * max_tokens, num_kv_heads, head_dim]`` (K rows
    first half, V rows second). ``kv_cache_dtype`` of ``torch.uint8`` stores fp8-e4m3 bytes;
    the dequant scales come from the strategy ``get_kv_cache_method`` returns, so write and
    read cannot disagree. Both dtype args follow vLLM's auto convention (``None`` defers to
    ``torch.get_default_dtype()``; the model passes the checkpoint's dtype).
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if kv_cache_dtype is None:
            kv_cache_dtype = torch.get_default_dtype()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.kv_cache_dtype = kv_cache_dtype
        self.params_dtype = params_dtype
        self.kv_cache_method = get_kv_cache_method(kv_cache_dtype)
        # Scales live on the strategy, not this layer: the quantise-on-write and
        # dequantise-on-read halves are only correct as a pair.
        method = self.kv_cache_method
        self.k_scale = method.k_scale if method is not None else 1.0
        self.v_scale = method.v_scale if method is not None else 1.0

        # Bind after placement: construction may happen on the meta device.
        self._bound_device = None

        # K/V halves of this layer's cache row, rebuilt only when the backing buffer
        # changes. The live buffers are allocated once (the executor owns them, the CUDA
        # graph shares the list), so after the first step the identity check hits and each
        # decode step skips two slice-view constructions per layer. Profiling is the one
        # legitimate swap (its dummy forward runs on scratch rows) and the check absorbs it.
        # Keeping the source alive alongside the views makes ``is`` safe: the comparison
        # never sees a recycled address because the cached buffer cannot be freed.
        self._kv_view_pair: tuple[torch.Tensor, torch.Tensor] | None = None
        self._kv_view_source: torch.Tensor | None = None

    def _bind_kernels(self, device_type: str) -> None:
        """Select from the actual input device, including CPU on CUDA hosts."""
        if self._bound_device == device_type:
            return
        if device_type == "cpu":
            from ..kernels.backend import cpu

            self._kv_write = cpu.update_kv_buffer
            self._prefill = cpu.flash_attention2_no_pad
            self._chunked = cpu.flash_attention2_chunked
            self._decode = cpu.flash_decoding
        else:
            # One scheme for both cache readers: the chunked prefill and decode
            # kernels dequantise the same e4m3 rows, so a row that serves one
            # and not the other would be a dispatch that disagrees with itself.
            scheme = "fp8_kv" if self.kv_cache_dtype == torch.uint8 else "unquantized"
            self._kv_write = dispatch(
                "kv_write", dtype=self.kv_cache_dtype, layout=PAGED_KV_TAGS
            ).load()
            self._prefill = dispatch("attention.prefill", dtype=self.params_dtype).load()
            self._chunked = dispatch(
                "attention.chunked_prefill", dtype=self.params_dtype, scheme=scheme
            ).load()
            self._decode = dispatch(
                "attention.decode",
                dtype=self.params_dtype,
                scheme=scheme,
                layout=PAGED_KV_TAGS,
            ).load()
        self._bound_device = device_type

    def _write_cache(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> None:
        """Scatter this step's K/V into their allocated cache rows."""
        self._bind_kernels(xk.device.type)
        if self.kv_cache_method is not None:
            xk, xv = self.kv_cache_method.quantize_kv(xk, xv)

        self._kv_write(xk, xv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index])

    def _kv_views(self, kv_buffer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V halves of a layer's cache row, held while the buffer holds.

        The cache packs K into the first ``num_kv_heads`` rows and V into the rest; both
        kernels want the halves as separate tensors, and both call sites used to slice them
        afresh every step.
        """
        views = self._kv_view_pair
        if views is None or self._kv_view_source is not kv_buffer:
            self._kv_view_source = kv_buffer
            views = (
                kv_buffer[:, : self.num_kv_heads, :],
                kv_buffer[:, self.num_kv_heads :, :],
            )
            self._kv_view_pair = views
        return views

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Prefill: causal attention over the freshly computed q/k/v.

        The plain kernel reads the unquantised tensors it was handed, not the cache (the
        write above only makes them available to later decode steps). A chunk resuming on
        cached rows (``b_prefix_len`` armed) cannot use it — self-attention over the grid
        alone would drop the prefix — so its queries run through the chunked kernel, whose
        keys/values are the slot's cache rows (prefix plus this chunk, contiguous from
        ``b_kv_base``) at tensor-core prefill prices, not the one-row-per-token extend path.
        An fp8 cache takes that same route: the strategy's scales travel with the call, and
        the kernel widens the e4m3 rows on load.

        Returns:
            ``[tokens, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        if atten_info.b_prefix_len is not None:
            k_cache, v_cache = self._kv_views(atten_info.kv_buffer[layer_index])
            return self._chunked(
                xq,
                k_cache,
                v_cache,
                self.scale,
                atten_info.b_start_loc,
                atten_info.b_kv_base,
                atten_info.b_prefix_len,
                atten_info.b_seq_len,
                atten_info.max_chunk_len,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
        return self._prefill(
            xq,
            xk,
            xv,
            self.scale,
            atten_info.b_start_loc,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )

    def token_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Decode: attend the single new query against the whole cached history.

        Returns:
            ``[batch, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        k_cache, v_cache = self._kv_views(atten_info.kv_buffer[layer_index])
        return self._decode(
            xq,
            k_cache,
            v_cache,
            self.scale,
            atten_info.b_req_tokens_table,
            atten_info.b_req_idx,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        if is_prefill:
            return self.context_forward(xq, xk, xv, atten_info, layer_index)
        return self.token_forward(xq, xk, xv, atten_info, layer_index)
