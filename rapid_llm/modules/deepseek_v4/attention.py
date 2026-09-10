"""DeepSeek-V4 attention: shared-KV MQA with sliding / compressed context.

Eager torch attention everywhere. ``head_dim=512`` exceeds the FlashAttention
2/3/4 ceiling (256), torch SDPA carries no per-head sink term and FlexAttention
cannot resize a BlockMask around the compressor's variable-length output —
transformers disables every backend for the same reasons and so does rapid_llm.

Tensor parallelism applies where the head axis allows it: ``q_b_proj`` is
column-parallel (head split), ``o_b_proj`` row-parallel; ``o_a_proj`` groups
stay whole on a rank, which requires ``o_groups % world == 0`` (each rank then
owns ``o_groups // world`` complete groups whose input is exactly its local
heads). The shared single KV head, the compressors and the indexer replicate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...distributed.parallel_state import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from ...models.config import ModelConfig
from ..linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from ..quantization import QuantizationConfig
from .cache import V4LayerCache
from .compressor import DeepseekV4CSACompressor, DeepseekV4HCACompressor
from .grouped_linear import DeepseekV4GroupedLinear
from .norm import DeepseekV4RMSNorm, DeepseekV4UnweightedRMSNorm
from .rope import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb

_COMPRESSOR_CLASSES = {
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


def _sink_loader(param, loaded, shard_id) -> torch.Tensor:
    """Narrow the per-head sinks to this rank's heads."""
    world = get_tensor_model_parallel_world_size()
    if world == 1:
        param.data.copy_(loaded)
        return param.data
    size = loaded.shape[0] // world
    param.data.copy_(loaded.narrow(0, get_tensor_model_parallel_rank() * size, size))
    return param.data


class DeepseekV4Attention(nn.Module):
    """Shared-KV MQA with per-head sinks, partial interleaved RoPE applied
    (and inverted) around an eager core, and the grouped output projection.

    Args:
        config: Model config with the V4 field group populated.
        layer_index: Position in the stack; selects the layer type (and
            therefore the compressor variant, or none for SWA layers).
        quant: Quantisation layout of an fp8 checkpoint — every attention
            projection (and the indexer's query projection) is blockwise fp8;
            the compressors' own projections stay bf16 whatever the checkpoint.
    """

    def __init__(
        self, config: ModelConfig, layer_index: int, *, quant: QuantizationConfig | None = None
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        self.head_dim = config.head_dim
        self.sliding_window = int(config.sliding_window)
        self.scale = self.head_dim**-0.5
        dtype = config.dtype

        self.num_heads = divide(
            config.num_heads, get_tensor_model_parallel_world_size(), "attention heads"
        )
        self.o_groups = int(config.o_groups)
        self.o_lora_rank = int(config.o_lora_rank)
        world = get_tensor_model_parallel_world_size()
        if self.o_groups % world != 0:
            raise ValueError(
                f"o_groups ({self.o_groups}) must be divisible by the tensor-parallel "
                f"world size ({world}) so each rank owns whole groups"
            )
        self.local_groups = self.o_groups // world

        self.q_a_proj = ReplicatedLinear(
            config.hidden_size, config.q_lora_rank, params_dtype=dtype, quant=quant
        )
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank, config.num_heads * self.head_dim, params_dtype=dtype, quant=quant
        )
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = ReplicatedLinear(
            config.hidden_size, self.head_dim, params_dtype=dtype, quant=quant
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            config.num_heads * self.head_dim // self.o_groups,
            self.local_groups * self.o_lora_rank,
            self.local_groups,
            dtype,
            quant=quant,
        )
        self.o_b_proj = RowParallelLinear(
            self.o_groups * self.o_lora_rank, config.hidden_size, params_dtype=dtype, quant=quant
        )
        self.sinks = nn.Parameter(torch.empty(self.num_heads, dtype=dtype))
        self.sinks.weight_loader = _sink_loader

        self.compressor = (
            _COMPRESSOR_CLASSES[self.layer_type](config, quant=quant)
            if self.layer_type != "sliding_attention"
            else None
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        # The rope table follows the reference's rope_layer_type rule: SWA
        # layers stay on the plain ``main`` table (theta 10 000); CSA/HCA
        # layers rotate their shared KV — and invert it on the output — with
        # the yarn-scaled ``compress`` table their compressors share.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self._cache = V4LayerCache(self.sliding_window)

    def reset(self) -> None:
        """Clear the sliding window and compressor state for new sequences."""
        self._cache.reset()
        if self.compressor is not None:
            self.compressor.reset()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """One attention step over the sliding window plus compressed entries.

        Args:
            hidden_states: ``[B, S, hidden]`` this step's tokens.
            position_ids: ``[B, S]`` absolute positions.
            position_embeddings: ``(cos, sin)`` of the *main* rope table — read
                by SWA layers only; CSA/HCA layers slice their own *compress*
                table from the module's cached buffers instead.
            valid: ``[B, S]`` which positions carry real tokens (padding on a
                mixed-length prefill never enters the window or compressors).
        """
        B, S, hidden = hidden_states.shape
        flat = hidden_states.view(B * S, hidden)
        if self.rope_layer_type == "main":
            cos, sin = position_embeddings
        else:
            # CSA/HCA need the compress table, which the caller does not
            # carry (it threads the main pair); every attention already owns
            # a table module, so the lookup is a cached-table slice.
            cos, sin = self.rotary_emb(hidden_states, position_ids, self.rope_layer_type)

        q_residual = self.q_a_norm(self.q_a_proj(flat))
        q = self.q_b_proj(q_residual).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(flat)).view(B, S, 1, self.head_dim).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        # Sliding K==V update: the attention reads the full concatenation,
        # the cache retains the trailing (window - 1) entries.
        pos = torch.where(valid, position_ids, torch.full_like(position_ids, -1))
        full_kv, full_pos = self._cache.update_sliding(kv.squeeze(1), pos)
        kv_all = full_kv.unsqueeze(1)  # [B, 1, T_sliding, head_dim]

        # Compressed entries (CSA indexer-picked / HCA running) append to the
        # KV axis; block_bias (per-query causality + indexer validity) extends
        # the mask. A decode step's None bias means every entry is visible.
        if self.compressor is not None:
            if self.layer_type == "compressed_sparse_attention":
                extra_kv, extra_mask = self.compressor(
                    hidden_states, q_residual, position_ids, valid
                )
            else:
                extra_kv, extra_mask = self.compressor(hidden_states, position_ids, valid)
            if extra_kv.shape[2]:
                kv_all = torch.cat([kv_all, extra_kv], dim=2)
        else:
            extra_kv, extra_mask = None, None

        # Mask over the sliding part: causal inside the window, padding slots
        # and padded queries stay masked out. Compressed entries carry their
        # own visibility (all valid entries visible — the reference semantics).
        q_pos = pos
        k_pos = full_pos
        sliding_visible = (
            (k_pos[:, None, :] >= 0)
            & (k_pos[:, None, :] <= q_pos[:, :, None])
            & (q_pos[:, :, None] - k_pos[:, None, :] < self.sliding_window)
        )
        mask = torch.zeros(B, 1, S, full_pos.shape[1], device=flat.device, dtype=q.dtype)
        mask = torch.where(sliding_visible[:, None], mask, torch.finfo(q.dtype).min)
        if extra_kv is not None and extra_kv.shape[2]:
            # CSA's block_bias is per-query [B, 1, S, T]; HCA's carries the
            # causal rule in the same shape. None (decode / empty) exposes
            # every entry, the reference's zero-pad.
            if extra_mask is not None:
                mask = torch.cat([mask, extra_mask.to(q.dtype)], dim=-1)
            else:
                mask = torch.cat([mask, mask.new_zeros(B, 1, S, extra_kv.shape[2])], dim=-1)

        # Eager attention with the per-head sink column, max-subtracted.
        attn_weights = torch.matmul(q, kv_all.transpose(-1, -2)) * self.scale
        attn_weights = attn_weights + mask
        sinks = self.sinks.view(1, -1, 1, 1).expand(B, -1, S, -1)
        combined = torch.cat([attn_weights, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        scores = probs[..., :-1].to(kv_all.dtype)

        out = torch.matmul(scores, kv_all)  # [B, H, S, head_dim], K == V
        # Conjugate rotation (-sin) at the query position undoes the RoPE the
        # shared K==V picked up, so each entry's contribution is relative.
        out = apply_rotary_pos_emb(out, cos, -sin)
        # Head-major back to token-major for the grouped projection's layout.
        stacked = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        grouped = stacked.view(B, S, self.local_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped)


__all__ = ["DeepseekV4Attention"]
