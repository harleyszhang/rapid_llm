"""Interleaved partial RoPE for DeepSeek-V4.

V4 interleaves rope pairs instead of halving the head dimension, and keeps one
frequency table per rope type: ``main`` for core attention, ``compress`` for
the compressor / indexer positions. The tables are half-sized (one entry per
pair) and :func:`apply_rotary_pos_emb` doubles them next to the rotation math.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...models.config import ModelConfig


class DeepseekV4RotaryEmbedding(nn.Module):
    """Per-rope-type frequency tables; V4 interleaves pairs instead of halving.

    ``config.rope_parameters`` nests one sub-dict per rope type — ``main``
    (theta 10 000) for core attention, ``compress`` (theta 160 000) for the
    compressor/indexer positions. ``forward`` returns *half-size* cos/sin
    (one entry per interleaved pair); :func:`apply_rotary_pos_emb` repeats
    them to the full rope width next to the rotation math.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        tc = config.text_config
        self.rope_dim = int(config.head_dim * float(getattr(tc, "partial_rotary_factor", 1.0)))
        nested = getattr(tc, "rope_parameters", None) or {}
        defaults = {
            "main": float(getattr(tc, "rope_theta", 10000.0)),
            "compress": float(getattr(tc, "compress_rope_theta", 160000.0)),
        }
        for name in ("main", "compress"):
            sub = nested.get(name, {}) if isinstance(nested, dict) else {}
            theta = float(sub.get("rope_theta", defaults[name]))
            # ``dim = head_dim * sub.factor`` exactly as the reference computes
            # it: the config's ``__post_init__`` copies the global
            # ``partial_rotary_factor`` into both sub-dicts, so multiplying the
            # already-shrunk ``rope_dim`` again would quarter the table.
            factor = float(
                sub.get("partial_rotary_factor", getattr(tc, "partial_rotary_factor", 1.0) or 1.0)
            )
            dim = int(config.head_dim * factor)
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim)
            )
            self.register_buffer(f"{name}_inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        inv_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].to(torch.float32)
        freqs = (inv_expanded @ pos_expanded).transpose(1, 2)
        return freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate V4's interleaved pairs: ``(x[2k], x[2k+1]) -> (-x[2k+1], x[2k])``."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Interleaved RoPE over the *trailing* rope slice, in fp32.

    ``cos``/``sin`` arrive half-sized from :class:`DeepseekV4RotaryEmbedding`;
    ``repeat_interleave(2)`` doubles them to the rope width here, where the
    pairing with :func:`rotate_half` is local.
    """
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos) + (rotate_half(rope).float() * sin)
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)


__all__ = [
    "DeepseekV4RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]
