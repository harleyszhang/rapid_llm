"""HCA / CSA compressors and the Lightning Indexer for DeepSeek-V4.

Both compressors pool every closed window of ``rate`` tokens into one
compressed entry, norm it and rope it at the window's deterministic absolute
position. CSA adds the Ca/Cb overlap layout (window ``w`` pools window
``w-1``'s Ca slice with window ``w``'s Cb slice) and the Lightning Indexer,
which scores queries against its own scaled-down entry sequence and returns
the ``index_topk`` entries each query attends. The two entry sequences grow in
lockstep because they consume identical windows.

Per-row rolling state (buffer, running entries, entry count, overlap slices)
lives on ``self._rows``; see :class:`~rapid_llm.modules.deepseek_v4.cache.V4LayerCache`
for the row-stability contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...models.config import ModelConfig
from ..linear import ReplicatedLinear
from ..quantization import QuantizationConfig
from .norm import DeepseekV4RMSNorm
from .rope import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb


def _fresh_row() -> dict:
    """One sequence's compressor-side rolling state (both name slots)."""
    return {
        "buf_kv": None,
        "buf_gate": None,
        "compressed": None,
        "entry_count": 0,
        "overlap_kv": None,
        "overlap_gate": None,
    }


def _combine_buffer(
    row: dict, kv: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepend the rolling buffer, peel off the window-aligned prefix.

    Mirrors ``store_compression_weights``: only whole windows leave the
    buffer, the remainder waits for the next call so chunked prefill stays
    position-aligned.
    """
    buf_kv, buf_gate = row["buf_kv"], row["buf_gate"]
    if buf_kv is not None and buf_kv.shape[0]:
        kv = torch.cat([buf_kv, kv], dim=0)
        gate = torch.cat([buf_gate, gate], dim=0)
    usable = (kv.shape[0] // row["rate"]) * row["rate"]
    row["buf_kv"], row["buf_gate"] = kv[usable:], gate[usable:]
    return kv[:usable], gate[:usable], row["entry_count"] * row["rate"]


def _softmax_pool(
    chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, rate: int, bias: torch.Tensor
) -> torch.Tensor:
    """Softmax-gated window aggregation: ``sum_j softmax(g_j + B)_j * kv_j``."""
    n_windows = chunk_kv.shape[0] // rate
    ck = chunk_kv.view(n_windows, rate, -1)
    cg = chunk_gate.view(n_windows, rate, -1) + bias
    return (ck * cg.softmax(dim=1, dtype=torch.float32).to(ck.dtype)).sum(dim=1)


class DeepseekV4HCACompressor(nn.Module):
    """Heavily-Compressed Attention compressor: one entry per ``rate`` tokens.

    Every closed window emits ``C_i = sum_j softmax(Z_j + B)_j * C_j``, normed
    and rope'd at the window's deterministic absolute position. State (rolling
    buffer, running entries, entry count) lives per batch row on
    ``self._rows``; see :class:`V4LayerCache` for the row-stability contract.
    """

    rope_layer_type = "compress"

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.compress_rate = int(config.compress_rates["heavily_compressed_attention"])
        self.head_dim = config.head_dim
        # The V4 checkpoints leave the compressors' projections in bf16 even on
        # quantised models; the quant argument exists for signature parity with
        # the CSA compressor and is unused here.
        self.kv_proj = ReplicatedLinear(
            config.hidden_size, self.head_dim, params_dtype=config.dtype
        )
        self.gate_proj = ReplicatedLinear(
            config.hidden_size, self.head_dim, params_dtype=config.dtype
        )
        self.position_bias = nn.Parameter(
            torch.zeros(self.compress_rate, self.head_dim, dtype=config.dtype)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self._rows: list[dict] = []

    def reset(self) -> None:
        self._rows = []

    def _ensure_rows(self, n: int) -> None:
        while len(self._rows) < n:
            row = _fresh_row()
            row["rate"] = self.compress_rate
            self._rows.append(row)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Emit the running compressed entries with their causal visibility.

        Args:
            hidden_states: ``[B, S, hidden]`` this step's tokens.
            position_ids: ``[B, S]`` absolute positions.
            valid: ``[B, S]`` which positions carry real tokens.

        Returns:
            ``(compressed_kv, block_bias)`` — ``compressed_kv``
            ``[B, 1, T, head_dim]`` every closed window's entry, and
            ``block_bias`` ``[B, 1, S, T]`` that is ``0`` where the causal
            rule lets query ``t`` see entry ``w`` (``w < (t+1)//rate``) and
            ``-inf`` elsewhere. A decode step (``S == 1``) or an empty
            sequence returns ``None``: every entry is visible, the reference
            zero-pads the mask in that case.
        """
        B, S, hidden = hidden_states.shape
        self._ensure_rows(B)
        flat = hidden_states.view(B * S, hidden)
        kv = self.kv_proj(flat).view(B, S, self.head_dim)
        gate = self.gate_proj(flat).view(B, S, self.head_dim)

        row_entries = []
        for b in range(B):
            row = self._rows[b]
            kv_b, gate_b = kv[b][valid[b]], gate[b][valid[b]]
            chunk_kv, chunk_gate, first_pos = _combine_buffer(row, kv_b, gate_b)
            if chunk_kv.shape[0]:
                compressed = self.kv_norm(
                    _softmax_pool(chunk_kv, chunk_gate, self.compress_rate, self.position_bias)
                )
                n_windows = compressed.shape[0]
                positions = (
                    torch.arange(n_windows, device=compressed.device) * self.compress_rate
                    + first_pos
                )
                cos, sin = self.rotary_emb(compressed, positions.unsqueeze(0), self.rope_layer_type)
                compressed = apply_rotary_pos_emb(compressed[None, None], cos, sin)[0, 0]
                row["entry_count"] += n_windows
                row["compressed"] = (
                    compressed
                    if row["compressed"] is None
                    else torch.cat([row["compressed"], compressed])
                )
            row_entries.append(row["compressed"])

        compressed_kv, _ = self._pad_entries(row_entries)
        compressed_kv = compressed_kv.unsqueeze(1)  # [B, 1, T, head_dim]
        t_max = compressed_kv.shape[2]
        if S == 1 or t_max == 0:
            return compressed_kv, None
        # Query t sees entry w only while w < (t+1)//rate — an entry pools
        # tokens up to position (w+1)*rate-1, so it is "future" for earlier
        # queries.
        entry_indices = torch.arange(t_max, device=compressed_kv.device)
        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
        block_bias = compressed_kv.new_full((B, 1, S, t_max), float("-inf"))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) < causal_threshold.view(B, 1, S, 1), 0.0
        )
        return compressed_kv, block_bias

    def _pad_entries(
        self, row_entries: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(row_entries)
        t_max = max((e.shape[0] if e is not None else 0) for e in row_entries)
        entries = next(
            (e for e in row_entries if e is not None),
            torch.zeros(0, self.head_dim),
        ).new_zeros(B, t_max, self.head_dim)
        mask = torch.full((B, 1, 1, t_max), float("-inf"), device=entries.device)
        for b, e in enumerate(row_entries):
            if e is not None and e.shape[0]:
                entries[b, : e.shape[0]] = e
                mask[b, 0, 0, : e.shape[0]] = 0.0
        return entries, mask


class DeepseekV4Indexer(nn.Module):
    """Lightning Indexer: top-``k`` compressed entries per query.

    Runs its own scaled-down compressor over the same windows (Ca/Cb overlap
    layout at ``index_head_dim``), scores queries against it with
    ``sum_h w_h * ReLU(q_h . K_h)`` and returns per-query indices into the
    *indexer's* entry sequence. The outer CSA compressor gathers its
    ``head_dim``-wide entries at the same indices — the two entry sequences
    grow in lockstep because they consume identical windows.
    """

    rope_layer_type = "compress"

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.compress_rate = int(config.compress_rates["compressed_sparse_attention"])
        self.num_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.index_topk = int(config.index_topk)
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5
        # Only the indexer's query projection is fp8 on quantised V4
        # checkpoints; the compressor-side projections stay bf16.
        self.kv_proj = ReplicatedLinear(
            config.hidden_size, 2 * self.head_dim, params_dtype=config.dtype
        )
        self.gate_proj = ReplicatedLinear(
            config.hidden_size, 2 * self.head_dim, params_dtype=config.dtype
        )
        self.position_bias = nn.Parameter(
            torch.zeros(self.compress_rate, 2 * self.head_dim, dtype=config.dtype)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = ReplicatedLinear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            params_dtype=config.dtype,
            quant=quant,
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size, self.num_heads, params_dtype=config.dtype
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self._rows: list[dict] = []

    def reset(self) -> None:
        self._rows = []

    def _ensure_rows(self, n: int) -> None:
        while len(self._rows) < n:
            row = _fresh_row()
            row["rate"] = self.compress_rate
            self._rows.append(row)

    def _compress_row(self, kv_b: torch.Tensor, gate_b: torch.Tensor, row: dict) -> None:
        """Advance the indexer's own entry sequence by the closed windows."""
        chunk_kv, chunk_gate, first_pos = _combine_buffer(row, kv_b, gate_b)
        if not chunk_kv.shape[0]:
            return
        rate = self.compress_rate
        n_windows = chunk_kv.shape[0] // rate
        ck = chunk_kv.view(n_windows, rate, -1)
        cg = chunk_gate.view(n_windows, rate, -1) + self.position_bias
        new_kv = ck.new_zeros(n_windows, 2 * rate, self.head_dim)
        new_gate = cg.new_full((n_windows, 2 * rate, self.head_dim), float("-inf"))
        new_kv[:, rate:] = ck[:, :, self.head_dim :]
        new_gate[:, rate:] = cg[:, :, self.head_dim :]
        if n_windows > 1:
            new_kv[1:, :rate] = ck[:-1, :, : self.head_dim]
            new_gate[1:, :rate] = cg[:-1, :, : self.head_dim]
        prior_kv, prior_gate = row["overlap_kv"], row["overlap_gate"]
        if prior_kv is not None:
            new_kv[0, :rate] = prior_kv
            new_gate[0, :rate] = prior_gate
        row["overlap_kv"] = ck[-1, :, : self.head_dim].clone()
        row["overlap_gate"] = cg[-1, :, : self.head_dim].clone()
        compressed = self.kv_norm(
            (new_kv * new_gate.softmax(dim=1, dtype=torch.float32).to(new_kv.dtype)).sum(dim=1)
        )
        positions = torch.arange(n_windows, device=compressed.device) * rate + first_pos
        cos, sin = self.rotary_emb(compressed, positions.unsqueeze(0), self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed[None, None], cos, sin)[0, 0]
        row["entry_count"] += n_windows
        row["compressed"] = (
            compressed if row["compressed"] is None else torch.cat([row["compressed"], compressed])
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Per-query top-k indices into each row's indexer entry sequence.

        Returns:
            ``[B, S, k]`` indices with ``k = min(index_topk, T_max)``. Picks
            the causal rule forbids (``w >= (t+1)//rate``) and picks that fell
            on another row's padding carry the reference's ``-1`` sentinel —
            the compressor scatters those past the real entries and drops
            them from the mask.
        """
        B, S, hidden = hidden_states.shape
        self._ensure_rows(B)
        flat = hidden_states.view(B * S, hidden)
        kv = self.kv_proj(flat).view(B, S, 2 * self.head_dim)
        gate = self.gate_proj(flat).view(B, S, 2 * self.head_dim)

        for b in range(B):
            self._compress_row(kv[b][valid[b]], gate[b][valid[b]], self._rows[b])

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids, self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(B, S, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q.transpose(1, 2), cos_q, sin_q).transpose(1, 2)
        weights = self.weights_proj(flat).view(B, S, self.num_heads).float() * self.weights_scaling

        # Per-row scores padded to the longest entry sequence; padding slots
        # start at -inf so top-k never prefers them over a real pick.
        t_max = max(
            (row["compressed"].shape[0] if row["compressed"] is not None else 0)
            for row in self._rows
        )
        index_scores = torch.full(
            (B, S, t_max), float("-inf"), device=flat.device, dtype=torch.float32
        )
        for b in range(B):
            entries = self._rows[b]["compressed"]
            if entries is None or not entries.shape[0]:
                continue
            scores = torch.relu(
                torch.matmul(q[b].float(), entries.float().transpose(-1, -2)) * self.softmax_scale
            )
            index_scores[b, :, : entries.shape[0]] = (scores * weights[b].unsqueeze(-1)).sum(dim=-2)

        top_k = min(self.index_topk, t_max)
        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
        entry_indices = torch.arange(t_max, device=flat.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]
        # A -inf slot can still win a pick when fewer than k entries are
        # causally available — either a future entry or another row's padding
        # (a row with fewer live entries). Both are the -1 sentinel; the
        # effective ceiling per row is min(causal_threshold, that row's
        # entry count).
        row_len = torch.tensor(
            [
                row["compressed"].shape[0] if row["compressed"] is not None else 0
                for row in self._rows
            ],
            device=flat.device,
        )
        ceiling = torch.minimum(causal_threshold.unsqueeze(-1), row_len.view(B, 1, 1))
        invalid = top_k_indices >= ceiling
        return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)


class DeepseekV4CSACompressor(nn.Module):
    """Compressed-Sparse Attention compressor (Ca/Cb overlap) + indexer.

    ``kv_proj``/``gate_proj`` produce two independent compressed series per
    token; window ``w``'s entry pools window ``w-1``'s Ca slice with window
    ``w``'s Cb slice — effective width ``2 * rate``, stride ``rate``. The
    Lightning Indexer then picks the ``index_topk`` entries each query
    attends; those gathered entries (all of them valid) are what the
    attention concatenates onto the sliding window.
    """

    rope_layer_type = "compress"

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.compress_rate = int(config.compress_rates["compressed_sparse_attention"])
        self.head_dim = config.head_dim
        self.kv_proj = ReplicatedLinear(
            config.hidden_size, 2 * self.head_dim, params_dtype=config.dtype
        )
        self.gate_proj = ReplicatedLinear(
            config.hidden_size, 2 * self.head_dim, params_dtype=config.dtype
        )
        self.position_bias = nn.Parameter(
            torch.zeros(self.compress_rate, 2 * self.head_dim, dtype=config.dtype)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        # The indexer's q_b_proj is the one fp8 projection inside a quantised
        # CSA block; the compressor's own projections stay bf16.
        self.indexer = DeepseekV4Indexer(config, quant=quant)
        self._rows: list[dict] = []

    def reset(self) -> None:
        self._rows = []
        self.indexer.reset()

    def _ensure_rows(self, n: int) -> None:
        while len(self._rows) < n:
            row = _fresh_row()
            row["rate"] = self.compress_rate
            self._rows.append(row)

    def _compress_row(self, kv_b: torch.Tensor, gate_b: torch.Tensor, row: dict) -> None:
        chunk_kv, chunk_gate, first_pos = _combine_buffer(row, kv_b, gate_b)
        if not chunk_kv.shape[0]:
            return
        rate = self.compress_rate
        n_windows = chunk_kv.shape[0] // rate
        ck = chunk_kv.view(n_windows, rate, -1)
        cg = chunk_gate.view(n_windows, rate, -1) + self.position_bias
        new_kv = ck.new_zeros(n_windows, 2 * rate, self.head_dim)
        new_gate = cg.new_full((n_windows, 2 * rate, self.head_dim), float("-inf"))
        new_kv[:, rate:] = ck[:, :, self.head_dim :]
        new_gate[:, rate:] = cg[:, :, self.head_dim :]
        if n_windows > 1:
            new_kv[1:, :rate] = ck[:-1, :, : self.head_dim]
            new_gate[1:, :rate] = cg[:-1, :, : self.head_dim]
        prior_kv, prior_gate = row["overlap_kv"], row["overlap_gate"]
        if prior_kv is not None:
            new_kv[0, :rate] = prior_kv
            new_gate[0, :rate] = prior_gate
        row["overlap_kv"] = ck[-1, :, : self.head_dim].clone()
        row["overlap_gate"] = cg[-1, :, : self.head_dim].clone()
        compressed = self.kv_norm(
            (new_kv * new_gate.softmax(dim=1, dtype=torch.float32).to(new_kv.dtype)).sum(dim=1)
        )
        positions = torch.arange(n_windows, device=compressed.device) * rate + first_pos
        cos, sin = self.rotary_emb(compressed, positions.unsqueeze(0), self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed[None, None], cos, sin)[0, 0]
        row["entry_count"] += n_windows
        row["compressed"] = (
            compressed if row["compressed"] is None else torch.cat([row["compressed"], compressed])
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Emit every compressed entry plus the indexer's per-query block bias.

        Returns:
            ``(compressed_kv, block_bias)`` — ``compressed_kv``
            ``[B, 1, T, head_dim]`` every closed window's entry (rows with
            fewer entries padded with zeros), and ``block_bias``
            ``[B, 1, S, T]`` carrying ``0`` exactly on each query's valid
            indexer picks (a ``-1`` sentinel scatters to a dropped column
            past ``T``) and ``-inf`` everywhere else.
        """
        B, S, hidden = hidden_states.shape
        self._ensure_rows(B)
        flat = hidden_states.view(B * S, hidden)
        kv = self.kv_proj(flat).view(B, S, 2 * self.head_dim)
        gate = self.gate_proj(flat).view(B, S, 2 * self.head_dim)

        for b in range(B):
            self._compress_row(kv[b][valid[b]], gate[b][valid[b]], self._rows[b])

        # Pad the per-row running entries into one batched sequence; padding
        # entries can never be picked (the indexer's sentinel covers them).
        row_entries = [row["compressed"] for row in self._rows]
        t_max = max((e.shape[0] if e is not None else 0) for e in row_entries)
        compressed = flat.new_zeros(B, t_max, self.head_dim)
        for b, entries in enumerate(row_entries):
            if entries is not None and entries.shape[0]:
                compressed[b, : entries.shape[0]] = entries
        compressed_kv = compressed.unsqueeze(1)  # [B, 1, T, head_dim]

        top_k_indices = self.indexer(hidden_states, q_residual, position_ids, valid)  # [B, S, k]
        t = compressed_kv.shape[2]
        picks = top_k_indices >= 0
        safe_indices = torch.where(picks, top_k_indices, torch.full_like(top_k_indices, t))
        block_bias = compressed.new_full((B, 1, S, t + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :t]


__all__ = [
    "DeepseekV4CSACompressor",
    "DeepseekV4HCACompressor",
    "DeepseekV4Indexer",
]
