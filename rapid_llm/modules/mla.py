"""DeepSeek MLA attention: the latent-cache block shared by V2/V3 models.

:class:`DeepseekV2MLAAttention` replaces the standard ``Attention`` block on DeepSeek
families (mirroring vLLM's ``DeepseekV2MLAAttention``): q (or the ``q_a``/``q_b`` LoRA pair),
the fused ``kv_a_proj_with_mqa``, the kv_a layernorm, ``kv_b_proj`` and a row-parallel
``o_proj``, composed around the native MLA kernels (prefill up-projects the fresh latent
chunk by chunk, decode keeps q absorbed and attends the latent cache directly).

Usage:
    attn = DeepseekV2MLAAttention(config, quant=None)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..distributed.parallel_state import divide, get_tensor_model_parallel_world_size
from ..kernels import dispatch, rope_emb_forward, skip_rmsnorm
from ..kernels.dispatcher import MLA_LATENT_TAGS
from ..models.config import ModelConfig
from .linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from .quantization import QuantizationConfig


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """DeepSeek YaRN magnitude factor ``0.1 * mscale * ln(scale) + 1``; 1.0 unscaled."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _pair_to_neox(x: torch.Tensor) -> torch.Tensor:
    """Re-pair a rope slice from DeepSeek's adjacent pairs into the kernel's neox pairs.

    DeepSeek rotates ``(x[2k], x[2k+1])``, the shared Triton kernel rotates ``(x[k], x[k +
    D/2])`` with the same frequency per index. Gathering is exact and far cheaper than a
    second rope kernel for one 64-wide slice.
    """
    half = x.shape[-1] // 2
    return x.view(*x.shape[:-1], half, 2).transpose(-1, -2).reshape(*x.shape)


def _pair_from_neox(y: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`_pair_to_neox`."""
    half = y.shape[-1] // 2
    return y.view(*y.shape[:-1], 2, half).transpose(-1, -2).reshape(*y.shape)


class DeepseekV2MLAAttention(nn.Module):
    """Multi-head latent attention — the whole replacement for ``Attention``.

    Reference: the DeepSeek-V2 paper (https://arxiv.org/abs/2405.04434); the absorbed decode
    path plays the role of vLLM's ``MLACommonImpl``. Projections follow HF
    ``DeepseekV2Attention``: ``q_proj`` (or the ``q_a``/``q_b`` pair when ``q_lora_rank`` is
    set), the fused ``kv_a_proj_with_mqa``, ``kv_a_layernorm`` over the c_kv half, ``kv_b_proj``
    producing per-head ``[k_nope | v]``, and a row-parallel ``o_proj``. RoPE touches only the
    rope-wide pe slices, so the rotary table is built at that width. TP splits along the heads
    (q and kv_b column-parallel, o_proj row-parallel); kv_a is replicated because its latent has
    no head axis to shard, so every rank caches it in full.
    """

    def __init__(self, config: ModelConfig, *, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        if quant is not None:
            raise ValueError(
                "DeepseekV2MLAAttention needs the plain kv_b weight for its absorbed "
                "views; quantised DeepSeek checkpoints are out of scope here"
            )
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.qk_nope_head_dim != self.v_head_dim:
            raise ValueError(
                f"kv_b_proj halves must be equal-width for the per-head views, "
                f"got qk_nope_head_dim={self.qk_nope_head_dim} "
                f"and v_head_dim={self.v_head_dim}"
            )
        # Head count is divided here, not left to ColumnParallelLinear: a world size that
        # does not divide the heads then fails on the count it actually breaks, and the
        # equal output split provably lands on head boundaries.
        self.num_heads = divide(
            config.num_heads, get_tensor_model_parallel_world_size(), "attention heads"
        )
        self.scale = self.qk_head_dim**-0.5

        dtype = config.dtype
        bias = bool(getattr(config, "attention_bias", False))
        q_lora_rank = config.q_lora_rank
        if q_lora_rank is None:
            self.q_proj: nn.Module | None = ColumnParallelLinear(
                self.hidden_size, config.num_heads * self.qk_head_dim, params_dtype=dtype
            )
        else:
            self.q_proj = None
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size, q_lora_rank, bias=bias, params_dtype=dtype
            )
            self.q_a_layernorm_weight = nn.Parameter(torch.ones(q_lora_rank, dtype=dtype))
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank, config.num_heads * self.qk_head_dim, params_dtype=dtype
            )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
            params_dtype=dtype,
        )
        self.kv_a_layernorm_weight = nn.Parameter(torch.ones(self.kv_lora_rank, dtype=dtype))
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            params_dtype=dtype,
        )
        self.o_proj = RowParallelLinear(
            config.num_heads * self.v_head_dim, self.hidden_size, bias=bias, params_dtype=dtype
        )

        # YaRN: the softmax scale rides the mscale-squared factor; the rope generator only
        # applies the ratio to cos/sin.
        rope_parameters = config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            factor = rope_parameters.get("factor")
            if mscale_all_dim and factor:
                mscale = yarn_get_mscale(factor, float(mscale_all_dim))
                self.scale = self.scale * mscale * mscale

        # Bind after weights and inputs have been placed on their execution device.
        self._bound_device = None
        self._params_dtype = dtype

    def _bind_kernels(self, device_type: str) -> None:
        if self._bound_device == device_type:
            return
        if device_type == "cpu":
            from ..kernels.backend import cpu

            self._prefill = cpu.mla_prefill
            # No partial: the contract fix pinned ``qk_rope_head_dim`` to the
            # kernel's internal 64-token constant, so there is nothing to pass.
            self._decode = cpu.mla_decode
        else:
            self._prefill = dispatch(
                "attention.mla_prefill",
                dtype=self._params_dtype,
                layout=MLA_LATENT_TAGS,
                backend="native",
            ).load()
            self._decode = dispatch(
                "attention.mla_decode",
                dtype=self._params_dtype,
                layout=MLA_LATENT_TAGS,
                backend="native",
            ).load()
        self._bound_device = device_type

    @property
    def w_uk(self) -> torch.Tensor:
        """``[heads, kv_lora_rank, qk_nope]`` K up-projection — a view of kv_b."""
        weight = self.kv_b_proj.weight
        return weight.view(self.num_heads, 2, -1, self.kv_lora_rank)[:, 0].transpose(-1, -2)

    @property
    def w_uv(self) -> torch.Tensor:
        """``[heads, kv_lora_rank, v]`` V up-projection — same source."""
        weight = self.kv_b_proj.weight
        return weight.view(self.num_heads, 2, -1, self.kv_lora_rank)[:, 1].transpose(-1, -2)

    def _project_q(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten, project to per-head layout, split into ``(q_nope, q_pe)``."""
        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            q, _ = skip_rmsnorm(
                self.q_a_proj(x), None, self.q_a_layernorm_weight, self.rms_norm_eps
            )
            q = self.q_b_proj(q)
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        return torch.split(q, (self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self._bind_kernels(x.device.type)
        batch, seq_len, _ = x.shape
        tokens = batch * seq_len
        flat = x.view(tokens, self.hidden_size)

        q_nope, q_pe = self._project_q(flat)

        kv_a = self.kv_a_proj_with_mqa(flat)
        c_kv, k_pe = torch.split(kv_a, (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        c_kv, _ = skip_rmsnorm(c_kv, None, self.kv_a_layernorm_weight, self.rms_norm_eps)

        # RoPE on the pe slices only: q_pe is already [tokens, heads, rope], k_pe rides
        # along as a single-head tensor. The re-pairing around the kernel is the contract
        # documented on _pair_to_neox.
        cos, sin = position_embeddings
        q_pe, k_pe = rope_emb_forward(
            _pair_to_neox(q_pe), _pair_to_neox(k_pe).unsqueeze(1), cos, sin
        )
        q_pe, k_pe = _pair_from_neox(q_pe), _pair_from_neox(k_pe.squeeze(1))

        # The cache row is the compressed latent exactly as computed here (normed c_kv
        # followed by the rotated k_pe, the same pair HF caches).
        latent = torch.cat((c_kv, k_pe), dim=-1)
        atten_info.kv_buffer[layer_index][atten_info.cur_select_index] = latent.unsqueeze(1)

        if atten_info.is_prefill:
            out = self._prefill(
                q_nope,
                q_pe,
                c_kv,
                k_pe,
                self.w_uk,
                self.w_uv,
                self.scale,
                atten_info.b_start_loc,
                atten_info.b_seq_len,
                atten_info.max_actual_seq_len,
            )
        else:
            # Absorb w_uk into q, attend the latent, up-project with w_uv: head-wide GEMMs
            # against the 576-dim latent row instead of materialising per-head K/V (the point
            # of the latent cache).
            q_absorbed = torch.einsum("bhd,hld->bhl", q_nope, self.w_uk)
            q_latent = torch.cat((q_absorbed, q_pe), dim=-1)
            block_table = atten_info.b_req_tokens_table[atten_info.b_req_idx]
            attended = self._decode(
                q_latent,
                atten_info.kv_buffer[layer_index],
                block_table,
                atten_info.b_seq_len,
                max_seq_len=atten_info.max_actual_seq_len,
                sm_scale=self.scale,
            )
            out = torch.einsum("bhl,hld->bhd", attended, self.w_uv)

        # reshape, not view: the decode einsum output can be non-contiguous, and the copy is
        # free next to the o_proj GEMM.
        out = out.reshape(batch, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out)
