"""DeepSeek-V4 model definition: the mHC stream stack over mixed attention.

:class:`DeepseekV4DecoderLayer` is deliberately *not* the shared
:class:`~rapid_llm.models.base.DecoderLayer`: the mHC residual turns each
block into a stream mixer ``streams = post ⊙ sublayer(collapsed) +
comb @ streams`` over ``[B, S, hc_mult, hidden]``, which the standard
attention/MLP two-stage split cannot express (and TBO with it; see
:meth:`DeepseekV4Model.forward_tbo`). :class:`DeepseekV4Model` overrides the
full forward template while keeping the shared skeleton's weight-loading,
tied-embedding and vocabulary-parallel plumbing. The MoE side
(:class:`DeepseekV4MoE`) subclasses :class:`~rapid_llm.modules.SparseMoeBlock`
to swap in V4's ``sqrtsoftplus`` router (``noaux_tc``'s additive
``e_score_correction_bias`` selection, softmax-free), the hash router the
leading ``hash_moe`` layers use, and the bounded SwiGLU (``swiglu_limit``).

Usage:
    model = DeepseekV4Model(config)   # model_type deepseek_v4
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batch_overlap import current_deferred_ar
from ..distributed.parallel_state import (
    expert_parallel_enabled,
    get_ep_rank,
    get_ep_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from ..kernels import skip_rmsnorm
from ..modules import SparseMoeBlock
from ..modules.deepseek_v4.attention import DeepseekV4Attention
from ..modules.deepseek_v4.hyper_connection import DeepseekV4HyperConnection, DeepseekV4HyperHead
from ..modules.deepseek_v4.rope import DeepseekV4RotaryEmbedding
from ..modules.quantization import QuantizationConfig, RawParameter, UnquantizedFusedMoEMethod
from ..modules.quantization.mxfp4 import e8m0_to_fp32, repack_mxfp4_pairs
from .base import CausalLM
from .config import ModelConfig


def _sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    """V4's router score: ``sqrt(softplus(x))`` — concave, always positive."""
    return torch.sqrt(F.softplus(x))


class DeepseekV4FusedMoEMethod(UnquantizedFusedMoEMethod):
    """The fused grouped-GEMM path with V4's bounded SwiGLU epilogue.

    Only the routed experts clamp (``gate ≤ limit``, ``|up| ≤ limit``); the
    shared expert runs the plain SwiGLU every other family uses, exactly as
    the reference's two MLP classes differ.
    """

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ..kernels import fused_moe

        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            swiglu_limit=float(getattr(block, "swiglu_limit", float("inf"))),
        )


def _load_ep_expert_stack(param, loaded) -> torch.Tensor | None:
    """Load this EP rank's contiguous slice from a globally stacked tensor."""
    if not expert_parallel_enabled() or get_ep_world_size() <= 1:
        return None
    local = param.shape[0]
    loaded = loaded.narrow(0, get_ep_rank() * local, local)
    if param.shape != loaded.shape:
        raise ValueError(
            f"expert stack of shape {tuple(loaded.shape)} does not fit "
            f"parameter of shape {tuple(param.shape)}"
        )
    param.data.copy_(loaded)
    return param.data


def _stacked_gate_up_loader(param, loaded, shard_id) -> torch.Tensor:
    """Fill the fused gate/up expert stack from V4's 3D checkpoint tensor.

    V4 ships ``mlp.experts.gate_up_proj`` as one ``[E, 2I, D]`` parameter
    (gate rows first, up rows second — the layout ``chunk(2)`` reads back).
    Under tensor parallelism each rank owns the same slice of the
    intermediate dimension inside *both* halves, so its local view stays a
    contiguous ``[E, 2I_local, D]`` gate-then-up stack.
    """
    ep_view = _load_ep_expert_stack(param, loaded)
    if ep_view is not None:
        return ep_view
    world = get_tensor_model_parallel_world_size()
    if world == 1:
        if param.shape != loaded.shape:
            raise ValueError(
                f"expert stack of shape {tuple(loaded.shape)} does not fit "
                f"parameter of shape {tuple(param.shape)}"
            )
        param.data.copy_(loaded)
        return param.data
    inter = loaded.shape[1] // 2
    local = param.shape[1] // 2
    rank = get_tensor_model_parallel_rank()
    gate = loaded.narrow(1, rank * local, local)
    up = loaded.narrow(1, inter + rank * local, local)
    param.data.copy_(torch.cat([gate, up], dim=1))
    return param.data


def _stacked_down_loader(param, loaded, shard_id) -> torch.Tensor:
    """Fill the down-projection expert stack; TP narrows the contracted dim."""
    ep_view = _load_ep_expert_stack(param, loaded)
    if ep_view is not None:
        return ep_view
    world = get_tensor_model_parallel_world_size()
    if world == 1:
        if param.shape != loaded.shape:
            raise ValueError(
                f"expert stack of shape {tuple(loaded.shape)} does not fit "
                f"parameter of shape {tuple(param.shape)}"
            )
        param.data.copy_(loaded)
        return param.data
    local = param.shape[2]
    param.data.copy_(loaded.narrow(2, get_tensor_model_parallel_rank() * local, local))
    return param.data


# --------------------------------------------------------------------------- #
# DSpark (the vendor's native) checkpoint layout -> the HF-style names the
# loader's translator already speaks
# --------------------------------------------------------------------------- #

#: ``layers.N.ffn.experts.E.w{1,3,2}.{weight,scale}`` — w1 is the gate half,
#: w3 the up half (the reference's chunk(2) read-back order), w2 the down
#: projection. Output keys use the ``{gate,up,down}_proj`` names
#: :func:`rapid_llm.models.weights.translate_text_key` stacks per expert.
_DSPARK_EXPERT_KEY = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|scale)$")
_DSPARK_EXPERT_PROJ = {"1": "gate_proj", "3": "up_proj", "2": "down_proj"}

#: Layer-local paths, keyed after stripping ``layers.N.``. The fp8 projections
#: carry a ``.scale`` twin that renames to ``.weight_scale_inv`` — the suffix
#: the blockwise loader pairs with its ``.weight``.
_DSPARK_LAYER_KEYS: dict[str, str] = {
    "attn_norm.weight": "input_layernorm_weight",
    "ffn_norm.weight": "post_attention_layernorm_weight",
    "attn.attn_sink": "self_attn.sinks",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.wq_a.scale": "self_attn.q_a_proj.weight_scale_inv",
    "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wq_b.scale": "self_attn.q_b_proj.weight_scale_inv",
    "attn.wkv.weight": "self_attn.kv_proj.weight",
    "attn.wkv.scale": "self_attn.kv_proj.weight_scale_inv",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight",
    "attn.wo_a.scale": "self_attn.o_a_proj.weight_scale_inv",
    "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.wo_b.scale": "self_attn.o_b_proj.weight_scale_inv",
    # Compressor-side projections stay bf16 in the checkpoints; only the
    # layout names differ.
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    # Lightning Indexer: its query projection is the one fp8 tensor inside.
    "attn.indexer.wq_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.wq_b.scale": "self_attn.compressor.indexer.q_b_proj.weight_scale_inv",
    "attn.indexer.weights_proj.weight": "self_attn.compressor.indexer.weights_proj.weight",
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    # Router: ``gate`` is a module in the reference, bare parameters here.
    "ffn.gate.weight": "mlp.gate_weight",
    "ffn.gate.bias": "mlp.gate_e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate_tid2eid",
    # Shared expert: w1/w3 feed the fused gate/up through the packed-module
    # rule, w2 renames to down_proj.
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w1.scale": "mlp.shared_experts.gate_proj.weight_scale_inv",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
    "ffn.shared_experts.w3.scale": "mlp.shared_experts.up_proj.weight_scale_inv",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w2.scale": "mlp.shared_experts.down_proj.weight_scale_inv",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
}

#: Keys outside the decoder stack.
_DSPARK_TOP_KEYS: dict[str, str] = {
    "embed.weight": "embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "norm_weight",
    "hc_head_fn": "hc_head.hc_fn",
    "hc_head_base": "hc_head.hc_base",
    "hc_head_scale": "hc_head.hc_scale",
}

_DSPARK_LAYER_PREFIX = re.compile(r"^layers\.(\d+)\.(.+)$")


def adapt_dspark_key(key: str) -> str:
    """Rewrite one DSpark checkpoint key into the loader's HF-style name.

    The pure-rename half of :func:`_adapt_dspark_checkpoint`, shared with
    :meth:`DeepseekV4Model.translate_weight_key` so key-level checks (the
    checkpoint-index tests) see the same mapping a load performs.
    """
    expert = _DSPARK_EXPERT_KEY.match(key)
    if expert is not None:
        layer, index, proj, leaf = expert.groups()
        suffix = "weight_scale_inv" if leaf == "scale" else "weight"
        return f"layers.{layer}.mlp.experts.{index}.{_DSPARK_EXPERT_PROJ[proj]}.{suffix}"
    mapped = _DSPARK_TOP_KEYS.get(key)
    if mapped is not None:
        return mapped
    layer_key = _DSPARK_LAYER_PREFIX.match(key)
    if layer_key is not None and layer_key.group(2) in _DSPARK_LAYER_KEYS:
        return f"layers.{layer_key.group(1)}.{_DSPARK_LAYER_KEYS[layer_key.group(2)]}"
    return key


def _adapt_dspark_checkpoint(weights):
    """Rewrite a DSpark checkpoint stream into the loader's HF-style names.

    The vendor layout differs in three ways the translator cannot see: the
    module paths (``attn.wq_a`` vs ``self_attn.q_a_proj``), the scale leaf
    (``.scale`` vs ``.weight_scale_inv``) and the byte formats — fp8 weights
    are native ``float8_e4m3fn``, their scales e8m0, routed experts I8-packed
    nibble pairs. The first two are pure renames; the byte formats convert
    here (``view`` for fp8, :func:`e8m0_to_fp32`, :func:`repack_mxfp4_pairs`)
    so every downstream loader sees the storage dtype it allocated.
    """
    for key, tensor in weights:
        mapped = adapt_dspark_key(key)
        if tensor.dtype == torch.float8_e4m3fn:
            # Ampere cannot compute on fp8; the w8a16 kernel widens the raw
            # bytes itself, and the parameters hold uint8.
            tensor = tensor.view(torch.uint8)
        elif tensor.dtype == torch.float8_e8m0fnu:
            tensor = e8m0_to_fp32(tensor)
        elif tensor.dtype == torch.int8 and ".ffn.experts." in key and key.endswith(".weight"):
            tensor = repack_mxfp4_pairs(tensor)
        yield mapped, tensor


class DeepseekV4MoE(SparseMoeBlock):
    """V4's routed + shared FFN: sqrtsoftplus (or hash) routing, bounded SwiGLU.

    Routing differences from the families :class:`SparseMoeBlock` serves out
    of the box: scores are ``sqrtsoftplus(logits)``; the additive
    ``e_score_correction_bias`` shifts *selection* only while the gathered
    original scores are renormalised with a ``+1e-20`` floor; ``hash_moe``
    layers replace selection entirely with the frozen ``tid2eid`` lookup.
    The shared expert is one plain SwiGLU MLP at ``moe_intermediate_size``
    wide (V4 defines exactly one — not the ``n_shared_experts``-scaled
    width DeepSeek-V2 uses) and, unlike the routed half, carries no
    ``routed_scaling_factor``.
    """

    def __init__(
        self, config: ModelConfig, layer_index: int, quant: QuantizationConfig | None = None
    ) -> None:
        if str(getattr(config, "scoring_func", "")) != "sqrtsoftplus":
            raise ValueError(
                f"DeepSeek-V4 routes through sqrtsoftplus; config says "
                f"{getattr(config, 'scoring_func', None)!r}"
            )
        if int(getattr(config, "n_shared_experts", 0) or 0) != 1:
            raise ValueError(
                "DeepSeek-V4 defines exactly one shared expert at moe width; "
                f"n_shared_experts={getattr(config, 'n_shared_experts', None)} is unsupported"
            )
        # With ``quant`` the block hands the routed half to the config's
        # method (MXFP4 on fp4-expert checkpoints): the parent builds the
        # stacked int32/scale parameters and keeps the per-expert loader a
        # DSpark checkpoint's ``experts.E.w{1,2,3}`` tensors need.
        super().__init__(config, quant)
        self.is_hash = str(config.mlp_layer_types[layer_index]) == "hash_moe"
        self.swiglu_limit = float(getattr(config, "swiglu_limit", float("inf")))
        if quant is None:
            # The bounded SwiGLU rides inside the quant-method apply, where the
            # activation kernel is launched.
            self.quant_method = DeepseekV4FusedMoEMethod()
        # V4's router is a module in the reference (``gate.weight``,
        # ``gate.e_score_correction_bias``, ``gate.tid2eid``); rapid_llm keeps
        # the same checkpoint keys through the bare-parameter ``mlp.gate*``
        # suffix rules. Both stay in their storage dtype: the bias is an
        # absolute additive term the fp32 router GEMM reads, and the table is
        # integer indices no cast could round-trip.
        if self.is_hash:
            # The hash router replaces selection outright, so drop the bias the
            # parent's ``noaux_tc`` branch registered for this layer — a DSpark
            # hash layer ships no such tensor and the loader's coverage check
            # would reject the checkpoint for the gap.
            self.gate_e_score_correction_bias = None
            self.gate_tid2eid = RawParameter(
                torch.zeros(config.vocab_size, self.top_k, dtype=torch.long)
            )
        else:
            self.gate_e_score_correction_bias = RawParameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )
        if quant is None:
            # Unquantised V4 checkpoints ship experts pre-stacked in 3D —
            # replace the per-expert loader the skeleton bound with the
            # stacked-aware pair. A quantised checkpoint is per-expert and
            # keeps the parent's loader.
            self.experts["gate_up_proj"].weight_loader = _stacked_gate_up_loader
            self.experts["down_proj"].weight_loader = _stacked_down_loader

    def _route(
        self, x: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token expert ids and weights (reference-compatible ordering).

        Args:
            x: ``[tokens, hidden]``.
            input_ids: ``[B, S]`` token ids — required by ``hash_moe`` layers,
                ignored by the top-k ones.

        Returns:
            ``(weights, ids)``, each ``[tokens, top_k]``; weights in x.dtype.
        """
        # The reference routes in the activation dtype — no fp32 widening —
        # so scores, the gather and the normalisation all round at bf16. A
        # widened GEMM here drifts every expert weight by up to a bf16 ulp,
        # which shows up downstream as a systematic parity offset. Only the
        # selection stays exact: the fp32 correction bias promotes the topk
        # comparison, matching the reference's buffer semantics.
        router_logits = F.linear(x, self.gate_weight)
        scores = _sqrtsoftplus(router_logits)
        if self.is_hash:
            if input_ids is None:
                raise ValueError("a hash_moe layer needs input_ids to route")
            indices = self.gate_tid2eid[input_ids.reshape(-1)].long()
        else:
            indices = torch.topk(
                scores + self.gate_e_score_correction_bias, self.top_k, dim=-1, sorted=False
            ).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return weights.to(x.dtype), indices

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)

        weights, ids = self._route(x, input_ids)
        out = self.quant_method.apply(self, x, weights, ids)
        out = tensor_model_parallel_all_reduce(out)
        if self.shared_experts is not None:
            shared = self.shared_experts(x)
            # Same deferred-AR fence discipline as the skeleton: the shared
            # MLP's down_proj may still hold a pending reduction promise.
            ar = current_deferred_ar()
            if ar is not None:
                ar.fence_pending_reads()
            out = out + shared
        return out.reshape(*leading_shape, self.hidden_size)


class DeepseekV4DecoderLayer(nn.Module):
    """One V4 block: mHC mixing around mixed attention and the MoE FFN.

    The forward contract differs from the skeleton's
    ``DecoderLayer.forward(hidden, residual)``: it consumes and returns the
    full ``[B, S, hc_mult, hidden]`` stream stack (there is no separate
    residual), and it threads ``input_ids``/``valid`` through for the hash
    router and the padding-aware compressors.
    """

    def __init__(
        self, config: ModelConfig, layer_index: int, quant: QuantizationConfig | None = None
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.rms_norm_eps = config.rms_norm_eps
        self.self_attn = DeepseekV4Attention(config, layer_index, quant=quant)
        self.mlp = DeepseekV4MoE(config, layer_index, quant=quant)
        # Bare norm gains: the checkpoint writes ``input_layernorm.weight`` /
        # ``post_attention_layernorm.weight`` and the loader's suffix rules
        # fold them onto these names (the same convention the skeleton uses).
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normed, _ = skip_rmsnorm(x, None, weight, self.rms_norm_eps)
        return normed

    def forward(
        self,
        hidden_streams: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """One block over the stream stack.

        Args:
            hidden_streams: ``[B, S, hc_mult, hidden]``.
            position_ids: ``[B, S]`` absolute positions.
            position_embeddings: ``(cos, sin)`` of the *main* rope, half-size.
            input_ids: ``[B, S]`` token ids (hash routing reads them).
            valid: ``[B, S]`` real-token mask.
        """
        # post/comb arrive fp32 (the Sinkhorn projection runs in float); cast
        # to the stream dtype at the mixing sites so the block is
        # dtype-preserving end to end.
        # comb is consumed transposed — ``comb.T @ residual`` — because the
        # Sinkhorn projection's doubly-stochastic output is not symmetric;
        # the reference sums over the FIRST hc axis (``sum_j comb[j, k] *
        # residual[j, d]``), so the direction is part of the semantics.
        dtype = hidden_streams.dtype
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_output = self.self_attn(
            self._norm(collapsed, self.input_layernorm_weight),
            position_ids,
            position_embeddings,
            valid,
        )
        hidden_streams = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_streams
        )

        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_output = self.mlp(
            self._norm(collapsed, self.post_attention_layernorm_weight), input_ids=input_ids
        )
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_streams
        )


class DeepseekV4Model(CausalLM):
    """DeepSeek-V4: mHC streams, SWA/CSA/HCA layers, Hash/TopK MoE.

    Everything the skeleton owns — vocabulary-parallel embeddings and head,
    weight translation, tied embeddings — is inherited; the forward itself
    is overridden because the stream stack replaces the hidden/residual
    pair, and the rope is V4's interleaved two-theta table.
    """

    #: The shared expert fuses its gate/up pair; every other projection in a
    #: V4 block keeps its checkpoint name (the attention modules are real
    #: submodules, the router is bare parameters).
    packed_modules_mapping: ClassVar[dict[str, tuple[str, ...]]] = {
        "mlp.shared_experts.gate_up_proj": (
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
        ),
    }

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # The skeleton built the standard halved-pair rope from config; V4's
        # interleaved partial variant replaces it wholesale.
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.hc_mult = int(config.hc_mult)

    def _build_decoder_layer(self, config: ModelConfig, layer_index: int) -> DeepseekV4DecoderLayer:
        return DeepseekV4DecoderLayer(config, layer_index, quant=self._layer_quant(layer_index))

    def load_weights(self, checkpoint: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Fill the parameters from a DSpark or an HF-style checkpoint stream.

        A quantised checkpoint is DSpark-native (``attn.wq_a`` paths, ``.scale``
        leaves, fp8/mxfp4 byte formats) and is rewritten on the way in; an
        unquantised one is the transformers-style state dict the parity tests
        checkpoint, already in the loader's vocabulary.
        """
        if self.quant is not None:
            checkpoint = _adapt_dspark_checkpoint(checkpoint)
        super().load_weights(checkpoint)

    def translate_weight_key(self, key: str):
        """Map a checkpoint key onto this model's parameters.

        A quantised checkpoint speaks DSpark's native vocabulary (``attn.wq_a``
        paths, ``.scale`` leaves), so its keys get the same rewrite a load
        applies before the translator sees them. transformers >= 5.9 nests the
        indexer's scorer projection one module deeper
        (``indexer.scorer.weights_proj``) than the DSpark checkpoints and
        rapid_llm do; the rename keeps one parameter name serving both
        layouts.
        """
        if self.quant is not None:
            key = adapt_dspark_key(key)
        return super().translate_weight_key(key.replace(".indexer.scorer.", ".indexer."))

    def reset_v4_caches(self) -> None:
        """Clear every layer's sliding window and compressor state.

        Call before a fresh prefill: the per-layer caches are row-indexed by
        the batch position of the sequences being served, so a new batch on
        the same rows must start from empty state.
        """
        for layer in self.layers:
            layer.self_attn.reset()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        inputs_embeds: torch.Tensor | None = None,
        layer_context: dict | None = None,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the stream stack and project to vocabulary logits.

        Args mirror :meth:`rapid_llm.models.base.CausalLM.forward`;
        ``atten_info`` contributes only step metadata here — ``is_prefill``
        triggers the per-layer cache reset and ``b_seq_len`` builds the
        real-token mask the compressors consume (decode steps are single
        real tokens by construction).
        """
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        )
        batch, seq_len = hidden_states.shape[:2]
        if atten_info.is_prefill:
            self.reset_v4_caches()
            lens = torch.as_tensor(atten_info.b_seq_len, device=hidden_states.device)
            valid = torch.arange(seq_len, device=hidden_states.device)[None, :] < lens[:, None]
        else:
            valid = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, position_ids, "main")
        streams = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        for layer in self.layers:
            streams = layer(streams, position_ids, position_embeddings, input_ids, valid)

        hidden_states, _ = skip_rmsnorm(
            self.hc_head(streams), None, self.norm_weight, self.rms_norm_eps
        )
        if logits_positions is not None:
            rows = torch.arange(batch, device=hidden_states.device)
            hidden_states = hidden_states[rows, logits_positions]
        return self.lm_head(hidden_states)
