"""Layer/model skeleton shared by the decoder-only models.

:class:`DecoderLayer` composes attention and MLP behind flag-driven seams
(qkv bias, qk norm, quant, MoE) and :class:`CausalLM` stacks the layers
with the embedding, LM head and weight-loading plumbing.

Usage:
    model = CausalLM(config)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn

from ..distributed.sequence_parallel import (
    SequenceParallelPass,
    entry_scatter,
    exit_gather,
    sequence_parallel_region,
)
from ..kernels import (
    fused_add_rmsnorm,
    qk_rmsnorm,
    rope_emb_forward,
    skip_rmsnorm,
)
from ..modules import (
    FusedMLP,
    LinearBase,
    PagedAttention,
    ParallelLMHead,
    QKVParallelLinear,
    RotaryEmbedding,
    RowParallelLinear,
    SparseMoeBlock,
    VocabParallelEmbedding,
)
from ..modules.quantization import QuantizationConfig, adapt_packed_checkpoint
from . import weights
from .config import ModelConfig


class Attention(nn.Module):
    """Fused-QKV self-attention with RoPE and optional per-head q/k normalisation.

    The model-layer half of the attention block: this class owns the
    projections and their composition (project → per-head reshape → q/k norm →
    RoPE → :class:`~rapid_llm.modules.attention.PagedAttention` → output
    projection), while the paged-cache write and the prefill/decode kernel
    call live in :class:`~rapid_llm.modules.attention.PagedAttention`.

    Args:
        config: Model config supplying the head geometry.
        qkv_bias: Whether q/k/v projections carry a bias (true for Qwen2).
        use_qk_norm: Whether q and k are RMSNormed per head before RoPE (Qwen3).
        quant: Quantisation layout of the projections, or ``None``.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        quant: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            bias=qkv_bias,
            quant=quant,
            params_dtype=config.dtype,
        )
        # This rank's share of the head geometry, read back from the layer
        # that owns the weight rather than divided a second time here.
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.qkv_proj.q_size
        self.kv_size = self.qkv_proj.kv_size

        self.o_proj = RowParallelLinear(
            config.q_size,
            self.hidden_size,
            quant=quant,
            params_dtype=config.dtype,
            what="query features",
        )

        if use_qk_norm:
            # RMSNorm over head_dim, i.e. independently per head; replicated
            # rather than sharded because of that.
            self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=config.dtype))
            self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=config.dtype))

        self.attn = PagedAttention(
            self.num_kv_heads,
            self.head_dim,
            kv_cache_dtype=config.kv_cache_torch_dtype,
            params_dtype=config.dtype,
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, reshape to per-head layout, normalise (optionally) and apply RoPE."""
        x = x.reshape(-1, self.hidden_size)

        xq, xk, xv = self.qkv_proj.project(x)

        # Rows come from the projection, not from ``x``: inside a
        # sequence-parallel region the fused QKV layer gathers this rank's token
        # shard back to the full grid, so it returns more rows than it was given
        # -- and the RoPE tables and attention metadata were built for that grid.
        num_tokens = xq.shape[0]
        xq = xq.view(num_tokens, self.num_heads, self.head_dim)
        xk = xk.view(num_tokens, self.num_kv_heads, self.head_dim)
        xv = xv.view(num_tokens, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            # RMSNorm over head_dim, i.e. independently per head -- both
            # tensors in one launch instead of two.
            xq, xk = qk_rmsnorm(xq, xk, self.q_norm_weight, self.k_norm_weight, self.rms_norm_eps)

        cos, sin = position_embeddings
        xq, xk = rope_emb_forward(xq, xk, cos, sin)
        return xq, xk, xv

    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self._project_qkv(x, position_embeddings)

        # The phase comes from whoever prepared the metadata, not from seq_len:
        # a single-token prompt is still a prefill, and guessing ``seq_len > 1``
        # would route it through the decode kernel by accident.
        attn_output = self.attn(
            xq, xk, xv, atten_info, layer_index, is_prefill=atten_info.is_prefill
        )
        # Flat rows into the output projection, which is where a sequence-parallel
        # region's ``g-bar`` boundary sits: it reduce-scatters, so its row count is
        # ``x``'s again rather than the attention grid's. Viewing to ``x``'s
        # leading dims afterwards is therefore right on both paths -- it is the
        # residual-stream layout the block hands on either way.
        attn_output = attn_output.reshape(-1, self.q_size)
        return self.o_proj(attn_output).view(batch_size, seq_len, -1)


class DecoderLayer(nn.Module):
    """Pre-norm transformer block with a fused add-and-normalise.

    ``skip_rmsnorm`` returns ``(normalised, residual)`` where ``residual`` is the
    running sum ``x + residual``. Threading that pair through the stack lets the
    residual add happen inside the norm kernel instead of as a separate op, which
    is why :meth:`forward` takes and returns a ``residual`` tensor.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        quant: QuantizationConfig | None = None,
        mlp: nn.Module | None = None,
        attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        # Same injection seam as the MLP: ``CausalLM._build_attention`` supplies
        # the block; an MLA variant replaces it whole.
        self.self_attn = (
            attention
            if attention is not None
            else Attention(config, qkv_bias=qkv_bias, use_qk_norm=use_qk_norm, quant=quant)
        )
        # MoE variants inject a SparseMoeBlock via ``CausalLM._build_mlp``;
        # the default is the dense SwiGLU.
        self.mlp = mlp if mlp is not None else FusedMLP(config, quant)

    def forward_attn_stage(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First half of the block: input norm through the attention output.

        The two-stage split exists for two-batch overlap
        (:mod:`rapid_llm.batch_overlap.two_batch_overlap`): the stage boundary is exactly the
        o_proj row-parallel all-reduce, whose deferral is what the *other*
        micro-batch's compute overlaps. Stage callers fence that reduction
        at the head of :meth:`forward_mlp_stage`; everyone else keeps using
        :meth:`forward`, which runs the stages back to back.

        Nothing here knows whether a sequence-parallel region is open. The norm
        is per-row, so it computes on whatever rows it is handed, and the two
        projections take the region's boundaries themselves.

        Returns the *un-normalised* attention output plus the running
        residual -- :meth:`forward_mlp_stage`'s fused add-and-norm consumes
        the pair.
        """
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.input_layernorm_weight, self.rms_norm_eps
        )
        return (
            self.self_attn(hidden_states, atten_info, layer_index, position_embeddings),
            residual,
        )

    def _post_attention_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fused add-and-norm both MLP paths start with (dense and EP).

        The row-parallel dispatcher has completed its reduction by the time this
        runs -- an all-reduce over the full grid, or a reduce-scatter to this
        rank's token shard when a sequence-parallel region is open. Either way the
        rows arriving here are reduced, and the kernel treats them one at a time.
        """
        return fused_add_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight,
            self.rms_norm_eps,
        )

    def forward_mlp_stage(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Second half of the block: post-attention norm through the MLP.

        The fence point for the attention stage's deferred o_proj
        all-reduce is this method's head: the post-attention norm is the
        first op that reads the reduction's result.
        """
        hidden_states, residual = self._post_attention_norm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    # ------------------------------------------------------------------ #
    # TBO op stream (sglang ``batch_overlap`` building blocks)
    # ------------------------------------------------------------------ #
    #
    # The layer hands these bound methods to
    # :class:`~rapid_llm.batch_overlap.operations_strategy.OperationsStrategy`,
    # which orders them and places the yields. Ops pop what they consume from
    # the micro-batch StateDict and write under a *new* key, so clobbering a
    # live key raises instead of feeding a predecessor's stale result
    # downstream. ``op_attn``'s ``layer_index`` is bound by the strategy at
    # build time.

    def op_attn(self, state, *, layer_index: int) -> None:
        """Attention segment: fence this half's promise, then run attention.

        The fence sits at the head because this stage is the first to read the
        previous stage's deferred reduction (the MLP's down_proj, or nothing on
        the first layer); the o_proj reduction this stage defers is what the
        *other* micro-batch's compute overlaps.
        """
        ar = state.ar
        ar.fence(state.ar_events)
        with (
            ar.collecting(state.ar_events),
            state.timeline.region(f"tbo.attn.{state.tag}", "compute"),
        ):
            state.attn_output, state.residual_after_attn = self.forward_attn_stage(
                state.pop("hidden_states"),
                state.pop("residual"),
                state.atten_info,
                layer_index,
                state.position_embeddings,
            )

    def op_mlp(self, state) -> None:
        """MLP segment: the dense stream's tail, writing the next layer's input."""
        ar = state.ar
        ar.fence(state.ar_events)
        with (
            ar.collecting(state.ar_events),
            state.timeline.region(f"tbo.mlp.{state.tag}", "compute"),
        ):
            state.hidden_states, state.residual = self.forward_mlp_stage(
                state.pop("attn_output"), state.pop("residual_after_attn")
            )

    def op_gate(self, state) -> None:
        """Post-attention norm (consumes the o_proj promise), then routing."""
        ar = state.ar
        ar.fence(state.ar_events)
        with (
            ar.collecting(state.ar_events),
            state.timeline.region(f"tbo.gate.{state.tag}", "compute"),
        ):
            normed, residual = self._post_attention_norm(
                state.pop("attn_output"), state.pop("residual_after_attn")
            )
            state.mlp_input = self.mlp.op_gate(normed, state.moe_ctx)
            state.residual = residual

    def op_dispatch_a(self, state) -> None:
        """Post the forward a2a; the other half's compute covers the wire."""
        with state.timeline.region(f"tbo.dispatch_a.{state.tag}", "compute"):
            self.mlp.op_dispatch_a(state.mlp_input, state.moe_ctx)

    def op_shared_experts(self, state) -> None:
        """Shared expert MLP — the longest shadow-filler for the dispatch."""
        with state.timeline.region(f"tbo.shared.{state.tag}", "compute"):
            self.mlp.op_shared_experts(state.mlp_input, state.moe_ctx)

    def op_dispatch_b(self, state) -> None:
        """Fence the dispatch; the stream now carries this rank's expert batch."""
        with state.timeline.region(f"tbo.dispatch_b.{state.tag}", "compute"):
            state.local_batch = self.mlp.op_dispatch_b(state.pop("mlp_input"), state.moe_ctx)

    def op_experts(self, state) -> None:
        """Grouped GEMM over the received batch."""
        with state.timeline.region(f"tbo.experts.{state.tag}", "compute"):
            state.expert_output = self.mlp.op_experts(state.pop("local_batch"), state.moe_ctx)

    def op_combine_a(self, state) -> None:
        """Post the return a2a; the other half's experts cover the wire."""
        with state.timeline.region(f"tbo.combine_a.{state.tag}", "compute"):
            self.mlp.op_combine_a(state.expert_output, state.moe_ctx)

    def op_combine_b(self, state) -> None:
        """Fence the combine; the layer output is complete on every rank.

        ``op_combine_b`` folds in the shared expert and fences its deferred
        all-reduce promise itself (the moe.py sum discipline), so
        ``hidden_states`` is final for the next layer's attention stage.
        """
        with state.timeline.region(f"tbo.combine_b.{state.tag}", "compute"):
            state.hidden_states = self.mlp.op_combine_b(state.pop("expert_output"), state.moe_ctx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One whole block: attention stage then MLP stage, back to back."""
        hidden_states, residual = self.forward_attn_stage(
            hidden_states, residual, atten_info, layer_index, position_embeddings
        )
        return self.forward_mlp_stage(hidden_states, residual)


class CausalLM(nn.Module):
    """Forward skeleton shared by every decoder-only text model.

    Subclasses only set the class-level switches below; the token->logits pipeline
    itself is fixed here (template method).

    Class attributes:
        qkv_bias: Whether q/k/v projections carry a bias.
        use_qk_norm: Whether q and k are RMSNormed per head.
        rotary_class: RoPE implementation; multimodal variants swap in an
            mrope-aware subclass.
        hf_prefix: Checkpoint prefix wrapping the decoder stack. HF text models
            nest everything except ``lm_head`` under ``model.``.
    """

    qkv_bias: ClassVar[bool] = False
    use_qk_norm: ClassVar[bool] = False
    rotary_class: ClassVar[type[RotaryEmbedding]] = RotaryEmbedding
    hf_prefix: ClassVar[str] = "model."
    #: ``{fused module path: (checkpoint module paths, in block order)}`` — the
    #: projections this model fuses, consumed by
    #: :func:`~rapid_llm.models.weights.translate_text_key`. The sources' index
    #: becomes the ``shard_id`` handed to the fused parameter's loader.
    packed_modules_mapping: ClassVar[dict[str, tuple[str, ...]]] = {
        "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    }

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Weight format of the checkpoint being loaded, and therefore of every
        # projection built below. ``None`` is unquantised; an fp8 checkpoint declares its
        # own block layout, which the layers keep as-is instead of widening.
        self.quant = config.quant
        dtype = config.dtype

        # The vocabulary tensors are split along the vocabulary itself (see
        # :mod:`rapid_llm.modules.vocab_parallel`): they are the largest pair of
        # weights in a large-vocabulary model, the decode-step head GEMM scales with
        # them, and a tied model cannot honestly shard one without the other.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, params_dtype=dtype
        )
        self.layers = nn.ModuleList(
            self._build_decoder_layer(config, i) for i in range(config.num_layers)
        )
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=dtype))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, params_dtype=dtype)

        self.rotary_emb = self.rotary_class(config.rope_config)
        self.rms_norm_eps = config.rms_norm_eps

        # Optional module pass; the region it marks for opens per forward, and
        # only for steps whose runtime gates clear.
        self.sequence_parallel_pass = SequenceParallelPass()
        self.sequence_parallel_pass.apply(self)

    def _layer_quant(self, layer_index: int) -> QuantizationConfig | None:
        """Quantisation layout for layer ``layer_index``, honouring the checkpoint's
        ``modules_to_not_convert``.

        Checkpoints exclude modules by HF path, so the question is asked with
        the HF path a projection of this layer would have. The modules excluded
        by name (layer norms, the MoE router, ``lm_head``) are not built from
        :class:`~rapid_llm.modules.linear.LinearBase` at all.
        """
        if self.quant is None:
            return None
        return self.quant if self.quant.quantizes(f"{self.hf_prefix}layers.{layer_index}") else None

    def _build_attention(self, config: ModelConfig, layer_index: int) -> nn.Module:
        """Per-layer attention block factory.

        The standard GQA composition (fused QKV, q/k norm, RoPE, paged
        attention) lives in :class:`Attention`; an MLA variant overrides this
        hook to swap the whole block.
        """
        return Attention(
            config,
            qkv_bias=self.qkv_bias,
            use_qk_norm=self.use_qk_norm,
            quant=self._layer_quant(layer_index),
        )

    def _build_mlp(self, config: ModelConfig, layer_index: int) -> nn.Module:
        """Per-layer MLP factory; MoE variants override it to pick per layer."""
        return FusedMLP(config, self._layer_quant(layer_index))

    def _build_decoder_layer(self, config: ModelConfig, layer_index: int) -> DecoderLayer:
        """Per-layer block factory: the default pairs the two factories above.

        A family that assembles the block differently (DeepSeek picks MLA plus
        dense-or-MoE per position inside its own ``DecoderLayer`` subclass)
        overrides this instead of the two narrower hooks.
        """
        return DecoderLayer(
            config,
            attention=self._build_attention(config, layer_index),
            mlp=self._build_mlp(config, layer_index),
        )

    # ---- weight loading --------------------------------------------------- #
    def translate_weight_key(self, key: str) -> weights.Target:
        """Map a checkpoint key onto this model's parameters.

        Strips :attr:`hf_prefix` (``lm_head.weight`` sits outside it) and defers
        the rest to :func:`rapid_llm.models.weights.translate_text_key`, with
        :attr:`packed_modules_mapping` supplying the fused-projection rules.
        Layer keys at or past ``num_layers`` are dropped first: their weights
        belong to modules this model never built — the MTP/nextn layers a
        DeepSeek checkpoint ships past its stack, or whatever an
        ``hf_overrides`` ``num_hidden_layers`` trim cut away — and failing
        the load on them would make trimming impossible.
        """
        stripped = key.removeprefix(self.hf_prefix)
        if stripped.startswith("layers."):
            index, _, _ = stripped[len("layers.") :].partition(".")
            if index.isdigit() and int(index) >= self.config.num_layers:
                return None
        return weights.translate_text_key(stripped, self.packed_modules_mapping)

    def load_weights(self, checkpoint: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Fill every parameter from a HuggingFace checkpoint stream.

        Args:
            checkpoint: ``(key, tensor)`` pairs as produced by
                :func:`rapid_llm.executor.weight_utils.hf_weights_iterator`.
        """
        if self.quant is not None and self.quant.is_packed:
            # A packed checkpoint (AWQ/GPTQ, either bit width) stores weights
            # in its producer's word layout; rewrite the stream to the
            # canonical layout on the way in.
            checkpoint = adapt_packed_checkpoint(checkpoint, self.quant)
        weights.load_weights(
            self,
            checkpoint,
            self.translate_weight_key,
            tied={"lm_head.weight": "embed_tokens.weight"}
            if self.config.tie_word_embeddings
            else None,
        )
        # Post-load weight transforms: a quant method whose kernel layout
        # differs from the checkpoint's (int4's byte packing) repacks here,
        # once, while the parameters sit on the load device. Most methods
        # consume exactly what they loaded and the hook is a no-op.
        for module in self.modules():
            if isinstance(module, (LinearBase, SparseMoeBlock)):
                module.quant_method.process_weights_after_loading(module)

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Convert every loaded fp16 projection to the requested scheme, in place.

        The ``--quantization <scheme>`` path: the checkpoint has no scales of
        its own, so they are computed here after loading. Already-quantised
        layers (an fp8 checkpoint) are left alone.
        """
        for module in self.modules():
            if isinstance(module, (LinearBase, SparseMoeBlock)):
                module.quantize_(quant)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _after_layer(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
        layer_context: dict[str, Any],
    ) -> torch.Tensor:
        """Extension point invoked after each decoder layer.

        The default is a no-op. Qwen3-VL overrides it to add its DeepStack visual
        features into the first few layers' hidden states.
        """
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        inputs_embeds: torch.Tensor | None = None,
        layer_context: dict[str, Any] | None = None,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder stack and project to vocabulary logits.

        Args:
            input_ids: ``[batch, seq_len]`` token ids.
            position_ids: Absolute positions; ``[batch, seq_len]`` for plain RoPE,
                ``[3, batch, seq_len]`` for mrope.
            atten_info: KV-cache bookkeeping for this step.
            inputs_embeds: Pre-computed embeddings; when given, ``input_ids`` is
                only used for shape information. Multimodal models pass the
                merged text+vision embeddings here.
            layer_context: Optional per-step payload handed to :meth:`_after_layer`.
            logits_positions: Optional ``[batch]`` position per sequence whose
                logits the caller wants. Given, the hidden states are gathered
                at exactly those positions *before* the lm_head projection, so
                a prefill of a 2 048-token prompt pays one vocabulary row
                instead of 2 048. ``None`` projects every position (decode
                steps want their single row anyway).

        Returns:
            ``[batch, seq_len, vocab_size]`` logits, or ``[batch, vocab_size]``
            when ``logits_positions`` was given. Under tensor parallelism the
            vocabulary dimension is this rank's slice; the sampler completes
            the distribution from a scalar per row instead of gathering logits.
        """
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        )
        # Built on the full grid and left there: the RoPE tables index absolute
        # positions, and every consumer of them runs between a ``g`` and a
        # ``g-bar``, where the rows are the whole step's again.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        grid_shape = hidden_states.shape
        with sequence_parallel_region(
            grid_shape[:-1].numel(),
            # ``layer_context`` rules the region out: :meth:`_after_layer` adds
            # whole-grid features (Qwen3-VL's DeepStack) into the stream between
            # blocks, where a region is holding one rank's token shard.
            marked=bool(self.sequence_parallel_pass.matched) and not layer_context,
        ) as region:
            if region is not None:
                hidden_states = entry_scatter(hidden_states, region)

            residual = None
            for layer_index, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    hidden_states, atten_info, layer_index, position_embeddings, residual
                )
                if layer_context:
                    # Adding into `hidden_states` before the next fused add-and-norm is
                    # equivalent to adding into the post-layer output, because that norm
                    # computes `hidden_states + residual`.
                    hidden_states = self._after_layer(hidden_states, layer_index, layer_context)

            hidden_states, _ = skip_rmsnorm(
                hidden_states, residual, self.norm_weight, self.rms_norm_eps
            )
            if region is not None:
                hidden_states = exit_gather(hidden_states, region, grid_shape)

        if logits_positions is not None:
            # Prompts differ in length, so each sequence's next-token prediction
            # sits at its own last real position; pick it before the GEMM. After
            # the gather above, necessarily: these are global row numbers.
            rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            hidden_states = hidden_states[rows, logits_positions]
        return self.lm_head(hidden_states)
