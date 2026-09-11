"""Logical-operator ABCs: the call contracts behind each ``dispatch()``.

Each :class:`LogicalOp` subclass (``LinearOp``, ``MoeOp``, ...) pins the
signature an implementation must expose, torch-free, so specs, registry
and dispatch can reason about ops without importing the kernels.

Usage:
    from rapid_llm.kernels.ops import LinearOp
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch


class LogicalOp(ABC):
    """Base of every logical-operator contract.

    ``op_id`` is the string the registry and dispatch key on; concrete
    implementation classes do not need to redefine it — the KernelSpec they
    are registered under already names the op. ``__call__`` stays abstract so
    an unfinished contract cannot be instantiated silently.
    """

    op_id: ClassVar[str]

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the operator (overridden by each interface with its full signature)."""
        raise NotImplementedError


class AttentionPrefillOp(LogicalOp):
    """Prefill attention over ragged (no-pad) batches."""

    op_id = "attention.prefill"

    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Attend over freshly projected K/V for a ragged batch.

        Args:
            q: ``[total_tokens, num_heads, head_dim]`` packed query rows.
            k, v: ``[total_kv, num_kv_heads, head_dim]`` packed key/value rows.
            sm_scale: Softmax scale, ``1 / sqrt(head_dim)``.
            b_start_loc: ``[batch]`` first packed row of each sequence.
            b_seq_len: ``[batch]`` true length of each sequence.
            max_seq_len: Longest sequence, sizing the flash tiles.

        Returns:
            ``[total_tokens, num_heads, head_dim]`` attention output.
        """
        raise NotImplementedError


class AttentionChunkedPrefillOp(LogicalOp):
    """Causal prefill for chunks resuming on K/V that is already cached."""

    op_id = "attention.chunked_prefill"

    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sm_scale: float,
        b_start_loc: torch.Tensor,
        b_kv_base: torch.Tensor,
        b_prefix_len: torch.Tensor,
        b_seq_len: torch.Tensor,
        max_chunk_len: int,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        """Attend over the paged cache for a resumed chunk.

        Args:
            q: ``[total_rows, num_heads, head_dim]`` packed query rows of the
                chunk grid, padding columns included (masked out inside).
            k_cache: ``[max_tokens, num_kv_heads, head_dim]`` paged key cache,
                already holding this chunk's rows; ``v_cache`` the same for V.
            sm_scale: Softmax scale, ``1 / sqrt(head_dim)``.
            b_start_loc: ``[batch]`` first packed query row of each sequence.
            b_kv_base: ``[batch]`` cache row holding each sequence's KV row 0.
            b_prefix_len: ``[batch]`` cached rows preceding this chunk.
            b_seq_len: ``[batch]`` total length once the chunk lands.
            max_chunk_len: Widest chunk, sizing the query-block grid.
            k_scale: Dequantisation scale of an fp8 key cache (1.0 otherwise).
            v_scale: Same for the value cache.

        Returns:
            ``[total_rows, num_heads, head_dim]`` attention output.
        """
        raise NotImplementedError


class AttentionDecodeOp(LogicalOp):
    """Decode attention reading a paged KV cache."""

    op_id = "attention.decode"

    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qk_scale: float,
        b_req_tokens_table: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        max_actual_seq_len: int,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        """One-token-per-sequence attention against the paged cache.

        Args:
            q: ``[batch, num_heads, head_dim]`` — decode has ``seq_len == 1``.
            k_cache: ``[max_tokens, num_kv_heads, head_dim]`` paged key cache.
            v_cache: Paged value cache, same layout as ``k_cache``.
            qk_scale: Softmax scale, ``1 / sqrt(head_dim)``.
            b_req_tokens_table: ``[max_requests, max_seq_len]`` position-to-
                cache-row map.
            b_req_idx: ``[batch]`` request slot owning each batch row; batch
                order is not slot order once requests come and go.
            b_seq_len: ``[batch]`` history length per row, this step included.
            max_actual_seq_len: Longest row, sizing the split-K grid.
            k_scale: Dequantisation scale of an fp8 key cache (1.0 otherwise).
            v_scale: Same for the value cache.

        Returns:
            ``[batch, num_heads, head_dim]`` attention output.
        """
        raise NotImplementedError


class MlaDecodeOp(LogicalOp):
    """Multi-head latent attention decode over the latent KV cache.

    The latent cache is single-head (MQA over the compressed ``c_kv``), which
    is the part every MLA backend shares; backend-specific head layouts stay
    inside the impl and its LayoutRequirement.
    """

    op_id = "attention.mla_decode"

    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        max_seq_len: int,
        sm_scale: float = 1.0,
    ) -> torch.Tensor:
        """Decode attention over the MLA latent KV cache.

        Args:
            q: ``[batch, num_heads, qk_head_dim]`` query per step.
            kv_cache: ``[num_pages, page_size, kv_lora_dim]`` latent cache.
            block_table: ``[batch, max_pages]`` page ids per sequence.
            cache_seqlens: ``[batch]`` cached length per sequence.
            max_seq_len: Longest row, sizing the kernel grid.
            sm_scale: Softmax scale.

        Returns:
            ``[batch, num_heads, v_head_dim]`` attention output.
        """
        raise NotImplementedError


class MlaPrefillOp(LogicalOp):
    """MLA prefill: up-project the fresh latent, then varlen attention.

    Decode keeps ``q`` absorbed and attends the latent cache directly; prefill
    cannot absorb — every new token queries every other new token, so the
    absorbed form would pay the wide latent dot per pair. The contract takes
    the up-projection weights so each backend picks its own workspace
    strategy (the native row chunks them; an external one may fuse them).
    """

    op_id = "attention.mla_prefill"

    @abstractmethod
    def __call__(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        w_uk: torch.Tensor,
        w_uv: torch.Tensor,
        sm_scale: float,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Up-project the fresh latent and attend the packed batch.

        Args:
            q_nope: ``[tokens, num_heads, qk_nope_head_dim]`` un-absorbed query.
            q_pe: ``[tokens, num_heads, qk_rope_head_dim]`` rope segment of q.
            c_kv: ``[tokens, kv_lora_rank]`` fresh latent rows.
            k_pe: ``[tokens, qk_rope_head_dim]`` rope keys, one per token,
                shared by all heads.
            w_uk: ``[num_heads, kv_lora_rank, qk_nope_head_dim]`` per-head K
                up-projection (a transposed view of the ``kv_b_proj`` K half).
            w_uv: ``[num_heads, kv_lora_rank, v_head_dim]`` per-head V
                up-projection, same source.
            sm_scale: Softmax scale.
            b_start_loc: ``[batch]`` first packed row of each sequence.
            b_seq_len: ``[batch]`` true length of each sequence.
            max_seq_len: Longest sequence, sizing the flash tiles.

        Returns:
            ``[tokens, num_heads, v_head_dim]`` attention output.
        """
        raise NotImplementedError


class LinearOp(LogicalOp):
    """Dense ``x @ weight.T (+ bias)``, quantised or not.

    The signature is the superset of the quantised GEMMs (fp8 / int8 block /
    smoothquant / int4 / nvfp4): a scheme is a dispatch key dimension, the impl
    knows which of the optional tensors it consumes and ignores the rest.
    """

    op_id = "linear"

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        weight_zeros: torch.Tensor | None = None,
        weight_global_scale: torch.Tensor | None = None,
        group_n: int = 0,
        group_k: int = 0,
    ) -> torch.Tensor:
        """Project activations.

        Args:
            x: ``[tokens, in_features]`` activations.
            weight: ``[out_features, in_features]``, packed when quantised.
            bias: Optional ``[out_features]`` additive bias.
            weight_scale: Dequantisation scales; ``None`` for plain GEMM.
            weight_zeros: Zero points for asymmetric int4; ``None`` symmetric.
            weight_global_scale: Second-level per-tensor scale, for formats whose
                block scales are themselves quantised (NVFP4). ``None`` for the
                single-level schemes, which is all of the others.
            group_n: Rows per scale block (``0`` = per-tensor).
            group_k: Columns per scale block.

        Returns:
            ``[tokens, out_features]`` in ``x``'s dtype.
        """
        raise NotImplementedError


class MoeOp(LogicalOp):
    """Routed expert FFN: ``sum_k w_k * SwiGLU_k(x) @ W2_k.T``."""

    op_id = "moe"

    @abstractmethod
    def __call__(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        w1_zeros: torch.Tensor | None = None,
        w2_zeros: torch.Tensor | None = None,
        group_n: int = 0,
        group_k: int = 0,
        swiglu_limit: float = float("inf"),
        mxfp4: bool = False,
    ) -> torch.Tensor:
        """Run all dispatched experts for every token.

        Args:
            hidden_states: ``[tokens, hidden]`` activations.
            w1: ``[E, 2 * intermediate, hidden]`` fused gate/up projections.
            w2: ``[E, hidden, intermediate]`` down projections.
            topk_weights: ``[tokens, top_k]`` routing weights.
            topk_ids: ``[tokens, top_k]`` expert indices.
            w1_scale, w2_scale: Per-expert dequant scales; ``None`` = plain.
            w1_zeros, w2_zeros: Zero points for the asymmetric formats — int4
                experts and GPTQ ``bits=8`` int8; ``None`` for symmetric.
            group_n, group_k: Scale block geometry.
            swiglu_limit: Clamp the SwiGLU gate at ``+limit`` and up at
                ``+-limit`` (gpt-oss semantics, DeepSeek-V4's ``7.0``);
                ``inf`` keeps plain SwiGLU.
            mxfp4: Read ``w1``/``w2`` as packed MXFP4 e2m1 nibbles (two
                elements per int8 byte, low nibble first) with e8m0-derived
                fp32 scales in ``w1_scale``/``w2_scale`` — DeepSeek-V4's
                weight-only expert format.

        Returns:
            ``[tokens, hidden]`` combined expert output.
        """
        raise NotImplementedError


class DispatchOp(LogicalOp):
    """All-to-all token routing to expert ranks (EP).

    No native row, and deliberately so: MoE in this repo is *tensor* parallel —
    every rank runs every expert over its slice of the intermediate dimension
    (:class:`rapid_llm.modules.moe.SparseMoeBlock`), so there is no EP group to
    all-to-all across, and the local permute half of the job already lives
    inside ``fused_moe``'s ``moe_align_block_size``. A placeholder written now
    would be untestable code duplicating an existing sort. The EP group and the
    deepep rows arrive together in M2.5; the contract is declared here so the
    receive layout is fixed before then.

    The exact receive layout (seg_indptr vs masked_m) is declared in the impl's
    LayoutRequirement, not invented ad hoc.
    """

    op_id = "comm.dispatch"

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        *,
        num_experts: int,
    ) -> tuple[Any, ...]:
        """Send each token copy to the ranks owning its experts.

        Args:
            x: ``[tokens, hidden]`` activations to route.
            topk_idx: ``[tokens, top_k]`` global expert ids.
            num_experts: Total experts across all ranks.

        Returns:
            Backend-specific receive bundle (received tokens plus the index
            metadata :class:`CombineOp` needs); see the impl's layout tags.
        """
        raise NotImplementedError


class CombineOp(LogicalOp):
    """Gather expert outputs back and merge the routed copies.

    Shares :class:`DispatchOp`'s reason for having no native row: the two are
    one protocol and must land in the same version as the EP group.
    """

    op_id = "comm.combine"

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        unsorted_src_idx: torch.Tensor,
        unsorted_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse of :class:`DispatchOp` for the same receive layout.

        Args:
            x: ``[recv_tokens, hidden]`` processed expert outputs.
            unsorted_src_idx: Token-order metadata from dispatch.
            unsorted_weights: Routing weight applied per received copy.

        Returns:
            ``[tokens, hidden]`` weighted sum back in request order.
        """
        raise NotImplementedError


class RmsNormOp(LogicalOp):
    """RMSNorm, optionally fused with the residual add (skip path)."""

    op_id = "rmsnorm"

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        weight: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise the last dimension, folding in the residual add when asked.

        Args:
            x: ``[..., hidden]`` activations.
            residual: When given, added to ``x`` before normalising; ``None``
                runs the plain norm. Positional and mandatory because the two
                paths return the same pair either way and the caller has to
                say which one it wants.
            weight: ``[hidden]`` learned scale.
            eps: Variance floor.

        Returns:
            ``(normalised, residual)``, always a pair. The second element is the
            summed input on the fused path and ``x`` itself on the plain one, so
            a decoder layer can thread it into the next norm without branching.
        """
        raise NotImplementedError


class RopeOp(LogicalOp):
    """Rotary position embedding applied to query and key."""

    op_id = "rope"

    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q``/``k`` in place where the layout allows.

        Args:
            q: ``[tokens, num_q_heads, head_dim]`` queries.
            k: ``[tokens, num_kv_heads, head_dim]`` keys.
            cos, sin: ``[batch, seq_len, head_dim]`` rotation tables; the
                batch/seq geometry is derived from them, not passed twice.

        Returns:
            ``(q, k)`` after rotation — the input tensors when they were
            adjacent-head contiguous, fresh copies otherwise.
        """
        raise NotImplementedError


class KvWriteOp(LogicalOp):
    """Scatter freshly computed K/V rows into the paged KV buffer."""

    op_id = "kv_write"

    @abstractmethod
    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        select_index: torch.Tensor,
        kv_buffer: torch.Tensor,
    ) -> None:
        """Write each token's K and V into its allocated cache slot.

        Args:
            k: ``[tokens, num_kv_heads, head_dim]`` new key rows.
            v: New value rows, same layout as ``k``.
            select_index: ``[tokens]`` target cache row per token.
            kv_buffer: ``[max_tokens, 2 * num_kv_heads, head_dim]`` buffer whose
                head axis holds the K heads first, then the V heads — so one
                token's K and V are adjacent in memory rather than a pool apart.

        The buffer is modified in place; nothing is returned.
        """
        raise NotImplementedError


class SampleOp(LogicalOp):
    """Token sampling core (top-k / top-p) over the final logits.

    No native row registers against this contract yet: sampling in this repo
    lives in :class:`rapid_llm.engine.sampler.Sampler`, which is torch ops over
    a *tensor-parallel slice* of the vocabulary and carries repetition penalties
    and per-row parameters that no kernel signature covers. Adding a second
    sampler under ``kernels/`` just to fill the catalogue would give sampling two
    places to disagree. The contract exists for flashinfer's fused top-k/top-p
    sampler (M2.1), which is when the engine path becomes the native row.
    """

    op_id = "sample"

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Draw one token id per row.

        Args:
            logits: ``[batch, vocab]`` this rank's slice under TP.
            temperature: Softmax temperature; ``1.0`` skips the division.
            top_k: Keep only the k best logits (``0`` = off).
            top_p: Nucleus sampling mass (``1.0`` = off).
            deterministic: Greedy argmax, used for golden runs and tests.

        Returns:
            ``[batch]`` sampled token ids.
        """
        raise NotImplementedError


class ElementwiseOp(LogicalOp):
    """Pointwise transforms under the open ``elementwise.*`` namespace.

    Concrete members register their own KernelSpec rows and each pins its own
    arity: ``elementwise.swiglu`` takes the packed ``[tokens, 2 * inter]``
    gate/up tensor a fused projection produces, ``elementwise.swiglu_split``
    takes the two halves separately. This ABC only pins what they share:
    row-major over the flattened token dimension, output dtype tracks the input.
    """

    op_id = "elementwise"

    @abstractmethod
    def __call__(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """Apply the transform; extra operands are member-specific."""
        raise NotImplementedError


#: The closed op catalogue dispatch and registration validate against.
LOGICAL_OPS: dict[str, type[LogicalOp]] = {
    cls.op_id: cls
    for cls in (
        AttentionPrefillOp,
        AttentionChunkedPrefillOp,
        AttentionDecodeOp,
        MlaDecodeOp,
        MlaPrefillOp,
        LinearOp,
        MoeOp,
        DispatchOp,
        CombineOp,
        RmsNormOp,
        RopeOp,
        KvWriteOp,
        SampleOp,
        ElementwiseOp,
    )
}


def is_logical_op(op: str) -> bool:
    """Return True when *op* is a known logical operator id.

    ``elementwise.*`` is an open namespace, so any child of the root counts.
    """
    return op in LOGICAL_OPS or op.startswith("elementwise.")
