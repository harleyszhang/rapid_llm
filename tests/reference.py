"""Pure-PyTorch reference implementations for the Triton kernels.

Every function here is the slow, obvious maths —
``varlen_causal_attention``, ``paged_decode_attention``, ``swiglu`` —
that the kernel tests diff against; correctness from readability.

Usage:
    from tests.reference import skip_rmsnorm
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def _expand_kv_heads(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Map ``num_kv_heads`` onto ``num_q_heads`` the way the kernels do.

    Both kernels resolve a query head to its KV head with
    ``kv_head = q_head // num_kv_groups``, so KV head ``j`` serves the
    contiguous query-head run ``[j * groups, (j + 1) * groups)``. That is
    exactly ``repeat_interleave`` on the head axis — ``repeat`` would tile the
    heads instead and silently pair the wrong ones when ``groups > 1``.

    Args:
        x: ``[num_kv_heads, seq, head_dim]``.
        num_groups: Query heads per KV head.
    """
    return x.repeat_interleave(num_groups, dim=0)


def varlen_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Reference for :func:`flash_attention2_no_pad` (prefill).

    Sequences are packed back-to-back in the token axis rather than padded to a
    rectangle, so each one is sliced out by ``b_start_loc``/``b_seq_len`` and
    attended independently under a causal mask.

    Args:
        q: ``[total_tokens, num_q_heads, head_dim]``.
        k: ``[total_tokens, num_kv_heads, head_dim]``.
        v: Same shape as ``k``.
        b_start_loc: ``[batch]`` offset of each sequence in the token axis.
        b_seq_len: ``[batch]`` length of each sequence.
        sm_scale: Softmax scale *without* the ``log2(e)`` factor the kernel
            needs, i.e. plain ``1 / sqrt(head_dim)``.

    Returns:
        ``[total_tokens, num_q_heads, head_dim]``, dtype of ``q``.
    """
    num_groups = q.shape[1] // k.shape[1]
    out = torch.zeros_like(q)

    for i in range(b_seq_len.shape[0]):
        start, length = int(b_start_loc[i]), int(b_seq_len[i])
        sl = slice(start, start + length)

        qi = q[sl].transpose(0, 1).float()  # [HQ, n, D]
        ki = _expand_kv_heads(k[sl].transpose(0, 1).float(), num_groups)
        vi = _expand_kv_heads(v[sl].transpose(0, 1).float(), num_groups)

        scores = (qi @ ki.transpose(-1, -2)) * sm_scale
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        probs = scores.softmax(dim=-1)
        out[sl] = (probs @ vi).transpose(0, 1).to(out.dtype)

    return out


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    b_req_tokens_table: torch.Tensor,
    b_seq_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Reference for :func:`flash_decoding` (single-token decode).

    The one new query per sequence attends the whole cached history, so there is
    no causal mask. History rows are gathered through ``b_req_tokens_table``,
    which is what makes the cache pageable: row ``i`` of the table lists the
    cache slots holding sequence ``i``'s tokens in order, and those slots need
    not be contiguous.

    Args:
        q: ``[batch, num_q_heads, head_dim]``.
        k_cache: ``[max_tokens, num_kv_heads, head_dim]``.
        v_cache: Same shape as ``k_cache``.
        b_req_tokens_table: ``[batch, max_seq_len]`` cache row indices.
        b_seq_len: ``[batch]`` valid history length per sequence.
        sm_scale: ``1 / sqrt(head_dim)``.

    Returns:
        ``[batch, num_q_heads, head_dim]``, dtype of ``q``.
    """
    num_groups = q.shape[1] // k_cache.shape[1]
    out = torch.zeros_like(q)

    for i in range(q.shape[0]):
        length = int(b_seq_len[i])
        rows = b_req_tokens_table[i, :length].long()

        ki = _expand_kv_heads(k_cache[rows].transpose(0, 1).float(), num_groups)
        vi = _expand_kv_heads(v_cache[rows].transpose(0, 1).float(), num_groups)
        qi = q[i].float().unsqueeze(1)  # [HQ, 1, D]

        scores = (qi @ ki.transpose(-1, -2)) * sm_scale  # [HQ, 1, n]
        probs = scores.softmax(dim=-1)
        out[i] = (probs @ vi).squeeze(1).to(out.dtype)

    return out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm, reducing in fp32 like the kernel does."""
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale).to(x.dtype) * weight


def skip_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the fused skip-connection + RMSNorm kernel.

    Returns:
        ``(normed, new_residual)`` where ``new_residual`` is the pre-norm sum
        ``x + residual`` that the next block adds onto. With ``residual=None``
        the kernel degenerates to plain RMSNorm and echoes ``x`` back.
    """
    if residual is None:
        return rmsnorm(x, weight, eps), x
    summed = x + residual
    return rmsnorm(summed, weight, eps), summed


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference for :func:`swiglu_forward`: ``silu(gate) * up``.

    The kernel computes the sigmoid in fp32 before multiplying back down, so the
    reference does too — an fp16 sigmoid drifts enough to mask real errors.
    """
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
    swiglu_limit: float = float("inf"),
) -> torch.Tensor:
    """Reference for :func:`fused_moe`: gather per expert, matmul, scatter-add.

    The kernel reaches the same result by sorting token slots into per-expert runs
    and padding each run to a row-block, so this loop is what that machinery is
    allowed to be equivalent to. Everything is fp32 and the expert loop is
    explicit; the point is that the routing is legible, not that it is fast.

    Quantised weights are *not* handled here. Pass the dequantised weights and
    keep the dequant in the caller, where it can be written with plain torch ops
    that do not share code with the kernel under test — a reference that called
    the kernel's own unpacking would certify it against itself.

    Args:
        hidden_states: ``[num_tokens, hidden]`` activations.
        w1: ``[E, 2 * intermediate, hidden]`` fused gate/up weights, float.
        w2: ``[E, hidden, intermediate]`` down weights, float.
        topk_weights: ``[num_tokens, top_k]`` routing weights.
        topk_ids: ``[num_tokens, top_k]`` expert indices.
        act_quant: Round-trip applied to each GEMM's *activation* operand, for
            ``fused_moe_w8a8_fp8`` where the activation is quantised too. It is a
            round trip (quantise then dequantise) rather than a byte tensor
            because the reference multiplies in fp32 throughout: what has to be
            reproduced is the rounding, not the storage. ``None``, the
            weight-only default, leaves the activations exact.

            Applying it to the whole ``[num_tokens, hidden]`` input before the
            gather is equivalent to applying it per gathered row, which is what
            the kernel does: the round trip is per row, and a gather copies rows
            without changing their amax.
        swiglu_limit: Clamp gate at ``+limit`` and up at ``+-limit`` before the
            activation, mirroring the fused kernels' ``LIMIT`` (DeepSeek-V4's
            bounded SwiGLU); ``inf``, the default, keeps plain silu.

    Returns:
        ``[num_tokens, hidden]`` fp32 output, summed over the ``top_k`` slots.
    """
    x = hidden_states.float()
    if act_quant is not None:
        x = act_quant(x)
    inter = w1.shape[1] // 2
    out = torch.zeros_like(x)

    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1).float()
    # Slot i belongs to token i // top_k, the same mapping the kernel applies as
    # ``offs_token // top_k``.
    token_of_slot = torch.arange(x.shape[0], device=x.device).repeat_interleave(topk_ids.shape[1])

    for e in flat_ids.unique():
        sel = flat_ids == e
        rows = token_of_slot[sel]
        gate_up = x[rows] @ w1[e].float().T
        gate = gate_up[:, :inter].clamp(max=swiglu_limit)
        up = gate_up[:, inter:].clamp(min=-swiglu_limit, max=swiglu_limit)
        h = F.silu(gate) * up
        if act_quant is not None:
            # The kernel quantises the silu output per slot row before GEMM2, so
            # the second rounding has to be modelled too, or the reference would
            # be tighter than anything the kernel can achieve.
            h = act_quant(h)
        # index_add_ rather than indexed assignment: a token routed to top_k
        # experts accumulates one contribution per expert.
        out.index_add_(0, rows, (h @ w2[e].float().T) * flat_weights[sel, None])
    return out


#: The eight e2m1 magnitudes, indexed by the code's low three bits ``ee.m``.
#: ``ee == 0`` is the subnormal row (``{0, 0.5}``); above it the value is
#: ``2**(ee-1) * (1 + m/2)``.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nvfp4_dequant(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    block: int = 16,
) -> torch.Tensor:
    """Reference reconstruction of an NVFP4 weight, for :func:`nvfp4_matmul`.

    Deliberately shares nothing with the kernel: the nibble becomes a value by
    indexing an explicit eight-entry table, where the kernel assembles an fp32
    bit pattern, and the block scale is widened by ``view(float8_e4m3fn)``, where
    the kernel uses a shift-based bit trick plus a compensating factor of 256.
    Two independent decoders agreeing is the evidence; one decoder checked
    against itself is not.

    ``.view(torch.float4_e2m1fn_x2)`` looks like it would shorten this and must
    not be used: torch 2.13 accepts the view but ``.to(torch.float32)`` on the
    result raises a device-side assert, so it cannot serve as a reference.

    Args:
        packed: ``[N, K // 2]`` uint8, two e2m1 nibbles per byte, low nibble at
            the even k index.
        block_scale: ``[N, K // block]`` uint8 e4m3 bit patterns.
        global_scale: One-element fp32 tensor.
        block: Weight elements per block scale.

    Returns:
        ``[N, K]`` fp32 reconstruction.
    """
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)

    low = packed & 0xF
    high = (packed >> 4) & 0xF
    # Interleave back to k order: stacking on a new trailing axis and flattening
    # it puts low at even k, high at odd, which is the packing convention.
    codes = torch.stack([low, high], dim=-1).flatten(-2).long()

    magnitude = values[codes & 0x7]
    w = torch.where(codes & 0x8 != 0, -magnitude, magnitude)

    n, k = w.shape
    scales = block_scale.view(torch.float8_e4m3fn).float() * global_scale.float().reshape(())
    return (w.unflatten(-1, (k // block, block)) * scales.unsqueeze(-1)).reshape(n, k)


def rope_half_split(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    """Reference for :func:`rope_emb_forward` on one of q/k.

    Uses the *half-split* convention (rotate ``x[:d/2]`` against ``x[d/2:]``),
    which is what the kernel implements, and reads only the first ``d/2``
    entries of ``cos``/``sin`` — the tables are stored duplicated across the
    full head dim, and feeding the second half in would rotate twice.

    Args:
        x: ``[batch * seq_len, num_heads, head_dim]``.
        cos: ``[batch, seq_len, head_dim]``; only ``[..., : head_dim // 2]`` is read.
        sin: Same shape as ``cos``.
        batch_size: Leading dimension folded into ``x``'s token axis.
        seq_len: Tokens per sequence.

    Returns:
        Rotated copy of ``x``.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2

    c = cos[..., :half].reshape(batch_size * seq_len, 1, half).float()
    s = sin[..., :half].reshape(batch_size * seq_len, 1, half).float()

    x1 = x[..., :half].float()
    x2 = x[..., half : 2 * half].float()

    rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    if 2 * half < head_dim:  # odd head_dim: the tail rides along untouched
        rotated = torch.cat([rotated, x[..., 2 * half :].float()], dim=-1)
    return rotated.to(x.dtype)
