"""CPU kernels: plain-PyTorch implementations of every op the dispatcher targets.

Two roles in one file. At inference this is the CPU backend — the ``cpu/*``
spec rows in :mod:`rapid_llm.kernels.backend.cpu_specs` point here, and the
functions keep their CUDA counterparts' signatures so dispatch can swap them
in. In tests it is the golden baseline: the eager PyTorch reference the GPU
kernels' outputs are pinned against (``tests/cpu``), so clarity outranks speed
throughout — per-request loops, eager dequant, no Triton.

Usage:
    from rapid_llm.kernels.backend.cpu import fused_moe, skip_rmsnorm
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def skip_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    values = x.float()
    if residual is None:
        residual = x
    else:
        values = values + residual.float()
        residual.copy_(values)
    normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return normed.to(x.dtype) * weight, residual


fused_add_rmsnorm = skip_rmsnorm


def fused_allreduce_rmsnorm(partial, residual, weight, eps=1e-5):
    from ...distributed.parallel_state import tensor_model_parallel_all_reduce

    return skip_rmsnorm(tensor_model_parallel_all_reduce(partial), residual, weight, eps)


def qk_rmsnorm(q, k, q_weight, k_weight, eps=1e-5):
    return skip_rmsnorm(q, None, q_weight, eps)[0], skip_rmsnorm(k, None, k_weight, eps)[0]


@torch.no_grad()
def rope_emb_forward(q, k, cos, sin):
    if q.shape[-1] % 2 or cos.shape != sin.shape or cos.numel() != q.shape[0] * q.shape[-1]:
        raise ValueError("RoPE requires matching cos/sin tables and an even head dimension")
    half = q.shape[-1] // 2
    c = cos.reshape(q.shape[0], 1, -1)[..., :half]
    s = sin.reshape(q.shape[0], 1, -1)[..., :half]
    for tensor in (q, k):
        left, right = tensor[..., :half].float(), tensor[..., half:].float()
        rotated = torch.cat((left * c - right * s, right * c + left * s), dim=-1)
        tensor.copy_(rotated)
    return q, k


def vocab_parallel_embedding(input_ids, weight, shard_start, local_vocab):
    ids = input_ids.reshape(-1) - shard_start
    owned = (ids >= 0) & (ids < local_vocab)
    return F.embedding(ids.clamp(0, local_vocab - 1), weight) * owned.unsqueeze(-1)


def update_kv_buffer(k, v, select_index, kv_buffer):
    if k.shape != v.shape or k.shape != (
        select_index.numel(),
        kv_buffer.shape[1] // 2,
        kv_buffer.shape[2],
    ):
        raise ValueError("K/V shapes must match the selected cache rows")
    kv_buffer[select_index.long()] = torch.cat((k, v), dim=1)


def update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index):
    if req_to_token_indexs.ndim != 2 or any(
        t.ndim != 1 for t in (b_req_idx, b_seq_len, select_index)
    ):
        raise ValueError("expected a 2-D token table and 1-D indices")
    if not b_req_idx.shape == b_seq_len.shape == select_index.shape:
        raise ValueError("request, length and cache indices must have the same shape")
    if torch.any(b_seq_len <= 0):
        raise ValueError("sequence lengths must be positive")
    req_to_token_indexs[b_req_idx.long(), b_seq_len.long() - 1] = select_index.to(
        req_to_token_indexs.dtype
    )


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, prefix: int | None = None
) -> torch.Tensor:
    if q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    groups = q.shape[1] // k.shape[1]
    q = q.transpose(0, 1).float()
    k = k.repeat_interleave(groups, dim=1).transpose(0, 1).float()
    v = v.repeat_interleave(groups, dim=1).transpose(0, 1).float()
    mask = None
    if prefix is not None:
        mask = torch.arange(k.shape[1])[None, :] <= torch.arange(q.shape[1])[:, None] + prefix
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale).transpose(0, 1)


def flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len):
    out = torch.zeros_like(q)
    for start, length in zip(b_start_loc.tolist(), b_seq_len.tolist(), strict=True):
        if length:
            rows = slice(start, start + length)
            out[rows] = _attention(q[rows], k[rows], v[rows], sm_scale, prefix=0).to(q.dtype)
    return out


def flash_attention2_chunked(
    q, k_cache, v_cache, sm_scale, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, max_chunk_len
):
    out = torch.zeros_like(q)
    for start, base, prefix, length in zip(
        b_start_loc.tolist(),
        b_kv_base.tolist(),
        b_prefix_len.tolist(),
        b_seq_len.tolist(),
        strict=True,
    ):
        if length > prefix:
            rows = slice(start, start + length - prefix)
            cached = slice(base, base + length)
            out[rows] = _attention(q[rows], k_cache[cached], v_cache[cached], sm_scale, prefix).to(
                q.dtype
            )
    return out


def flash_decoding(
    q,
    k_cache,
    v_cache,
    qk_scale,
    b_req_tokens_table,
    b_req_idx,
    b_seq_len,
    max_actual_seq_len,
    k_scale=1.0,
    v_scale=1.0,
):
    out = torch.zeros_like(q)
    for row, (slot, length) in enumerate(zip(b_req_idx.tolist(), b_seq_len.tolist(), strict=True)):
        if length <= 0:
            continue
        ids = b_req_tokens_table[slot, :length].long()
        k, v = k_cache[ids], v_cache[ids]
        if k.dtype == torch.uint8:
            k = k.view(torch.float8_e4m3fn).float() * k_scale
            v = v.view(torch.float8_e4m3fn).float() * v_scale
        out[row : row + 1] = _attention(q[row : row + 1], k, v, qk_scale).to(q.dtype)
    return out


def _dequant_weight(weight, scales, zeros=None, group_n=1, group_k=0):
    if weight.dtype == torch.int32:
        shifts = torch.arange(8) * 4
        weight = ((weight.unsqueeze(-1) >> shifts) & 15).flatten(-2)
    elif weight.dtype == torch.uint8 and zeros is None:
        weight = weight.view(torch.float8_e4m3fn)
    elif weight.dtype == torch.uint8:
        weight = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    values = weight.float()
    if scales is None:
        if not weight.is_floating_point():
            raise ValueError("quantized weights require scales")
        return values
    n, k = values.shape[-2:]
    rows = torch.arange(n) // max(group_n, 1)
    cols = torch.arange(k) // (group_k or k)
    scale = scales[..., rows[:, None], cols[None, :]].float()
    if zeros is not None:
        values = values - zeros[..., rows[:, None], cols[None, :]].float()
    return values * scale


def linear(
    scheme: str,
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
    if scheme == "unquantized":
        return F.linear(x, weight, bias)
    if scheme in {"w8a8_fp8", "w8a8_int8"}:
        values = x.float()
        limit = 448.0 if scheme == "w8a8_fp8" else 127.0
        scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-12) / limit
        quantized = (values / scale).clamp(-limit, limit)
        if scheme == "w8a8_fp8":
            quantized = quantized.to(torch.float8_e4m3fn).float()
        else:
            quantized = quantized.round()
        x_values = quantized * scale
        values = _dequant_weight(weight, weight_scale, weight_zeros, group_n, group_k)
        return F.linear(x_values, values, None if bias is None else bias.float()).to(x.dtype)
    if scheme not in {"fp8", "blockwise_int8", "gptq_int8", "awq", "gptq"}:
        raise NotImplementedError(f"CPU linear does not yet support {scheme!r}")
    values = _dequant_weight(weight, weight_scale, weight_zeros, group_n, group_k)
    return F.linear(x.float(), values, None if bias is None else bias.float()).to(x.dtype)


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale=None,
    w2_scale=None,
    w1_zeros=None,
    w2_zeros=None,
    group_n=0,
    group_k=0,
    swiglu_limit=float("inf"),
    mxfp4=False,
    activation_scheme: str | None = None,
) -> torch.Tensor:
    def project(x, weights, scales, zeros, expert):
        weight = weights[expert]
        scale = None if scales is None else scales[expert]
        zero = None if zeros is None else zeros[expert]
        if activation_scheme is not None:
            # W8A8 experts use per-channel scales; the down projection has a different K.
            return linear(activation_scheme, x, weight, weight_scale=scale, group_n=1, group_k=0)
        if mxfp4:
            from ...modules.quantization.mxfp4 import dequant_mxfp4

            weight = dequant_mxfp4(repack_int4_experts(weight), scale)
        else:
            weight = _dequant_weight(weight, scale, zero, group_n, group_k)
        return F.linear(x.float(), weight).to(x.dtype)

    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        rows, choices = torch.where(topk_ids == expert)
        if not rows.numel():
            continue
        gate, up = project(hidden_states[rows], w1, w1_scale, w1_zeros, expert).chunk(2, dim=-1)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        values = project(
            (F.silu(gate.float()) * up.float()).to(hidden_states.dtype),
            w2,
            w2_scale,
            w2_zeros,
            expert,
        )
        out.index_add_(0, rows, values.float() * topk_weights[rows, choices, None])
    return out.to(hidden_states.dtype)


def fused_moe_w8a8_fp8(hidden_states, w1, w2, topk_weights, topk_ids, **kwargs):
    return fused_moe(
        hidden_states, w1, w2, topk_weights, topk_ids, activation_scheme="w8a8_fp8", **kwargs
    )


def fused_moe_w8a8_int8(hidden_states, w1, w2, topk_weights, topk_ids, **kwargs):
    return fused_moe(
        hidden_states, w1, w2, topk_weights, topk_ids, activation_scheme="w8a8_int8", **kwargs
    )


def mla_decode(
    q, kv_cache, block_table, cache_seqlens, *, max_seq_len, sm_scale=1.0, qk_rope_head_dim=64
):
    latent_dim = kv_cache.shape[-1] - qk_rope_head_dim
    page_size = kv_cache.shape[1]
    out = q.new_zeros((*q.shape[:2], latent_dim))
    for row, length in enumerate(cache_seqlens.tolist()):
        if length <= 0:
            continue
        pages = block_table[row, : (length + page_size - 1) // page_size].long()
        latent = kv_cache[pages].flatten(0, 1)[:length].float()
        probs = (q[row].float() @ latent.T * sm_scale).softmax(-1)
        out[row] = (probs @ latent[:, :latent_dim]).to(q.dtype)
    return out


def mla_prefill(
    q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len, max_seq_len
):
    out = q_nope.new_zeros((*q_nope.shape[:2], w_uv.shape[-1]))
    for start, length in zip(b_start_loc.tolist(), b_seq_len.tolist(), strict=True):
        if not length:
            continue
        rows = slice(start, start + length)
        keys = torch.einsum("tl,hld->thd", c_kv[rows], w_uk)
        values = torch.einsum("tl,hld->thd", c_kv[rows], w_uv)
        queries = torch.cat((q_nope[rows], q_pe[rows]), -1)
        keys = torch.cat((keys, k_pe[rows, None].expand(-1, keys.shape[1], -1)), -1)
        out[rows] = _attention(queries, keys, values, sm_scale, prefix=0).to(out.dtype)
    return out


def repack_int4_experts(packed):
    if packed.dtype != torch.int32:
        raise ValueError("packed weights must be int32")
    return ((packed.unsqueeze(-1) >> (torch.arange(4) * 8)) & 255).to(torch.uint8).flatten(-2)


def unpack_int8_experts(packed):
    return repack_int4_experts(packed).view(torch.int8)
