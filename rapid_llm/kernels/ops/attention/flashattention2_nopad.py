"""FlashAttention-2 (Triton) for variable-length, unpadded prefill batches.

One kernel iterates query blocks against KV blocks of each sequence,
using the ``B_Start_Loc`` / ``B_Seqlen`` row tables to skip padding — so
a ragged batch needs no tensor padding at all.

The sibling :func:`flash_attention2_chunked` answers the same question for
a chunk resuming on cached K/V: queries are the chunk's rows, keys and
values come from the paged cache, and the causal mask is expressed in
absolute positions. That cache may hold fp8-e4m3 bytes, which the chunked
kernel dequantises on load the way flash-decoding does.

Usage:
    from rapid_llm.kernels import flash_attention2_no_pad
"""

import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd

from ..quantization.w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3

#: ``exp(x) == exp2(x * log2(e))`` — the kernel takes the exp2 route.
_LOG2E = 1.4426950408889634

configs_tma = [
    triton.Config({"BLOCK_M_SIZE": BM, "BLOCK_N_SIZE": BN}, num_stages=stages, num_warps=warps)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for warps in [4, 8, 16]
    for stages in [2, 3, 4, 6]
]


def keep_tma(conf):
    BLOCK_M_SIZE = conf.kwargs["BLOCK_M_SIZE"]
    BLOCK_N_SIZE = conf.kwargs["BLOCK_N_SIZE"]
    return not (
        torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M_SIZE * BLOCK_N_SIZE < 128 * 128
        and conf.num_warps == 8
    )


# key 参数列表(['B_Seqlen', 'HEAD_DIM'])的值会直接影响最佳配置的选择，因为不同的输入尺寸或问题规模可能需要不同的内核调度策略。
# @triton.autotune(
#     configs=list(filter(keep_tma, configs_tma)),
#     key=['B_Seqlen', 'HEAD_DIM']
# )
@triton.jit
def flash_attention2_nopad_kernel(
    Q,
    K,
    V,
    O,
    B_Start_Loc,
    B_Seqlen,
    sm_scale,
    heads,
    num_kv_groups,  # group of kv heads
    stride_q_bs,
    stride_q_heads,
    stride_q_dim,  # Q 的 strides
    stride_k_bs,
    stride_k_heads,
    stride_k_dim,  # K 的 strides
    stride_v_bs,
    stride_v_heads,
    stride_v_dim,  # V 的 strides
    stride_o_bs,
    stride_o_heads,
    stride_o_dim,
    HEAD_DIM: tl.constexpr,  # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr,  # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr,  # n_size dimension
):
    """
    flashattentionv1 内核实现, 支持 nopad 计算, 输入为 3 维张量
    """
    block_m_idx = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch_idx = cur_bh // heads
    cur_head_idx = cur_bh % heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups

    # 计算当前批次的序列长度和请求序列的起始位置
    cur_seq_len = tl.load(B_Seqlen + cur_batch_idx)
    # cur_seq_start_loc = tl.load(b_req_tokens_table + cur_batch_idx * stride_req_to_tokens_b)
    cur_seq_start_loc = tl.load(B_Start_Loc + cur_batch_idx)

    block_start_loc = block_m_idx * BLOCK_M_SIZE  # 计算当前 block 的起始和结束索引

    offs_n = tl.arange(0, BLOCK_N_SIZE)  # head_dim 维度偏移
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M_SIZE)

    # Compute offsets for the first block on matrix Q K V Output
    q_offs = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    k_offs = (
        offs_n[None, :] * stride_k_bs
        + cur_kv_head_idx * stride_k_heads
        + offs_d[:, None] * stride_k_dim
    )
    v_offs = (
        offs_n[:, None] * stride_v_bs
        + cur_kv_head_idx * stride_v_heads
        + offs_d[None, :] * stride_v_dim
    )

    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    # 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, HEAD_DIM), dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_seq_len)

    # 每次循环按 BLOCK_N_SIZE 来处理 K, V 的列（即 key/value 的序列维度）。
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)
        # 计算 qk^t
        k = tl.load(
            k_ptrs + (cur_seq_start_loc + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )

        qk = tl.dot(q, k)

        # 应用因果遮罩, 下三角矩阵 causal mask
        casual_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(casual_mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # 求 qk 的最大值
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)  # qk - m_ij[:, None]更新为安全的 qk 分子项
        d_ij = tl.sum(p, 1)  # 1d vector

        # -- 更新归一化项 d_new
        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij

        # -- update output accumulator --
        acc = acc * alpha[:, None]  # acc scaling

        # compute O = PV
        v = tl.load(
            v_ptrs + (cur_seq_start_loc + start_n) * stride_v_bs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        # update the normalizer (l and d) for next iteration
        m_i = m_ij

    acc = acc / d_i[:, None]
    off_o = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_o_bs
        + cur_head_idx * stride_o_heads
        + offs_d[None, :] * stride_o_dim
    )
    out_ptrs = O + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_seq_len)


# --------------------------------------
# Flashattention NoPad 实现（Triton 内核）
# --------------------------------------
@torch.no_grad()
@custom_fwd(device_type="cuda")
def flash_attention2_no_pad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    b_start_loc,
    b_seq_len,
    max_seq_len,
):
    """Causal prefill attention over a packed (no-pad) ragged batch.

    Args:
        q: ``[total_tokens, num_heads, head_dim]`` packed query rows.
        k: ``[total_kv, num_kv_heads, head_dim]`` freshly projected keys; ``v``
            has the same layout. Grouped-query attention is handled inside the
            kernel from the head-count ratio.
        v: Value rows, laid out like ``k``.
        sm_scale: Plain softmax scale, ``1 / sqrt(head_dim)``.
        b_start_loc: ``[batch]`` first packed row of each sequence.
        b_seq_len: ``[batch]`` true length of each sequence.
        max_seq_len: Longest sequence, sizing the query-block grid.

    Returns:
        ``[total_tokens, num_heads, head_dim]`` attention output. fp32 inputs
        are not supported (``custom_fwd`` casts them down).
    """
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    BLOCK_M, BLOCK_N, num_warps, num_stages = _nopad_blocks(max_seq_len, HEAD_DIM, q.dtype)

    num_kv_groups = q.shape[1] // k.shape[1]  # num_q_heads // num_k_heads
    grid = (triton.cdiv(max_seq_len, BLOCK_M), batchs * n_heads, 1)

    flash_attention2_nopad_kernel[grid](
        q,
        k,
        v,
        output,
        b_start_loc,
        b_seq_len,
        # Fold log2(e) here, once, instead of trusting every caller to know
        # that this kernel's softmax runs on exp2.
        sm_scale * _LOG2E,
        n_heads,
        num_kv_groups,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M_SIZE=BLOCK_M,
        BLOCK_N_SIZE=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


@triton.jit
def flash_attention2_chunked_kernel(
    Q,
    K,
    V,
    O,
    B_Start_Loc,  # [batch] 每序列在(带 padding 的)query grid 里的行偏移
    B_KV_Base,  # [batch] 每序列 KV 第 0 行所在的 cache 绝对行号
    B_Prefix_Len,  # [batch] 前缀长度: KV 中先于本 chunk 落地的行数
    B_Seqlen,  # [batch] 总长 = prefix + chunk
    sm_scale,
    k_scale,  # fp8 KV cache 反量化标量: 与 qk 同为线性因子, 折在 sm_scale 上
    v_scale,  # 同, 折在输出归一化一步
    heads,
    num_kv_groups,
    stride_q_bs,
    stride_q_heads,
    stride_q_dim,  # Q 的 strides
    stride_k_bs,
    stride_k_heads,
    stride_k_dim,  # K 的 strides
    stride_v_bs,
    stride_v_heads,
    stride_v_dim,  # V 的 strides
    stride_o_bs,
    stride_o_heads,
    stride_o_dim,
    HEAD_DIM: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    KV_FP8: tl.constexpr,  # cache 以 e4m3 字节存储(uint8 容器)
):
    """Causal attention for chunk queries against the paged KV cache.

    Query 行来自本 chunk 的 grid(padding 布局同 prefill), K/V 则直接从
    cache buffer 读: 行 ``[0, prefix)`` 是更早的 chunk 或 prefix-cache 拷贝
    落下的, 行 ``[prefix, prefix + chunk)`` 刚由本 pass 的 KV-write 写入。
    每个 slot 的 cache 行连续, 所以一次基址加法就完成寻址, 无逐 token 表
    间接——这是它比逐行 decode kernel 快一个量级的关键。因果按绝对位置
    判定: query ``prefix + m`` 只看 KV 行 ``<= prefix + m``。

    ``KV_FP8`` 时 cache 行是 e4m3 字节: 每块 K/V 在进 ``tl.dot`` 前先升到
    q 的 dtype(e4m3 的 3 bit 尾数在 fp16/bf16 下精确), 逐张量的反量化标量
    是常数, 从内层循环提到 qk 与输出两步上。
    """
    block_m_idx = tl.program_id(0)  # chunk 内 query 块索引
    cur_bh = tl.program_id(1)
    cur_batch_idx = cur_bh // heads
    cur_head_idx = cur_bh % heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups

    cur_seq_len = tl.load(B_Seqlen + cur_batch_idx)
    cur_prefix = tl.load(B_Prefix_Len + cur_batch_idx)
    cur_chunk_len = cur_seq_len - cur_prefix
    cur_q_start = tl.load(B_Start_Loc + cur_batch_idx)
    cur_kv_base = tl.load(B_KV_Base + cur_batch_idx)

    block_start_loc = block_m_idx * BLOCK_M_SIZE  # chunk 内的 query 块起始
    offs_n = tl.arange(0, BLOCK_N_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M_SIZE)

    # Query 行住在带 padding 的 grid 展平里, 短 chunk 的尾部列被 mask 掉。
    q_offs = (
        (cur_q_start + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_chunk_len, other=0.0)

    # K 以 [D, N] 转置布局加载, 供 tl.dot(q, k) 走 tensor core; V 保持 [N, D]。
    k_offs = (
        offs_n[None, :] * stride_k_bs
        + cur_kv_head_idx * stride_k_heads
        + offs_d[:, None] * stride_k_dim
    )
    v_offs = (
        offs_n[:, None] * stride_v_bs
        + cur_kv_head_idx * stride_v_heads
        + offs_d[None, :] * stride_v_dim
    )
    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, HEAD_DIM), dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_chunk_len, 1, 0)
    # 本 query 块要读到的最远 KV 行(绝对) = prefix + 块内最后一个真实 query 位置 + 1。
    block_q_end = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_chunk_len)
    kv_end = cur_prefix + block_q_end

    for start_n in range(0, block_mask * kv_end, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)
        # KV 绝对行 start_n + offs_n, 换算成 cache 行号: 基址一次加法,
        # 列偏移由 k_ptrs/v_ptrs 自带的 offs_n stride 补齐。
        n_mask = (start_n + offs_n) < kv_end
        kv_shift = (cur_kv_base + start_n) * stride_k_bs
        k = tl.load(k_ptrs + kv_shift, mask=n_mask[None, :], other=0.0)

        if KV_FP8:
            # 屏蔽列的字节为 0 -> 0.0, 与 fp16 路径的 other=0.0 语义一致;
            # bit-trick 输出欠 2**8, 已由调用方折入 k_scale。
            k = dequant_fp8e4m3(k).to(q.dtype)

        qk = tl.dot(q, k)

        # 因果遮罩: query 的绝对位置是 prefix + m, KV 的绝对位置是 n。
        casual_mask = (cur_prefix + offs_m[:, None]) >= (start_n + offs_n[None, :])
        qk = tl.where(casual_mask, qk * (sm_scale * k_scale), -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        d_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij

        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + (cur_kv_base + start_n) * stride_v_bs, mask=n_mask[:, None], other=0.0)
        if KV_FP8:
            v = dequant_fp8e4m3(v).to(q.dtype)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        m_i = m_ij

    acc = acc * v_scale / d_i[:, None]
    off_o = (
        (cur_q_start + offs_m[:, None]) * stride_o_bs
        + cur_head_idx * stride_o_heads
        + offs_d[None, :] * stride_o_dim
    )
    tl.store(O + off_o, acc, mask=offs_m[:, None] < cur_chunk_len)


def _nopad_blocks(max_len: int, head_dim: int, dtype: torch.dtype) -> tuple[int, int, int, int]:
    """Tile shape for the nopad kernels: tuned record if present, heuristic else.

    Shared by :func:`flash_attention2_no_pad` and :func:`flash_attention2_chunked`
    so a persisted tuning for one cannot silently disagree with the other.
    """
    from ...dispatcher.autotune import get_best_config

    dtype_key = "bf16" if dtype == torch.bfloat16 else "fp16"
    tuned = get_best_config("flash_attn_nopad", m=max_len, n=head_dim, k=head_dim, dtype=dtype_key)
    if tuned is not None:
        return (
            tuned.get("BLOCK_M_SIZE", 64),
            tuned.get("BLOCK_N_SIZE", 64),
            tuned.get("num_warps", 4),
            tuned.get("num_stages", 1),
        )
    # For Ampere Architecture, 3090ti, set BLOCK_M 128.
    return 64, 64, 4 if head_dim <= 64 else 8, 1


@torch.no_grad()
@custom_fwd(device_type="cuda")
def flash_attention2_chunked(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sm_scale,
    b_start_loc,
    b_kv_base,
    b_prefix_len,
    b_seq_len,
    max_chunk_len,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Causal prefill for chunks resuming on K/V that is already cached.

    The mirror of :func:`flash_attention2_no_pad` for the resumed half of
    chunked prefill: same padded query grid, same ``tl.dot`` tiling, but the
    keys and values are the sequence's own cache rows ``[0, seq_len)`` —
    prefix included — read straight out of the paged buffer. A resumed chunk
    therefore pays tensor-core prefill prices instead of one decode-style
    row per token, which the extend path charges and which costs roughly an
    order of magnitude more per token.

    An fp8 cache is served by the same kernel: ``uint8`` rows are e4m3 bytes
    that widen to ``q``'s dtype on load, so a quantised cache no longer costs
    a resumed chunk its grid pass.

    Args:
        q: ``[total_rows, num_heads, head_dim]`` packed query rows of the
            chunk grid, padding columns included (their rows are masked out).
        k_cache: ``[max_tokens, num_kv_heads, head_dim]`` the K half of the
            paged buffer; ``v_cache`` the same for V. Must hold this pass's
            rows already: the KV write precedes attention on the same stream.
            ``uint8`` rows are read as fp8-e4m3.
        sm_scale: Plain softmax scale, ``1 / sqrt(head_dim)``.
        b_start_loc: ``[batch]`` first packed query row of each sequence
            (``i * grid_width`` — the padded layout prefill uses).
        b_kv_base: ``[batch]`` cache row holding each sequence's KV row 0.
        b_prefix_len: ``[batch]`` cached rows preceding this chunk.
        b_seq_len: ``[batch]`` total length once the chunk lands.
        max_chunk_len: Widest chunk, sizing the query-block grid.
        k_scale: Dequantisation scale of an fp8 key cache; ignored for fp16.
        v_scale: Same for the value cache.

    Returns:
        ``[total_rows, num_heads, head_dim]`` attention output; rows past a
        sequence's chunk length carry garbage and are never read.
    """
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    kv_fp8 = k_cache.dtype == torch.uint8
    if kv_fp8:
        # dequant_fp8e4m3 的 bit-trick 输出欠 2**8; 折进 scale, kernel 内免补偿
        k_scale = k_scale * FP8_E4M3_BIT_TRICK_SCALE
        v_scale = v_scale * FP8_E4M3_BIT_TRICK_SCALE

    BLOCK_M, BLOCK_N, num_warps, num_stages = _nopad_blocks(max_chunk_len, HEAD_DIM, q.dtype)
    num_kv_groups = q.shape[1] // k_cache.shape[1]  # num_q_heads // num_k_heads
    grid = (triton.cdiv(max_chunk_len, BLOCK_M), batchs * n_heads, 1)

    flash_attention2_chunked_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        b_start_loc,
        b_kv_base,
        b_prefix_len,
        b_seq_len,
        sm_scale * _LOG2E,
        k_scale,
        v_scale,
        n_heads,
        num_kv_groups,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M_SIZE=BLOCK_M,
        BLOCK_N_SIZE=BLOCK_N,
        KV_FP8=kv_fp8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
