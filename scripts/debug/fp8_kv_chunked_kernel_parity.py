"""内核级隔离: chunked vs extend(packed decode) 在同一份 fp8 cache 上。

复刻引擎 long-a 的最后一块 chunk 场景 (P=3328, C=173, Qwen2.5-1.5B 形状),
四条路径两两对比:

  chunked-fp8  flash_attention2_chunked(q, k8, v8)
  extend-fp8   flash_decoding(q, k8, v8)        (packed stage1, uint8 自动识别)
  chunked-bf16 / extend-bf16  同上, 未量化 cache

参考实现: torch fp32, K/V 用 widen 后的"量化真值"(与内核看到的输入一致)。

用法:
    python -m scripts.debug.fp8_kv_chunked_kernel_parity

结果与解读见 docs/quantization.md "分块预填充下的 fp8 KV cache 数值实测",
原始输出归档于 docs/benchmark_logs/quantization/fp8_kv_chunked_20260911/kernel_parity.txt。
"""

import math

import torch

from rapid_llm.kernels.ops.attention.flashattention2_nopad import flash_attention2_chunked
from rapid_llm.kernels.ops.attention.flashdecoding import flash_decoding
from rapid_llm.modules.quantization.utils import quantize_fp8_per_tensor

PREFIX, CHUNK = 3328, 173
H, KVH, D = 12, 2, 128
TOTAL = PREFIX + CHUNK
GROUPS = H // KVH


def reference(q, k_true, v_true, prefix, chunk):
    """fp32 逐行 causal attention, GQA repeat_interleave。"""
    scale = 1.0 / math.sqrt(D)
    k = k_true.float().repeat_interleave(GROUPS, dim=1)  # [T, H, D]
    v = v_true.float().repeat_interleave(GROUPS, dim=1)
    out = torch.zeros(chunk, H, D, device=q.device, dtype=torch.float32)
    for m in range(chunk):
        row = q[m].float()
        scores = torch.einsum("hd,thd->ht", row, k[: prefix + m + 1]) * scale
        w = torch.softmax(scores, dim=-1)
        out[m] = torch.einsum("ht,thd->hd", w, v[: prefix + m + 1])
    return out


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"

    k16 = torch.randn(TOTAL, KVH, D, device=dev, dtype=torch.bfloat16) * 0.3
    v16 = torch.randn(TOTAL, KVH, D, device=dev, dtype=torch.bfloat16) * 0.3
    q = torch.randn(CHUNK, H, D, device=dev, dtype=torch.bfloat16) * 0.3
    k8 = quantize_fp8_per_tensor(k16)
    v8 = quantize_fp8_per_tensor(v16)
    scale = 1.0 / math.sqrt(D)

    b_start_loc = torch.zeros(1, dtype=torch.int64, device=dev)
    b_kv_base = torch.zeros(1, dtype=torch.int64, device=dev)
    b_prefix_len = torch.tensor([PREFIX], dtype=torch.int64, device=dev)
    b_seq_len = torch.tensor([TOTAL], dtype=torch.int64, device=dev)

    table = torch.arange(TOTAL, dtype=torch.int32, device=dev).unsqueeze(0)
    b_req_idx = torch.zeros(CHUNK, dtype=torch.int32, device=dev)
    b_seq_len_d = torch.arange(PREFIX + 1, TOTAL + 1, dtype=torch.int32, device=dev)

    def chunked(k_cache, v_cache):
        return flash_attention2_chunked(
            q, k_cache, v_cache, scale, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, CHUNK
        ).float()

    def extend(k_cache, v_cache):
        return flash_decoding(
            q, k_cache, v_cache, scale, table, b_req_idx, b_seq_len_d, TOTAL
        ).float()

    kt8 = k8.view(torch.float8_e4m3fn).float()
    vt8 = v8.view(torch.float8_e4m3fn).float()
    ref8 = reference(q, kt8, vt8, PREFIX, CHUNK)
    ref16 = reference(q, k16, v16, PREFIX, CHUNK)

    out_c8, out_e8 = chunked(k8, v8), extend(k8, v8)
    out_c16, out_e16 = chunked(k16, v16), extend(k16, v16)

    def rel(a, b):
        return (a - b).abs().max().item() / b.abs().max().item()

    print(f"P={PREFIX} C={CHUNK}  |ref|max={ref8.abs().max():.4f}")
    print("\n--- 对 torch 参考 (fp8 真值) ---")
    print(f"  chunked-fp8 vs ref : {rel(out_c8, ref8):.2e}")
    print(f"  extend-fp8  vs ref : {rel(out_e8, ref8):.2e}")
    print("\n--- 对 torch 参考 (bf16 真值) ---")
    print(f"  chunked-bf16 vs ref: {rel(out_c16, ref16):.2e}")
    print(f"  extend-bf16  vs ref: {rel(out_e16, ref16):.2e}")
    print("\n--- 两内核互比 (读同一输入) ---")
    print(f"  fp8 : chunked vs extend: {rel(out_c8, out_e8):.2e}")
    print(f"  bf16: chunked vs extend: {rel(out_c16, out_e16):.2e}")
    print("\n--- 量化本身带来的漂移 (同内核) ---")
    print(f"  chunked: fp8 vs bf16: {rel(out_c8, out_c16):.2e}")
    print(f"  extend : fp8 vs bf16: {rel(out_e8, out_e16):.2e}")

    # 逐行看最后一行 (引擎里生成首个 token 的那行) 的差异
    print("\n最后一行 (row 172) 的逐头最大绝对差:")
    for name, a, b in (
        ("chunked-fp8 vs extend-fp8", out_c8, out_e8),
        ("chunked-bf16 vs extend-bf16", out_c16, out_e16),
        ("chunked-fp8 vs chunked-bf16", out_c8, out_c16),
        ("extend-fp8 vs extend-bf16", out_e8, out_e16),
    ):
        d = (a[-1] - b[-1]).abs().max().item()
        print(f"  {name:<32} {d:.2e}")


if __name__ == "__main__":
    main()
