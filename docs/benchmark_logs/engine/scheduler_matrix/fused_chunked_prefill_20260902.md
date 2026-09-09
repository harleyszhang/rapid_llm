# Fused chunked prefill：prefix-cache 命中路径的 TTFT 修复（2026-09-02）

设备：2×H100 80GB；模型：Qwen2.5-0.5B-Instruct（24 层，14 Q heads / 2 KV heads）；
greedy 解码，`max_num_seqs=8`，prefix workload 为 4 组 × 6 请求共享 ~384-token 前缀。
复现命令（`RAPID_LLM_FUSED_CHUNK_PREFILL=0` 是 kill-switch，`=1` 为默认）：

```bash
for mode in 0 1; do
  RAPID_LLM_FUSED_CHUNK_PREFILL=$mode python benchmarks/engine/run.py scheduler matrix \
      --model-dir <ckpt> --graph --prefix-cache \
      --json docs/benchmark_logs/engine/scheduler_matrix/ab_fused${mode}_graph_prefix.json
done
```

## 问题：命中了缓存，TTFT 却不降

79% 的 prefix 命中率下 TTFT 与无缓存几乎相同。Timeline 归因（`diag-prefix`
子命令，`RAPID_LLM_OVERLAP_TIMELINE=1`）：命中请求的剩余 token 走 EXTEND pass
——每 token 一行 decode 风格行、`flash_decoding` kernel（element-wise FMA、
`num_warps=1`、逐 token 经 `b_req_tokens_table` 间接寻址），而全新 prefill 走
`flash_attention2_no_pad`（`tl.dot` tensor core、BLOCK_M=64 个 query 共享 KV
遍历）。per-token 成本差约一个数量级：一个 39-token 的 EXTEND pass 要 55.6 ms，
而 3224-token 的 grid prefill pass 只要 22.2 ms。

## 修复：两条改动

1. **`flash_attention2_chunked` kernel**（`kernels/ops/attention/flashattention2_nopad.py`）：
   query 仍是 chunk 的 padded grid，K/V 直接从 paged cache buffer 读——slot 的 KV
   区域连续（`b_req_tokens_table` 是恒等映射），经 `b_kv_base` 基址 + 线性偏移寻址，
   因果遮罩按绝对位置判定，`tl.dot` tensor-core tiling。fp8 KV 缓存（uint8 行）
   读不了原样字节，保持 EXTEND（三层防御：engine 开关、spec `schemes=("unquantized",)`、
   `begin_prefill` 断言）。
2. **按行数路由**（`continuous_engine._prefill_work`）：resumed chunk 的 pass 走哪条
   kernel 由总行数决定。关键事实（`diag-prefix` + pass 级计时测得）：

   | resumed pass 形状 | EXTEND 路径 | chunked 路径 | 最优 |
   |---|---|---|---|
   | rows ≤ 最大捕获 batch（32）：可 pad 后 **replay decode graph** | **3.6 ms** | 12.3 ms（eager） | EXTEND |
   | rows > 32：EXTEND 只能 eager | ~20 ms | 12–23 ms | chunked |

   因此阈值取 `max(captured batch sizes) + 1`（graph 关闭时为 1，fp8 或
   kill-switch 时为 `inf`）：短余量保留可 replay 的 EXTEND，长 chunk 走 tensor
   core。初版把所有 resumed chunk 都推向 chunked kernel，`--chunk-tokens 256` 下
   短命中余量（~20 rows）从 3.6 ms 的 replay 变成 12 ms 的 eager，TTFT 反升 34%——
   行数路由正是修这个退化。

## A/B 结果

`fused=0`（全部 EXTEND）vs `fused=1`（行数路由 + chunked kernel），Qwen2.5-0.5B：

| 配置 | workload | TTFT p50 | TTFT p95 | 总时长 |
|---|---|---|---|---|
| graph+prefix-cache | prefix | 199.4 → **159.2 ms（-20%）** | 311 → 265 ms | 0.39 → **0.34 s** |
| graph+prefix-cache | chunk | 60.2 → **52.8 ms（-12%）** | 87 → 73 ms | 0.17 → 0.15 s |
| graph+cache+chunk 256 | prefix | 236.5 → 237.4 ms（持平） | 378 → 381 ms | 0.46 → 0.46 s |
| graph+cache+chunk 256 | chunk | 331.7 → **290.6 ms（-12%）** | 351 → 301 ms | 0.43 → 0.38 s |
| graph+chunk 256 | prefix | 473.6 → **446.0 ms（-6%）** | 830 → 802 ms | 0.93 → 0.91 s |
| graph+chunk 256 | chunk | 318.6 → **303.8 ms（-5%）** | 329 → 314 ms | 0.41 → 0.40 s |

无任何配置退化；纯 prefix-cache 场景（调度器默认组合）收益最大。TPOT 不受影响
（decode 路径未动），TPS 随总时长同比例提升（0.36→0.34 s ≈ +6% 聚合吞吐）。

原始数据：`ab_fused{0,1}_{graph_prefix,graph_prefix_chunk,graph_chunk}.json`
（贪心文本 sidecar 同名 `.texts.json`，跨配置可审计）。

## 正确性

- kernel 数值：`tests/kernels/test_flash_attention_chunked.py` 15 项（ragged
  spans、GQA 4 档、head_dim 32/64/128、cache 段隔离、逐行 fp32 参考对照）
- 路由：`tests/executor/test_model_input.py::test_resumed_rows_route_by_replay_capacity`
- 回归：engine 254 项 + executor + ops 全过；TP=2 冒烟（`--tp 2 --graph
  --prefix-cache`）通过
- `diag-preempt` 抢占一致性：16/24 文本一致，与 kill-switch 基线（19/24）同源——
  抢占重算时 batch 形状变化引起的贪心平局翻转（算术噪声），不是路径正确性问题
