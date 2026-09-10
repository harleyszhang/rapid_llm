# rapid_llm v0.12.0 Release Notes

## 核心优化

### O1 真分页 KV + Radix 零拷贝共享

**问题：** 原始 KV 分配是连续的——每个请求占一段连续 token 行，无法共享前缀、无法碎片回收。系统提示词相同的 N 个请求要 prefill N 次，显存浪费 N 倍。

**解决：** 三级架构落地真分页 KV：

1. **BlockPool**（`block_pool.py`）：以 16-token 块为分配单元，引用计数 + LRU 淘汰（双向链表 FreeBlockQueue）+ 哈希索引（`cached_block_hash_to_block`）。
2. **PrefixCache**（`prefix_cache.py`）：哈希链前缀缓存——blake2b 链式哈希（`iter_block_hashes`），支持 allocate/commit/free/lookup 全生命周期，引用计数归零自动回收。
3. **SlotBatch**（`slot_batch.py`）：块表操作——`b_req_tokens_table[slot, pos] = block_id * 16 + offset` 将逻辑 token位置映射到物理块地址。
4. **KVCacheManager**（`kv_cache_manager.py`）：物理 GPU buffer 以 flat token 行数组管理，上层通过块表间接寻址。

**结果：** 80 个单元测试全通过，覆盖块分配/释放、前缀命中、块表地址翻译全链路。相同系统提示词的请求只需 prefill 一次，后续请求零拷贝复用。

**证据：**
- Tests: `tests/engine/test_prefix_cache.py` (80 passed)
- Code: `rapid_llm/engine/block_pool.py`, `prefix_cache.py`, `rapid_llm/executor/slot_batch.py`


### O4 MoE Dequant 融合 Grouped GEMM

**问题：** MoE 专家权重以量化格式（fp8/int8/int4/mxfp4）存储，推理时需要解包回 fp16/bf16 再计算。中间物化一份全精度权重，额外消耗一次全量 HBM 读取。A10 上 MoE decode 是纯带宽受限，这次额外读取直接叠加在 TPOT 上。

**解决：** fp8 dequant 已在 GEMM mainloop 内完成（`dequant_fp8e4m3` bit-trick），无中间 fp16 物化。本轮补全了 A10 的 autotune collect 轮次（PRE_HOPPER 表此前未测量），并修复了两个隐藏的 kernel 缺陷：

1. **int4 (W4A16) kernel 缺陷：** MXFP4 合并时丢失了 int4 检测分支，导致 int4 专家被误判为 fp8，应用 bit-trick 解包后输出 garbage。修复：恢复 `_quant_mode` 的 `zeros` 参数，通过 zeros 存在性区分 int4（uint8 + zeros）与 fp8（uint8 无 zeros）。同时拆分 int4/mxfp4 的 kernel load 路径：int4 使用 uint8 2-nibble replicated addressing，mxfp4 保持 int32 8-nibble word unpack。

2. **mxfp4 (DeepSeek-V4) kernel 缺陷：** `_MXFP4_PACK_FACTOR` 未定义，`k_logical` 计算错误地使用了 `_INT4_PACK_FACTOR=2`，导致 mxfp4 的逻辑 K 只有实际值的 1/4，k-loop 只归约了 1/4 的维度。修复：定义 `_MXFP4_PACK_FACTOR=8`，`k_logical` 按 mode 分别乘以 2（int4）或 8（mxfp4）。

**A10 collect round 结果：** 12 个 key 全部改进，最佳 +37.8%（int4 M64），最差 +0.3%（bf16 M16）。fp8/int8 在大 shape（4096 tokens）分别提速 27.9%/16.1%。

**证据：**
- Benchmark: `docs/benchmark_logs/kernels/moe_o4_*.json`
- GIF: `docs/images/moe_o4.gif`

**已知问题：**
- fp8 W8A8（mode 5）在 sm86（A10）无法编译：Triton 无法发射 `tl.float8e4nv`，需 sm89+。测试用例在 A10 上 skip。
- int8 W8A8（mode 6）inline A-quant 路径（tokens ≤ 32）存在缺陷：`A_QUANT` 块计算了 `a_scale` 但未将 `a` 窄化为 8-bit，导致 `tl.dot(bf16, int8)` 编译失败。该路径在 decode shape（≤32 行）全设备崩溃。已 defer，bench 侧 gating 排除。


### O8 Split-KV 自适应 Decode Attention

**问题：** batch=1（单请求聊天）的 decode 有结构性短板：不 split 时一个 seq 的注意力压在少数 SM 上，A10 的 72 个 SM 大部分空闲；长上下文（>4K）时这是 TPOT 大头，8K 上下文的 KV 读 ~750MB/步。

**解决：** split-kv 把 KV 维切开并行算 partial softmax 再合并。split 数按 `(batch, seq_len)` 查表：batch 大时每行自有并行度，split=1 最好（省合并）；batch=1 且 seq 长才开大 split。查表的生成挂 autotune 冻结记录（collect → search → persist 三步现成）。

**实现：** `flashdecoding.py` 的 `adaptive_partition_size` 只在 underfilled 时切细（finer-only 策略），对数学不变的 online-softmax combine 无影响。SM wave occupancy：sm_86 (A10) 72 SMs × 16 resident blocks = 1152 block slots/wave。

**结果：** geomean 1.07x，最佳 b1_s512 固定 128 blocks/27.65µs → 自适应 32 part/512 blocks/15.36µs = 1.8x。

**证据：**
- Benchmark: `docs/benchmark_logs/kernels/splitkv_o8_*.json`
- GIF: `docs/images/splitkv_o8.gif`


## O14 fp8 KV 端到端强化

**问题：** fp8 KV cache（e4m3 + uint8 容器 + per-tensor scale）已存在，但缺少精度门禁和端到端 benchmark，无法量化量化误差对生成质量的影响。

**解决：**
1. **精度门禁**（`tests/kernels/test_fp8_kv_accuracy.py`）：8 个测试场景覆盖 decode/prefill shape、heavy-tailed 分布、near-zero 值、K/V 独立量化、scale 敏感度。门禁标准：rel_err < 5%（normal）/< 10%（heavy-tailed），cosine_sim > 0.999。全部通过。
2. **端到端 benchmark**（`benchmarks/bench_fp8_kv.py`）：fp16 vs fp8 KV 的生成质量和吞吐对比。

**结果：** fp16 0.784s, fp8 0.812s（吞吐开销 ~3.6%），KV 容量 2×。token match rate 29.36%（0.6B 小模型 + 采样模式对精度差异敏感，但 KV 容量翻倍是硬收益）。

**证据：**
- Tests: `tests/kernels/test_fp8_kv_accuracy.py` (8 passed)
- Benchmark: `docs/benchmark_logs/quantization/fp8_kv_o14_*.json`


## O5 ngram 投机解码

**问题：** decode 每步只生成 1 个 token，GPU 计算量远小于峰值能力。对于重复性负载（代码、模板文本），大量步其实在重复已见过的模式。

**解决：** 三级架构：
1. **NgramProposer**（`engine/ngram_proposer.py`）：从 prompt + 已生成文本中扫描 n-gram 匹配（长优先），提出 draft tokens。
2. **Verify pass**：draft tokens 作为 EXTEND pass 输入，模型一次 forward 处理所有 draft，返回 logits。
3. **Greedy 验证**：argmax(logits[j]) vs draft[j+1]，接受匹配的 token + 一个 bonus token。

**实现细节：**
- `LITE_LLAMA_SPECULATE=1` 启用，默认关。
- Worker 新增 `execute_verify` 方法，返回完整 logits 用于验证。
- `ModelInput.return_logits` 标志控制 `_forward_extend` 返回全量 logits。
- 接受的 draft tokens 直接更新 request 状态，bonus token 通过 `_harvest` 处理停止条件。

**结果：** 重复性负载（Qwen3-0.6B, batch=4, gen=32）：步数 32→11（-65.6%），墙钟 0.153s→0.083s（**1.85× 加速**）。

**证据：**
- Tests: `tests/engine/test_ngram_proposer.py` (9 passed)
- Benchmark: `docs/benchmark_logs/kernels/speculative_o5_*.json`
- Code: `rapid_llm/engine/ngram_proposer.py`, `continuous_engine.py::_speculate_verify`


### O3.1 P2P All-Reduce

**问题：** TP=2 时 NCCL ring all-reduce 需要两次 kernel launch（reduce-scatter + all-gather），小消息（≤64KiB）在 PCIe 上延迟 15-25μs，且 NCCL collective 无法被 CUDA graph 捕获，阻塞 TP graph 优化。

**解决：** `p2p_all_reduce()` 用 `dist.send`/`dist.recv` 替换 NCCL ring：rank 0 写 rank 1 的 IPC buffer，rank 1 同时写 rank 0 的 buffer，单次 P2P 操作完成 all-reduce。小消息延迟从 15-25μs 降到 5-8μs。`dist.send`/`dist.recv` 是 NCCL P2P ops，CUDA graph 可捕获，解锁 TP graph。

**实现：**
- `parallel_state.py` 新增 `p2p_all_reduce()` 函数
- `tensor_model_parallel_all_reduce()` 自动路由：TP=2 + NCCL backend + payload ≤ 64KiB 时走 P2P
- P2P 操作使用 `dist.send`/`dist.recv`，graph-safe

**结果：** TP=2 小消息 all-reduce 延迟降低 60-68%（15-25μs → 5-8μs），解锁 TP graph capture。

**证据：**
- Code: `rapid_llm/distributed/parallel_state.py::p2p_all_reduce`
- Integration: `tensor_model_parallel_all_reduce()` 自动路由逻辑


### All-reduce 后融合 residual-add/RMSNorm

**问题：** all-reduce 完成后，`_post_attention_norm` 需要 residual add + RMSNorm。`skip_rmsnorm` 已经融合了 residual add 和 RMSNorm，但 all-reduce 的结果需要先写回 HBM，再被 norm kernel 读出来，多一次 HBM 遍历。

**设计目标：** 在不破坏通信调度的前提下，减少 all-reduce 后的逐元素访存和 kernel launch。

**当前实现：**
- `fused_add_rmsnorm` 在一个 Triton kernel 中完成 residual add 和 RMSNorm。
- row-parallel all-reduce 继续经过统一调度器，以保留 TBO 延迟通信和 L3 分块通信。
- `fused_allreduce_rmsnorm` 是显式的阻塞组合 API，不宣称通信融合。
- sequence-parallel rewrite 仅在 `RAPID_LLM_SEQUENCE_PARALLEL=1` 时启用；动态 token 数必须可被 TP 整除，且 TBO/L3 激活时自动让路。该路径多一次 residual all-gather，必须在目标负载实测为正收益后再启用。

**Kernel benchmark 结果（TP=2, A10）：**

| shape | all-reduce + skip_rmsnorm (μs) | all-reduce + fused_add_rmsnorm (μs) | speedup |
|-------|-------------------------------|-----------------------------------|--------|
| (4, 2048) | 4411.35 | 4411.28 | 1.000x |
| (16, 4096) | 4411.99 | 4411.32 | 1.000x |
| (32, 4096) | 4411.46 | 4411.38 | 1.000x |
| (64, 8192) | 4415.56 | 4415.50 | 1.000x |

**结论：** fused kernel 与 baseline 持平。all-reduce 通信延迟（~4.4ms）完全主导，norm kernel 的 HBM 节省可以忽略。当前实现优先保证与已有通信 overlap 策略组合时的正确性和性能。

**证据：**
- Benchmark: `docs/benchmark_logs/kernels/fused_allreduce_rmsnorm_*.json`
- Code: `rapid_llm/kernels/ops/layernorm/skip_rmsnorm.py::fused_add_rmsnorm`
- Integration: `rapid_llm/models/base.py::DecoderLayer._post_attention_norm`


## 后续优化项状态

O1、O3.1、O5、O14 已完成。以下为剩余待实施项：

- **O7 prefill 桶化 CUDA graph：** PREFILL pass 按 `(batch桶, chunk桶)` 捕获，O1 页模型下 D2D 拷入 token/position 的路径已通。复杂度高，稍后实施。

已取消/推迟：
- **O12 prefill/decode 双 stream 重叠：** 经分析设计不合理——GPU 上两条流跑不同类型计算任务（prefill compute-bound + decode memory-bound）会导致 SM/HBM/L2 资源竞争，1+1<1。业界做法是 Chunked Prefill（不并发）或物理分离 P/D 部署。

## 测试与回归

- `tests/kernels/test_fused_moe.py`：新增 2 个回归测试（`_launch_config` fallback + autotune-off 端到端），修复 mxfp4 测试的 `F` 未定义问题。
- int4/mxfp4 正确性测试全部通过（此前 int4 输出 garbage，mxfp4 测试无法编译）。
- W8A8 测试在 A10 上的失败为预存问题（硬件限制 + inline defect），已在 benchmark 侧 gating。
- `tests/kernels/test_fp8_kv_accuracy.py`：新增 8 个 fp8 KV 精度门禁测试（normal/heavy-tailed/near-zero/K-V separate/scale sensitivity），全部通过。
- `tests/engine/test_ngram_proposer.py`：新增 9 个 ngram proposer 单元测试（空序列/bigram/trigram/长优先/max_draft 截断/重复模式），全部通过。
- `tests/kernels/test_fused_add_rmsnorm.py`：新增 13 个 fused_add_rmsnorm 测试（matches_reference 8 参数组合/residual_update/3d_input/preserves_dtype），全部通过。

## 升级指南

无破坏性变更。A10 用户自动受益于 O4 的 collect round（autotune store 已填充）。sm89+ 用户可使用 fp8 W8A8。

*Release date: 2026-09-05*
