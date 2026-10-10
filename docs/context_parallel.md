# 上下文并行（DCP / PCP）

decode 上下文并行（DCP）把一个 attention-TP 组的 KV 历史按**位置**分到 DCP 组内的多个 rank：位置 `p` 存在 rank `p % dcp` 上。每个 rank 只对自己那份历史做 decode attention，再用 log-sum-exp（LSE）把各 rank 的部分结果合并。合并是 softmax 的恒等变形，因此 DCP 引擎的输出应当与同宽度的纯 TP 引擎一致（只差求和顺序）。

prefill 上下文并行（PCP）只建了进程组，**模型数据路径尚未接入**，`prefill_context_parallel_size > 1` 会被拒绝，见[约束](#约束与拒绝的组合)。

## 使用

DCP 复用 TP 进程组：`decode_context_parallel_size` 必须整除 `tensor_parallel_size`，TP 组被切成 `tp / dcp` 个 DCP 组（连续的 TP rank 组成一个 DCP 组）。

```python
from rapid_llm import Scheduler

engine = Scheduler.from_pretrained(
    "my_weight/Qwen3-0.6B",
    tensor_parallel_size=2,
    decode_context_parallel_size=2,
    dcp_comm_backend="ag_rs",  # 或 "a2a"
)
```

`rapid-llm serve`、`rapid-llm batch` 等命令接受同名参数，`DataParallelController` 会把它们透传给每个副本：

```bash
rapid-llm serve --model-dir my_weight/Qwen3-0.6B \
    --tensor-parallel-size 2 --decode-context-parallel-size 2 --dcp-comm-backend ag_rs
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--decode-context-parallel-size` | 1 | 每个 DCP 组的 rank 数，必须整除 TP 和 KV block size（16） |
| `--dcp-comm-backend` | `ag_rs` | LSE 合并方式：`ag_rs` = all-gather LSE + reduce-scatter fp32 加权输出；`a2a` = all-to-all bf16 输出与 LSE |
| `--prefill-context-parallel-size` | 1 | 保留参数；>1 当前报 `NotImplementedError` |
| `--prefill-context-parallel-strategy` | `zigzag` | PCP 的 token 排布，PCP 接入前不生效 |

## 数据流

**权重与 KV 头**：query 头仍按 TP 切分；key/value 头只按 `tp / dcp` 切分，DCP 组内的 rank 加载相同的 k/v 投影分片（`QKVParallelLinear`），各自算出整组 query 头所需的 KV 头。代价是 k/v 投影在组内重复计算一次。

**写 KV**：block table 仍记录全局位置，但每个 block 在 DCP rank 上只占 `block_size / dcp` 行。`slot_batch` 把位置 `p` 映射到 owner rank 上的本地行，非 owner 的写入落到 null 行，所以每个 rank 只存自己拥有的位置。

**prefill attention**：每个 rank 在自己的 query 头上 attend 本步新算的 K/V（`dcp_kv_head_range` 给出这些 query 头对应的 KV 子区间），不需要跨 rank 通信。

**decode attention**（每层，`modules/attention.py` 的 DCP 分支 + `distributed/dcp.py`）：

1. **Q all-gather**：本地 query 头在 DCP 组内 all-gather，组内每个 rank 拿到整组的 query 头。
2. **本地 attention**：对本 rank 的份额 `table[:, rank::dcp]` 调用 `flash_decoding(..., return_lse=True)`，序列长度为本地长度 `ceil((L - rank) / dcp)`。
3. **LSE 合并**：`merge_dcp_attention` 合并各 rank 的 `(output, lse)`，并把结果按 query 头散回各 rank（`ag_rs` 或 `a2a`）。

### KV 布局：什么时候省显存

| 模型 | TP 下每 rank 的 KV 行 | DCP 下每 rank 的 KV 行 | 每 rank 每 token 字节 |
| --- | --- | --- | --- |
| GQA，`tp ≤ kv_heads` | `kv_heads / tp` 个头 | `kv_heads × dcp / tp` 个头，每 block `16 / dcp` 行 | **与同宽 TP 相同** |
| GQA，`tp > kv_heads` | KV 头被复制 | 复制减少 `dcp` 倍 | 减少 |
| MLA | latent 无头轴，每 rank 存完整行 | 每 rank 存 `1 / dcp` 的位置 | **减为 `1 / dcp`** |

所以 DCP 的容量收益集中在 MLA 和 KV 头少于 TP 宽度的 GQA 模型上。`tests/distributed/test_dcp_engine.py` 在合成 MLA checkpoint 上验证了每 rank 每 token 字节数减半（`test_an_mla_dcp_rank_caches_half_the_bytes_per_token`），并验证 GQA 下行宽是 dcp 倍、每 token 字节不变（`test_a_dcp_rank_caches_dcp_times_the_heads…`）。

### 执行路径

- **decode**：DCP 的 Q gather 和合并 collective 一起录进 decode CUDA graph，所有 rank 锁步 replay。graph 覆盖的历史长度最大为 4096（`DEFAULT_SEQ_LEN_BUCKETS`），更长的历史走 eager。
- **prefill**：首个 chunk 走普通 prefill kernel。续接 chunk（chunked prefill 的后续块、prefix-cache 命中后的续算）**全部走 EXTEND**：每 token 一行，replay decode graph。原因是 chunked prefill kernel 读一段连续的 cache 行，而 DCP rank 持有的是交错份额。因此 DCP 下 prefill graph 和 fused chunk 都关闭。如果续接 chunk 被错误地规划进 prefill 网格，`slot_batch` / attention 会抛 `ValueError`，不会静默算错。
- **kernel 选择**：DCP 直接调用原生 Triton `flash_decoding`（需要 LSE 输出），不走 dispatch 的外部后端。

## 约束与拒绝的组合

| 组合 | 状态 | 原因 / 报错位置 |
| --- | --- | --- |
| DCP × TP（dcp ∣ tp） | **已验证** | e2e 对齐 TP=2/4，dcp=2/4 |
| DCP × CUDA Graph | **已验证** | decode graph 安装并 replay；与 eager 在平局许可下一致 |
| DCP × prefix cache | **已验证** | 命中后的续算全走 EXTEND，答案与冷启动一致 |
| DCP × chunked prefill | **已验证** | 续接 chunk 走 EXTEND（有 TTFT 代价，见下方实测） |
| DCP × FP8 KV cache | **已验证** | eager 与 graph 均对齐 TP |
| DCP × MLA（DeepSeek-V2 结构） | **已验证（合成权重）** | 仓库无真实 MLA checkpoint，用合成两层 DeepSeek-V2 做门禁 |
| DCP × `ag_rs` / `a2a` | **已验证** | 两种后端都对齐 TP |
| DCP × KV offload | 拒绝 | offload 按 `block_size` 行整块搬运，DCP rank 每块只有 `block_size / dcp` 行（`Scheduler`，`ValueError`） |
| DCP × TBO | 拒绝 | TBO 的两半 micro-batch 不调度 DCP 的 gather/merge collective（`LLMEngine`，`ValueError`） |
| DCP × DeepSeek-V4 | 拒绝 | V4 按位置维护滑窗和 compressor 状态，未按 DCP 切分（`ModelRunner`，`NotImplementedError`） |
| DCP × one-shot 分配器 | 拒绝 | 只有 slot block table 有位置到 rank 的映射（`prefill_alloc_kv_cache`，`NotImplementedError`） |
| DCP × CPU | 拒绝 | DCP decode 只有 CUDA kernel（`NotImplementedError`） |
| `block_size % dcp != 0` | 拒绝 | 轮转分配会让各 rank 拿到不等的份额（`dcp_rows_per_block`，`ValueError`） |
| GQA 头无法整块映射 | 拒绝 | 本 rank 的 query 头横跨 KV 头，无法用单一 GQA 比例表达（`dcp_kv_head_range`，`ValueError`） |
| PCP > 1 | 拒绝 | 进程组已建立，但模型各层仍按整个 TP 组切分（`LLMEngine`，`NotImplementedError`） |
| DCP × DP | 未验证 | 参数会透传给每个副本，没有端到端测试 |
| DCP × EP（MoE） | 未验证 | 没有 MoE + DCP 的端到端测试 |
| DCP × 量化权重 | 未验证 | 权重量化与 attention 正交，但没有端到端测试 |
| DCP × 投机解码 | 未验证 | verify 步的多 token decode 未在 DCP 下测试 |

## 正确性

### 判据

`tests/distributed/test_dcp_engine.py` 把 DCP 引擎与同宽 TP 引擎对照：贪心解码，`logprobs=2`，24 token，4 组 prompt，每组分 batched/alone 两种提交方式。LSE 合并只改变 softmax 分母的求和顺序，所以两个引擎允许在**平局**上分叉。平局要求双方对称地满足：

1. 在第一个分叉步，双方的 top-1/top-2 差（gap）都 ≤ `_TIE_GAP = 0.5` nat；
2. 双方选的 token 互为对方的 top-2（top-2 互换）；
3. 双方对参考 token 的 logprob 定价差 ≤ 0.5；
4. 分叉前的一致前缀上，所选 token 的 logprob 漂移 ≤ 0.5（logit 级对齐），防止已退化为噪声的引擎靠逐步平局通过。

rank 偏移错误、丢掉一份历史、未屏蔽的填充行，都会让 logprob 偏移整数个 nat，不会被判为平局。合成 MLA 门禁的灵敏度已验证：只给 `kv_a_proj` 加 10% 扰动，第 0 个 token 就被拒绝。

### graph 与 eager 为什么不是逐字节相同

`flash_decoding` 的分区大小由 `_resolve_partition_size(batch, heads, max_actual_seq_len, device)` 决定。eager 用本步真实的最大历史长度，graph 在捕获时固化的是 bucket 长度，两者分区数不同，split-K 的求和顺序也不同。所以 DCP 的 graph 与 eager 只能做到平局许可下一致，测试名 `test_dcp_decode_graphs_install_and_replay_up_to_ties` 就是这个意思。

### 结果

`tests/distributed/test_dcp_engine.py`，4× H100（环境同下），15 passed，耗时 456 s。每行 480 = 20 条序列 × 24 token。

| 对照 | 一致 token | 前缀 logprob 漂移 | 分叉处最大 gap |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B：tp2 vs tp2+dcp2（ag_rs / a2a） | 424 / 424 | 0.114 | 0.125 |
| Qwen3-0.6B：tp2 vs tp2+dcp2（ag_rs / a2a） | 278 / 278 | 0.116 | 0.125 |
| Qwen3-0.6B：tp4 vs tp4+dcp2 / tp4+dcp4 | 372 / 438 | 0.108 / 0.111 | 0.125 |
| tp2+dcp2 eager vs graph（Qwen2.5 / Qwen3） | 480 / 432 | 0 | — / 0.062 |
| 合成 MLA：tp2 vs tp2+dcp2（eager / graph） | 374 / 374 | 0.244 | 0.250 |
| Qwen3-0.6B FP8 KV：tp2 vs tp2+dcp2（eager） | 374 | 0.142 | 0.125 |

FP8 KV 的 graph 版本只用一次性脚本跑过（365/480，漂移 ≤ 0.142），没有收进测试。

一致 token 少于 480 都是因为平局在早期分叉，之后两条序列各走各的；分叉步全部满足上面四条判据。

## 性能实测（2026-10-09）

### 环境

- 4× NVIDIA H100 80GB HBM3（sm_90，132 SM），NV18 全互联，驱动 580.159.03
- CUDA 12.8，torch 2.7.0a0+7c8ec84dab.nv25.03，triton 3.2.0，NCCL 2.25.1，transformers 5.17.0，Python 3.12.3

### 方法

- 模型：Qwen3-0.6B（GQA，16 query 头 / 8 KV 头，head_dim 128，28 层），bf16 权重和 KV
- 离线：所有 prompt 一次提交，引擎跑完为止；每个请求生成 128 token；指标按请求计算（TTFT 取请求自己的首 token 时间，TPOT 取请求自己的 decode 区间）
- 所有 arm 都开 CUDA graph；prompt 一次性 prefill（`--chunk-size 0`）
- 每个 arm 一个引擎：`max_seq_len=8448`，`max_num_seqs=32`，每 rank 40960 行 KV
- 扩展效率 = TPS / (tp1 TPS × 卡数)；“vs 同宽 TP” = DCP arm 的 TPS / 同 TP 宽度纯 TP arm 的 TPS
- 通信：在每个场景的 decode 形状下，把 DCP 的 Q gather 和 merge collective 录进 CUDA graph 单独计时（graph replay 绕过 Python，引擎内的 collective 统计看不到它们）；comm/step = (gather + merge) × 28 层
- parity：6 条 prompt，各生成 32 token，所有 arm 都对照 tp1（N 卡对单卡），判据同上

复现：

```bash
cd rapid_llm && export PYTHONPATH=$PWD/.deps:$PWD
python benchmarks/parallelism/bench_context_parallel.py --model my_weight/Qwen3-0.6B
python benchmarks/parallelism/bench_context_parallel.py --model my_weight/Qwen3-0.6B \
    --arms tp1,tp2,tp2_dcp2 --scenarios bs8-3.5k --chunk-size 512
```

原始数据：`docs/benchmark_logs/parallel/context_parallel_Qwen3-0.6B_20261009_132853.json`（全量）、`context_parallel_Qwen3-0.6B_20261009_133024.json`（chunk 512 对照）。

### 端到端

| 场景 | 配置 | TTFT ms | TPOT ms | TPS | 扩展效率 | vs 同宽 TP | 峰值显存 GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bs1-3.5k | tp1 | 32.2 | 2.00 | 446.8 | 100% | | 7.28 |
| | tp2 | 42.2 | 3.45 | 266.2 | 30% | | 4.18 |
| | tp2_dcp2 | 50.5 | 5.99 | 157.7 | 18% | 0.59x | 6.43 |
| | tp2_dcp2_a2a | 48.9 | 5.93 | 159.5 | 18% | 0.60x | 6.43 |
| | tp4 | 34.3 | 3.48 | 268.8 | 15% | | 2.63 |
| | tp4_dcp2 | 47.7 | 6.00 | 157.9 | 9% | 0.59x | 3.76 |
| | tp4_dcp4 | 49.8 | 6.23 | 152.0 | 9% | 0.57x | 6.02 |
| bs1-8k（eager） | tp1 | 100.2 | 22.42 | 43.4 | 100% | | 7.40 |
| | tp2 | 68.8 | 33.08 | 30.0 | 35% | | 4.25 |
| | tp2_dcp2 | 70.5 | 56.33 | 17.7 | 20% | 0.59x | 6.52 |
| | tp2_dcp2_a2a | 70.3 | 55.12 | 18.1 | 21% | 0.60x | 6.52 |
| | tp4 | 51.4 | 33.74 | 29.5 | 17% | | 2.69 |
| | tp4_dcp2 | 52.4 | 56.05 | 17.8 | 10% | 0.60x | 3.82 |
| | tp4_dcp4 | 56.2 | 58.23 | 17.2 | 10% | 0.58x | 6.09 |
| bs8-3.5k | tp1 | 185.5 | 4.08 | 1446.4 | 100% | | 7.37 |
| | tp2 | 159.0 | 5.05 | 1262.6 | 44% | | 4.24 |
| | tp2_dcp2 | 183.9 | 7.98 | 845.8 | 29% | 0.67x | 6.50 |
| | tp2_dcp2_a2a | 178.8 | 7.54 | 892.5 | 31% | 0.71x | 6.50 |
| | tp4 | 165.3 | 4.77 | 1313.0 | 23% | | 2.67 |
| | tp4_dcp2 | 181.2 | 7.55 | 888.3 | 15% | 0.68x | 3.81 |
| | tp4_dcp4 | 178.6 | 7.86 | 860.4 | 15% | 0.66x | 6.08 |
| bs32-1k | tp1 | 173.4 | 4.44 | 5515.1 | 100% | | 7.40 |
| | tp2 | 202.2 | 5.68 | 4385.5 | 40% | | 4.25 |
| | tp2_dcp2 | 215.7 | 8.68 | 3075.6 | 28% | 0.70x | 6.52 |
| | tp2_dcp2_a2a | 207.9 | 8.27 | 3224.8 | 29% | 0.74x | 6.52 |
| | tp4 | 212.8 | 5.61 | 4378.8 | 20% | | 2.69 |
| | tp4_dcp2 | 218.9 | 8.42 | 3144.4 | 14% | 0.72x | 3.82 |
| | tp4_dcp4 | 216.0 | 8.76 | 3051.2 | 14% | 0.70x | 6.09 |

bs1-8k 的历史超过 graph 最大 bucket 4096，所有 arm 都走 eager（graph replay 次数为 0）；其余场景每个 arm 都 replay 了 127–130 次。

### 通信开销（每层，µs）

| 场景 | 配置 | attention kernel | Q gather | merge | comm/step ms | 占 TPOT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bs1-3.5k | tp2 → tp2_dcp2 | 13.5 → 10.4 | 12.6 | 52.8 | 1.83 | 31% |
| | tp4 → tp4_dcp4 | 13.2 → 7.8 | 17.4 | 58.4 | 2.12 | 34% |
| bs1-8k | tp2 → tp2_dcp2 | 25.0 → 17.3 | 12.5 | 52.5 | 1.82 | 3% |
| | tp4 → tp4_dcp4 | 23.4 → 11.0 | 16.7 | 58.5 | 2.11 | 4% |
| bs8-3.5k | tp2 → tp2_dcp2 | 22.9 → 23.6 | 14.2 | 55.9 | 1.96 | 25% |
| | tp2 → tp2_dcp2_a2a | 22.9 → 23.5 | 14.2 | 58.1 | 2.02 | 27% |
| bs32-1k | tp2 → tp2_dcp2 | 27.6 → 28.6 | 15.0 | 60.8 | 2.12 | 24% |
| | tp4 → tp4_dcp4 | 15.9 → 15.8 | 19.2 | 66.4 | 2.40 | 27% |

bs1-8k 的 comm 占比低，是因为该场景走 eager，TPOT 被 Python 发射开销主导。bs8/bs32 下 DCP 的 attention kernel 不比 TP 快：DCP rank 读一半位置，但每个位置的 KV 头是 TP 的 dcp 倍，读的字节数相同。

### KV 容量

| 配置 | 每 rank 行数 | token 容量 | 每 rank 每 token 字节 |
| --- | ---: | ---: | ---: |
| tp1 | 40960 | 40960 | 114688 |
| tp2 / tp2_dcp2 | 40960 | 40960 / 81920 | 57344 / 57344 |
| tp4 / tp4_dcp2 / tp4_dcp4 | 40960 | 40960 / 81920 / 163840 | 28672 / 28672 / 28672 |

各 arm 行数相同，但 GQA 下 DCP 的一行是 dcp 倍宽，所以 token 容量多 dcp 倍，KV 显存也多 dcp 倍（峰值 6.4 vs 4.2 GiB），每 token 字节与同宽 TP 相同。

### parity（对照 tp1）

所有 arm 都判为 tie-licensed：分叉处最大 gap 0.125，前缀 logprob 漂移 ≤ 0.21（tp2 0.153，tp2_dcp2 0.151，tp4 0.208，tp4_dcp2 0.110，tp4_dcp4 0.155）。

### chunked prefill 的 TTFT 代价（bs8-3.5k，chunk 512）

| 配置 | TTFT ms | TPOT ms | TPS |
| --- | ---: | ---: | ---: |
| tp1 | 255.5 | 3.43 | 1298.2 |
| tp2 | 201.8 | 4.61 | 1221.2 |
| tp2_dcp2 | 967.3 | 7.43 | 535.2 |

DCP 下首个 chunk 之后的 6 个 chunk 全部按每 token 一行走 EXTEND，TTFT 是 tp2 的 4.8 倍。DCP 与 chunked prefill 一起使用时，应尽量调大 `max_chunk_size`，让长 prompt 一次 prefill 完。

### 结论

- **这个模型上 DCP 没有收益**。Qwen3-0.6B 是 GQA 且 `tp ≤ kv_heads`：每 token 字节与同宽 TP 相同（不省容量），attention kernel 每层只省 0–14 µs，而 gather + merge 每层固定 65–85 µs。DCP 的 TPS 是同宽 TP 的 0.57–0.74 倍；batch 越大，merge 越能被摊薄，a2a 比 ag_rs 略好（bs8/bs32 下约 +5%）。
- **通信是延迟主导**：每层 wire 字节只有 3–300 KB，NV18 上 gather 12–19 µs、merge 52–66 µs，基本不随 batch 变化，是 collective 的延迟地板。每层多出的这 2 次 collective（dense TP 本身每层还有 2 次 all-reduce）就是 TPOT 差距的主要来源。
- **DCP 的目标场景是 MLA 和 `tp > kv_heads` 的 GQA**：这时 KV 每 rank 字节减为 `1/dcp`，能放下更长的上下文或更多并发。MLA 下字节减半已经验证（合成 checkpoint），但缺少真实 MLA checkpoint（DeepSeek-V2-Lite 不在仓库内，V4 被拒绝），**MLA 的性能没有实测**。
- 以上数字只适用于本机、本模型和上述形状，不能概括成通用加速比。

## 测试

```bash
cd rapid_llm && export PYTHONPATH=$PWD/.deps:$PWD
# 单元：rank 映射、LSE 合并、头映射、slot 行映射、scheduler 路由
python -m pytest -q tests/distributed/test_dcp.py tests/executor/test_slot_batch_blocks.py \
    tests/engine/test_scheduler.py -k dcp
# kernel：flash_decoding / MLA decode 的 return_lse 与分片合并
python -m pytest -q tests/kernels/test_flash_decoding.py tests/kernels/test_mla.py
# 端到端：DCP vs TP（需 ≥2 GPU；tp4 用例需 4 GPU 和 Qwen3-0.6B）
python -m pytest -q tests/distributed/test_dcp_engine.py
```
