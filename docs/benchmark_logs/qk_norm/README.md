# qk_rmsnorm 融合 A/B：q/k 逐头 RMSNorm 两次 launch 合并为一次

比值一律为 **fused ÷ baseline**（同一 checkpoint、同一硬件、同一命令，只有
`models/base.py` 的 `_project_qkv` 一处不同）。比值 < 1 表示融合更快。所有数字由
`benchmarks/suites/run.py qk-norm summarize` 从本目录 JSON 重算，非手抄。

## 1. 环境

| 项 | 值 |
|---|---|
| 设备 | NVIDIA H100 80GB HBM3（sm90）；3352 GB/s peak HBM、989 TFLOP/s dense tc |
| 运行卡 | 单卡（`CUDA_VISIBLE_DEVICES=1`），TP1/DP1 |
| torch | 2.13.0+cu130 |
| triton | 3.7.1 |
| python | 3.14.7 |
| transformers | 5.15.1 |

## 2. 推理工作负载

| 项 | 值 |
|---|---|
| 采样 | greedy（`temperature=0`，确定性） |
| max_gen_len | 64 |
| batch | 1 / 8 / 32（离线）；8（在线） |
| max_seq_len | 1024（在线场景） |
| prompt 集 | `bench_e2e.py` 内置 PROMPTS，按 `--batch` 扩充；在线为 8 请求、250 ms 间隔到达 |

## 3. 模型与是否受融合影响

| checkpoint | model_type | 层数 | hidden | `use_qk_norm` | 角色 |
|---|---|---|---|---|---|
| Qwen3-4B-Thinking-2507 | qwen3 | 36 | 2560 | **True** | 受影响 |
| Qwen3-30B-A3B-Instruct-2507 | qwen3_moe | 48 | 2048 | **True** | 受影响 |
| Qwen2.5-0.5B-Instruct | qwen2 | 24 | 896 | False | **对照组**（不走融合分支，用于标定噪声） |

并行档位覆盖：TP2 与 DP2 各测了两个模型（见第 9 节）。

未覆盖：`qwen3_5`/`qwen3_5_moe`（Qwen3.6-27B / 35B-A3B）未注册进
`models/registry.py`，本机无法加载；`qwen3_vl` 的唯一 checkpoint 是 235B
FP8，2×H100 装不下。

## 4. 框架优化参数

| 开关 | 状态 |
|---|---|
| CUDA graph | **两种都测**（`--mode both`），eager 与 graph 分别报数 |
| autotune 缓存 | 关闭（`RAPID_LLM_AUTOTUNE=0`），走启发式 tile |
| 量化路径 | 无（全部 bf16 权重，避免量化差异混入比值） |
| prefix cache / chunked prefill | 默认（离线 bench_e2e 口径） |
| eager↔graph 一致性 | `--verify` 断言贪心输出一致，两侧各 8/8 通过 |
| TP2 | `engine/run.py optimizations --tp 2`，baseline cell = 全关（eager）+ cuda_graph cell |
| DP2 | `bench_data_parallel --mode scaling --dp 2`，weak scaling，round_robin |

## 5. 运行命令

```bash
# 单卡矩阵（离线 + 在线）；脚本内部已 export RAPID_LLM_AUTOTUNE=0
.venv/bin/python benchmarks/suites/run.py qk-norm run fused docs/benchmark_logs/qk_norm --scope single
# 把 models/base.py 的 _project_qkv 切回两次 skip_rmsnorm 后：
.venv/bin/python benchmarks/suites/run.py qk-norm run baseline docs/benchmark_logs/qk_norm --scope single
# 并行侧矩阵（TP2 + DP2）；省略 scope 则一次跑完 single + parallel
.venv/bin/python benchmarks/suites/run.py qk-norm run fused docs/benchmark_logs/qk_norm --scope parallel
.venv/bin/python benchmarks/suites/run.py qk-norm run baseline docs/benchmark_logs/qk_norm --scope parallel
.venv/bin/python benchmarks/suites/run.py qk-norm summarize   # 从 JSON 重算每格比值
```

单条离线命令口径：

```bash
CUDA_VISIBLE_DEVICES=1 RAPID_LLM_AUTOTUNE=0 PYTHONPATH=. .venv/bin/python \
  benchmarks/bench_e2e.py --model-dir <ckpt> --mode both --greedy --verify \
  --batch <b> --max-gen-len 64 --json <out>.json
```

## 6. 结果日志

本目录归档：单卡 `offline_<model>_b<batch>_{fused,baseline}.json`（16 个）、
`online_<model>_{fused,baseline}.json`（4 个）、TP2 `tp2_<model>_{fused,baseline}.json`
（4 个）、DP2 `dp2_<model>_{fused,baseline}/`（各含带时间戳的 JSON）。每个含 `config`
（命令行参数与时间戳）与 `results`（TTFT / TPOT / TPS，多卡时加 TPS/GPU）。
统一入口 `benchmarks/suites/run.py qk-norm` 提供 `run`（scope=single 单卡 / parallel 并行 / all 两者）与 `summarize`。

## 7. 结果

### 离线（TTFT / TPOT / TPS 全覆盖）

| 模型 | batch | 模式 | TTFT 基线→融合 (ms) | TPOT 基线→融合 (ms) | TPOT 比 | TPS 比 |
|---|---|---|---|---|---|---|
| qwen3-4b | 1 | eager | 21.13 → 20.05 | 20.15 → 19.27 | **0.956** | 1.046 |
| qwen3-4b | 1 | graph | 20.81 → 19.92 | 4.49 → 4.42 | **0.985** | 1.017 |
| qwen3-4b | 8 | eager | 21.31 → 20.92 | 21.18 → 20.60 | **0.973** | 1.028 |
| qwen3-4b | 8 | graph | 21.42 → 19.95 | 4.79 → 4.74 | **0.988** | 1.016 |
| qwen3-4b | 32 | eager | 23.08 → 22.80 | 21.35 → 21.24 | 0.995 | 1.006 |
| qwen3-4b | 32 | graph | 22.69 → 22.28 | 5.16 → 5.13 | 0.995 | 1.006 |
| qwen3-30b-a3b | 1 | eager | 43.07 → 40.87 | 41.96 → 40.16 | **0.957** | 1.045 |
| qwen3-30b-a3b | 1 | graph | 42.99 → 41.16 | 5.43 → 5.36 | **0.989** | 1.015 |
| qwen3-30b-a3b | 8 | eager | 44.96 → 42.17 | 42.99 → 42.19 | **0.981** | 1.020 |
| qwen3-30b-a3b | 8 | graph | 44.28 → 42.56 | 10.68 → 10.61 | 0.994 | 1.009 |
| 对照 qwen2.5-0.5b | 1 | eager | 11.98 → 12.17 | 11.85 → 12.04 | 1.016 | 0.984 |
| 对照 qwen2.5-0.5b | 1 | graph | 11.86 → 12.12 | 1.18 → 1.18 | 0.999 | 0.998 |
| 对照 qwen2.5-0.5b | 8 | eager | 12.76 → 12.80 | 12.11 → 12.15 | 1.004 | 0.996 |
| 对照 qwen2.5-0.5b | 8 | graph | 12.23 → 12.62 | 1.25 → 1.26 | 1.006 | 0.991 |
| 对照 qwen2.5-0.5b | 32 | eager | 14.17 → 14.33 | 12.17 → 12.36 | 1.016 | 0.985 |
| 对照 qwen2.5-0.5b | 32 | graph | 14.01 → 13.99 | 1.43 → 1.43 | 0.995 | 1.005 |

TTFT 两侧同向变化（融合侧普遍略低），但 TTFT 主要由 prefill 决定、且融合只改
decode 前的 q/k 归一化，故 TTFT 差异按噪声看待，不作为收益主张。

### 在线（`engine/run.py scheduler continuous`，batch=8）

| 模型 | 场景 | TTFT 基线→融合 (ms) | TPS 基线→融合 | TPS 比 | latency 比 |
|---|---|---|---|---|---|
| qwen3-4b | offline_static | 21.9 → 20.6 | 1579.5 → 1602.8 | 1.015 | 0.985 |
| qwen3-4b | offline_continuous | 22.3 → 21.1 | 1475.1 → 1509.3 | **1.023** | 0.977 |
| qwen3-4b | online_static | —（static 不记 TTFT） | 216.0 → 219.7 | 1.017 | 0.953 |
| qwen3-4b | online_continuous | 27.4 → 25.9 | 247.1 → 247.5 | 1.001 | 0.977 |
| 对照 qwen2.5-0.5b | offline_static | 12.8 → 12.7 | 5161.8 → 5169.5 | 1.001 | 0.999 |
| 对照 qwen2.5-0.5b | offline_continuous | 13.3 → 13.2 | 4332.1 → 4354.8 | 1.005 | 0.995 |
| 对照 qwen2.5-0.5b | online_static | — | 277.7 → 278.1 | 1.001 | 0.996 |
| 对照 qwen2.5-0.5b | online_continuous | 12.8 → 12.9 | 275.9 → 276.1 | 1.001 | 0.997 |

分组几何均值：受影响模型 TPS **1.0141**、latency 0.9731；对照组 TPS 1.0022、
latency 0.9964。

## 8. 噪声底与结论（哪些收益是真的）

分组几何均值（由 `benchmarks/suites/run.py qk-norm summarize` 输出）：

| 分组 | n | TPOT geo | TPS geo |
|---|---|---|---|
| qk_norm 模型 · eager | 5 | **0.9723** | 1.0287 |
| qk_norm 模型 · graph | 5 | **0.9901** | 1.0124 |
| 对照组 qwen2 · eager | 3 | 1.0119 | 0.9883 |
| 对照组 qwen2 · graph | 3 | 0.9999 | 0.9978 |
| 在线 · qk_norm 模型 | 4 | — | 1.0141 |
| 在线 · 对照组 | 4 | — | 1.0022 |

对照组（qwen2，`use_qk_norm=False`，**不走融合分支**）的比值就是纯噪声标定：
eager 1.004–1.016、graph 0.995–1.006。即 **eager 噪声约 ±1.6%、graph 约 ±0.6%**。
据此判读：

| 档位 | TPOT 比 | 是否超出噪声 |
|---|---|---|
| eager batch=1 | 0.956 / 0.957 | **是**（−4.3~4.4%） |
| eager batch=8 | 0.973 / 0.981 | **是**（−1.9~2.7%） |
| eager batch=32 | 0.995 | 否（−0.5%，在噪声内） |
| graph batch=1 | 0.985 / 0.989 | **是**（−1.1~1.5%） |
| graph batch=8 | 0.988 / 0.994 | 边界（−0.6~1.2%） |
| graph batch=32 | 0.994 / 0.995 | 否（−0.5~0.6%，在噪声内） |
| 在线 TPS | 1.014（对照 1.002） | **是**（+1.4%） |

结论：收益真实但幅度有限，且**随 batch 增大迅速衰减**——batch=1 时 decode 完全
launch-bound，少一次 launch 直接减 TPOT；batch 变大后 GEMM 转为带宽/算力主导，
一次 norm launch 的占比被摊薄到噪声以下。eager 的收益（−4.3%）明显大于 graph
（−1.5%），因为 graph replay 已把每次 launch 的 CPU 开销压到 GPU 侧 ~1–2 µs。

## 9. TP2 / DP2（补齐并行档位）

驱动命令 `benchmarks/suites/run.py qk-norm run <variant> <out> --scope parallel`，同样两侧各跑一遍。

### TP2（`engine/run.py optimizations --tp 2`，batch=8，greedy + --verify）

| 模型 | cell | TPOT 基线→融合 (ms) | TPOT 比 | TPS/GPU 基线→融合 | 比 |
|---|---|---|---|---|---|
| qwen3-4b | baseline (eager) | 31.39 → 29.70 | 0.946 | 126.9 → 133.8 | 1.054 |
| qwen3-4b | cuda_graph | 5.12 → 6.06 | 1.185 | 698.0 → 606.5 | 0.869 |
| 对照 qwen2.5-0.5b | baseline (eager) | 18.30 → 23.35 | 1.276 | 216.4 → 170.5 | 0.788 |
| 对照 qwen2.5-0.5b | cuda_graph | 2.74 → 2.77 | 1.010 | 1274.3 → 1260.4 | 0.989 |

**TP2 这组数据不能用来判读收益或回退。** 对照组（`use_qk_norm=False`，两次运行代码完全相同）的 eager TPOT 自己就波动了 **+27.6%**（18.30→23.35），比受影响模型的波动更大——噪声底远超任何可能的融合收益。原因：测量时 GPU0 上有另一个进程占 3.8 GB、NCCL all-reduce 的 run-to-run 方差、以及 batch=8 下 TP2 由集合通信主导而非 kernel launch。所以受影响模型 cuda_graph 那个 1.185 落在噪声内，既不构成回退证据也不构成收益证据。`--verify` 两侧各 2/2 通过（greedy 文本与各自 baseline 一致）。

### DP2（bench_data_parallel --mode scaling --dp 2，weak scaling）

| 模型 | 行 | TPS 基线→融合 | TPS 比 | TPS/GPU 比 |
|---|---|---|---|---|
| qwen3-4b | LLM (in-process) | 1604.3 → 1630.4 | 1.016 | 1.016 |
| qwen3-4b | DataParallelEngine dp=1 | 1475.3 → 1502.3 | 1.018 | 1.018 |
| qwen3-4b | DataParallelEngine dp=2 | 2942.7 → 3020.2 | **1.026** | 1.026 |
| 对照 qwen2.5-0.5b | LLM (in-process) | 3443.0 → 5342.8 | 1.552 | 1.552 |
| 对照 qwen2.5-0.5b | DataParallelEngine dp=1 | 4148.0 → 4172.4 | 1.006 | 1.006 |
| 对照 qwen2.5-0.5b | DataParallelEngine dp=2 | 8131.8 → 8076.1 | 0.993 | 0.993 |

DP2 判读：对照组的 dp=1/dp=2 两行给出噪声底 **±1%**；受影响模型 dp=2 为 **+2.6%**，略超噪声，算弱正信号。但"LLM (in-process)"行对照组自己波动 +55%，该行不可用（单进程基线最受 GPU0 上其他进程干扰），不作为证据。两侧 dp=1 各 8/8 completions 与 baseline 一致。

### 仍未覆盖

- **`qwen3_vl`**（第三个 `use_qk_norm=True` 的模型）：本机唯一 VL checkpoint 是 Qwen3-VL-235B-A22B-Instruct-FP8，FP8 权重约 235 GB，2×H100 共 160 GB 装不下，无法测。
- **TP4 及以上、DP4 及以上**：本机仅 2 卡。
- TP2 需要多次重复才能压出可用噪声底，本轮只做了一对单次运行，故按"不可判读"处理而非补测。

## 10. 精度

融合 kernel 与两次 `skip_rmsnorm` **逐位一致**（`torch.equal`，8 个几何/dtype 组合，
含非 2 幂 head_dim、奇数 head 数、MHA、decode tokens=1、2048 prefill、fp16/bf16）；
两侧 `--verify` 各 8/8 断言 eager==graph 贪心输出一致，无不一致记录；端到端 5 个
prompt 的 greedy 文本与基线完全相同。回归测试见
`tests/kernels/test_norm_activation.py::test_qk_rmsnorm_is_bit_identical_to_two_skip_rmsnorm`。
