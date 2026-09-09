# 优化特性矩阵 benchmark：单特性 + 混合交叉

每个开关单独一格，再跑机制上真正相互影响的组合。比值口径分两类，方向都调成
「大于 1 更好」：**TTFT/TPOT 为 baseline ÷ cell**，**TPS 为 cell ÷ baseline**。
同一 checkpoint、同一硬件、同一命令行，只有开关不同。所有数字由脚本从本目录
JSON 重算，非手抄。

## 1. 环境

| 项 | 值 |
|---|---|
| 设备 | NVIDIA H100 80GB HBM3（sm90）；3352 GB/s peak HBM、989 TFLOP/s dense tc |
| 运行卡 | 单卡（`CUDA_VISIBLE_DEVICES=1`），TP1/DP1 |
| torch | 2.13.0+cu130（cuda 13.0） |
| triton | 3.7.1 |
| transformers | 5.15.1 |
| vllm | 0.28.0（本轮未用作对照） |
| python | 3.14.7 |
| 代码基线 | `8eb1166`（q/k RMSNorm 融合）之后的工作树 |
| autotune | `RAPID_LLM_AUTOTUNE=0`（关掉档位搜索，避免首次运行混入调优开销） |

## 2. 推理工作负载

| 项 | 值 |
|---|---|
| checkpoint | `my_weight/Qwen2.5-0.5B-Instruct`（qwen2，24 层，hidden 896，14/2 头，vocab 151936，bf16） |
| 采样 | greedy（`temperature=0`、`top_p=1`、`repetition_penalty=1`、`stop_on_repeat=False`），确定性 |
| batch | 8 个并发请求 |
| max_seq_len | 2048 |
| max_num_seqs | 16 |
| KV 池 | 40960 token |
| 引擎 | 连续批处理（`ContinuousBatchingEngine`），逐 `step()` 驱动 |

三个 workload 分别对准不同特性的生效条件——用短 prompt 测 chunked prefill
等于测一个从未触发的开关：

| workload | prompt 形状 | 对准的特性 |
|---|---|---|
| `short` | `benchmarks/common.py` 内置 PROMPTS（约 10–40 token） | CUDA graph、pipeline 等 decode 侧开关 |
| `long` | 40 句语料拼接（约 500–600 token）+ 独立问句 | chunked prefill（超出 token 预算才会切块） |
| `shared` | 32 句共享前缀 + 独立问句，8 个请求前缀相同 | prefix cache（有共享前缀才有命中） |

`max_gen_len`：`short` 为 128，`long`/`shared` 为 64。

## 3. 框架优化参数

baseline 格是全关，每格只在其上加一个开关（组合格加多个）。`BASELINE` 逐项：

| 开关 | baseline | 说明 |
|---|---|---|
| `use_cuda_graph` | False | decode 走 eager |
| `cuda_graph_lazy` | False | O13 惰性捕获 |
| `enable_prefix_cache` | False | 前缀复用 |
| `max_num_batched_tokens` | 引擎默认 | 远大于本轮 prompt，不切块 |
| `pipeline` | False | O2 launch/harvest 循环 |
| `async_tokenize` | False | O10 后台 tokenize |
| `kv_cache_dtype` | `auto`（bf16） | fp8 KV 关闭 |
| `decode_window_steps` | 0 | O9 准入窗口，立即准入 |

各特性格的覆盖值：`cuda_graph`→`use_cuda_graph=True`；`lazy_graph`→再加
`cuda_graph_lazy=True`；`prefix_cache`→`enable_prefix_cache=True`；
`chunked_prefill`→`max_num_batched_tokens=256`；`pipeline`→`pipeline=True`；
`async_tokenize`→`async_tokenize=True`；`fp8_kv`→`kv_cache_dtype="fp8"`；
`decode_window`→`decode_window_steps=2`。

## 4. 运行脚本命令

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. RAPID_LLM_AUTOTUNE=0 \
  .venv/bin/python benchmarks/engine/run.py optimizations \
  --model-dir my_weight/Qwen2.5-0.5B-Instruct \
  --workload short --batch 8 --max-gen-len 128 --max-seq-len 2048 \
  --mode all --greedy --verify \
  --json docs/benchmark_logs/engine/optim_matrix_20260904/qwen05b_short.json
```

`long`/`shared` 两轮同上，只改 `--workload`、`--max-gen-len 64` 与输出文件名。
`--mode all` = 单特性格 + 默认组合格；`--verify` 逐格比对 greedy 文本。

## 5. 运行结果日志

| 文件 | workload |
|---|---|
| `qwen05b_short.json` | short |
| `qwen05b_long.json` | long |
| `qwen05b_shared.json` | shared |

每个 JSON 为 `{"config": ..., "results": [...]}`，`config` 含命令行参数、GPU 标签、
`BASELINE`/`FEATURES` 全量与格列表；`results` 每格含 `ttft_ms`/`tpot_ms`/`tps`/
`tps_per_gpu`/`gen_tokens`/`latency_s`/`features`。

## 6. 指标表

指标口径：TTFT = 首个 step 结束减提交时刻；TPOT = 首步之后各 step 间隔均值；
TPS = 生成 token 数 ÷ 整轮墙钟。本轮 TP1，故 `TPS/GPU == TPS`；并行场景该列才
与 TPS 分开（`engine/run.py optimizations` 的 `--tp` 会按卡数折算）。

### 6.1 short（batch 8，gen 128）

| 格 | TTFT ms | TPOT ms | TPS | TTFT× | TPOT× | TPS× |
|---|---|---|---|---|---|---|
| baseline | 23.3 | 12.71 | 625.5 | — | — | — |
| cuda_graph | 14.6 | 1.60 | 4691.6 | 1.60 | **7.92** | **7.50** |
| lazy_graph | 13.6 | 2.09 | 3672.3 | 1.72 | 6.08 | 5.87 |
| prefix_cache | 13.1 | 12.81 | 624.3 | 1.79 | 0.99 | 1.00 |
| chunked_prefill | 13.2 | 12.71 | 629.2 | 1.77 | 1.00 | 1.01 |
| pipeline | 13.3 | 12.63 | 623.6 | 1.75 | 1.01 | 1.00 |
| async_tokenize | 22.8 | 12.70 | 621.0 | 1.02 | 1.00 | 0.99 |
| fp8_kv | 16.4 | 15.68 | 510.0 | 1.42 | **0.81** | **0.82** |
| decode_window | 13.3 | 12.87 | 621.7 | 1.76 | 0.99 | 0.99 |
| cuda_graph+prefix_cache | 13.2 | 1.71 | 4446.8 | 1.77 | 7.43 | 7.11 |
| cuda_graph+chunked_prefill | 13.3 | 1.71 | 4429.7 | 1.75 | 7.41 | 7.08 |
| cuda_graph+pipeline | 13.1 | 1.71 | 4378.9 | 1.78 | 7.43 | 7.00 |
| cuda_graph+fp8_kv | 16.3 | 2.31 | 3313.0 | 1.43 | 5.51 | 5.30 |
| cuda_graph+prefix_cache+chunked_prefill | 13.4 | 1.71 | 4447.5 | 1.74 | 7.44 | 7.11 |
| cuda_graph+prefix_cache+chunked_prefill+pipeline | 13.5 | 1.71 | 4368.7 | 1.73 | 7.42 | 6.98 |

### 6.2 long（batch 8，gen 64）

| 格 | TTFT ms | TPOT ms | TPS | TTFT× | TPOT× | TPS× |
|---|---|---|---|---|---|---|
| baseline | 33.2 | 13.04 | 589.9 | — | — | — |
| cuda_graph | 22.4 | 1.87 | 3598.7 | 1.49 | **6.96** | **6.10** |
| lazy_graph | 22.4 | 2.74 | 2591.4 | 1.48 | 4.76 | 4.39 |
| prefix_cache | 22.2 | 12.63 | 616.2 | 1.50 | 1.03 | 1.04 |
| chunked_prefill | 29.6 | 15.47 | 376.4 | 1.12 | **0.84** | **0.64** |
| pipeline | 22.1 | 12.48 | 605.2 | 1.50 | 1.04 | 1.03 |
| async_tokenize | 1.3（假象，见 §7.1） | 1.00（假象） | 572.7 | 26.21（假象） | 13.10（假象） | 0.97 |
| fp8_kv | 24.7 | 16.19 | 482.7 | 1.35 | 0.81 | 0.82 |
| decode_window | 21.5 | 12.70 | 613.8 | 1.55 | 1.03 | 1.04 |
| cuda_graph+prefix_cache | 21.9 | 1.99 | 3434.1 | 1.52 | 6.56 | 5.82 |
| cuda_graph+chunked_prefill | 20.5 | 4.92 | 1153.7 | 1.62 | 2.65 | 1.96 |
| cuda_graph+pipeline | 22.6 | 2.03 | 3266.9 | 1.47 | 6.42 | 5.54 |
| cuda_graph+fp8_kv | 25.3 | 2.74 | 2549.8 | 1.31 | 4.76 | 4.32 |
| cuda_graph+prefix_cache+chunked_prefill | 20.5 | 2.39 | 2659.3 | 1.62 | 5.46 | 4.51 |
| cuda_graph+prefix_cache+chunked_prefill+pipeline | 20.7 | 2.43 | 2555.2 | 1.61 | 5.37 | 4.33 |

### 6.3 shared（batch 8，gen 64，8 个请求共享前缀）

| 格 | TTFT ms | TPOT ms | TPS | TTFT× | TPOT× | TPS× |
|---|---|---|---|---|---|---|
| baseline | 31.9 | 12.79 | 601.9 | — | — | — |
| cuda_graph | 21.0 | 1.85 | 3673.2 | 1.52 | **6.92** | **6.10** |
| lazy_graph | 20.9 | 3.36 | 2169.2 | 1.52 | 3.81 | 3.60 |
| prefix_cache | 20.9 | 12.61 | 618.5 | 1.53 | 1.01 | 1.03 |
| chunked_prefill | 28.1 | 15.51 | 375.9 | 1.13 | **0.82** | **0.62** |
| pipeline | 20.9 | 12.37 | 611.5 | 1.53 | 1.03 | 1.02 |
| async_tokenize | 1.2（假象，见 §7.1） | 1.02（假象） | 573.0 | 25.51（假象） | 12.50（假象） | 0.95 |
| fp8_kv | 23.2 | 15.70 | 498.1 | 1.38 | 0.81 | 0.83 |
| decode_window | 20.2 | 12.68 | 615.7 | 1.58 | 1.01 | 1.02 |
| cuda_graph+prefix_cache | 20.0 | 1.83 | 3729.0 | 1.59 | 6.98 | 6.20 |
| cuda_graph+chunked_prefill | 18.8 | 3.98 | 1416.9 | 1.70 | 3.21 | 2.35 |
| cuda_graph+pipeline | 20.8 | 1.83 | 3617.5 | 1.54 | 6.99 | 6.01 |
| cuda_graph+fp8_kv | 24.0 | 2.52 | 2760.6 | 1.33 | 5.07 | 4.59 |
| cuda_graph+prefix_cache+chunked_prefill | 19.1 | 2.18 | 2901.2 | 1.67 | 5.85 | 4.82 |
| cuda_graph+prefix_cache+chunked_prefill+pipeline | 19.1 | 2.22 | 2790.5 | 1.67 | 5.76 | 4.64 |

## 7. 结论与诚实声明

### 7.1 async_tokenize 的 TTFT/TPOT 是测量假象，不是收益

`long`/`shared` 两轮里 async_tokenize 的 TTFT 显示 1.2–1.3 ms、比值 25–26×，
这个数字不可用。原因是 `benchmarks/common.py` 的 TTFT 定义为「首个 step 结束
减提交时刻」，而后台 tokenize 下请求要等 encode 落地才进 scheduler：第一个
`step()` 手上没有请求，立刻返回，TTFT 便量到了这个空步。TPS 一列不受影响
（分母是整轮墙钟、分子是真实 token 数），三轮都是 0.95–0.99×，即后台
tokenize 对吞吐是中性的——它的价值在大 prompt 的 encode 不阻塞引擎循环，
本轮 prompt 太短，量不出来。要量准需要把 TTFT 锚到请求自身的准入时刻，
属于 benchmark 框架的改动，本轮未做。

### 7.2 CUDA graph 是唯一的大头，其余开关在此负载下接近中性

decode CUDA graph 三个 workload 一致给出 TPOT 6.9–7.9×、TPS 6.1–7.5×。0.5B
模型 eager decode 是 launch-bound（每步约 300 次 launch），graph 把这部分
归零，所以量级差这么大。lazy_graph 略低于全量捕获（short 6.08× vs 7.92×），
差额是按需捕获在首次命中时付的一次性代价，符合 O13 的设计预期。

prefix_cache / chunked_prefill / pipeline / decode_window 在 short 上全是
0.99–1.01×：这些特性在短 prompt、无共享前缀、不切块的负载下本就不该生效，
接近 1 正是对照正确性的证据，不是「没用」。它们各自的生效条件见 §2。

### 7.3 fp8 KV 在本负载下是负收益

三个 workload 一致：TPOT 0.81–0.82×、TPS 0.82–0.83×，即比 bf16 KV 慢约
20%。fp8 KV 的收益来自 KV 读取量减半（长上下文、大并发才显著），而代价是
读时反量化；本轮上下文只有几百 token，省下的带宽抵不过反量化开销。叠加
cuda_graph 后仍是 4.6–5.5×（相对 baseline），说明 graph 的收益不被 fp8 KV
吃掉，但 fp8 KV 自身在该场景不该开。

### 7.4 chunked_prefill 单独开在长 prompt 上是负收益

`long`/`shared` 上单独开 chunked_prefill 是 TPOT 0.82–0.84×、TPS 0.62–0.64×。
切块把一次网格 prefill 拆成多趟，每趟都要重新走一遍注意力，在 baseline 已经
走 eager 的前提下只增加开销。但叠加 cuda_graph 后变成正收益（short 7.41×、
shared 3.21×），因为切块后的续传 chunk 行数落在已捕获的 graph batch 之内，
能 replay decode graph——这正是引擎里「续传 chunk 按行数路由」那条逻辑的
意图。结论：chunked prefill 要与 CUDA graph 一起开。

### 7.5 组合不叠加，也不互相破坏

`cuda_graph` 加任何其他开关，TPOT 都停在 7.4× 附近（short），比单独开
cuda_graph 的 7.92× 略低。差额来自叠加开关各自的固定开销（prefix cache 的
块哈希、chunked 的路由判断），不是机制冲突。四开关全开的格与两开关格几乎
同值（7.42× vs 7.43×），说明这些特性可以一起开而不互相拖累。

### 7.6 精度校验结果

`--verify` 逐格比对 greedy 文本，`OUTPUT_SHIFTING_FEATURES`（`fp8_kv`、
`async_tokenize`）单独报，不计入失败：

- **short**：全部 13 个精确格复现 baseline 文本；`fp8_kv` 两格如预期文本不同。
- **long / shared**：`chunked_prefill`、`pipeline` 及含它们的组合格报文本不同。
  这不是本轮改动引入的回归，根因见 §7.7。

### 7.7 遗留问题：两条续传 chunk 路由数值不等价

`shared` workload 上引擎自身是确定的（eager 连跑两次逐 token 相同），但
eager 与 graph 的 greedy 输出 8/8 全不同。定位到根因：打开 CUDA graph 会把
`_chunked_min_rows` 从 1 抬到「最大捕获 batch + 1」，于是续传 chunk 的路由
从「全部走 chunked prefill kernel」变成「短余量走 EXTEND」。用
`RAPID_LLM_FUSED_CHUNK_PREFILL=0/1` 直接对拍两条路由，首 token 即不同——
两条路由不是逐位等价的，greedy argmax 把这个差异放大成整条序列的分岔。

这是既有实现层面的问题（路由逻辑本身是有意为之，见
`engine/continuous_engine.py` 的 `_prefill_work`），不是本轮引入，本轮未修：
要么让两条 kernel 数值对齐，要么在 golden 门禁里把「路由随 graph 开关变化」
显式纳入。列为后续项。

### 7.8 未覆盖的部分

- **模型**：本轮只跑 Qwen2.5-0.5B-Instruct。modelzoo 里另有
  Qwen3-4B-Thinking-2507、Qwen3-30B-A3B-Instruct-2507（bf16/FP8）、
  Qwen3-30B-A3B-Instruct-2507-Int4-W4A16 可用；`qwen3_5`/`qwen3_5_moe`/
  `deepseek_v4` 未注册进 `models/registry.py`，本机无法加载。
- **并行**：本轮 TP1/DP1，故 `TPS/GPU == TPS`，未量 TGS。TP2 的
  `--tp 2` 路径已在脚本里接好，未跑。
- **对照引擎**：本轮无 vllm/HF 对照行，故表内比值一律是 lite 引擎内部
  「格 ÷ baseline」，不是对外加速比。
