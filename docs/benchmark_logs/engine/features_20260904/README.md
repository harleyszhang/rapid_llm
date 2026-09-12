# 推理引擎优化特性全量验证（2026-09-04）

对 8 个优化特性（`cuda_graph`、`lazy_graph`、`prefix_cache`、`chunked_prefill`、
`pipeline`、`async_tokenize`、`fp8_kv`、`decode_window`）单独开启 + 6 个组合，跨 3 个
模型、3 种 workload、TP1/TP2 做 `--verify` 校验（逐格比对 greedy 文本是否与 baseline 一致）。

## 1. 环境与命令

| 项 | 值 |
|---|---|
| 设备 | NVIDIA H100 80GB HBM3（sm90），2 卡 |
| torch / triton / python | 2.13.0+cu130 / 3.7.1 / 3.14.7 |
| 采样 | greedy，`--verify` 逐格断言文本一致 |
| batch / max_gen_len | 8 / 32 |
| autotune | 关闭（`RAPID_LLM_AUTOTUNE=0`） |

```bash
RAPID_LLM_AUTOTUNE=0 PYTHONPATH=. .venv/bin/python benchmarks/engine/run.py optimizations \
  --model-dir <ckpt> --mode all --workload <short|long|shared> --verify --greedy \
  --batch 8 --max-gen-len 32 --json <out>.json
# MoE 需 TP2 分片权重（单卡 OOM）：加 --tp 2 --mode single --features cuda_graph
```

workload 与特性匹配：`long` 触发 `chunked_prefill`，`shared` 触发 `prefix_cache`，
`short` 两者都不触发——这是区分"特性本身有问题"与"chunk 路由不等价"的关键对照。

## 2. 结论：特性本身数值正确

**short workload（不触发 chunk 路由）下全部精确格复现 baseline 文本：**

| 模型 | 类型 | 精确格 | 结果 |
|---|---|---|---|
| Qwen3-4B-Thinking-2507 | dense，qk_norm | 11/11 | **全部复现 baseline** |
| Qwen2.5-0.5B-Instruct | dense，无 qk_norm | 11/11 | **全部复现 baseline** |

只有 `fp8_kv`、`async_tokenize` 及含它们的组合文本不同——这两个是
`OUTPUT_SHIFTING_FEATURES` 里声明的合法漂移（fp8 KV 存 e4m3 换容量；async_tokenize
改变请求准入时机从而改变 prefill batch 形状），不计入失败。

**MoE 覆盖（TP2）**：Qwen3-30B-A3B-Instruct-2507 的 `cuda_graph` 格复现 baseline 文本，
且 `TP CUDA graphs verified: worst graph-vs-eager logit difference 0.000e+00`
（tolerance 1e-2）；TPOT 55.91 → 9.22 ms（**6.06×**），TPS/GPU 70.9 → 361.5。

## 3. long / shared 的 ERROR 格：根因是 chunk 路由不等价（既有遗留项）

| 模型 | workload | ERROR 格数 |
|---|---|---|
| Qwen3-4B | long | 6（`chunked_prefill`、`pipeline` 及含它们的组合） |
| Qwen3-4B | shared | 9（额外含 `cuda_graph`、`lazy_graph`） |
| Qwen2.5-0.5B | long | 6 |
| Qwen2.5-0.5B | shared | 9 |

根因已在代码中确认（`engine/continuous_engine.py`）：

```python
cap = max(manager.batch_sizes, default=0) if manager else 0
self._chunked_min_rows = cap + 1 if fused else math.inf
```

`_prefill_work` 用它路由续传 chunk：graphs off 时阈值为 1，所有续传 chunk 走 chunked
prefill kernel；graphs on 时阈值为「最大捕获 batch + 1」，短余量改走 EXTEND。两条
kernel 不逐位等价（`RAPID_LLM_FUSED_CHUNK_PREFILL=0/1` 直接对拍首 token 即不同），
greedy argmax 把这个差异放大成整条序列分岔。

这解释了为何 short 干净而 long/shared 报错：**问题不在特性本身，而在续传 chunk 的
两条路由数值不等价**。与 [`optim_matrix_20260904/README.md`](../optim_matrix_20260904/README.md)
§7.7 的记录一致，是既有实现层面的遗留项。

`pipeline` 出现在 ERROR 列表里也是同一根因（它本身是 launch/harvest overlap，
device-side 回喂使 `seq_lens` 用乐观长度），short 下它干净即为证据。

### 3.1 已修：让路由阈值不随 graph 开关变化

两条 kernel 无法做到逐位等价——EXTEND 是 fp32 向量 GEMV（代码里明确注释了为何必须
升 fp32：保持 fp16 乘积会让 64 项点积的 ~1e-2 舍入误差逐层放大成 logits ~5e-2 噪声，
翻转 greedy argmax），chunked kernel 走 tensor core `tl.dot`，累加顺序本质不同。
所以改为让**路由不随 graph 开关变化**：graph replay 与 eager 本身已验证逐位一致
（捕获时 logit diff 0.000e+00），阈值一致即可保证两者输出一致。

> **2026-09-11 注**：本段对 EXTEND 的机制描述已过时——packed stage-1（默认开启）下
> extend 的 stage-1 也走 tensor core `tl.dot` + fp32 累加，不再存在“fp32 向量 GEMV”；
> 逐头版（`RAPID_LLM_PACKED_DECODE=0` 或 group 不满足打包条件）才是 fp32 逐元素乘加。
> 两条 kernel 仍非逐位等价，但默认路径下的差异实为 tl.dot 归约顺序层面的浮点重结合
> （bf16 6.5e-3 / fp8 6.6e-3），见 [`quantization.md` 分块预填充实测](../../../quantization.md#分块预填充下的-fp8-kv-cache-数值实测)。

```python
# 修复前
cap = max(manager.batch_sizes, default=0) if manager else 0
# 修复后：无 graph manager 时回落到配置的 batch sizes，阈值两边一致
cap = max(manager.batch_sizes, default=0) if manager else max(DEFAULT_BATCH_SIZES)
```

**效果：shared workload 的 9 个 ERROR 全部归零（11/11 精确格复现 baseline）。**
实测性能代价为零——shared 上 TTFT/TPOT 所有格比值 0.98–1.02；long 上无变化，因为其
续传 chunk 有 256–394 行，本来就超过两个阈值，路由原本就相同。全量测试 1860 passed。

### 3.2 仍存：long workload 的 chunked_prefill 差异是特性固有的

修复后 long workload 仍有 6 个 ERROR（`chunked_prefill`、`pipeline` 及含它们的组合）。
根因与路由无关：`DEFAULT_MAX_NUM_BATCHED_TOKENS=8192`，long prompt 650 token，
8×650=5200 < 8192，所以 baseline **不分块**（单 pass 覆盖整个 prompt）；开
`chunked_prefill` 后预算降到 256，prompt 被切成多块并跨块边界做 online softmax。
两者数学等价但 fp 累加顺序不同，greedy argmax 放大成分岔——这是分块本身固有的，
不是 bug。要消除只能让分块与非分块的 fp 顺序一致，代价是放弃 tensor core。

## 4. 新发现：TP2 多 cell 重建会挂起

`--tp 2 --mode all`（15 个 cell，每 cell 重建引擎）**在第二个 cell 挂起**：第一个 cell
正常完成并输出结果，随后日志出现

```
[W] executor.py:56 group teardown did not finish within 15s (a graph-captured
    NCCL communicator can wedge its abort); abandoning the group — it dies with
    this process
```

下一个 cell 的引擎重建即卡死：GPU 利用率 0%、进程处于 sleeping、5 分钟以上无进展。
两次独立运行均复现。单 cell（`--mode single`）不受影响，故 TP2 本身可用，问题在
**graph 捕获过的 NCCL communicator 拆除不干净 + 紧接着重建**。列为后续项。

## 5. 代码深度检查（本轮已核对的契约）

| 契约 | 位置 | 结论 |
|---|---|---|
| readback ring 生命周期 | `executor/overlap.py` | 正确：`_in_use` 按 `data_ptr` 持有直到 `release_readback`，两个方向都 `record_stream` 防分配器回收竞争；`_acquire` 跳过在途缓冲、退役不可用的 |
| `release_readback` 调用链 | overlap → worker → executor → continuous_engine | 完整，无断链 |
| cuda_graph TP 三道闸门 | `executor/cuda_graph.py`、`model_runner.py` | 完整：env kill-switch、grid 一致性 all-reduce、数值 parity（atol 1e-2）；实测 logit diff 0.000e+00 |
| prefix cache 哈希契约 | `engine/prefix_cache.py` | 完整：blake2b 链式哈希作为跨进程契约（DP router 与 replica 必须一致） |

## 6. 未覆盖

- **30B MoE 单卡**：bf16 权重约 60 GB，加 8 batch × 4 seq bucket 共 32 个 graph 的
  缓冲超出 80 GB，`--kv-blocks 20480` 仍 OOM。只能用 TP2 分片，故 MoE 只跑了单特性。
- **TP2 多特性组合**：受 §4 的挂起阻塞，只验证了 `cuda_graph` 单特性。
- **TP4+/DP4+**：本机仅 2 卡。
- **DP 下的特性组合**：本轮未跑（DP 与特性正交，DP 侧性能已在
  [`qk_norm`](../../qk_norm/README.md) §9 覆盖）。

## 7. 归档

7 个 JSON：`features_<model>_<workload>.json`（short/long/shared × 2 个 dense 模型）
+ `features_qwen3-30b-a3b_tp2.json`。每个含 `config`（命令行与时间戳）、`features`
定义与逐 cell 的 TTFT/TPOT/TPS/TPS-per-GPU。
