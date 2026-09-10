# Release v0.11.1 — decode 每步的 host 开销削减 + TP 图捕获关闭死锁修复

**Date:** 2026-09-03 **Branch:** `update_fp8_fp4` **Theme:** 每层每步都要做的事做一次就够；多进程的图捕获能开也要能关

## Summary

v0.11.1 是 v0.11.0 之上的优化与修复版：两处每步固定开销从 decode 路径上拿掉，一个堵住 TP 组合的关闭死锁，外加五个 rebase 遗留破损。

**两处每步开销：** MoE 路由的 fp32 语义要求每次 GEMM 前把 `gate_weight` 加宽成 fp32——这件事每层每步都 launch 一个 cast kernel，而权重在加载后是冻结的，加宽一次就该存下来（优化 A）。attention 的 paged cache 布局把 K 和 V 打包在同一行里，两个算子各要一半，于是每层每步切两次 view——live buffer 在引擎生命周期里从不换身份，切一次就该复用（优化 B）。两处都是 host 侧开销，收益随 batch 变小、层数变多而放大：Qwen3-30B-A3B（48 层 MoE）eager decode TPOT -2.6%（batch 8）/ -3.5%（batch 1），graph 模式吞吐 +2.0%。

**一个死锁：** TP=2 + CUDA graph 的引擎在 shutdown 时永远挂起。根因在 NCCL：`ncclCommAbort` 对被图捕获过的 communicator 会停在一个谁也叫不醒的 futex 里（PyTorch/NCCL 的交互），而关闭序列——先 join follower 再销毁组——恰好把两边的 abort 排成了先后而不是并肩。修复分三层：销毁提前到 join 之前、销毁前 gloo barrier 会合各 rank、销毁本身带 15 秒时限，超时则放弃组（communicator 随进程退出由驱动回收）。修复后 `tests/distributed/test_tp_cuda_graph.py` 从 900 秒超时死锁变成 11/11 通过（178 秒）——这组测试（bf16/fp8/nvfp4 × graph × TP=2 × parity × 逐字节贪心一致）是本仓库最重的交叉验证，此前从未在本机带真实 checkpoint 跑通过。

## Feature

### 优化 A：MoE 路由权重的一次性 fp32 加宽（`modules/moe.py`）

![router GEMM 三代演进](images/v0111_router_evolution.png)

```python
# 之前：每层每步一个 cast kernel
router_logits = F.linear(x.float(), self.gate_weight.float())

# 之后：首次路由时加宽一次，之后直接用
if self._gate_weight_fp32 is None:
    self._gate_weight_fp32 = self.gate_weight.detach().float()
router_logits = F.linear(x.float(), self._gate_weight_fp32)
```

fp32 语义本身不动：DeepSeek 的路由（显式 `.float()`）与 qwen3 的参考实现都要求 fp32 logits，bf16 GEMM 会在 near-tie 上翻转 topk 选择，选错一个专家的代价远大于加宽。权重本体仍按模型 dtype 存储——parity 测试把 `gate_weight.dtype` 读作模型 dtype 的代理，只有 GEMM 的操作数加宽。lazy 初始化（而非构造时）是刻意的：测试和 loader 可以在构造之后再填 `gate_weight`；CUDA graph capture 前的三次 eager warmup 保证缓存在录制前已经存在。

**后续演进（tier-4，e80fd63）**：fp32 缓存路径随后被 vllm 的 tier-4 router 路径取代——`torch.mm(x, gate_weight.T, out_dtype=fp32)`，单个 bf16 tensor-core GEMM 带 fp32 accumulate/output epilogue，把 fp32 权重副本和每步 `x.float()` 加宽一起拿掉。算子级（H100，hidden 2048 × 128 experts，topk parity 验证后计时）：decode 2.2×、batch 8 约 2.56×、2048 tokens 5.28×，geomean 3.23×（[`router_gemm_tier4_h100_20260903.json`](benchmark_logs/kernels/router_gemm_tier4_h100_20260903.json)）；e2e A/B（同一棵树 monkey-patch `_route`，隔离 router GEMM）：graph TPOT -2.6% / TPS +2.7%，eager 在噪声内（[`router_ab_h100_20260903.json`](benchmark_logs/engine/router_ab_h100_20260903.json)）。上图三代框即这条演进线。

### 优化 B：K/V 半区 view 的身份感知缓存（`modules/attention.py`）

paged 布局是 `[2 * max_tokens, num_kv_heads, head_dim]`：K 占前半行，V 占后半行，两个 kernel 各要一个单独张量。原来每次 forward 切两刀；现在缓存切好的 pair，keyed 在 buffer 身份上：

```python
views = self._kv_view_pair
if views is None or self._kv_view_source is not kv_buffer:
    self._kv_view_source = kv_buffer
    views = (kv_buffer[:, : self.num_kv_heads, :], kv_buffer[:, self.num_kv_heads :, :])
    self._kv_view_pair = views
return views
```

身份检查（`is`）覆盖唯一合法的 buffer 更换场景：KV profiling 的 dummy forward 跑在临时 buffer 上。缓存持有 source 的强引用，所以 `is` 比较永远不会看到被回收后复用的地址。MLA 路径不受益也不需要：它的 latent 布局把整行交给 kernel，没有半区切分。

### TP 图捕获关闭死锁的修复（`executor/executor.py` + `distributed/parallel_state.py`）

![teardown 时序对比](images/v0111_teardown_timeline.png)

三层修复，一层比一层防御：

1. **销毁提前**：rank0 的 `destroy_parallel()` 从 join follower 之后提到之前。follower 的清理要在所有 rank 同时销毁时才完成，一个 rank0 停在 join 里，follower 就停在自己的析构里等 rank0 手里的 communicator——鸡生蛋。原顺序对 eager 组无害（普通 communicator 的销毁不需要会合），图捕获过的 communicator 才把这个顺序变成死锁。
2. **gloo barrier 会合**：`tensor_model_parallel_barrier()`（新 API）在销毁前把各 rank 排齐。`ncclCommAbort` 在部分 NCCL 版本是集合调用——一个 rank 单独 abort 会把对端永久留在等待里。barrier 走 CPU 组，代价是零。
3. **带时限的销毁**：`_destroy_with_deadline()` 把 `destroy_process_group` 放到 daemon 线程跑，15 秒不返回则 `abandon_parallel()`（新 API）——parallel state 的全局复位为 world-of-one，communicator 留给进程退出时与 CUDA context 一起回收。eager 引擎毫秒级销毁，完全感知不到这条路径；卡住的只有图捕获过的 communicator，而它的正确出路本来就不是等待。

死锁的取证过程值得记录，因为它排除了三个错误假设：反演优化 A/B 后同样死锁（排除本版引入）；`torch.cuda.synchronize()` 后销毁同样死锁（排除在途 kernel）；先销毁图再销毁组同样死锁（排除图池的未决状态）。faulthandler 栈最终定位在 `destroy_process_group` → `pg.shutdown()` → `ncclCommAbort` 的 futex，且 follower 侧同样卡着——两边的 abort 都在等一个不会发生的事件。

## Bug fix

五个由测试暴露的破损（四 rebase 遗留 + 死锁）：

| 破损 | 表象 | 修复 |
|------|------|------|
| `per_token_group_quant` 未从 `kernels.ops.quantization` 包根导出 | `tests/kernels/test_quantization.py` 收集失败（ImportError） | 包根 re-export + `__all__` |
| pytest `--import-mode=importlib` 从 addopts 丢失 | `tests/kernels/` 与 `tests/models/` 的 `test_grouped_topk.py` basename 冲突，清缓存无效 | `pyproject.toml` addopts 补回 |
| `continuous_engine` 深入 `engine.model_runner.config.kv_cache_torch_dtype` | 16 个测试失败（测试替身 `SimpleNamespace` 无 `config`） | `getattr` 优雅降级 |
| golden 测试硬编码 `/data/shared/llm_weights/...` | `check_hardcoded_paths` 钩子失败 | 相对路径 + `parents[2]` 解析，保留 env override |
| TP + graph shutdown 死锁（上文） | `test_tp_cuda_graph` 900 秒超时 | 三层修复（上文） |

## Benchmark

![e2e A/B TPOT](images/v0111_e2e_tpot_ab.png)

H100 单卡，`bench_e2e.py` 口径（greedy，gen=256，每个配置两次进程级重复 + 每次两轮 in-process warmup），基线 = 反演两处优化后的同一棵树（A/B 对照，不是跨版本对比）。日志：[`docs/benchmark_logs/engine/optim_ab_h100_20260903.json`](benchmark_logs/engine/optim_ab_h100_20260903.json)。

| 模型 | 模式 | batch | TPOT（基线） | TPOT（优化后） | 差值 |
|------|------|-------|------------|---------------|------|
| Qwen3-30B-A3B-FP8（MoE，48 层） | eager | 1 | 45.21 ms | 43.64 ms | **-3.5%** |
| Qwen3-30B-A3B-FP8 | eager | 8 | 46.90 ms | 45.68 ms | **-2.6%** |
| Qwen3-30B-A3B-FP8 | graph | 8 | 11.26 ms | 11.05 ms | **-1.9%**（TPS +2.0%） |
| Qwen3-0.6B-FP8（dense，28 层） | eager | 8 | 19.97 ms | 19.62 ms | -1.8% |
| Qwen3-0.6B-FP8 | graph | 8 | 3.25 ms | 3.26 ms | 噪声内 |

读法：两处优化都是 host 开销，模型越深（层×专家路由次数）、batch 越小（固定开销占比越高），收益越大；0.6B graph 持平是预期——replay 不走 Python。graph 档位的 `--verify` 全部通过（eager 与 graph 的贪心输出逐字节一致），证明缓存不改变数值路径。

router tier-4 演进的 e2e A/B（Qwen3-30B-A3B-FP8，同一棵树 monkey-patch `_route`，两次进程级重复）：

| 模式 | TPOT（fp32 cache） | TPOT（tier-4） | TPS（fp32 cache） | TPS（tier-4） |
|------|-------------------|---------------|-------------------|---------------|
| graph | 10.88 ms | 10.60 ms（**-2.6%**） | 725.4 | 744.7（**+2.7%**） |
| eager | 46.46 ms | 46.16 ms（噪声内 ~0.7%） | 172.2 | 173.3 |

## 测试结果

```text
1739 passed                          tests/ 全量（非分布式 1599 + distributed 140）
11 passed in 178s                    tests/distributed/test_tp_cuda_graph.py，Qwen3-0.6B-FP8（首次全绿）
434 passed in 53s                    tests/engine/ 带 checkpoint e2e
```

已知失败两项，均与本版无关：`test_concatenated_local_logits[2]`（干净 HEAD 同样失败）；`test_dp_perf` 的 1.4× 扩展性断言对 host-bound 的 0.6B eager 不成立（单副本 1.28s vs 双副本并发 1.32s——串行会是 2.56s，副本确实并发，只是 eager TPOT 不随 batch 变化，固定工作量的墙钟 scaling 数学上就是 ~1.0×）。`test_tp_cuda_graph` 的 11 项覆盖 bf16/fp8/nvfp4 × graph × TP=2 的 capture 安装、replay 落桶、与 eager 的 logits parity、以及 32 步贪心输出的逐字节一致——是 graph + TP + quant 三特性的交叉验证。

## 文件清单

```text
rapid_llm/modules/moe.py                        优化 A：_gate_weight_fp32 lazy 缓存（后演进为 tier-4 out_dtype GEMM）
rapid_llm/modules/attention.py                  优化 B：_kv_view_pair 身份感知缓存
rapid_llm/executor/executor.py                  死锁修复：销毁顺序 + barrier + 时限 + abandon
rapid_llm/distributed/parallel_state.py         新 API：tensor_model_parallel_barrier / abandon_parallel
rapid_llm/engine/continuous_engine.py           kv_fp8 封装泄漏修复
rapid_llm/kernels/ops/quantization/__init__.py  per_token_group_quant 导出
pyproject.toml                                   pytest --import-mode=importlib 补回
tests/golden/test_deepseek_trimmed_parity.py     硬编码路径改相对
docs/benchmark_logs/engine/optim_ab_h100_20260903.json  A/B benchmark 日志
docs/benchmark_logs/kernels/router_gemm_tier4_h100_20260903.json  router 算子级 tier-4 vs fp32 SGEMM
docs/benchmark_logs/engine/router_ab_h100_20260903.json          router e2e A/B（tier-4 vs fp32 cache）
scripts/gen_v0111_release_figs.py                本版三张图的生成脚本（数据读自上述 JSON）
docs/images/v0111_{router_evolution,e2e_tpot_ab,teardown_timeline}.png
```

## 相关文档

- [docs/benchmark_logs/engine/optim_ab_h100_20260903.json](benchmark_logs/engine/optim_ab_h100_20260903.json) — 本版全部 A/B 数字
- [docs/release-v0.11.0.md](release-v0.11.0.md) — 上一版（MLA + 流式 reasoning/tool 解析）
