# Release v0.10.0 — 可观测性内置 + 算子分发定型

**Date:** 2026-09-01 **Branch:** `dev-v0.10` **Theme:** "模型为什么给出这个 token"与"引擎此刻在等谁"变成一等 API，"用哪个 kernel"从静态优先级换成冻结实测排序

## Summary

v0.10 做的是"看得见"这件事，分三层。

**给调用方看模型：** `logprobs` / `prompt_logprobs`（F6）报告每个采样 token 的对数概率与它压过的 top-k 备选，以及 prompt 每个位置的分数。两者都出自请求本来就要跑的那次 forward，没有第二次打分，也没有 vLLM 那样的独立 `PromptLogprobsWorker`——代价是 sampler、worker、executor、scheduler、engine、API 六层的签名都要能把"要不要收集"和"收集到什么"透传下去。

**给运维看引擎：** `tools/observability` 的 `metrics` + `trace`（A7）把引擎已有的时间戳变成能画图的数字——queue / TTFT / TPOT 三段直方图、in-flight gauge、token 计数器，`/metrics` 直出 Prometheus 文本格式，不引入 `prometheus_client`；配上 collector 地址后每个请求一条 OTLP span。实测两者全开落在基线自身 0.5% 抖动以内。

**给开发者看单层：** F1 单层 harness 只在 meta 骨架上材料化一层，权重可以是 checkpoint 里那一层的 key、transformers 同层的镜像、或随机初始化，prefill 与 decode 两形态各自对 HF 比 max-abs-diff，并给出逐模块 CUDA event 计时、峰值显存与这一层真实走到的 dispatch 决策。671B 模型的一层是单卡对象，MLA / 新路由这类改动因此能先在真机上验证一层再谈整网。

地基 2 的收尾项是**冻结实测排序接线**：autotune store 经 `set_perf_provider` 接进 dispatch 的 rank 步，同一 key 永远选同一实现。诚实结论写在下面的 benchmark 里——这张 A10 上实测赢家与静态 priority 的首选一致，端到端差值在噪声内；换成实测排序的意义是"外部后端真快时才翻盘"，不是无条件提速。

## Feature

### F6 logprobs / prompt_logprobs（commit c52ecb9）

```python
output = llm.generate(["The capital of France is"], SamplingParams(logprobs=5, prompt_logprobs=5))[0]
for record in output.outputs[0].logprobs:      # 每个生成 token 一条
    print(record.token_id, record.logprob, record.top_token_ids, record.top_logprobs)
print(output.prompt_logprobs[0])               # None:位置 0 没有预测者
```

![logprobs and prompt_logprobs](./images/logprobs.gif)

GIF 由 `scripts/gen_logprobs_gif.py` 驱动**真实 Qwen3-0.6B** 生成：prompt 位置 1 的 `' capital'` 是 -12.8，最后一个生成 token 是一场势均力敌的胜负——`' Italy'` -1.74 压过 `' France'` -1.86，正是 mean-logprob 过滤要抓的那种情况。

三处设计取舍值得记账：

- **采集点在 sampler 内部，不在外部重算。** 一次 `log_softmax` + `topk` 拿到备选，采样得到的 token 再按 id 取自己那格——所以"采样 token 不在 top-k 里"这件事被正确表达（低温度下罕见，高温度下常见），而不是被四舍五入掉。
- **TP 下先广播 id 再造记录。** 各 rank 的 log_softmax 数值可能末位不同，若各自独立取 argmax 会分叉；改成 rank0 广播已定的 token id，其余 rank 只负责为这个 id 填分数。
- **prompt 段与 chunked prefill 兼容。** 一次性 prefill 与分块 prefill 是两条采集路径，`tests/golden/test_logprob_parity.py` 各自与 transformers 的 teacher-forced log_softmax 逐位置比对。

### A7 运行时可观测性（commit ce01b75）

```bash
rapid-llm serve --model-dir my_weight/Qwen3-0.6B &
curl -s localhost:8000/metrics | grep -A2 time_to_first_token
```

`tools/observability/metrics.py` 是一个不到 310 行的进程内 registry：Counter / Gauge / Histogram 各自渲染 Prometheus 文本，桶网格照 vLLM 的粒度取（延迟 1 ms – 10 s，token 数 1 – 16K）。为一个"每种指标几行文本"的格式引入 `prometheus_client`，代价是每个离线用户都要多装一个包，所以没引。采集是 opt-out（`RAPID_LLM_METRICS=0`），因为它本身只是 finish 路径上的几次浮点加法。

`tools/observability/trace.py` 是 opt-in 的另一面：`RAPID_LLM_OTLP_ENDPOINT` 给了就每个请求一条 span（request_id / prompt_tokens / output_tokens / finish_reason），没给就返回 `None` 当 span——`start_span` / `end_span` 对 `None` 是无操作，OpenTelemetry SDK 保持可选依赖，不装也不报错。

### F1 单层 harness（本分支）

```bash
# 计时 + 每个算子实际派发到哪个实现（随机权重，不读 checkpoint）
python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B --layer 0

# 与 transformers 同层数值比对，并当门禁用
python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B \
    --layer 3 --weights mirror --tolerance 2e-2
```

真机输出（A10，Qwen3-0.6B 第 3 层，mirror 权重）：

```text
layer 3 of qwen3 (mlp=FusedMLP)
  parameters   0.029 GiB on cuda as bfloat16
  shape        batch=1 seq_len=128 decode_steps=8
  prefill      1.356 ms
  decode       1.093 ms/step
  peak memory  0.074 GiB
  per-module device time (a parent includes its children):
    self_attn            5.952 ms  x9
    mlp                  2.648 ms  x9
    self_attn.attn       1.806 ms  x9
  dispatched   native/flash_attention2_no_pad, native/flash_decoding, native/linear_torch, native/update_kv_buffer
  prefill vs reference: max_abs=3.125e-02 mean_abs=8.693e-04 rel=6.579e-03
  decode vs reference: max_abs=1.562e-02 mean_abs=6.709e-04 rel=4.016e-03
PASS: worst relative difference 6.579e-03 vs tolerance 2.000e-02
```

两个支撑改动：`hf_weights_iterator` 加 `key_filter`，谓词在读张量**之前**生效，所以从分片 checkpoint 里取一层只付那一层的代价（shard 是 mmap 的，没读的张量根本不出文件）；`OpRegistry.decisions()` 暴露 dispatch 已经做过的决策，报告里"这一层走到了哪些实现"是读出来的，不是照排序重新推的——shape 与 dtype 都是 key 的一部分，重推等于赌自己猜对了 key。

### 冻结实测排序接线（commit 4465b46）

autotune store 经 `set_perf_provider` 接进 rank 步：有冻结记录就按实测排，没有就退回 priority。`benchmarks/kernels/freeze_dispatch_ranking.py` 负责在目标 GPU 上跑出这份记录并写进 `docs/benchmark_logs/`。

### MoE 量化专家吃 bf16 激活（commit ab3d55d）

`fused_moe` 允许量化专家权重与 bf16 激活混用，30B-A3B-FP8 在 2×A10 上的 TP=2 数据随本版入库（`docs/benchmark_logs/quantization/quant_Qwen3-30B-A3B-Instruct-2507-FP8_tp2.json`）。

## Refactor

- benchmark 归一（commit 811c884）：`benchmarks/common.py` 抽出 Backend 抽象 + Factory + `steps_to_result` / `measure_generate`，`bench_e2e` / `bench_quant` / `bench_dp_prefix_cache` / `bench_overlap_l1` 都接工厂；`bench_all_kernels.py`、`flashattention.py`、`flashattentionv2.py`、`bench_hf_baseline.py` 等重复实现退场，净减约 1200 行。
- `bench_continuous` 的 `count_gen_tokens` 参数顺序 bug 修复（此前长短 prompt 混合负载下的 token 计数偏低）。
- `scripts/layer_harness.py` 补 repo root 的 `sys.path` 插入，`python scripts/layer_harness.py` 直接可跑，不必先设 `PYTHONPATH`。

## Benchmark

全部为**离线推理口径**（整批一次提交，不模拟到达间隔），A10 单卡，Qwen3-0.6B，batch=16，gen=128，greedy，best of 3。

### 观测面开销（`benchmarks/bench_observability.py`）

| 配置 | TTFT | TPOT | 吞吐 | 相对基线 |
| ------ | ------ | ------ | ------ | --------- |
| baseline | 23.2 ms | 4.77 ms | 3257.8 tok/s | — |
| metrics | 23.5 ms | 4.78 ms | 3246.5 tok/s | -0.1%（低于噪声） |
| metrics + trace | 23.3 ms | 4.78 ms | 3248.4 tok/s | -0.0%（低于噪声） |
| logprobs=5 | 23.9 ms | 5.35 ms | 2913.4 tok/s | **-10.4%** |
| prompt_logprobs=5 | 32.0 ms | 4.78 ms | 3201.7 tok/s | **-1.5%** |
| both | 32.3 ms | 5.43 ms | 2836.7 tok/s | **-12.7%** |

判据是基线跑两遍取自差：**噪声下限 0.5%**，小于它的差值只报"低于噪声"，不报成提速或变慢。metrics 与 trace 落在噪声里符合预期——它们是每请求几次浮点加法与一次落桶，不是每 token 的工作。logprobs 的 10.4% 则是真的 GPU 代价：每步多一次 `log_softmax` + `topk` 与一次 D2H 搬运，随 batch 与 vocab 走；prompt_logprobs 只压在 prefill 上，所以 TTFT 涨 9 ms 而吞吐几乎不动。

日志：[`docs/benchmark_logs/kernels/observability_v0.10.json`](benchmark_logs/kernels/observability_v0.10.json)。

### dispatch 开销（`benchmarks/kernels/bench_dispatch.py`）

| 档 | 耗时 | 占一步 decode（4.75 ms） |
| ---- | ------ | ------------------------ |
| 首次决策（含后端检测 import） | 761 ms | 一次性，只发生在启动 |
| 换 key 后的 filter + rank | 27.0 µs | 0.57% |
| 命中缓存 | 15.2 µs | 0.32% |

15 µs 里真正的字典查找只有 **0.48 µs**，其余是每次重取的平台快照（9.95 µs，`torch.cuda.get_device_name()`）与环境变量读取（1.72 µs）。这三档都只影响冷启动：调用点在构造期决策一次并存成属性，每步 forward 连命中缓存那次查找都不做。是否缓存 `detect()` 本版没动，记在这里。

日志：[`docs/benchmark_logs/kernels/dispatch_cost_v0.10.json`](benchmark_logs/kernels/dispatch_cost_v0.10.json)。

### v0.9.0 ↔ v0.10.0 端到端对照

同一份不依赖 `benchmarks/` 的探针脚本，分别以两个 worktree 的 `PYTHONPATH` 运行，保证只有 rapid_llm 不同、测量代码逐字相同：

| 路径 | 指标 | v0.9.0 | v0.10.0 | 差值 |
| ------ | ------ | -------- | --------- | ------ |
| continuous | 吞吐 | 3274.6 tok/s | 3269.9 tok/s | -0.14% |
| continuous | TTFT | 22.77 ms | 23.19 ms | +1.84% |
| static graph | 吞吐 | 3380.6 tok/s | 3387.0 tok/s | +0.19% |
| static graph | TPOT | 4.574 ms | 4.562 ms | -0.26% |

全部在噪声内：本版新增的都是默认关闭或按请求 opt-in 的路径，冻结实测排序在这张 GPU 上选出的赢家与 v0.9 静态 priority 的首选一致。**精度无损、性能无回归**是本版对性能的全部主张。

日志：[`docs/benchmark_logs/engine/version_compare_v0.9.0_v0.10.json`](benchmark_logs/engine/version_compare_v0.9.0_v0.10.json)。

## 测试结果

```text
243 passed, 1 skipped, 996 deselected in 199s     pytest -m "gpu or weights"（A10 ×2）
11 passed in 30s                                  tests/golden，Qwen3-0.6B-FP8（4 token parity + 7 logprob parity）
7 passed in 11s                                   tests/golden/test_logprob_parity.py，Qwen3-0.6B
```

唯一 skip 是本机缺 `Qwen2.5-0.5B` 的 gsm8k 项，没有静默变绿。logprob parity 的判据是双阈：单点最大漂移 ≤1.0 nat 且平均漂移 ≤0.15 nat——单点阈放宽是因为 bf16 下极低概率位置的 log 值本就发散，平均阈才是"分布对上了"的实际约束；两个 checkpoint（bf16 与 FP8）各跑一遍都过。

## 文件清单（相对 v0.9.0）

| 操作 | 路径 |
| ------ | ------ |
| 新建 | `rapid_llm/observe/{__init__,metrics,trace}.py`（A7 registry + OTLP tracer） |
| 新建 | `rapid_llm/tools/harness/`、`scripts/layer_harness.py`（F1 单层 harness） |
| 新建 | `rapid_llm/kernels/dispatcher/autotune/frozen.py`、`benchmarks/kernels/freeze_dispatch_ranking.py`（冻结实测排序） |
| 修改 | `rapid_llm/engine/{sampler,continuous_engine,llm_engine,llm,async_engine,scheduler,outputs}.py`、`rapid_llm/executor/{worker,executor}.py`、`rapid_llm/entrypoints/{api_server,protocol}.py`（logprobs 六层透传） |
| 修改 | `rapid_llm/executor/weight_utils.py`（`key_filter`，读张量前过滤）、`rapid_llm/kernels/dispatcher/registry.py`（`decisions()`） |
| 新建 | `benchmarks/bench_observability.py`、`benchmarks/kernels/bench_dispatch.py` |
| 修改 | `benchmarks/common.py` + 全部 bench 脚本接工厂；删除 `bench_all_kernels.py` / `flashattention*.py` / `bench_hf_baseline.py` |
| 新建 | `tests/golden/test_logprob_parity.py`、`tests/observe/test_metrics.py`、`tests/ops/test_frozen_rank.py`、`tests/tools/test_harness.py` |
| 新建 | `scripts/gen_logprobs_gif.py`、`docs/images/logprobs.gif` |
| 新建 | `docs/benchmark_logs/kernels/{observability,dispatch_cost}_v0.10.json`、`engine/version_compare_v0.9.0_v0.10.json` |

## Upgrade

```bash
git checkout dev-v0.10 && uv pip install -e .

# token 分数（默认关闭，按请求打开）
python -c "
from rapid_llm import LLM, SamplingParams
llm = LLM(model='my_weight/Qwen3-0.6B')
print(llm.generate(['The capital of France is'], SamplingParams(logprobs=5))[0].outputs[0].logprobs[0])
"

# Prometheus 抓取
rapid-llm serve --model-dir my_weight/Qwen3-0.6B & curl -s localhost:8000/metrics | head

# OTLP 追踪（不设这个变量 tracer 就是 no-op）
RAPID_LLM_OTLP_ENDPOINT=http://localhost:4318 rapid-llm serve --model-dir my_weight/Qwen3-0.6B

# 单层验证
python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B --layer 3 \
    --weights mirror --tolerance 2e-2

# 复现本版全部 benchmark
python benchmarks/bench_observability.py --json docs/benchmark_logs/kernels/observability_v0.10.json
python benchmarks/kernels/bench_dispatch.py --json docs/benchmark_logs/kernels/dispatch_cost_v0.10.json

# 重新生成上面的 GIF
python scripts/gen_logprobs_gif.py
```

## 相关文档

- [ROADMAP](../ROADMAP.md)：v0.10 章的 feat / test / benchmark / 验收逐条状态
- [docs/release-v0.9.0.md](release-v0.9.0.md)：地基 2 三层与 L1 重叠的上一版交付
- [docs/online_serving.md](online_serving.md)：`/metrics` 所在的 server 面
