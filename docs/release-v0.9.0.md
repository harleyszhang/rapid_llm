# Release v0.9.0 — 多后端 kernel 分发 + L1 跨 stream 重叠

**Date:** 2026-09-01 **Branch:** `release-v0.9.0` **Theme:** "用哪个 kernel"收敛成一张声明式清单，"host 与 GPU 谁等谁"改写成跨 stream 重叠

## Summary

v0.9 有两条工作线。

**Part A（地基 2，本版主体）** 把 kernel 选择从散落的 if-else 收敛成三层机制：`kernels/ops/` 按算子域分组并持有全部注册行（11 个算子契约、21 行实现），`kernels/dispatcher/` 是 torch-free 的选择层（声明式 KernelSpec + 确定性 dispatch + 逐条拒绝理由的 explain），`kernels/backend/` 一库一包接入 flashinfer / deepgemm / flashmla / deepep，缺库是排名事件而不是崩溃。A9 Platform 抽象给出设备检测与 capability 声明，sm90+ 的行在 A10 上被窗口过滤自动拒绝。默认全 native，外部后端要等 v0.10 的冻结实测排序才可能翻盘。

**Part B（L1 跨 stream 重叠，ROADMAP 第六节第一级）** 引擎步内最多三个 pass（prefill / extend / decode），本版让每个 pass 的输入上传在独立 copy stream 上发出，不再等待上一个 forward 排空：prepare 阶段在 host 侧算好布局后立刻发起 pinned-staging H2D 拷贝，compute stream 只插一次 event 等待。配套改动是引擎步从"每 pass 一次 `tolist()`"改成**步末一次同步**（deferred harvest）——没有它，host 每 pass 自抽干一次，跨 pass 重叠在结构上不可能。

## Feature

### 地基 2 三层落地（commit 65f52a0、dd525f6、509e83d）

| 层 | 目录 | 职责 |
| ---- | ------ | ------ |
| 算子域 | `kernels/ops/<group>/__init__.py` | 谁来算：native 行与外部后端的行同处一地 |
| 选择 | `kernels/dispatcher/` | 怎么选：KernelSpec 六维声明（available / capability / dtypes+schemes / shape / layout / golden）+ filter → rank → cache → report |
| 接入 | `kernels/backend/<lib>/` | 能算什么：INSTALL 元数据 + 真 import 检测 + adapter |

三个刻意决策：**不造第二注册表**（量化 scheme 只是 dispatch key 的一维，`linear` 的量化实现与非量化实现进同一张清单）、**不写转发适配器**（KernelSpec 的 target 是 `"module:attr"` 字符串直指 kernel 函数，形参名是契约的一部分，由 `TestTargetsMatchTheirContract` 逐名比对）、**golden 门禁内建于 dispatch**（`verified=False` 的行默认不参与选择，只有显式 `backend=` 可越过——flashinfer 的 attention / rmsnorm / rope / sample 行已带 max-abs-diff 记录）。

### A9 Platform 抽象

设备检测与能力声明（`platform/`）让 dispatch 的 capability 过滤可 mock 测试：deepgemm / flashmla 声明 `>=sm90` 窗口，在 A10（sm86）上被拒且 explain 给出原因，而不是 import 时报错。

### L1 跨 stream 重叠（commit 07ee09e + 本分支集成）

`executor/overlap.py` 三件套：`OverlapPolicy`（`RAPID_LLM_OVERLAP` 环境变量，默认开）、`StreamPool`（copy stream + pinned staging 环 + event）、`Timeline`（CUDA event 区间记录，全部 region 挂在同一 epoch 事件上，跨 stream 可直接比较，`RAPID_LLM_OVERLAP_TIMELINE=1` 开启）。

本分支完成的集成把模块接进热路径：

- `ModelWorker.prepare()` 先做 host 侧布局（`slot_batch` 新增的 `flatten_extend_rows` / `plan_extend_rows` / `pad_decode_rows` 三个 prepare-path helper），随即把 input ids / positions / logits positions 经 pinned staging 异步上传，返回携带 event 的 `_PreparedPass`；
- 三个 `_forward_*` 消费 prepared pass，compute stream 用 `pool.consume(event)` 插一次等待，forward 本体包在 `timeline.region()` 里；
- `prepare` 被提到 `copy_prefix` 之前——否则 host 先阻塞在 D2D 前缀拷贝的启动上，upload 窗口被推迟到重叠机会之后；
- `ContinuousBatchingEngine.step()` 改为 deferred harvest：先执行全部 pass，步末统一读回 token。原来每 pass 一次 `tolist()`，GPU 在每个 pass 之间被抽干，重叠在结构上不可能发生。

![L1 cross-stream overlap](./images/overlap_l1.gif)

GIF 由 `scripts/gen_overlap_l1_gif.py` 驱动**真实引擎**录制：窗口内 extend forward 仍在 compute stream 上执行时，下一个 pass 的 input upload 已经落在 copy stream 上——两条泳道的相交就是重叠本身，不是渲染技巧。

### prefix caching 支持 DP（commit e4c2060）

DP 路由按前缀亲和选择副本：共享前缀的请求聚到同一 rank，各副本的 prefix cache 合成一个逻辑池。

### 多模态 CUDA graph replay（commit 1bc7e13）

multimodal decode 路径接入 CUDA graph replay，并补 TP 与多模态的 e2e benchmark。

## Fix

两处预存测试债（非本版回归，`git stash` 验证在 main 上同样失败）：

- `tests/utils/test_prompt_templates.py` 钉着已删除的 per-family 名称映射旧契约，重写为新契约并补反向用例；
- `tests/models/test_checkpoint_index.py` 对未注册 model_type（`qwen` / Qwen-1_8B）setup 直接报错，改为带理由的 `pytest.skip`。

## Refactor

- attention 接口拆薄：`PagedAttention` 下沉到 `modules/`（KV 写入 + prefill/decode 分派），`models/base.py` 的 Attention 只管投影与 RoPE；dispatch 在构造期一次决策，热路径是普通属性调用——这是 MLA 后续接入的前提。
- collective 记账从 log 变为工具（`tools/observability/`，commit b555a53）。
- v0.8 的 `kernels/backends/registry.py` 雏形与平铺 `backends/` 目录随三层迁移删除，其 per-op 环境变量能力泛化为对每个注册 op 生效的 `op_backend_env()`。

## 顺带提前入库（归属后继版本，在此记账）

- v0.11 的 MLA 算子侧：`MinimalMlaLayer` 单层 harness + flashmla 后端行（golden 未验证，默认不 dispatch）；
- v0.13 的 FlashInfer attention 后端行（prefill + decode 两行，golden 已验证）。

## Benchmark

### L1 overlap on/off（A10, Qwen2.5-1.5B-Instruct, batch=16 长 prompt 混合负载, best of 3）

| 配置 | 墙钟 |
|------|------|
| overlap off | 1.190 s |
| overlap on | 1.203 s（-1.1%） |

诚实结论：重叠**机制确实启动**——timeline 里 `upload.decode.tokens [106.742, 106.785]` 落在 `forward.prefill [88.389, 114.426]` 内部，copy 与 compute 泳道出现真实相交。但墙钟差是 ~1% 噪声级：A10 + eager prefill 下 host（Python 启动路径 ~22ms）与 GPU 计算（~22ms）几乎等长，host 是瓶颈，藏起来的 H2D 拷贝（每次 <0.1ms）不在关键路径上。真正的收益依赖 P5（prefill graph 化压缩 host 路径）与 P9（异步调度流水线）。本版的交付物是机制与可观测性，不是数字。

日志：[`docs/benchmark_logs/overlap/overlap_l1_v09.json`](benchmark_logs/overlap/overlap_l1_v09.json)（含完整 timeline region 表）。

### e2e 回归（A10, Qwen2.5-1.5B, batch=8, gen=128, greedy, CUDA graph）

| 指标 | v0.8.0 基线 | v0.9.0 | 变化 |
| ------ | ------------ | -------- | ------ |
| TTFT | 21.92 ms | 20.35 ms | -7.1% |
| TPOT | 9.36 ms | 8.40 ms | -10.3% |
| 吞吐 | 844.5 tok/s | 942.0 tok/s | +11.5% |

无回归且略有提升，主要受益于 deferred harvest 消除了每 pass 一次的 host 同步。

日志：[`e2e_Qwen2.5-1.5B_b8_g128_v09_release.json`](benchmark_logs/engine/e2e_Qwen2.5-1.5B_b8_g128_v09_release.json) vs 基线 [`Qwen2.5-1.5B_b8_g128_20260831_195724.json`](benchmark_logs/models/Qwen2.5-1.5B_b8_g128_20260831_195724.json)。

### native vs flashinfer 逐算子对照

同 shape 下 native 与 flashinfer 的逐一对照已随 bench 基建入库（`bench_flashinfer` 等）；静态 priority 顺序与实测顺序的出入即是 v0.10 冻结实测排序的输入。

## 测试结果

```text
1067 passed, 78 skipped, 4 xfailed in 164s    tests/ 全量（A10 ×2）
4 passed (RAPID_LLM_GOLDEN_STRICT=1)          tests/golden，Qwen3-0.6B 逐 token 基线一致
```

golden 门禁在 overlap 默认开启下通过：prepared 路径与 inline 路径逐 token 一致，证明 prepare 重构与 deferred harvest 没有改变任何一个输出 token。skip 项均为本机缺 checkpoint（`Qwen2.5-0.5B` 等）或需 4 卡，无静默变绿。

## 文件清单（本分支相对 main 的增量）

| 操作 | 路径 |
| ------ | ------ |
| 修改 | `rapid_llm/executor/slot_batch.py`（prepare-path helpers：`flatten_extend_rows` / `plan_extend_rows` / `pad_decode_rows`） |
| 修改 | `rapid_llm/executor/worker.py`（`_PreparedPass` + `prepare()` + 三个 `_forward_*` 消费 prepared 并记录 timeline） |
| 修改 | `rapid_llm/engine/continuous_engine.py`（step 改 deferred harvest，一步一次同步） |
| 修改 | `benchmarks/overlap/levels.py (L1)`（长短不齐的长 prompt 负载 + token 预算参数） |
| 修改 | `tests/utils/test_prompt_templates.py`、`tests/models/test_checkpoint_index.py`（测试债修复） |
| 新建 | `scripts/gen_overlap_l1_gif.py`、`docs/images/overlap_l1.gif` |
| 新建 | `docs/benchmark_logs/overlap/overlap_l1_v09.json`、`engine/e2e_Qwen2.5-1.5B_b8_g128_v09_release.json` |

地基 2 三层、A9 Platform、prefix affinity 路由、多模态 graph replay 等已随 main 上的 v0.9 提交入库（dd525f6、509e83d、65f52a0、e4c2060、1bc7e13、b555a53、07ee09e 等）。

## Upgrade

```bash
git checkout release-v0.9.0 && uv pip install -e .

# overlap 默认开启；关闭对照
RAPID_LLM_OVERLAP=0 python -m benchmarks.overlap.levels --level l1

# 录制 timeline 证据（copy/compute 泳道相交）
RAPID_LLM_OVERLAP_TIMELINE=1 python -m benchmarks.overlap.levels --level l1 --timeline

# 重新生成上面的 GIF
python scripts/gen_overlap_l1_gif.py

# 查看某个算子的分发决策链
RAPID_LLM_KERNEL_TRACE=1 python -m rapid_llm.cli chat --model-dir my_weight/Qwen3-0.6B
```

## 相关文档

- [ROADMAP](../ROADMAP.md)：地基 2 五根支柱、第六节三条 ping-pong 轴
- [docs/tensor_parallel.md](tensor_parallel.md)：Executor 接缝（L1 重叠挂在它的 `ModelWorker` 上）
- [docs/data_parallel.md](data_parallel.md)：prefix affinity 路由的上下文
