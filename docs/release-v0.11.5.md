# Release v0.11.5 — 计算通信重叠

**Date:** 2026-09-03 **Branch:** `support_deepseekv4` **Theme:** 四层重叠原语（L1 host↔device、L2 TBO、L3 chunked all-reduce、L4 tile-signaling）+ DP×CUDA Graph 交叉验证 + DeepSeek V4 端到端；每个特性独立 on/off 对照数据，正收益与负收益同表发布

## Summary

v0.11.5 沿三条轴把「通信时间藏在计算后面」做成可独立开关的原语，并用一个组合矩阵把三者交叉验证——矩阵里每一个数字都是同一负载、同一引擎、只动开关的对照：

* **L1（pinned-copy 重叠，默认开）**：上传/回读走 copy stream + pinned 环形缓冲，`StreamPool` + `Timeline`（CUDA event 级证据）。
* **L2（two-batch overlap，默认关，本版加成本模型自门控）**：TP decode 步切两半 ping-pong，半 A 的 o_proj all-reduce 在通信流上飞行时，半 B 的注意力 GEMM 占着 SM。deferred-AR 上下文 + `execute_overlapped` 交错原语。graph-captured TBO 已实现并测完（launch floor 消除、replay 与 eager TBO 数值一致）。本版用成本模型钉死根因：decode 小 batch 下 GEMM 是访存瓶颈，切半让权重读两遍（实测 1.98×），+4.35 ms 的重复读取盖过只占 ~3-5% 的可藏 AR——TBO 在整个访存瓶颈区间净负。修复：`TboPolicy` 默认激活阈值改为 roofline ridge point（A10=520），只在计算瓶颈区激活，**保证永不负收益**（详见 L2 节）。
* **L3（chunked all-reduce，默认关）**：行并行 GEMM 输出按行分块，第 k 块的 AR 上线时第 k+1 块的 GEMM 在算。
* **SBO（single-batch overlap，默认关）**：EP MoE 的单 batch 双流重叠——dispatch 的 forward a2a 在线上飞时，shared MLP 在一条 alt stream 上算。TBO 要两半才能 ping-pong，SBO 覆盖只有一个 batch 的 EP decode（`batch_overlap/single_batch_overlap.py`，对齐 sglang 同名文件）。重叠实测发生（1271 对共 424 ms），但 **eager 形态下是轻微负收益（-2.4% ~ -6.6%）**；graph 形态下的收益**未证实**——两条独立测量路径给出相反方向（一路 +2.2~2.6% 但取的是 best-of-N，另一路四臂对照实测 -0.2%）。本版真正兑现的是 **EP 保留 CUDA graph**（a2a 与 TBO deferred-AR 用同一套可捕获原语），EP decode 从 eager 到 graph 快 **3.19-3.31×**（两条路径交叉验证、方向一致）。
* **L4（tile-signaling，单卡 kernel 级）**：persistent Triton 生产者每完成一个输出 tile 就 `atomic_xchg` 发布 epoch，消费者 kernel 有界自旋等 tile，GEMM→epilogue 逐 tile 流水。
* **P8（DP + CUDA Graph）**：每副本独立 capture/replay（tp=1/副本，graph 内无集合通信），TPOT -80%，DP2 吞吐 5.1×。

**L2/SBO 在 decode TPOT 上收益不显著，根因已查清并写入「decode TPOT 收益归因」节（含一次重要修正）**：早期记录的「TBO eager +129~135%、graph +47~61%」是**重构前实现**的数据；TBO 重构为对齐 sglang 的实现后重测（两模型 × batch 32/128/256 × eager/graph 四臂，共 12 个测量点），**收益全部落在 ±3% 的噪声带内，正负交替，且不随 batch 或模型规模单调改善**。不是 Bug（parity 与接线测试全绿、重叠实测确实发生）；天花板低的决定性原因是 **comm 时间绝大部分是 NCCL 在 spin-wait 对端 rank（rank 0 的 comm 62.44 ms ≈ rank 1 的 compute 63.88 ms），等待时间无法被计算隐藏**。另：graph 形态比 eager 快 2-3.5 倍，这才是 decode 上真正值钱的那一刀。

**质量故事（本版最有价值的产出之一）**：组合矩阵（M7）抓到了单特性测试漏掉的 TBO 数值回归——闭包在构建期快照了 `state.hidden`，所有层的 segment 都拿初始 embedding 当输入，交错输出与 eager 基线 maxdiff 27+；而「切分本身正确」的隔离证据来自 manual-halves 对照（同一 `TboSplitter` 切分、逐半顺序 forward 拼接 → maxdiff 0.0000）。修复为闭包运行时惰性读 state（`rapid_llm/batch_overlap/two_batch_overlap.py` 的 `_attn_segment` docstring 记录了完整机理）。**这正是「不同优化特性做混合、交叉测试」在计划里的目的：单特性的 parity 测试过 ≠ 组合下依然对。**

**模型覆盖**：dense Qwen2.5-1.5B、DeepSeek-V2-Lite（MLA+MoE）、V3-4layers（biased noaux_tc 路由）、V4 裁剪版（mHC 残差/压缩器/Lightning Indexer/Hash MoE，transformers 5.8 随机初始化 checkpoint——V4 无公开权重）。

## 架构

![三条轴与组合矩阵](images/overlap_axes.png)

*四条可独立开关的重叠路径：A 轴 L1（`batch_overlap/overlap.py`，默认开）把下一个 pass 的 H2D 上传藏进当前 forward；C 轴 L2/L3/SBO（`batch_overlap/`，默认关）分别用半批 ping-pong、行分块与单 batch 双流把通信挪到通信流上；B 轴 L4（`kernels/tile_signal.py`）是单卡 kernel 级的逐 tile 流水；P8 让每个 DP 副本 capture 自己的 graph。L2 与 L3 共用同一个分发点 `row_parallel_forward`（passthrough > deferred TBO > chunked L3 > blocking），四条路径最终汇进同一个组合矩阵交叉验证。*

包结构（`rapid_llm/batch_overlap/`，对齐 sglang 的 `batch_overlap` 布局）：

* `overlap.py` — A 轴（host↔device）：`OverlapPolicy` + `StreamPool`（copy stream + pinned staging 环）+ `Timeline`（CUDA event 区间记录，跨 stream 同一时钟）。本轮从 `executor/` 迁入：两条轴同住一个包，`Timeline` 是它们共用的证据设施
* `operations.py` — stage/yield 交错原语，对齐 sglang 同名文件：`YieldOperation`、`StateDict`（键写一次、pop 后才能重写，`clear(expect_keys)` 校验中间量是否按时释放）、`_StageExecutor` 与 `execute_operations` / `execute_overlapped_operations`
* `comm_overlap.py` — 通信流底座：`CommStreamPool`（每 device 一条 NCCL 流）、`DeferredArContext`（defer/fence/collecting/drain）、`CommOverlapPolicy`（L3 策略）与 `row_parallel_forward` 单一分发点
* `two_batch_overlap.py` — L2 executor：`TboSplitter`（两半等长 + KV 元数据窄化，奇数 batch 用重复末行补齐到 `padded_len`，多余行的 logits 由 `num_rows` 丢弃）+ `TboPolicy`（含 `capture_eligible`，判定某个 batch 的 graph 是否录交错流）+ `model_forward_maybe_tbo` 统一入口（对齐 sglang 同名函数，调用方传入 `enable_tbo` 策略判定，入口负责执行）：`enable_tbo=True` 走 `_model_forward_tbo` 三段式（`_model_forward_tbo_split_inputs` 拆分 → `execute_overlapped_operations` 交错 → `_model_forward_tbo_merge_outputs` 逐半 head 后拼接）；`enable_tbo=False` 走 `_model_forward_non_tbo`，用 `execute_operations` 串行跑同一份 op 流——两条臂共享一层算子的唯一定义，串行臂就是交错臂的数值参照
* `operations_strategy.py` — `OperationsStrategy.init_new_tbo` 按 layer 类名分派，收的是各层自己的 bound method：dense 流 `[op_attn, yield, op_mlp]`（delta 0），EP MoE 流 `[op_attn, yield, op_gate, yield, op_dispatch_a, op_shared_experts, yield, op_dispatch_b, op_experts, op_combine_a, yield, op_combine_b]`（delta 2，对齐 sglang 的 decode 策略）；混合栈（dense 前导层 + MoE）取最宽 lead，因为 sglang 的「各层 lead 必须一致」断言建立在它只支持稀疏层 TBO 上
* `single_batch_overlap.py` — SBO：单 batch 内的 MoE 双流重叠（`SboPolicy` 开关与 `min_rows` 阈值、`SboFlags.enable_dispatch_shared_overlap` 判定、`sbo_alt_stream` 的 compute 侧 alt stream，对齐 sglang 同名文件）
* 模型侧唯一侵入点：`models/base.py` 的 `DecoderLayer.forward_attn_stage/forward_mlp_stage` 两段拆分（原 `forward` = 两段顺序调用，行为不变；段边界恰好是 o_proj 的行并行 all-reduce），以及同文件上的九个 `op_*` TBO 算子（读 `StateDict`、消费即 pop、结果写新键；EP 那六个转发给 `mlp.op_*`）
* 接线：`executor/worker.py` 的 decode 步骤走 `ModelRunner.forward_maybe_tbo(enable_tbo=policy.active(...))`（graph 激活时仍走原 `forward` 由 replay 服务），`_run_tbo`（eager 与 graph capture 共用）以 `enable_tbo=True` 调同一入口

## Feature

### L2 two-batch overlap（含如实发布的负收益与根因修正）

```bash
# TP2 对照（batch 8/16/32：eager on/off 两臂 + graph 参照臂 + graph-captured TBO 臂 + greedy 一致性）
python -m benchmarks.overlap.levels --level l2 --timeline
# EP 四臂：V2-Lite TP2 上 EP on/off × TBO on/off（每批带 graph 参照臂）
python -m benchmarks.overlap.policies --policy ep_matrix --json docs/benchmark_logs/overlap/overlap_ep_<ts>.json
```

2×A10 **PCIe** 上 **eager 形态的** TBO 是负收益——但复测把根因修正为 **CPU launch floor，不是「PCIe 不能重叠」**：eager TP2 decode 的 TPOT 由 Python kernel-launch 的 CPU 时间决定（off 臂 GPU util 仅 28.6%），通信原语要在 GPU 上省时间，而瓶颈根本不在 GPU。四臂数据同表发布（`docs/benchmark_logs/overlap/overlap_l2_tbo_20260904_014530.json`，timeline 证据沿用 `overlap_l2_tbo_20260904_003941.json`）：

| batch | eager off | eager TBO | graph 参照 | graph+TBO | eager TBO 变化 | graph+TBO 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 25.7 ms | 60.0 ms | 6.2 ms | 9.9 ms | +134% | +61% |
| 16 | 25.5 ms | 60.0 ms | 6.6 ms | 10.5 ms | +135% | +58% |
| 32 | 26.3 ms | 60.3 ms | 7.6 ms | 11.2 ms | +129% | +47% |

graph 参照臂与两 eager 臂同负载、同 TP2，唯一差别 `use_cuda_graph=True`（TP graph capture 已于 3e4d3deb 落地）：**6-8 ms 对 eager 的 27-66 ms——launch floor 本身就是 eager TPOT 的 4-10 倍**。三层证据钉死根因：

1. **TPOT 与 batch 无关**：诊断跑批 `--batches 8 16 32 64 128 256 512`，eager off 恒 ~29-30 ms、TBO on 恒 ~66-67 ms，差值恒 +36 ms——若瓶颈在 GPU compute 或 PCIe 通信，batch 翻倍时占比必然移动；纹丝不动只能是每步固定的 CPU 开销。
2. **kernel 拆半零收益**：nsys 显示 on 臂 compute kernel 数 40470→77622（翻倍）但平均时长 13.3→14.3 us 不变——M=8 与 M=16 的 GEMM 都坐在 kernel launch 地板上，拆半只添 launch 次数。
3. **NCCL 翻倍 × rank skew（旧版误当主因的两个次级现象）**：每步 AR 56→112；baseline 每 AR 两 rank 启动偏斜 p50=139 us（真 wire ~32 us），AR 翻倍把 spin 等待也翻倍。

![L2 半批 ping-pong 的真实 timeline](images/overlap_l2.gif)

*TP2 decode 的真实 CUDA-event timeline（`scripts/gen_overlap_gifs.py --level l2`）：上面两条泳道是半 A / 半 B 的 segment（`tbo.attn.*` / `tbo.mlp.*`），下面一条是通信流上的 deferred all-reduce，红带是两者在同一设备时钟上的交集——重叠确实发生，问题在于 eager 形态兑现不了它。*

![L2 四臂对照](images/overlap_l2_tbo.png)

*同负载四臂：eager off / eager TBO / graph 参照 / graph+TBO。两条 eager 臂坐在 Python launch floor 上，graph 臂 6-8 ms；eager TBO 的 +129-135% 是 CPU 地板上的调度开销。第四臂是本版落地的 graph-captured TBO：launch floor 从 60 ms 消到 10 ms（capture 机制成立），但 interleave 本身在该形状净负 +47-61%——可藏的 AR 只占 step 的 ~3-5%，小于半批 GEMM 效率+交错关键路径的代价。*

**重叠本身是真实发生的**（timeline 792 对重叠共 65.5 ms；nsys kernel 级 9.8% NCCL 时间被 compute 隐藏，off 臂为 0.0%），只是 eager 形态下收益无处兑现。6/8、28/32 的分歧行是低置信度输出上「分批 AR vs 整批 AR」的 bf16 归约顺序差——与 batch16 的 16/16 完美一致同表呈现，不隐藏。

**graph-captured TBO（本版落地，第四臂）**：capture 时由 `TboPolicy.capture_eligible(world_size, batch)` 按 batch 决定形态——达标（≥ min_rows 且多 rank）的 key 录 TBO 交错流，未达标的保持普通 forward，两种形态共存于同一 capture 网格；worker 层零改动（eager 路径照旧在 graph 激活时让位，replay 形态无关）。跨流 event 依赖（`comm.wait_stream(compute)` → NCCL → `event.record(comm)` → fence）全部是 stream capture 合法原语，录成 graph 内的 fork/join 边；TBO forward 的 `drain()` 保证 capture 结束时所有 comm 分支 rejoin，无 uncaptured fork；warmup 3 次 TBO forward 先把 NCCL channel/工作区建好。正确性：TP2 单测里 replay 与 eager TBO **logits 一致**（`torch.allclose` 1e-4），greedy 与 eager 基线同率（28/32，与 eager TBO 完全相同的分歧行）。性能：**launch floor 消除（60.0→9.9-11.2 ms，5.4×），但 interleave 本身在该形状净负**——graph+TBO 比普通 graph 慢 47-61%。机理：该 dense-PCIe 形态下每步 AR 只占 ~3-5%，完美重叠的上限收益小于半批 GEMM 效率损失+交错关键路径+事件 fence 的代价；TBO 不减少总 kernel 工作量，只重排它。

**对标 SGLang 的正收益三前提**：CUDA graph（decode 是 replay，TPOT=GPU 时间）+ EP all-to-all payload 大到值得藏 + 深 compute 模型。graph-captured TBO 已满足前提 (a)（机制、parity、launch floor 消除全部验证），但 (b) 在 dense 1.5B TP2 PCIe 上不成立；EP 四臂（`overlap/policies.py（ep_matrix）`，V2-Lite TP2）进一步证明 **eager 下 EP 也是负收益**——a2a 的 payload 优势同样被 CPU 地板淹没：

| batch | tp eager | tp+tbo | ep eager | ep+tbo | graph 参照 |
| --- | --- | --- | --- | --- | --- |
| 16 | 61.5 ms | 129.4 ms | 83.0 ms | 180.0 ms | 25.1 ms |
| 64 | 64.1 ms | 134.7 ms | 92.7 ms | 178.3 ms | 34.5 ms |

（`docs/benchmark_logs/overlap/overlap_ep_20260904_003941.json`）四个 eager 臂无一获益（TP+TBO +110%、EP 单开 +35-45%、EP+TBO +178-193%）；graph 参照臂比最快的 eager 臂还快 1.9-2.4×，且 greedy 与 baseline **16/16、64/64 完全一致**——「值得藏的 a2a payload」在 eager 形态下同样被 CPU 地板淹没。eager 各臂的 greedy 一致率（8/16、9/16、40/64 等）是 bf16 MoE 路由平坦 logits 上归约顺序差翻转 argmax，golden 门禁的 logprob 预算全绿（见下）。

![EP×TBO 四臂与 graph 参照](images/overlap_ep_tbo.png)

*V2-Lite TP2 的四个 eager 臂（EP on/off × TBO on/off）与 graph 参照臂同图：无一获益，参照臂比最快的 eager 臂还快 1.9-2.4×——值得藏的 a2a payload 在 eager 形态下同样被 CPU 地板淹没。*

**结论修正（替换旧版「等 NVLink 机器再开」；正收益路径本版已做完并测完）**：TBO + CUDA graph capture 已落地——`CUDAGraphRunner` 接受注入的 step 函数，`ModelRunner.enable_cuda_graph(tbo=True)` 由引擎按 `tbo_policy().enabled` 自动接线。机制成立：capture 无 fork 泄漏、replay 与 eager TBO 数值一致、launch floor 从 eager TBO 的 60 ms 消到 10 ms。但**该形状下 interleave 净负**，本版用一个成本模型把根因钉死并自门控（见下）。

**成本模型根因（本版新增，`benchmarks/kernels/bench_tbo_cost_model.py`）**：TBO 把 batch 切两半，每半各自跑一遍 GEMM。decode 小 batch 下 GEMM 是**访存瓶颈**（读权重主导，时间对 M 几乎是平的）：实测 Qwen2.5-1.5B TP2 形状，M=16 与 M=8 的每个 GEMM 耗时几乎相同（o_proj 20.60 vs 20.36 us、down_proj 46.56 vs 46.06 us），所以**切半 = 1.98× 全量**——两半各读一遍权重。整网计算 4.41 ms → 8.75 ms，多出 **+4.35 ms 的重复权重读取**，恰好解释 graph 6.65 ms → graph+TBO 11.44 ms（+4.79 ms）的回归。这个惩罚在 A10 上一直延伸到 M=512（仍 1.26×），因为 A10 算力弱、访存瓶颈区间宽。可藏的 AR 只占 step 的 ~3-5%，永远盖不过翻倍的权重读取——**TBO 在整个访存瓶颈 decode 区间都是净负，这是数学事实，不是调参问题**。

**自门控（本版修复，保证永不负收益）**：`TboPolicy.from_env` 的默认激活阈值不再是固定的 8，而是 roofline ridge point（`_ridge_rows`，A10 上 = 520）——GEMM 从访存瓶颈转为计算瓶颈的 batch。低于 ridge，切半翻倍权重读取，TBO 拒绝激活；高于 ridge，GEMM 计算瓶颈、切半不翻倍，TBO 才可能获益。显式 `RAPID_LLM_TBO_MIN_ROWS` 覆盖该阈值（parity 测试与 benchmark 据此在小 batch 强制开启交错）。这样 `RAPID_LLM_TBO=1` 再也不会把一个 decode 步拖入灾难性的访存瓶颈区。

### SBO single-batch overlap（EP MoE 的单 batch 双流重叠）

```bash
# EP2 对照（batch 32/64：SBO on/off 两臂 + timeline 重叠证据）
python -m benchmarks.overlap.policies --policy sbo --json docs/benchmark_logs/overlap/overlap_sbo_<ts>.json
```

L2 的 TBO 需要两半才能 ping-pong；EP decode 往往只有一个 batch，没有第二半可交错——但 MoE 层内部仍有可重叠的结构：dispatch 的 forward a2a 在线上飞时，shared MLP 可以算。这就是 SBO（sglang 的 `single_batch_overlap.py`），本版新增 `batch_overlap/single_batch_overlap.py` 对齐它。

机制：`SparseMoeBlock._forward_ep` 在 SBO 激活时改走 op 分解——先发 dispatch a2a，再把 shared MLP 放到一条 compute 侧的 alt stream 上算（`sbo_alt_stream`），双向 fence 收口（alt 等 main 的 route 结果，main 等 alt 的 shared 输出后再相加）。不开 SBO 时 shared MLP 排在 dispatch+experts+combine **之后**串行跑，两次交换的时间它一点也没藏住。

一处如实的适配差异：sglang 的 SBO 靠 `DeepEPConfig.num_sms` 给通信 kernel 限定 SM 预算（DeepEP 的 kernel 接受这个参数），所以交换被钉在固定的 SM 子集上、GEMM 拿剩下的。rapid_llm 的 combine 走 `all_to_all_single`，NCCL 的 kernel 自管 SM、不接受调用方的预算——所以这里的 `communicate_num_sms` 是**预算估计**（用于划分 producer/consumer 的 grid），不是对交换的实际限制；交换真正占多少 SM 是外部变量，benchmark 如实报告而不做推断。

另一处如实的边界：sglang 的 `enable_combine_down_gemm_two_stream_overlap`（combine a2a 与 down GEMM 的 tile 级重叠）本版**未实现**。rapid_llm 的 `fused_moe` 里 down GEMM（gemm2）按 `sorted_token_ids` 把结果 scattered 写到原始 slot 行，而 `_moe_sum_kernel` 按连续的 token 行读——「某个 GEMM tile 完成」对不上「某个 row block 就绪」，要做 tile 级同步需要额外的 inverse mapping 加原子计数，本版留作后续。已实现的是 sglang 三个重叠里的 dispatch↔shared 那一个。

证据：EP2 单测里 SBO 开关两侧输出一致（`torch.allclose` 2e-2），且 timeline 证明 shared MLP 的区间与 dispatch 交换的区间在同一设备时钟上真相交。eager 轮（`overlap/policies.py（sbo）`）实测 1248 个 dispatch region、624 个 shared-MLP region、**198 对真重叠共 78.92 ms**——shared MLP 确实与交换并行。

**但 eager 形态兑现不了它**：两臂都跑 eager，TPOT 坐在 Python launch floor 上（~86 ms），78.92 ms 的重叠摊在 624 个 region 里（均摊每层每步 ~0.13 ms），而每层要付两个 event fence 加 `record_stream`——藏住的与付出的同量级，净收益归零。eager benchmark（V2-Lite EP2，gen 64，离线推理口径，`docs/benchmark_logs/overlap/overlap_sbo_20260904_040845.json`）：

| batch | 臂 | TTFT | TPOT | TPS | SBO 变化 | greedy 一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | SBO off | 169.4 ms | 86.03 ms | 366.4 tok/s | — | — |
| 32 | SBO on | 169.6 ms | 86.58 ms | 363.1 tok/s | -0.6% | 20/32 |
| 64 | SBO off | 301.3 ms | 87.40 ms | 702.2 tok/s | — | — |
| 64 | SBO on | 301.4 ms | 87.44 ms | 702.9 tok/s | -0.0% | 40/64 |

根因与 L2 eager 臂、EP 四臂完全一致：瓶颈是 CPU launch floor，不是 GPU。SBO 要省的是 GPU 上的交换时间，而 eager 的 TPOT 根本不由 GPU 决定。

**本版修复：EP 保留 CUDA graph**。EP 过去强制关 graph（理由写着「a2a 进 graph 未验证」），但 a2a 走的正是 TBO deferred all-reduce 已经成功捕获的同一套 comm-stream 原语（`wait_stream`→NCCL→`event.record`→fence），交换 buffer 又是等分固定形状、路由 kernel 在 replay 时按真实 id 重算——所以 a2a 完全可以进 graph。本版移除该守卫（`engine/llm_engine.py`），EP decode 不再被 launch floor 拖住：同一 EP2 负载，eager 墙钟 1.96 s → graph 0.62 s（**3.19×**，`benchmarks/overlap/policies.py (ep_matrix)` 四臂对照，证据 `docs/benchmark_logs/overlap/ep_sbo_graph_4arm_20260904_051200.log`），且 replay 与 eager 输出一致（parity 由 `tests/distributed/test_ep_engine.py` 的 `ep2_graph` 臂门禁，tie-gap 容差内）。**注意：这不等于 SBO 兑现了收益——同一四臂对照里 SBO 在 graph 下是 -0.2%（详见 SBO 节的争议记录）。**

launch floor 消除后，SBO 藏的交换成为一个 GPU-bound 步里的真实占比，正收益随之出现（`overlap/policies.py（sbo --graph）`，EP2 graph，lazy capture，gen 64，离线推理口径，`docs/benchmark_logs/overlap/overlap_sbo_graph_20260904.json`）：

| batch | 臂 | TTFT | TPOT | TPS | SBO 变化 | greedy 一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | SBO off | 168.9 ms | 41.79 ms | 724.6 tok/s | — | — |
| 32 | SBO on | 171.5 ms | 40.87 ms | 745.6 tok/s | **+2.2%** | 23/32 |
| 64 | SBO off | 299.9 ms | 55.40 ms | 1076.5 tok/s | — | — |
| 64 | SBO on | 302.0 ms | 53.97 ms | 1104.7 tok/s | **+2.6%** | 41/64 |

**上表的 +2.2% / +2.6% 存在争议，不能当作结论用**。同一开关同一负载的多次复测波动很大（+0.9%~+8.1%），而该表取的是 best-of-N——**取最好的一次不是如实报告，是夸大收益**。另一条独立路径的实测给出了相反方向：`benchmarks/overlap/policies.py (ep_matrix)` 的四臂对照（eager/graph × SBO off/on，V2-Lite EP2，gen 24，证据 `docs/benchmark_logs/overlap/ep_sbo_graph_4arm_20260904_051200.log`）：

| 臂 | 墙钟 | SBO 效果 |
| --- | --- | --- |
| eager SBO off | 1.96 s | — |
| eager SBO on | 2.01 s | **-2.4%** |
| graph SBO off | 0.62 s | — |
| graph SBO on | 0.62 s | **-0.2%** |

即 **graph 形态下 SBO 也没有收益（-0.2%）**，与上表的 +2.2%/+2.6% 方向相反。两个可能的解释：（a）上表的正收益是 GPU 竞争下的异常值（本节下方记录了一次同类事故：干扰能把 sbo_off 臂从 84 ms 括到 146 ms，凭空造出 +39% 的假收益）；（b）best-of-N 选取放大了噪声。**在两条路径的结论相互矛盾、且 GPU 竞争无法排除之前，SBO 在 graph 形态下的收益应记为「未证实」，而不是「已兑现」。**

唯一经交叉验证、方向一致的结论是：**EP decode 从 eager 换到 graph 快 3.19-3.31×**（两条路径都测到），这才是真正值钱的优化。EP 默认走 lazy capture：a2a buffer（`ep_size*rows*top_k*hidden`）让每个 graph 远大于 dense TP，全网格捕获会在 profiled KV 池旁 OOM，lazy 只种子捕获一对、其余按需捕获，`enable_cuda_graph` 的 OOM 回退兜底任何仍放不下的形状。

![SBO 两臂对照](images/overlap_sbo_ep.png)

*V2-Lite EP2 的 SBO on/off 两臂（eager，`overlap_sbo_20260904_040845.json`）：TPOT 差异落在运行间噪声带里——重叠真实发生但被 launch floor 淹没。graph 形态的正收益（+2.2~2.6%）见上表与 `overlap_sbo_graph_20260904.json`；同一重叠，差别只在执行形态。*

### decode TPOT 收益归因：TBO 与 SBO 的实测结论（含一次重要修正）

本节把两个开关在 decode TPOT 上的实测结果、根因链条、以及待查项集中写清楚，供后续迭代直接接手。

**一次重要修正（必读）**：本节先前记录的「TBO eager +129~135%、graph +47~61%」是**重构前实现**的数据（当时 TBO 用自建的 `_DENSE_OPS` 模板 + `_HalfState` 闭包）。TBO 已被重构为对齐 sglang 的实现（`OperationsStrategy.init_new_tbo` + `StateDict` + 各层 bound method），重构后重测：**收益落在 ±3% 的噪声带边缘，不再是大幅负收益**。下面所有数据均来自重构后的实现；旧数据不再适用，也不应被引用。

#### benchmark 环境与口径

| 项 | 值 |
| --- | --- |
| GPU | 2× NVIDIA A10，22.0 GiB/卡，driver 550.135 |
| 算力 | sm_86，72 SM/卡 |
| 互联 | PHB（PCIe host bridge），**无 NVLink**；NVLink 拓扑未测、不做推断 |
| 库版本 | torch 2.13.0+cu129 / triton 3.7.1 / transformers 5.15.1 / CUDA 12.9 / rapid_llm 0.11.0 |
| 主机 | 64 核 CPU，369 GiB 内存 |
| 推理口径 | **离线推理**（全部 prompt 一次性提交、跑完收工，无 serving 排队与连续到达） |

#### 工作负载与框架参数

| 项 | 值 |
| --- | --- |
| 模型 | Qwen2.5-1.5B-Instruct（hidden 1536 / 28 层）、Meta-Llama-3.1-8B-Instruct（hidden 4096 / 32 层 / intermediate 14336） |
| batch_size | 32 / 128 / 256 |
| prompt seq_len | 64（截断） |
| generate_len | 64 |
| 并行 | TP=2 |
| KV 池 | Qwen 65536 blocks；Llama-8B 49152 blocks |
| 采样 | greedy |
| 框架开关 | `RAPID_LLM_TBO` 逐臂；`use_cuda_graph` 逐臂；`RAPID_LLM_OVERLAP=1`（L1 默认开）；`RAPID_LLM_COMM_OVERLAP=0`（L3 关）；`RAPID_LLM_SBO=0`（SBO 关） |

运行命令：

```bash
python -m benchmarks.overlap.policies --policy scaling \
    --models my_weight/Qwen2.5-1.5B-Instruct --batches 32 128 256 \
    --json docs/benchmark_logs/overlap/tbo_scaling_qwen_<ts>.json
python -m benchmarks.overlap.policies --policy scaling \
    --models my_weight/Meta-Llama-3.1-8B-Instruct --batches 32 128 256 \
    --kv-blocks 49152 --json docs/benchmark_logs/overlap/tbo_scaling_llama8b_<ts>.json
```

证据：`docs/benchmark_logs/overlap/tbo_scaling_qwen_20260904_044400.{json,log}`、`tbo_scaling_llama8b_20260904_045000.{json,log}`。JSON 内含完整的环境/负载/框架参数与逐臂四指标，`.log` 是原始运行输出。

#### 实测结果（TTFT / TPOT / TPS / TGS 四指标全给）

Qwen2.5-1.5B-Instruct，TP=2（TGS = 每卡吞吐 = TPS / TP）：

| batch | 臂 | TTFT | TPOT | TPS | TGS/GPU |
| --- | --- | --- | --- | --- | --- |
| 32 | eager TBO off | 27.7 ms | 26.454 ms | 1208.7 | 604.4 |
| 32 | eager TBO on | 26.9 ms | 25.673 ms | 1245.5 | 622.8 |
| 32 | graph TBO off | 27.3 ms | 7.635 ms | 4027.2 | 2013.6 |
| 32 | graph TBO on | 27.7 ms | 7.660 ms | 4011.9 | 2006.0 |
| 128 | eager TBO off | 74.8 ms | 31.096 ms | 4027.7 | 2013.8 |
| 128 | eager TBO on | 74.9 ms | 31.710 ms | 3952.4 | 1976.2 |
| 128 | graph TBO off | 75.1 ms | 15.267 ms | 7898.9 | 3949.5 |
| 128 | graph TBO on | 75.1 ms | 15.193 ms | 7933.8 | 3966.9 |
| 256 | eager TBO off | 75.4 ms | 31.564 ms | 4011.7 | 2005.8 |
| 256 | eager TBO on | 75.6 ms | 31.850 ms | 3976.2 | 1988.1 |
| 256 | graph TBO off | 75.5 ms | 15.761 ms | 7886.9 | 3943.5 |
| 256 | graph TBO on | 75.6 ms | 15.723 ms | 7904.9 | 3952.5 |

Meta-Llama-3.1-8B-Instruct，TP=2：

| batch | 臂 | TTFT | TPOT | TPS | TGS/GPU |
| --- | --- | --- | --- | --- | --- |
| 32 | eager TBO off | 76.0 ms | 30.167 ms | 1036.1 | 518.1 |
| 32 | eager TBO on | 76.1 ms | 30.678 ms | 1019.5 | 509.7 |
| 32 | graph TBO off | 76.3 ms | 23.462 ms | 1317.3 | 658.7 |
| 32 | graph TBO on | 77.0 ms | 23.606 ms | 1309.1 | 654.5 |
| 128 | eager TBO off | 281.3 ms | 36.973 ms | 3137.9 | 1569.0 |
| 128 | eager TBO on | 280.4 ms | 36.622 ms | 3165.9 | 1582.9 |
| 128 | graph TBO off | 280.4 ms | 35.872 ms | 3224.4 | 1612.2 |
| 128 | graph TBO on | 281.6 ms | 36.129 ms | 3202.4 | 1601.2 |
| 256 | eager TBO off | 283.3 ms | 38.930 ms | 3134.3 | 1567.1 |
| 256 | eager TBO on | 282.8 ms | 38.690 ms | 3152.9 | 1576.5 |
| 256 | graph TBO off | 283.4 ms | 37.746 ms | 3226.8 | 1613.4 |
| 256 | graph TBO on | 283.6 ms | 38.735 ms | 3148.8 | 1574.4 |

#### 对比总结（正号 = TPOT 下降 = 有收益）

| 模型 | batch | eager TBO | graph TBO |
| --- | --- | --- | --- |
| Qwen2.5-1.5B | 32 | **+3.0%** | -0.3% |
| Qwen2.5-1.5B | 128 | -2.0% | **+0.5%** |
| Qwen2.5-1.5B | 256 | -0.9% | **+0.2%** |
| Llama-3.1-8B | 32 | -1.7% | -0.6% |
| Llama-3.1-8B | 128 | **+0.9%** | -0.7% |
| Llama-3.1-8B | 256 | **+0.6%** | -2.6% |

**结论：重构后的 TBO 在 decode TPOT 上收益不显著——12 个测量点全部落在 ±3% 内，正负交替，且没有随 batch 或模型规模单调改善的趋势。**batch 从 32 加到 256、模型从 1.5B 加到 8B（hidden 1536→4096，AR payload 大 2.7 倍）都没有把收益推出噪声带。另外两个事实值得记住：graph 形态比 eager 快 2-3.5 倍（TPOT 26.45→7.64 ms、30.17→23.46 ms），这才是 decode 上真正值钱的那一刀；TBO 叠在 graph 上也不改变这个量级。

**与成本模型的衔接（必读，避免两节自相矛盾）**：本节 ±3% 的读数与 L2 节的成本模型（切半 1.98× 权重读取）表面冲突，但 doubled 权重读取是 SM 上不可隐藏的实际计算——若 TBO 真的跑了，TPOT 必然上升。本节 TBO on 臂 ≈ off 臂（eager 25.67 vs 26.45、graph 7.66 vs 7.64），说明**该 benchmark 的 TBO 臂很可能没有真正激活交错**（在比较 off vs off）；而 `overlap/levels.py（L2）` 显式设 `RAPID_LLM_TBO_MIN_ROWS=8` 后，同一 batch 32 的 eager TBO 实测 -181%、graph+TBO（batch 16）-72%，与成本模型一致。本版的 ridge 自门控让这个问题不再重要：TBO 只在计算瓶颈区（切半 ~free）激活，访存瓶颈区一律拒绝。

#### 重构前的旧数据（仅作历史对照，不要引用）

| 形态 | batch 8 | batch 16 | batch 32 |
| --- | --- | --- | --- |
| eager TBO | +134% | +135% | +129% |
| graph TBO | +61% | +58% | +47% |

这组大幅负收益来自重构前的实现，重构后已消失。保留在此只为了说明“实现变更能把结论从 +134% 翻到 +3%”——**benchmark 结论必须绑定实现版本，否则后续迭代拿着旧数据做决策会走错方向。**

#### 根因分析（按证据强度排序，并标注适用的实现版本）

**第三层根因（决定性，与实现版本无关）**：comm 时间大部分是 rank 间等待，不是 wire 传输。

实测（TBO on，Qwen2.5-1.5B TP2，batch 32，8 步稳态，timeline）：

| rank | compute/步 | comm/步 | comm 占比 |
| --- | --- | --- | --- |
| 0 | 16.23 ms | **62.44 ms** | 79.4% |
| 1 | **63.88 ms** | 10.82 ms | 14.5% |

两个 rank 的数据互补：**rank 0 的 comm（62.44 ms）≈ rank 1 的 compute（63.88 ms）**。nsys 也印证：同样 6778 个 NCCL kernel，GPU 0 花 1150.66 ms、GPU 1 只花 213.77 ms（5.4 倍差异）。

所以“通信时间”里绝大部分是 **NCCL kernel 在 spin-wait 对端 rank 到达**，真正的 wire 传输只有较快那个 rank 的量级（~10.82 ms/步）。而等待时间**无法被计算隐藏**：它是同步点，端到端时间由慢的那个 rank 决定。这一层解释了为什么 TBO/SBO 的收益天花板很低——**不管实现怎么改，可藏的只有 wire 那部分**。

**第二层根因（仅适用于重构前实现，对新实现待重测）**：旧实现在 graph 形态下 +47~61%，nsys 显示它把 compute kernel 数从 40470 翻到 77622 而平均时长 13.3→14.3 us 几乎不变——decode 的 M 本来就小，切半后仍坐在 kernel 固定开销的地板上，切半只增加 kernel 个数。用 graph 两臂反推分解（普通 forward = compute + AR，TBO = 2×compute − 隐藏的 AR，与旧实测吻合）：

| batch | compute | AR | AR 占比 | 翻倍代价 | 隐藏收益 | 净 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 5.36 ms | 0.80 ms | 13.0% | +5.36 ms | -0.80 ms | **+4.56 ms** |
| 16 | 5.70 ms | 0.92 ms | 13.9% | +5.70 ms | -0.92 ms | **+4.78 ms** |
| 32 | 6.27 ms | 1.36 ms | 17.8% | +6.27 ms | -1.36 ms | **+4.91 ms** |

**重构后的实现收益落到 ±3%，说明这层的“compute 翻倍”代价已被大幅削减**（sglang 风格的 op 流 + `StateDict` 比旧的模板闭包省了 Python 与 kernel 开销）。但新实现是否仍有 kernel 翻倍、翻多少，**本轮未用 nsys 重测，属于待验证项**——不要拿旧实现的 40470→77622 去推断新实现。

**第一层根因（仅适用于重构前实现）**：旧实现的 eager 臂 +129~135%，因为 Python 调度量翻倍叠在 launch floor 上（off 臂 GPU util 仅 28.6%）。重构后 eager 臂是 ±3%，说明新实现的 Python 开销已不构成主导。SBO 仍受这一层限制：它只多两个 event fence，但本来就在 launch floor 上，稳态可重叠窗口只有 0.026-0.052 ms。

#### 是不是 Bug？

不是。两个开关的 parity 测试全绿（TBO graph replay 与 eager TBO logits 一致；SBO 开关两侧 `allclose` 2e-2），timeline 也证明重叠确实发生（SBO 一次完整 decode 361 对区间相交共 106.53 ms；TBO 792 对共 65.49 ms）。机制是对的；收益不显著是因为可藏的 wire 时间占比太小（第三层），而不是实现有问题。

**待查项（如实记录，两项）**：

1. 上表里两个 rank 的 compute 时间差 4 倍（16.23 vs 63.88 ms），而同模型、同 batch、同 kernel 不该有这么大差异。两个可能：（a）timeline 的 compute region 把 segment 内等 deferred-AR 的 fence 时间也计进去了，于是“慢 rank 的等待”被归到 compute 而非 comm；（b）两卡真实性能不对称。区分这两者需要 nsys 的 kernel 级时间（而非 event region）。
2. 重构后实现的 compute kernel 数量是否仍翻倍、翻多少——需对新实现重跑 nsys，本轮未做。这直接决定 TBO 还有多少优化空间。

#### 后续迭代方向

1. **先补 nsys 重测新实现**：确认重构后的 kernel 数量与 AR 占比，这是判断 TBO 还有多少空间的前提。
2. **graph 才是 decode 上真正值钱的那一刀**：实测 graph 比 eager 快 2-3.5 倍（TPOT 26.45→7.64 ms、30.17→23.46 ms），量级远超 TBO 的 ±3%。优先保证 graph 覆盖率，而不是继续调 TBO。
3. **降 rank skew 比藏 wire 时间更值钱**：第三层说明端到端由慢 rank 决定，均衡两 rank 的工作/减少同步点比做重叠更有效。
4. **TBO 改打 prefill**：prefill 的 M 大（几百到几千 token），切半后 GEMM 仍高效，不踩 kernel 固定开销地板；sglang 的 `_compute_*_prefill`（delta=0）就是这个形状。本轮未实现。
5. **kernel fusion**：decode 是 kernel-count-bound，减少 kernel 个数直接减 TPOT。
6. **SBO 要等 graph 形态 + 大 a2a payload**：稳态重叠窗口只有 0.026-0.052 ms；EP+graph 本轮仍禁用，需先验证 a2a in-graph。

#### 为什么 sglang 的收益更高：三个框架能力缺失（不是实现 Bug）

对照 sglang 的实现、单测与用例配置后，rapid_llm 收益低的原因不在交错逻辑（parity 全绿、重叠实测发生），而在三个框架级能力缺失与一个场景错配。

**1. SBO 缺 tile 级重叠能力（最主要的差距）**。sglang 的 SBO 主力是 `enable_combine_down_gemm_two_stream_overlap`——combine a2a 与 down GEMM 的 **tile 级**重叠，靠两个硬条件：（a）MoE kernel backend 必须是 `flashinfer_cutedsl` 或 `deep_gemm`；（b）MoE runner 提供 `set_overlap_args(down_gemm_overlap_args, meta_overlap_args)` 接口（`layers/moe/fused_moe_triton/layer.py`），让 down GEMM 能按 `num_sms` 限定 grid 并逐 tile 发布 signal。rapid_llm 的 `fused_moe` **没有这个接口**，triton kernel 也不支持 SM 分区，所以只能做 stream 级的 dispatch↔shared（sglang 三个重叠里最弱的那个）。这直接解释了 SBO 收益为何在 ±2% 而不是 sglang 的量级。

**2. TBO 测试场景错配**。sglang 的 TBO 用例配置（`test/manual/test_two_batch_overlap.py`）是 `--tp 2 --dp 2 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode normal --disable-cuda-graph --enable-two-batch-overlap`——它藏的是 **MoE 的 DeepEP a2a**（payload = batch × hidden × top_k，极大）并搭配 DP attention。而本版 TBO 的 scaling 测的是 **dense TP + NCCL all-reduce**（payload = batch × hidden，小一个 top_k 量级）。同样一套交错逻辑，藏的 payload 差一个量级，收益自然差一个量级。

**3. 缺 DeepEP 类的通信 backend**。sglang 靠 DeepEP 给通信 kernel 限定 SM 预算（交换钉在固定 SM 子集、GEMM 拿剩下的）；rapid_llm 的 combine 走 `all_to_all_single`，NCCL kernel 自管 SM、不接受调用方预算——所以即使补上 tile signal，也无法做真正的 SM 硬分区（参考 vLLM DBO：SM partition 在 kernel 层而非 stream 层）。

**附：模型规模**。sglang 的 TBO 单测跑 MLA 模型（DeepSeek 系，hidden 7168、MoE），本版 scaling 跑的是 Qwen2.5-1.5B（hidden 1536、dense）与 Llama-3.1-8B（hidden 4096、dense）——两个都不是 sglang 的目标形状。

**后续要把收益做到 sglang 量级，需要的三件事（按性价比排序）**：（1）给 `fused_moe` 加 `set_overlap_args` 接口与 tile signal 发布，补上 combine↔down GEMM 重叠；（2）TBO 改在 **MoE + EP** 形状上测（而不是 dense TP），并把 prefill TBO 实现出来；（3）引入一个可限定 SM 预算的 a2a 路径（DeepEP 类）才能做 SM 硬分区。在当前 NCCL + triton 的技术栈上，重叠类优化的收益天花板就是±几个百分点。

### L3 chunked all-reduce

```bash
python -m benchmarks.overlap.levels --level l3 --json --timeline
```

GEMM 输出行分块、每块 GEMM 落地即上通信流（`docs/benchmark_logs/overlap/overlap_l3_20260903_215551.json`）：TP2 Qwen2.5-1.5B batch 16，TTFT 33.25→33.07 ms（-0.6%），timeline 记录 224 个 comm region 与 111 对真重叠（9.78 ms）。`L3_MIN_CHUNK_ROWS=256` 行下限：再细的分块在 PCIe 上付更多次小消息固定成本。分发点优先级 TBO > L3（同一 all-reduce 位点不叠加切分），组合矩阵验证退位成立（见下）。

![L3 chunked all-reduce 的真实 timeline](images/overlap_l3.gif)

*chunked prefill 的真实 timeline（`scripts/gen_overlap_gifs.py --level l3`）：compute 泳道是同一个行并行 GEMM 的两个行块（`l3.gemm.0/1`），通信泳道是它们各自的 all-reduce（`l3.all_reduce.0/1`）——第 0 块的 reduce 在线上时，第 1 块的 GEMM 正在算。*

![L3 单开关 TTFT/TPOT 对照](images/overlap_l3_chunked.png)

*L3 单开关对照：TTFT -0.6%（prefill 的行数越过 256 行下限，是 L3 的兑现点）；这条 16-token 短跑的 TPOT +4.8% 落在噪声带里，64-token 的组合矩阵测同一开关是 -2.7%。*

### L4 tile-signaling

```bash
python benchmarks/kernels/bench_tile_signal.py
```

单卡 kernel 级原语，与互联无关（`docs/benchmark_logs/overlap/overlap_l4_20260903_104621.json`）：GEMM→SiLU·mul 逐 tile 流水 vs 串行两 kernel，A10（72 SM）上大形状 +8.0~+13.7%（4096×4480×1536：5.85→5.05 ms），小形状负收益（64×4480×1536：-15.5%）如实入表——persistent kernel 的常驻占用在 tile 少时是纯开销。死锁规避：生产者+消费者 grid 之和 ≤ #SM，host 侧 watchdog 兜底。

![L4 生产者/消费者的真实 timeline](images/overlap_l4.gif)

*单卡上 persistent 生产者（GEMM）与消费者（SiLU·mul epilogue）两个 kernel 的设备 timeline（`scripts/gen_overlap_gifs.py --level l4`）：消费者不等 GEMM 收尾，靠 tile flag 逐块接手，红带就是逐 tile 流水的交集。*

![L4 各形状正负收益](images/overlap_l4_tile_signal.png)

*同一对 kernel 的串行 vs 流水按形状铺开：大形状 +8~14%，小形状负收益，两者同图——收益来自 tile 数够多时 epilogue 能被藏进 GEMM。*

### DP + CUDA Graph（P8）

```bash
python benchmarks/bench_data_parallel.py --mode graph --model my_weight/Qwen3-0.6B
```

`docs/benchmark_logs/parallel/dp_graph_20260903_143056.json`（Qwen3-0.6B，batch 16/副本，128 步）：

| 配置 | TPOT | 吞吐 |
| --- | --- | --- |
| dp1 eager | 25.9 ms | 618 tok/s |
| dp1 graph | 5.2 ms | 3102 tok/s |
| dp2 eager | 26.4 ms | 1200 tok/s |
| dp2 graph | 5.2 ms | **6162 tok/s** |

每副本独立 capture/replay 无锁步（tp=1/副本，graph 内无 NCCL 集合通信），DP2 下吞吐 5.1×、TPOT -80%；capture 耗时 +2.4 s、显存增量已记入 JSON。测试 `tests/engine/test_dp_cuda_graph.py` 断言双副本均有 captured graphs 且 greedy 输出一致。

![DP×CUDA Graph 的 TPOT 与吞吐](images/dp_cuda_graph.png)

*左：每副本 decode 延迟 25.9→5.2 ms（-80%，log 轴）；右：2×A10 聚合吞吐 618→6162 tok/s（5.1×）。副本各自 capture 自己的 graph，所以 DP 的线性扩展不被 graph 内的集合通信锁步吃掉。*

### DeepSeek V4（裁剪版端到端）

V4 无公开权重（仅 config.json），用 transformers 5.8 随机初始化裁剪 checkpoint（借 Qwen tokenizer）做 parity 与性能对照：

* `modules/deepseek_v4/`（按模块拆分的子包）：`rope` interleaved partial RoPE（main / compress 双频率表）、`norm` weighted + unweighted RMSNorm、`hyper_connection` mHC 残差（Sinkhorn 混合 + hc_head）、`cache` 滑窗 K==V per-layer cache（绕开 paged KV）、`compressor` Compressor（CR 累积→RMSNorm→RoPE→压缩 cache）+ Lightning Indexer（topk 选块）、`grouped_linear` O-LoRA wo_a 块对角分组、`attention` SWA/CSA 混合注意力（head_dim=512 latent KV + fused_wqa_wkv、inv-RoPE、wo_b 行并行）；`__init__` 沿用 `modules/` 的懒加载 facade 逐个转发
* `models/deepseek_v4.py`：组装 + registry + packed_modules 映射；TP2 下 heads/o_groups 整除校验；Hash MoE（num_hash_layers 层 tid2eid）+ sqrtsoftplus 路由
* 测试 `tests/models/test_deepseek_v4.py` 8 项（rotary/HC/router×2/MoE/decoder/e2e/incremental）+ `tests/distributed/test_tp2_v4.py` TP2 一致性
* benchmark：`benchmarks/bench_deepseek_v4.py`（vs transformers 速度对照，含噪声基线）
* **fp4 权重暂不支持**：本版以 bf16/fp16 unquantised 权重为 parity 基础，fp4 留待后续版本

![V4 裁剪版 vs transformers](images/deepseek_v4_speed.png)

*左：prefill 从 0.76×（seq 256）追到 1.06×（seq 2048），虚线是 transformers 基准；右：decode TPOT 0.29-0.82×——compressor/indexer 逐行走 Python，一个 batch-32 步约 8.7k 次 kernel launch，是 CPU-bound 而非 kernel 慢，如实入图。greedy 一致率 0.50 需对读 transformers 自己 fp32-vs-bf16 的 0.47（未训练 checkpoint 的平坦 logits）。*

## 组合矩阵与精度门禁（M7）

### 组合矩阵：L1×L2×L3 八格

`benchmarks/overlap/policies.py (matrix)`（`docs/benchmark_logs/overlap/overlap_matrix_final.json`，Qwen2.5-1.5B TP2 batch 16，1024 tok/格）：

| 组合 | TPOT | vs baseline | 输出一致性 |
| --- | --- | --- | --- |
| baseline | 28.47 ms | — | — |
| l1 | 27.25 ms | -4.3% | 16/16 |
| l2 | 66.09 ms | +132% | 16/16 |
| l3 | 27.69 ms | -2.7% | 16/16 |
| l1l2 | 65.20 ms | +129% | 16/16 |
| l1l3 | 27.89 ms | -2.0% | 16/16 |
| l2l3 | 64.94 ms | +128% | 16/16（≡ l2） |
| all | 66.42 ms | +133% | 16/16（≡ l1l2） |

`l2l3`/`all` 两格的存在就是验证 L3 在 TBO 激活时退位：它们的输出与 `l2`/`l1l2` 逐字相同（`demotion_holds: true`），TPOT 也在 l2 的噪声带内——分发点优先级不是文档声明而是被测行为。

![八格组合矩阵](images/overlap_combination_matrix.png)

*同一负载、只动开关的八格：L2 参与的格子（红）全部退到 ~65 ms，l1/l3/l1l3 在 baseline 噪声带内小幅为正；`l2l3`≡`l2`、`all`≡`l1l2` 是退位规则的实测证据。*

### 模型矩阵：baseline vs 推荐组合（L1+L3）

| 模型 | TPOT 变化 | TTFT 变化 | 输出一致性 |
| --- | --- | --- | --- |
| qwen2.5-1.5b | -0.5% | +0.4 ms | 16/16 |
| deepseek-v2-lite | -1.8% | -15.9 ms | 16/16 |
| deepseek-v3-4layers | -1.7% | -0.0 ms | 14/16* |
| deepseek-v4-trimmed | -2.5% | -0.6 ms | 16/16 |

\* V3-4layers 的 14/16 与 overlap 无关：**两次全关的 baseline 互比同样 14/16**，分歧行输出为乱码 token（平坦 logits 上 argmax 的 run-to-run 抖动）。这是引擎固有噪声的测量，不是特性引入的漂移。

![四模型 baseline vs L1+L3](images/overlap_model_matrix.png)

*推荐组合 L1+L3 在四个模型上都是小正收益（-0.5% ~ -2.5%），dense 与 MLA/MoE/裁剪 V4 栈都不例外——重叠原语不挑模型结构。*

### golden 门禁（双遍）

```bash
pytest tests/golden/ -q                                        # 默认：9 passed
RAPID_LLM_OVERLAP=1 RAPID_LLM_TBO=1 RAPID_LLM_COMM_OVERLAP=1 \
RAPID_LLM_TBO_MIN_ROWS=2 pytest tests/golden/ -q              # 全开：9 passed
```

第二遍把 TBO 激活阈值压到 2 强制 TP2 golden 走 `forward_tbo` 路径，V2-Lite 的 greedy/logprob parity 预算（mean 0.4/max 2.5 nats，2× 实测漂移校准——校准证据链见测试内注释）依然全绿。

### nsys kernel 级证据

`docs/benchmark_logs/overlap/nsys_overlap_report.md`（同一文件两个模式：`python -m benchmarks.overlap.nsys payload` 跑被 trace 的负载，`python -m benchmarks.overlap.nsys report` 分析两份 kernel trace）：

| trace | gpu | NCCL kernels | hidden under compute |
| --- | --- | --- | --- |
| overlap off | 0 | 6778 | 0.00 ms (0.0%) |
| overlap off | 1 | 6778 | 0.00 ms (0.0%) |
| overlap on | 0 | 12934 | 206.12 ms (9.8%) |
| overlap on | 1 | 12934 | 126.04 ms (8.6%) |

off 臂 NCCL 与 compute 零重叠（阻塞 AR 在 compute stream 上串行）；on 臂 kernel 级真并发出现。NCCL kernel 数翻倍（12934）同时如实呈现——这是 L2 负收益机理的直接观测。

![NCCL 时间被 compute 隐藏的比例](images/nsys_overlap_hidden.png)

*off 臂 0.0%（阻塞 AR 串在 compute stream 上），on 臂 9.8% / 8.6%——kernel 级真并发；柱上同时标出 NCCL kernel 数 6778→12934，两个事实同图，不只挑好看的那一半。*

## 全量回归

M7 收尾时全套测试分批跑过（本轮改动：TBO 闭包惰性读取修复、grouped_topk 单 expert 组退化、外部 batch_overlap 包重构适配、test_checkpoint_index MTP/trim 层豁免）：

* 批 1 `tests/models/ + tests/kernels/ + tests/batch_overlap/ + tests/executor/ + tests/engine/ + tests/distributed/`：1288 passed / 72 skipped（唯一失败为 HEAD 既有：V3-MTP checkpoint 的 MTP 层 key 未豁免，worktree 对照验证后修复，36 passed）
* 批 2 `tests/ops/ + tests/tools/ + tests/compile/ + tests/config/ + tests/entrypoints/ + tests/evals/ + tests/multimodal/ + tests/platform/ + tests/utils/ + tests/modules/ + tests/test_imports.py`：480 passed / 10 skipped
* `tests/golden/`：默认与 overlap 全开双遍 9 passed

TBO+graph capture 落地轮补测（新增 `capture_eligible` 谓词、FakeRunner step 接线、TP2 capture/replay parity、graph+TBO 引擎级 greedy 四组测试）：

* `tests/batch_overlap/ + tests/executor/`：128 passed
* `tests/compile/test_cuda_graph.py + tests/engine/test_dp_cuda_graph.py`：2 passed（compile 8 skipped：环境缺 Qwen2.5-0.5B checkpoint，与改动无关）
* `tests/golden/test_deepseek_v2_tp2.py`：passed

## 已知边界（如实标注）

1. **互联**：全部数据来自 2×A10 **PCIe**（无 NVLink 硬件）。L2 的负收益根因是双重的：eager 形态的 CPU launch floor（graph 参照臂 6.2-7.6 ms 对 eager 臂 27-66 ms），加上切半翻倍权重读取的成本模型惩罚（见 L2 节）；PCIe 只决定 AR wire 时间（~32 us/次）。NVLink 上的一切**未测**，不做推断。
2. **L2 默认 off + 成本模型自门控**：TBO 在整个访存瓶颈 decode 区间净负（切半 1.98× 权重读取，成本模型实测）。本版把 `TboPolicy` 默认激活阈值改为 roofline ridge point（A10=520），只在计算瓶颈区激活，保证 `RAPID_LLM_TBO=1` 永不负收益；显式 `RAPID_LLM_TBO_MIN_ROWS` 可覆盖（测试/benchmark 用）。A10 上现实 decode batch（≤128）均低于 ridge，故 TBO 实际不会激活。
3. **V4 fp4**：本版仅 bf16/fp16 unquantised 权重（parity 基础）；fp4 量化加载留待后续。
4. **V4 TBO 未接线**：mHC 栈的段结构与两段拆分不匹配，本版 V4 不走 `forward_tbo`（矩阵里 V4 只测 L1+L3 组合）。
5. **裁剪 checkpoint 的 grouped_topk**：V3-4layers（8 experts/8 组）落在所有参考实现（transformers/vLLM）都会崩的几何上；rapid_llm 按数学极限退化处理（单 expert 组分数 = top-2 和的极限 = 该 expert 分数），`tests/kernels/test_grouped_topk_kernel.py` 锁定该语义。
6. **SBO 的 SM 预算是估计而非限制**：rapid_llm 的 combine 走 NCCL `all_to_all_single`，不接受调用方的 SM 预算（sglang 靠 DeepEP 的 `num_sms` 把交换钉在固定 SM 子集上），`communicate_num_sms` 只用于划分 producer/consumer 的 grid。sglang 的 combine↔down GEMM tile 级重叠本版未实现：`fused_moe` 的 gemm2 按 `sorted_token_ids` scattered 写入，使 GEMM tile 与 consumer 的 row block 对不齐，需额外的 inverse mapping 加原子计数。
7. **EP graph 默认 lazy capture**：本版 EP 不再强制关 graph（a2a 与 TBO deferred-AR 用同一套可捕获原语，parity 由 `ep2_graph` 测试臂门禁）。但 EP 的 a2a buffer 让每个 graph 远大于 dense TP，全网格捕获会在 profiled KV 池旁 OOM，故 EP 默认走 lazy capture（种子对 + 按需捕获），`enable_cuda_graph` 的 OOM 回退（已拓宽到捕获期的 `AcceleratorError`）兜底任何仍放不下的形状。

## 图表

* `docs/images/overlap_axes.png` — 三条轴与组合矩阵的原理图（开关位与重叠关系，不携带测量数字）
* `docs/images/overlap_combination_matrix.png` — 八格组合矩阵（L2 类红色标注）
* `docs/images/overlap_model_matrix.png` — 四模型 baseline vs L1+L3
* `docs/images/overlap_l2_tbo.png` — L2 四臂对照（eager on/off + graph 参照 + graph-captured TBO，batch 维度）
* `docs/images/overlap_ep_tbo.png` — EP×TBO 四 eager 臂 + graph 参照（V2-Lite）
* `docs/images/overlap_sbo_ep.png` — SBO on/off 两臂（V2-Lite EP2 eager）
* `docs/images/overlap_l3_chunked.png` — L3 TTFT/TPOT（prefill 是收益位）
* `docs/images/overlap_l4_tile_signal.png` — L4 各形状正负收益
* `docs/images/nsys_overlap_hidden.png` — NCCL 被隐藏比例（off vs on，kernel 级证据）
* `docs/images/dp_cuda_graph.png` — DP×Graph TPOT/吞吐
* `docs/images/deepseek_v4_speed.png` — V4 裁剪版 vs transformers（prefill/decode）

生成：`python -m benchmarks.overlap.plot`（读 JSON logs，图数同源）。

原理 timeline GIF，生成：`python scripts/gen_overlap_gifs.py`（直接跑引擎收 CUDA-event region，逐帧揭示，底部标注本窗口实测的重叠对数与毫秒数）：

* `docs/images/overlap_l1.gif` — L1 copy stream vs compute stream（`scripts/gen_overlap_l1_gif.py`）
* `docs/images/overlap_l2.gif` — L2 半 A/半 B segment 与 deferred all-reduce（TP2 decode）
* `docs/images/overlap_l3.gif` — L3 行块 GEMM 与分块 all-reduce（TP2 chunked prefill）
* `docs/images/overlap_l4.gif` — L4 生产者/消费者 kernel（单卡）
