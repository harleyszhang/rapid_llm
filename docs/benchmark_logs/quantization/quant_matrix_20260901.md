# 量化矩阵 — 2×H100 80GB，2026-09-01

以下全部数据在一天之内、同一台机器上测得：2× NVIDIA H100 80GB HBM3（sm90，3352 GB/s，989 TFLOP/s dense tensor core，50 MiB L2），torch 2.13.0+cu130、triton 3.7.1、python 3.14.7，commit `v0.8.0-32-g4f256d7`。`deep_gemm` 与 `flashinfer` 均未安装，因此每一行内核数据都是 native Triton 路径。

模型取自 `$RAPID_LLM_MODELZOO`：

| 代号 | Checkpoint | 用途 |
|---|---|---|
| `M_TINY` | Qwen2.5-0.5B-Instruct | 快速轴扫描、golden 基线 |
| `M_MAIN` | Qwen3-4B-Thinking-2507 | 离线矩阵与在线矩阵 |
| `M_MOE` | Qwen3-30B-A3B-Instruct-2507 | fp8 fused MoE 端到端 |

两个精度参照系，读任何一行数据之前必须先分清它们：

- **golden** — 与本引擎自身（bf16/eager/TP1）的贪心一致率，在 Phase 0 记录（`tests/golden/data/`）。引擎与 tokenizer 固定不动，因此隔离出来的是纯粹的*量化*误差。
- **HF fp16** — 与 `transformers` fp16 的一致率。只报一次，在 bf16 行上，结果为 **0.348**。这是*引擎*在同等精度下与 HuggingFace 的分歧——paged attention、不同的 softmax 归约顺序、fused RMSNorm——golden 参照系正是为绕开这些而存在。量化方案若直接对 HF 测，这些引擎差异会被一并算到量化头上。

## 1. 内核层

### 1.1 量化 dense GEMM

完整表格见 [`quant_gemm_h100_20260901.json`](../kernels/quant_gemm_h100_20260901.json)（2026-09-02 复测为 [`quant_gemm_h100_20260902.json`](../kernels/quant_gemm_h100_20260902.json)，数字与首测一致——dense 路径未受后续 fused MoE 改动影响）。一个**测试点**就是一个「模型 × 投影 × token 数 M」的组合：两个模型（Qwen3-4B、Qwen3-30B-A3B）各取 4 个真实投影（qkv / o / gate_up / down），M 取 `{1, 8, 32, 128, 512, 2048}` 六档，2 × 4 × 6 = **48 个测试点**；每点测 7 行实现（6 种格式加 1 行消融），共 336 行。运行时设 `RAPID_LLM_AUTOTUNE=0`。

**48 个测试点中，cuBLAS bf16 在 44 个上最快。**有用的是那 4 个例外：全部集中在 `qwen3-4b/gate_up`（N=19456, K=2560，bf16 权重约 100 MB，是全表最大的权重矩阵）的 M ≤ 128 四档。下表摘出这一投影的三个 M 档，外加 `qkv` 的首尾两档作对照；括号内为相对 bf16 基线的加速比（bf16 耗时 ÷ 该格式耗时），大于 1 快于基线：

| 投影 (µs) | bf16（基线） | fp8 W8A16 | fp8 W8A8 | int8 W8A8 | int4 awq | nvfp4 |
|---|---|---|---|---|---|---|
| gate_up, M=1 | 48.8 | **45.8 (1.06×)** | **36.1 (1.35×)** | **34.8 (1.40×)** | **44.4 (1.10×)** | 117.9 (0.41×) |
| gate_up, M=32 | 50.2 | 65.1 (0.77×) | **38.8 (1.29×)** | **37.7 (1.33×)** | 66.6 (0.75×) | 124.2 (0.40×) |
| gate_up, M=2048 | **304.0** | 1125.4 (0.27×) | 513.6 (0.59×) | 313.2 (0.97×) | 1008.6 (0.30×) | 2267.9 (0.13×) |
| qkv, M=1 | **21.7** | 28.2 (0.77×) | 24.0 (0.90×) | 22.7 (0.96×) | 22.2 (0.98×) | 49.0 (0.44×) |
| qkv, M=2048 | **89.1** | 370.0 (0.24×) | 166.1 (0.54×) | 103.8 (0.86×) | 334.8 (0.27×) | 755.3 (0.12×) |

在 `gate_up` M=1，五种量化格式里有**四种快过 bf16**，包括 int4；同一投影到 M=2048，无一胜出，nvfp4 落后 7.5×。

**fp8 W8A8 为什么总比 int8 W8A8 慢一点？**这两行都是本仓库的 native Triton 内核（`fp8.py` 与 `w8a8.py`；cuBLAS 只出现在 bf16 基线，deep_gemm / flashinfer 未安装），差距来自 Triton 对两种 dtype 的下沉不同。int8 内核把两个 int8 操作数直接送 `tl.dot`、int32 累加，整数张量核从 `BLOCK_M=16` 起就能用；e4m3 要等 `BLOCK_M ≥ 64` 才能走 Hopper `wgmma`（§1.2 记录了同一条规则），decode 档 `BLOCK_M=16` 的块宽不够，两个 e4m3 操作数先加宽成 fp16 再走 `mma.sync`——省权重字节的收益还在，但张量核峰值退到 fp16 档，每个权重元素还多一次转换。所以 decode 档 int8 快约 3% 合理。M=2048 差距放大到 1.64×（313.2 对 513.6 µs）：gate_up 该档算术量约 204 GFLOP，int8 折合 651 TFLOP/s（H100 int8 峰值 1979 的 33%），fp8 只有 397 TFLOP/s（同峰值的 20%）——两种 dtype 的张量核峰值相同，fp8 输在 Triton 的 fp8 流水线成熟度。结论：合理，且这是内核现状而非硬件上限；要兑现 fp8 的理论收益需要 deep_gemm 一类专用后端（§5：行已注册、未安装、被 golden 门过滤）。在当前栈上 int8 W8A8 同时赢速度与精度（§2：0.907 对 0.659）。

这 4 个测试点暴露的规律：**量化只在 bf16 真正处于带宽受限时获胜，其余场合都不赢。**M=1 时 bf16 行占 HBM 峰值的比例从 10.4%（`qwen3-30b-a3b/down`，3 MB 权重）到 60.9%（`qwen3-4b/gate_up`），赢的测试点全部落在这段区间的顶部。低于约 50% 时内核并不在等显存，删掉权重字节什么也省不下来，反量化是纯增量成本。预测请看 bf16 的 %bw 列，而不是压缩比——本节早先一版用的就是压缩比，得出了方向完全相反的结论（nvfp4 字节少 3.6× 所以 decode 最快）；实际它在 48 个测试点里全部垫底。

分工作负载再看两条：

- **Decode（M ≤ 32）**：原理上受显存限制，但决定排名的不是 `moved`，是反量化速率。int4 在 `qkv` 上追平 bf16（22.2 对 21.7 µs）是两个不同瓶颈相遇——bf16 流 31 MB、占峰值 43%，带宽受限；int4 流 7.9 MB、占 11.9%，解包受限——这个平局换一个形状就不成立。
- **Prefill（M ≥ 512）**：受算力限制。**量化行在任何测试点都没有赢。**cuBLAS 达到张量核峰值的 73%，Triton 各行被解包循环钉死（int8 慢 1.17× 到 nvfp4 慢 8.5×）。fp8 W8A8 达到峰值的 39%——它是唯一在算术上*可能*赢的格式，但没有赢。

### 1.1a 量化什么时候赢、什么时候输：roofline 判断

上面 48 个测试 case 的胜负不是随机的，判据只有一条：这次 GEMM 卡在带宽还是算力上。

decode（M≤32）算术强度低，时间花在从 HBM 搬权重上，是 bandwidth-bound。量化把权重字节砍半（int8/fp8）或砍到 1/4（int4），直接缩短搬运——但只有 bf16 基线本身在等显存时才兑现。看 bf16 行的带宽占比：赢的测试点全落在占比高的投影（gate_up 60.9%），占比低的（down 10.4%）内核没在等显存，删字节省不下时间，反量化是纯增量成本。本节早先一版用压缩比预测，得出 nvfp4 字节少 3.6× 所以 decode 最快的反向结论，实际它 48 个测试点全部垫底——判据是 bf16 的 %bw，不是压缩比。

prefill（M≥512）算术强度高，是 compute-bound，省字节没用，能不能赢只看量化后的 MMA 吞吐是否高过 bf16。本表（0901/0902 口径）里量化行 prefill 全输：cuBLAS bf16 达张量核峰值 73%，Triton 各行被解包钉死。其中只有真 W8A8（fp8/int8，激活也量化）在算术上*可能*赢——两个操作数都进 tensor core，fp8/int8 MMA 峰值是 bf16 的 2×。fp8 W8A8 此表只到峰值 39%、0.59×，是 Triton fp8 wgmma codegen（0.7–1.05× cuBLAS）加激活量化 pass（5–17%）的编译器层边界；0903 的 epilogue 化把它拉到残余 1.05–1.5×，int8 W8A8 因 imma 无 `BLOCK_M≥64` 门槛、scale 在 epilogue，修复后 prefill 能到 1.58×。W8A16/W4A16 反量化后走 fp16-rate dot，算力上限就是 bf16 同档，prefill 结构性无胜机。完整推导见 [quantization.md](../../quantization.md) 的「量化为什么常常比 bf16 慢：roofline 判断」一节。

### 1.1b `--tune` 找到的 int4 启发式缺陷

五个量化 dense 内核里，**只有 `w4a16_matmul` 会读 `ConfigStore`**；fp8 W8A8、fp8/int8 W8A16 与 NVFP4 无条件计算 launch config，因此 `bench_quant_gemm.py --tune` 对它们的报告是"没有消费者"，而不是写入没人会读的缓存条目。（v0.5 的 changelog 宣称 autotune 覆盖"量化 GEMM"；对 dense 路径而言，那只是五个内核里的一个。）

在这个内核上，搜索找到的是启发式缺陷，不是逐形状的调优空间。`m ≤ 32` 分支用的是 `GROUP_M=1, num_stages=2`；同样的 16×64 tile 换成 `GROUP_M=8, num_stages=4` 后，在**全部 16 个** `m ≤ 32` 缓存键（两种几何 × 四个投影 × M16/M32 两个桶）上全胜，提升 9.0–41.5%——tile 保持不变，所以变量只有这两个旋钮。`GROUP_M=1` 什么都没分组，相邻 program 按行主序走遍网格，在 L2 里共享不到任何权重 tile。由于提升全局一致，它属于内核回退逻辑而不是形状键控缓存：现在不开调优也自带这份配置，`qkv` 上 M=1 的 int4 也因此从 34.0 降到 22.2 µs——上文那个"追平"是这次修复的结果，不是 int4 的天赋。

修复之后，逐形状调优仍有大量空间：32 个键里 29 个能再压过修正后的启发式，幅度 9.7–46.0%。只有三个键报告"启发式已是最优"——`qwen3-4b/qkv`、`qwen3-4b/gate_up` 与 `qwen3-30b-a3b/qkv` 的 M16 键。其余地方，decode 段的最优 tile 比启发式更*窄*（M16/M32 桶上是 16×32 或 64×32），prefill 段则宽得多（M512 达 128×64–128×256）——三分支回退覆盖不了这个跨度，所以它是唯一值得进缓存的 dense 内核。注意：桶条目是按共享该桶的 token 数*总和*选的，桶内某个宽度可能变差而条目整体仍是净赢。在 `qwen3-30b-a3b/qkv` 与 `qwen3-4b/qkv` 上抽查过，两个键的两种宽度都有提升（t512 +0.7% / +12.2%，t2048 +25.5% / +24.3%），未观察到回归——但只跑 decode 的部署仍应把 `--tokens` 收窄到自己服务的宽度。

### 1.2 Fused MoE

Qwen3-30B-A3B 几何（E=128, top_k=8, h=2048, i=768），bf16 激活，单位 µs，运行时设 `RAPID_LLM_AUTOTUNE=0`——即用户在没有调优缓存时拿到的启发式 tile。括号内为相对 bf16 基线的加速比（bf16 耗时 ÷ 该格式耗时），大于 1 快于基线；「仅 `moe_align`」是消融行、不含 GEMM 计算，不参与对比；末列旧 tile 是 §1.3 的回归哨兵，其比值就是旧缺陷的量级。数据来自 [`fused_moe_h100_20260902_int4byte.json`](../kernels/fused_moe_h100_20260902_int4byte.json)：

| tokens | bf16（基线） | fp8 W8A16 | fp8 W8A8 | int8 W8A8 | int8 W8A16 | int4 | 仅 `moe_align` | bf16 @ 旧 BLOCK_K=32 |
|---|---|---|---|---|---|---|---|---|
| 1 | **108.2** | 113.1 (0.96×) | 120.7 (0.90×) | 117.5 (0.92×) | 111.8 (0.97×) | 113.2 (0.96×) | 5.6 | 104.1 (1.04×) |
| 8 | 186.1 | 118.7 (1.57×) | 123.8 (1.50×) | 132.2 (1.41×) | **114.9 (1.62×)** | 130.0 (1.43×) | 6.7 | 254.0 (0.73×) |
| 64 | 415.9 | 240.0 (1.73×) | 236.3 (1.76×) | 234.4 (1.77×) | **234.1 (1.78×)** | 242.2 (1.72×) | 9.1 | 606.4 (0.69×) |
| 512 | 469.7 | 355.0 (1.32×) | 280.4 (1.68×) | **275.9 (1.70×)** | 313.0 (1.50×) | 403.6 (1.16×) | 18.0 | 640.7 (0.73×) |
| 4096 | 1062.4 | 1320.8 (0.80×) | 868.9 (1.22×) | **673.1 (1.58×)** | 1148.0 (0.93×) | 1701.9 (0.62×) | 40.2 | 1189.1 (0.89×) |

与首测（0901 JSON）相比这是另一条曲线：`moe_align` Triton 化后从约 188 µs（半层开销）降到 5.6–40.2 µs；W8A8 的激活量化在 ≤32 行时融进 GEMM、silu 输出在 store 时量化；int4 换 byte 级打包 + 双 dot kernel（t4096 从 2598.7 → 1701.9 µs，基线同步从 1573.2 → 1062.4 µs，相对位置 0.61×→0.62× 几乎不动——中间档才是它的受益区间）。三个负载区间：

- **1 token**：launch 开销主导，六行挤在 12% 以内（0.90–0.97×）——graph 分解测量证实 device 端量化行全部 ≤ bf16（35–47 µs 对 47 µs），差距纯在 launch，CUDA graph 模式下消失。
- **8–512 tokens**：全面赢。int8 W8A16 在 t8/t64 最快（1.62×/1.78×），int8 W8A8 自 t512 起接棒（1.70×）；weight-only 权重字节减半（int4 减至 1/4）的收益，在 `moe_align` 不再吞掉半层之后终于显形。
- **4096 tokens**：算力受限，排序由张量核吞吐决定。int8 W8A8 达 **459.4 TFLOP/s**（本轮所有数据中最高的算术速率，1.58×）；fp8 W8A8 355.9 TFLOP/s（1.22×）；fp8 W8A16 234.1 TFLOP/s 落败（0.80×）——逐 row-block 反量化的摊销极限；int4 181.7 TFLOP/s、219.3 GB/s 全表最低（0.62×），读字节最少却最慢，是反量化路径而非流量的结构性成本。

> **历史注**（原「后记/后记二」，2026-09-02）：上表是 `moe_align` Triton 化、W8A8 激活量化融合、fp8 W8A16 的 e4m3→fp16 加宽改走单条硬件 `cvt`（kernel 开关 `FP8_CVT`，pre-sm89 不变、golden 逐位一致）、int4 byte 级打包双 dot kernel 之后的当前快照。同轮证伪并留档：**BLOCK_M=256**（旨在消 t4096 专家权重 2× 重读）全格式 0.26–0.90×——shared memory 逼 `num_stages=2`、256 行 accumulator 压垮 occupancy，`GROUP_M=8` 的 L2 分组已吸收大部分重读；**int4 改 vLLM 式复制寻址**（逻辑 k 直读所在 word、免 3D reshape）在 int32 打包格式上慢约 10×（t4096 18109 µs）——vLLM 是每字节 2 nibble、重复率 2×，本框架是每 int32 8 nibble、重复率 8×，结论固化在 kernel 注释里（byte 布局落地后该对比不再适用）；**int4 BLOCK_K=256** 全档 0.38–0.79×。现行权威数据见 [`fused_moe_h100_20260902_int4byte.json`](../kernels/fused_moe_h100_20260902_int4byte.json) 与 [`quantization.md`](../../quantization.md)（0902 早间的 0902/0902_fp8cvt JSON 是修复中途快照，其 t1 行与自身消融行自相矛盾，勿引用）。

TFLOP/s 里藏着一个陷阱：Triton 只在 `BLOCK_M ≥ 64` 时才发射 Hopper fp8 `wgmma`，而 `_launch_config` 的 fp8 W8A8 分档要到 4096 token 才达到这个行块（t512 的 tier-1 tile 是 `BLOCK_M=32`）——其余各档把两个 e4m3 操作数加宽成 fp16 走 `mma.sync`，没有测到 fp8 张量核。t512 的 1.68× 领先来自字节与跳过的 bit-trick，不是 MMA；fp8 W8A8 的张量核速率要到 t4096（355.9 TFLOP/s）才见于表中，decode 档的数字低估了该格式的潜力。

### 1.3 基线列暴露的缺陷

`_launch_config` 原来返回 `BLOCK_K = 128 if quant_mode else 32`。tile 搜索（`bench_fused_moe.py --tune`）在所有 token 档都**找不到**任何 `BLOCK_K` 低于 64 的获胜 fp16 配置；窄 tile 在 t8 损失 26.7%、t64 损失 31.4%、t512 损失 26.7%、t4096 损失 10.7%。它压低的从来只是*未量化基线*——所以没有任何测试抓住它，t512 的 W8A16 fp8 也一度被读成 18% 的赢面而非首测表的 5.5% 落败（新表的 1.32× 已是 `FP8_CVT` 之后的数字，见历史注）。已修复：所有模式都用 128。表格最后一列保留旧 tile 作为回归哨兵——两列 fp16 收敛即说明旧问题没有回来。

同一次搜索写入 `ConfigStore` 后，15 个缓存键中 **13 个**改善（最大：fp16 在 M512 桶，2502.8 → 1694.1 µs，+32.3%；fp8 W8A8 的 M512 与 int4 的 M512 是启发式本来就读对的两个）。缓存按设备存放、不入库；换 GPU 请重跑 `--tune`。

## 2. 离线矩阵 — `M_MAIN`（Qwen3-4B-Thinking-2507）

batch 4、生成 64 token；KV 容量来自 profiling 而非固定值，因此显存列本身就是测量结果。[A](quant_main_A_h100_20260901.json) · [B/C](quant_main_BC_h100_20260901.json) · [D/E](quant_main_DE_h100_20260901.json) · [F](quant_main_F_h100_20260901.json) · [G](quant_main_G_h100_20260901.json)

五行 `int4` 是 §1.1b 的 `w4a16` tile 修复之后，用相同命令、`RAPID_LLM_AUTOTUNE=0` 重测的，表格里放的就是重测值：[BC](quant_main_int4fix_BC_h100_20260901.json) · [DE](quant_main_int4fix_DE_h100_20260901.json) · [F](quant_main_int4fix_F_h100_20260901.json)。修复前的数值保留在表下的行注里而没有删——这个差值就是该修复在端到端的量级，也是本节不是单次一致运行的全部原因。

| 配置 | TTFT ms | TPOT ms | TPS | model GB | KV tok | golden | prefix |
|---|---|---|---|---|---|---|---|
| HF fp16（参照） | 35.4 (0.62×) | 56.47 (0.08×) | 71.3 (0.09×) | — | — | — | — |
| **A** bf16+tp1+graph（基线） | 22.0 | 4.77 | 793.0 | 7.49 | 447,830 | **1.000** | 1.000 |
| **A** bf16+tp1+eager | 21.6 (1.02×) | 22.23 (0.21×) | 180.0 (0.23×) | 7.49 | 458,752 | 1.000 | 1.000 |
| **B** fp8+tp1+eager | 32.0 (0.69×) | 32.17 (0.15×) | 124.3 (0.16×) | 4.11 | 476,672 | 0.659 | 0.617 |
| **B** int8+tp1+eager | 25.1 (0.88×) | 25.32 (0.19×) | 158.0 (0.20×) | 4.11 | 476,653 | 0.907 | 0.822 |
| **B** int4+tp1+eager | 25.7 (0.86×) | 26.12 (0.18×) | 153.1 (0.19×) | 2.63 | 488,320 | 0.157 | 0.139 |
| **B** smoothquant+tp1+eager | 29.3 (0.75×) | 29.43 (0.16×) | 135.9 (0.17×) | 4.11 | 476,667 | 0.181 | 0.051 |
| **B** nvfp4+tp1+eager | 25.3 (0.87×) | 25.75 (0.19×) | 155.3 (0.20×) | 2.63 | 488,420 | 0.249 | 0.233 |
| **C** fp8+tp1+graph | 32.0 (0.69×) | 5.60 (0.85×) | 664.3 (0.84×) | 4.11 | 465,750 | 0.659 | 0.617 |
| **C** int8+tp1+graph | 25.4 (0.87×) | 7.11 (0.67×) | 540.5 (0.68×) | 4.11 | 465,730 | 0.907 | 0.822 |
| **C** int4+tp1+graph | 24.8 (0.89×) | 6.97 (0.68×) | 551.3 (0.70×) | 2.63 | 477,398 | 0.157 | 0.139 |
| **C** smoothquant+tp1+graph | 29.0 (0.76×) | 5.26 (0.91×) | 710.0 (0.90×) | 4.11 | 465,745 | 0.181 | 0.051 |
| **C** nvfp4+tp1+graph | 26.2 (0.84×) | 13.66 (0.35×) | 288.7 (0.36×) | 2.63 | 477,497 | 0.249 | 0.233 |
| **D** fp8+tp2+eager | 38.2 (0.58×) | 41.11 (0.12×) | 97.4 (0.12×) | 2.06 | 977,678 | 0.676 | 0.608 |
| **D** int4+tp2+eager | 31.4 (0.70×) | 34.32 (0.14×) | 116.7 (0.15×) | 1.31 | 989,653 | 0.160 | 0.139 |
| **E** fp8+tp2+graph | 40.0 (0.55×) | 5.89 (0.81×) | 622.2 (0.78×) | 2.06 | 955,832 | 0.676 | 0.608 |
| **E** int4+tp2+graph | 33.8 (0.65×) | 6.43 (0.74×) | 583.1 (0.74×) | 1.31 | 967,808 | 0.160 | 0.139 |
| **F** fp8+dp2+graph | — | — | 1275.3 (1.61×) | — | — | 0.659 | 0.617 |
| **F** int4+dp2+graph | — | — | 1057.6 (1.33×) | — | — | 0.157 | 0.139 |
| **G** bf16+kvfp8+tp1+graph | 26.1 (0.84×) | 5.47 (0.87×) | 690.2 (0.87×) | 7.49 | 895,692 | 0.718 | 0.703 |
| **G** fp8+kvfp8+tp1+graph | 37.2 (0.59×) | 6.34 (0.75×) | 586.0 (0.74×) | 4.11 | 931,503 | 0.617 | 0.574 |

性能三列的括号为相对 `bf16+tp1+graph` 基线行的比值，口径同 §1.1：TTFT/TPOT 是耗时、取基线 ÷ 该配置，TPS 是吞吐、方向相反、取该配置 ÷ 基线（793.0），大于 1 快于基线。比值同时含 CUDA graph 与引擎差异，scheme 内部的公平对比看下方结论；`model GB`/`KV tok` 是资源量、`golden`/`prefix` 是一致率，均不参与对比。

tp2 行的 `model GB` 是**每张卡**的值（`note: rank 0 shard`）；D/E 与 fp8 tp1 基线两侧都开连续批处理，tp1↔tp2 对比只差并行方式，不差调度器。

矩阵读出的结论：

- **CUDA graph 是单项最大收益，且与量化正交。**bf16 提升 4.4×（180 → 793 TPS）、fp8 5.3×、smoothquant 5.2×、int4 3.6×、int8 3.4×、nvfp4 1.9×；所有 scheme 都能捕获，包括 TP2 行。*量化*路径也能进图，这是 Phase 2/3/5 的成果：它们的 launch config 依赖形状，其中任何一处 host 同步都会让整层无法捕获。scheme 之间的增益差本身就有信息量——图消掉的是 launch 开销，所以增益最小的 scheme（nvfp4，1.9×）正是每次 launch 里实际计算最多的那个。int4 原本也在这个句子里（2.8×）；§1.1b 的 tile 修复把它抬到 3.6×——从结果看，这就是"内核每次 launch 带着可避免的工作"的样子。
- **`w4a16` tile 修复在端到端买到了什么。**修复前后、相同命令：C int4+tp1+graph 419.0 → 551.3 TPS（TPOT 9.28 → 6.97 ms），E int4+tp2+graph 471.9 → 583.1（8.06 → 6.43），F int4+dp2 816.8 → 1057.6；eager 行几乎不动（B 149.7 → 153.1，D 112.3 → 116.7），因为 eager decode 被 launch 开销主导，tile 摸不到它。精度逐位不变（前后都是 0.157/0.139）——tile 配置改动本该如此。int4 从最慢的 graph 行回到中游，但仍慢于 bf16。
- **这个尺寸上没有量化 scheme 比 bf16 快。**bf16+graph 是全表最快行。4B 的权重本来放得下，decode 又是 launch 受限，weight-only 格式等于在没有需要节省的流量上白付解包成本。量化在这里买到的是**显存**：int4 把权重砍到三分之一（7.49 → 2.63 GB），fp8+tp2 每卡砍到 27%（7.49 → 2.06 GB），省下的显存变成 KV 容量——447,830 → 967,808 tokens，2.2×。
- **这个尺寸上 TP 是容量特性，不是速度特性。**fp8+tp2+graph 比 fp8+tp1+graph 还*慢*（调度器固定时 622 对 644 TPS）：每步 all-reduce 的代价超过第二张卡在 4B 上的算力收益。它把 KV 容量翻倍，这才是用它的理由。
- **TP 扩不动的方向 DP 能扩。**fp8+dp2 达到 1275 TPS，对 tp1 的 664——1.92×，接近线性，因为副本之间每步零共享。int4+dp2 是 1058 对 551，同样 1.92×。
- **精度排序 int8 (0.907) > fp8 (0.659) > nvfp4 (0.249) > smoothquant (0.181) > int4 (0.157)**，且精度差距远大于速度差距。在*推理型*（reasoning）checkpoint 上，一个 token 分叉就会改写整条后续链路，所以这些数字更像"补全是否完整存活"，而不是逐 token 错误率。int8 是唯一称得上近似无损的行。smoothquant 是全表最差的组合——匹配率 0.181，*前缀*率只有 0.051，即很早就分叉且从此一路错下去——它是无校准数据的运行时量化路径，这很可能是原因。
- **KV fp8 在 bf16 上花掉 0.28 的精度**（1.000 → 0.718），换来 2.0× KV 容量（447,830 → 895,692）。见第 4 节。

### `M_MOE`（Qwen3-30B-A3B-Instruct-2507）— H 组

batch 4、生成 32 token、`--max-seq-len 2048`。该 checkpoint 没有 golden 基线，精度列为空是设计使然，不是遗漏。[fp8](quant_moe_fp8_h100_20260901.json) · [bf16](quant_moe_bf16_h100_20260901.json)

| 配置 | model GB | peak GB | KV tok | TTFT ms | TPOT ms | TPS |
|---|---|---|---|---|---|---|
| fp8+tp1+graph | 29.11 | 68.97 | 420,348 | 73.1 (1.05×) | 11.11 (1.01×) | 306.5 (1.01×) |
| fp8+tp2+graph | 14.59（rank 0） | 68.16 | 1,141,756 | 88.2 (0.87×) | 10.96 (1.02×) | 298.9 (0.99×) |
| bf16+tp2+graph（基线） | 28.45（rank 0） | 68.94 | 855,078 | 76.4 | 11.17 | 302.7 |

性能三列的括号为相对 `bf16+tp2+graph` 基线行的比值，口径同 §2；`model GB`/`peak GB`/`KV tok` 是资源量，容量倍数见下文。

三行的吞吐互相都在 2.5% 之内，TPOT 平坦在约 11 ms。这正是内核表的 decode 区间在端到端的显形：4 路并发时 MoE 层在 `moe_align_block_size` 后面处于 launch 受限，fp8-A8 在内核层面 33% 的 decode 惩罚被模型其余部分稀释到测不出差别。**fp8-A8 MoE 不是 decode 优化。**它买到的是容量：fp8+tp1 用 29.11 GB 把 57 GB 的 bf16 checkpoint 塞进*一张*卡；fp8+tp2 达到 1.14M KV tokens——是 tp1 的 2.7×、bf16+tp2 的 1.34×。内核层测到的 prefill 收益（512 tokens 处 18%）需要 prefill 重的负载才会显形，这个配置没有这样的负载。

### 伴测 — *原生* FP8 checkpoint

上面各行的量化都是对 bf16 checkpoint 在运行时做的。官方发布的 `Qwen3-30B-A3B-Instruct-2507-FP8` checkpoint（fp8-e4m3 + 128×128 block scales，`quant_method: fp8`）走的是 W8A16 路径——block-scale 权重与 `w8a8_fp8` 期望的 per-channel 布局不匹配——已于 2026-09-01 在全轴矩阵上测完（tp1/tp2 × graph/eager × kv auto/fp8，外加 dp2，并为其记录了 golden 基线）：tp1+graph 得 TPOT 13.16 ms / 285.9 TPS，tp2+graph 得 12.76 ms / 290.3 TPS 且 KV 容量 2.7×；这个尺寸上 CUDA graph 值 4.8×。完整表格与精度列见 [quantization.md § Qwen3-30B-A3B-Instruct-2507-FP8](../../quantization.md)，原始 JSON 在 [`bench_quant_Qwen3-30B-A3B-FP8_20260901.json`](quant_Qwen3-30B-A3B-FP8_20260901.json)（数据并行行见加 `-dp` 后缀的文件）。

## 3. 在线矩阵 — `M_MAIN`，`rapid-llm serve`

最多生成 64 token、`max_seq_len` 1024，对 `POST /v1/completions` 施加 1/8/32 三档并发，`temperature=0`。[tp](../serving/serving_main_tp_h100_20260901.json) · [dp](../serving/serving_main_dp_h100_20260901.json)

三行 `int4` 是 §1.1b 之后的重测值，命令相同（[tp](../serving/serving_main_int4fix_tp_h100_20260901.json) · [dp](../serving/serving_main_int4fix_dp_h100_20260901.json)）；其余六行是原始运行。

| 配置 | 并发 | TTFT mean | TTFT p99 | TPOT | TPS | batch | in-wave dup | offline | dup batch |
|---|---|---|---|---|---|---|---|---|---|
| bf16+tp1 | 1 | 24.2 | 24.2 | 4.70 | 199.7 | 1.000 | — | 1.000 | 1.000 |
| bf16+tp1 | 8 | 46.4 | 47.3 | 5.50 | 1294.3 | 0.883 | — | 1.000 | 1.000 |
| bf16+tp1（基线） | 32 | 179.2 | 187.2 | 7.28 | 3148.2 | 0.890 | 0.766 | 1.000 | 1.000 |
| bf16+tp2 | 32 | 222.3 (0.81×) | 249.3 (0.75×) | 7.91 (0.92×) | 2730.4 (0.87×) | 0.760 | 1.000 | 1.000 | 1.000 |
| fp8+tp1 | 32 | 202.7 (0.88×) | 217.2 (0.86×) | 7.89 (0.92×) | 2803.0 (0.89×) | 0.546 | 0.766 | 1.000 | 1.000 |
| fp8+tp2 | 32 | 147.2 (1.22×) | 160.0 (1.17×) | 8.54 (0.85×) | 2879.8 (0.91×) | 0.633 | 1.000 | 1.000 | 1.000 |
| int4+tp1 | 32 | 143.6 (1.25×) | 157.7 (1.19×) | 9.75 (0.75×) | 2585.0 (0.82×) | 1.000 | 1.000 | 1.000 | 1.000 |
| int4+tp2 | 32 | 232.0 (0.77×) | 253.3 (0.74×) | 9.12 (0.80×) | 2440.5 (0.78×) | 1.000 | 1.000 | 1.000 | 1.000 |
| bf16+dp2 | 32 | 144.7 (1.24×) | 161.8 (1.16×) | 5.45 (1.34×) | 3969.2 (1.26×) | 0.883 | 1.000 | 1.000 | 1.000 |
| fp8+dp2 | 32 | 146.6 (1.22×) | 159.7 (1.17×) | 6.26 (1.16×) | 3633.2 (1.15×) | 0.569 | 0.281 | 1.000 | 1.000 |
| int4+dp2 | 32 | 130.6 (1.37×) | 148.1 (1.26×) | 7.72 (0.94×) | 3226.1 (1.02×) | 1.000 | 1.000 | 1.000 | 1.000 |

9 配置 × 3 并发共 27 个组合，请求全部完成（`completed == issued`）。并发 32 档的括号为相对同档 `bf16+tp1` 的比值，口径同 §2（TPS 基线 3148.2）；并发 1/8 档只有 `bf16+tp1` 一个配置、没有可比对象，保持裸值。TTFT 是单次运行（下文按 ±50% 对待），TPOT 与 TPS 才是复现列；末四列是一致率与重复性指标，不参与对比。

- **服务与离线生成完全一致。**`offline` 列是服务端补全对同一 prompt 在 `temperature=0` 下 `LLM.generate` 的对比——**9 个配置全部 1.000**。连续批处理不改变答案。
- **并发 32 时每个组合的 TTFT p99 都在均值的 15% 以内**——现在包括 int4+tp1。修复前那一行是 30% 的例外（391.7 对 302.1），还是全表最慢的组合（1720 TPS）；`w4a16` tile 修复把它带到 157.7/143.6 与 2585 TPS——即尾延迟的根源是 decode 内核吃不下整批，不是调度副作用。int4+dp2 同样从 2654.9 → 3226.1。
- **int4+tp2 是重测唯一变*差*的组合**（2676.4 → 2440.5 TPS，TTFT 124.4 → 232.0），按实测原样报告。TPOT 变好（9.79 → 9.12），稳态更快，回归落在首 token 延迟上——两个 rank 同时放行 32 请求的波，是这张表里可重复性最差的事件；修复前 tp2 的 TTFT 也曾以 2.4× "胜过" tp1，模型里没有任何机制能解释这个排序。并发 32 下的单次运行 TTFT 按 ±50% 对待；TPOT 与 TPS 才是复现了的列。
- **在线场景 DP 同样是吞吐轴**：bf16+dp2 达 3969 TPS，是 bf16+tp1 的 1.26×；并发 32 时三个 dp2 配置的 TTFT 全部落在 131-147 ms 区间，与量化方案无关——在那里决定首 token 延迟的是 router，不是模型。

### 一个负结果：batch 不变性不成立

`batch` 列是并发 N 下服务出的补全对同一 prompt 单独服务的对比；`in-wave dup` 是同一波里两条相同 prompt 互相对比。两者都低于 1.000——最低 0.546（fp8+tp1）与 0.281（fp8+dp2）。这都不是 bug，证据在最后一列。

`dup batch` 把同一 prompt 在**一次** `engine.generate` 调用里提交 32 份：所有副本在第一步之前已入队、长度完全相同，必然共享每一个 batch。这一列**9 个配置全部 1.000**。所以并发请求互相看不到对方的状态——看得到的话，副本在这里也会分叉。

剩下的解释只有依赖 batch 大小的算术：GEMM 按 M 选 tile、捕获的图 pad 到桶、split-K 改变求和顺序。1e-3 的 logit 偏移足以翻转 bf16 `argmax` 平局，而在推理型 checkpoint 上，一个翻转的 token 会改写其余补全——所以一致率是 0.55-0.89 而不是 0.999。本基准的早先一版曾把 `in-wave dup < 1.000` 当成状态泄漏的证据；那是错的，因为调度器在 `max_num_seqs` 与 `max_num_batched_tokens` 约束下从 `_waiting` 准入，HTTP 到达又是异步的，同一波里的两份副本根本不保证共享 batch。把这个疑点关掉的是 duplicate-batch 对照组。

## 4. fp8 KV cache

[`kv_fp8_error_qwen3-4b_20260901.json`](kv_fp8_error_qwen3-4b_20260901.json)。阈值先于测量设定：任一层 `amax > 448`，**或** token 匹配率低于 0.98，即触发校准流程。

触发了一个门，而后续实验说明校准不是解药：

| 探针 | 结果 |
|---|---|
| 截断检查：36 层、47.1M 个值 | `max_amax` **294.0**（`layers.0...k`），0 个值被截断——门**未**触发 |
| 贪心一致率：4 prompt × 128 token | **0.316**（162/512），首个分叉点在 token 11-74——门**已**触发 |
| 对照组（`auto` 对 `auto`，同种子） | **1.000**——harness 本身是确定性的，分歧来自 dtype |
| subnormal 占比 | 最高 **0.567**（`layers.0...v`） |
| oracle 每 tensor 余量 | 平均 **1.030×**，最好的层 1.185× |
| GSM8K，500 题 | 0.192 → 0.164，Δ **−0.028**，未配对 stderr **0.024** |

`scale=1.0` 并没有发生截断——没有任何值到达 448，而 *oracle* scale（完美的 per-tensor 校准能选到的最优值）平均只有 1.03× 的余量。所以校准几乎没有东西可挽回：误差是 e4m3 的 3-bit 尾数摊在每一个被缓存的 token 上，最差层 57% 的值落在 subnormal 区间，那里的相对精度还要更差。**Phase 4b 因此未执行**——这是测量得出的决定，不是被跳过的步骤。

诚实的代价，两个独立参照：KV 脚本自己的探针集上贪心一致率 0.316；离线矩阵更长 prompt 上的 golden 前缀一致率 0.703（G 行，1.000 → 0.718 匹配 / 0.703 前缀）。收益是 2.0× KV 容量（447,830 → 895,692 tokens）。GSM8K 在 n=500 下分辨不出这个差别——−2.8 分对 ±2.4 分的 stderr——所以任务级精度的问题保持开放，并按开放记录在案。fp8 KV 是一笔带这些数字的容量/精度交易，默认保持关闭。

## 5. 失败与无结果，记录而非丢弃

| 事项 | 状态 |
|---|---|
| `M_MOE_INT4`（Qwen3-30B-A3B Int4 W4A16） | `quantization/__init__.py:105` 处 `ValueError: unsupported quant_method 'compressed-tensors'`。该 checkpoint 使用 compressed-tensors 序列化，这里没有对应的读取器；补一个超出本次范围。所有矩阵均排除，Phase 0 已记录。 |
| fp4 张量核（A4）GEMM | 未尝试。sm90 没有 fp4 硬件 MMA；Triton 的 `tl.dot_scaled`（microscaling，支持 uint8 打包 fp4）在 sm90 走软件模拟（upcast bf16），无 fp4 MMA 收益，原生支持要到 Blackwell（sm100）。NVFP4 是 weight-only：买到的是字节，不是算术。 |
| NVFP4 MoE 专家 | 未实现；`NVFP4Config.get_quant_method` 在 MoE 层上直接抛错，而不是静默回退。 |
| `deepgemm/fp8_gemm_nt` 与 flashinfer 各行 | 仍是 `GoldenRecord(verified=False)`。两个库都未安装；行已注册、被过滤掉，`explain()` 里可见。 |
| `tests/kernels/test_w4a16_accuracy.py::[w4a16_problem2]` | 在干净的 `HEAD` 上**曾是**失败；本次修复，改的是测试而非内核。用例是 M16/N1024/K2048，最大输出达 `abs(ref) = 139.8`，fp16 的一个 ULP 已是 0.125——平铺的 `max_diff < 0.1` 上限要求了超过可表示精度的准确度。实测 `max_diff` 恰为 0.1250，即一次舍入步，且在新旧 tile 下*完全相同*（强制两种配置核对过），所以从来不是 tile 或内核缺陷。上限现改为 `max(0.1, 2 * abs(ref).max() * 2**-10)`，即输出自身量级下的 2 个 fp16 ULP。4 个形状全部通过；另外 3 个在两种上限下本来就通过。 |
| tp1+graph 的 nvfp4 TPOT（13.66 ms） | 所有捕获行中最差的 decode 延迟，差 2×——最接近它的格式 int4+graph 是 6.97 ms。16 元素块的反量化每字节工作量比 int4 更大，而这个格式相对 int4 一个字节也没省。没有人扫过 nvfp4 的 tile：`nvfp4_matmul` 不读 `ConfigStore`，§1.1b 的 `w4a16` 修复在这里没有对应物，所以这个数字应读作"第一个能跑的 tile"，而不是该格式的下限。 |

## 复现

```bash
export RAPID_LLM_MODELZOO=/mnt/otto-temp/modelzoo_with_full_weights

# 内核层
RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_fused_moe.py --json out.json
python benchmarks/kernels/bench_fused_moe.py --tune          # 写入 ConfigStore
RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py --json out.json
python benchmarks/kernels/bench_quant_gemm.py --tune       # 仅 w4a16；见 §1.1b

# 离线
python benchmarks/engine/run.py quant --model-dir $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --schemes fp8 int4 --tp 1 2 --engine continuous --cuda-graph --no-cuda-graph --skip-hf

# 在线
python benchmarks/engine/run.py scheduler serving --model-dir $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --schemes bf16 fp8 int4 --tp 1 2 --concurrency 1 8 32 --max-tokens 64 --max-seq-len 1024

# kv fp8 误差
python scripts/quant_kv_error.py --model-dir $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507
```
