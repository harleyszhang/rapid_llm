## 一 量化 benchmark 性能测试

### 量化内核性能（W8A16 / W4A16 / SmoothQuant）

两台设备分开记，因为**同一格式在两台机器上可以给出相反的结论**：A10（sm86）的 HBM 约 600 GB/s，decode 档几乎全程在等显存，省下的字节直接兑现；H100（sm90）有 3.35 TB/s，bf16 在中尺寸投影上只占峰值带宽的 43.5%，内核根本没在等显存，反量化的 ALU 开销就是纯增量成本。两节的 shape 也不同（A10 是合成方阵，H100 是 checkpoint 的真实投影），**绝对值不要跨表比较**，只读各表内的相对位置。

#### A10 (24 GB, SM86)

以下为量化 Triton 内核在 `A10` 上的 `triton.testing.do_bench` 实测结果（2026-08-22 口径、合成 shape）。环境：NVIDIA A10 24GB（sm86，~600 GB/s HBM），当时的软件栈为 torch 2.11.0+cu129 / triton 3.6.0 / python 3.12；负载：合成方阵 shape（下表 M×N×K 列），`triton.testing.do_bench` 口径。**该批数字早于 JSON 环境日志功能，无独立日志留存**（数字仅见于本节表格）。此后内核经过多轮改动（v0.6 重写、`FP8_CVT` 硬件 `cvt` 分叉、0903 按 H100 扫描重定 tile fallback），**A10 未复测**，绝对值仅供参考，下面的 roofline 判据仍然成立。

##### W8A16 (fp8-e4m3, 128×128 block scales)

| Shape (M×N×K) | fp16 (ms) | w8a16 (ms) | 加速比 | 场景 |
| --------------- | ----------- | ------------ | -------- | ------ |
| 1×4096×4096 | 0.086 | 0.053 | **1.62×** | decode |
| 1×11008×4096 | 0.199 | 0.116 | **1.71×** | decode (MLP up) |
| 8×4096×4096 | 0.084 | 0.051 | **1.65×** | decode batch |
| 64×4096×4096 | 0.091 | 0.055 | **1.64×** | small prefill |
| 512×4096×4096 | 0.191 | 0.280 | 0.68× | prefill (compute-bound) |

判据（roofline）与 H100 矩阵一致，但 **W8A16 这一行的胜负在两台机器上是反的**。decode（M≤64）算术强度低、卡在 HBM 带宽上，W8A16 把权重字节减半直接缩短搬运，在 A10（~600 GB/s）稳定 1.6–1.7×；prefill（M≥512）算术强度高、卡在算力上，W8A16 反量化后走 fp16-rate dot，算力上限就是 cuBLAS fp16 同档，省下的字节不在瓶颈上，于是 0.68×——此时应回退 cuBLAS fp16。同一格式到 H100 上 decode 档反而输（qkv 的 M=1/8/32 为 0.76–0.79×；24 个 decode 测试点里只有 qwen3-4b/gate_up 的两个还赢，1.21–1.22×，见[下文](#h100-80-gb-sm90)）：3.35 TB/s 的带宽让 bf16 只占峰值 43.5%（gate_up 才到 60.9%），内核没在等显存，删字节省不下时间。A10（sm86）没有原生 fp8 GEMM，所以真 W8A8（激活也量化）在这里只能付量化开销、拿不到 MMA 收益；完整的「量化什么时候赢/输」推导见 [quantization.md](quantization.md) 的 roofline 一节与 [quant_matrix_20260901.md](benchmark_logs/quantization/quant_matrix_20260901.md) §1.1a。

##### W4A16 (int4, group_size=128)（初版内核，勿引用）

| Shape (M×N×K) | fp16 (ms) | w4a16 (ms) | 加速比 | 场景 |
| --------------- | ----------- | ------------ | -------- | ------ |
| 1×4096×4096 | 0.085 | 0.056 | **1.53×** | decode |
| 8×4096×4096 | 0.084 | 0.042 | **2.01×** | decode batch |
| 64×4096×4096 | 0.090 | 0.062 | **1.45×** | small prefill |
| 512×4096×4096 | 0.192 | 0.430 | 0.45× | prefill (compute-bound) |

结论：decode 阶段（M≤64）**1.4–2.0× 加速**——内核走 per-group `tl.dot` tensor core（v0.5 重写，`rapid_llm/kernels/ops/quantization/w4a16.py`），权重读取量减至 1/4；prefill 阶段（M≥512）compute-bound，int4 路径无优势，与 W8A16 同一结论。

tile 配置来自 autotune 落盘表（`~/.cache/rapid_llm/autotune/w4a16_matmul.json`，A10 实测）：M≤16 → `BLOCK_M=16/BLOCK_N=64/num_stages=4`，M=64 → `BLOCK_M=64/BLOCK_N=64/num_stages=4`；表未命中时 `w4a16_matmul._heuristic_config` 兜底（同一组 A10 sweep 的最优档）。内存节省仍然有效：30B 模型 int4 权重仅占 ~15 GB（fp16 需 ~61 GB）。

#### SmoothQuant W8A8 (dynamic per-token)

| Shape (M×N×K) | fp16 (ms) | smoothquant (ms) | 加速比 | 备注 |
|---------------|-----------|------------------|--------|------|
| 8×256×512 | — | ✓ | — | 精度验证通过 |
| 64×2048×2048 | — | ✓ | — | 精度验证通过 |

精度：相对 fp32 参考的相对误差 < 2%（含激活 + 权重量化双重噪声）。A10 上只做了精度验证（shape 太小，性能读数无意义）；它的性能行就是[下文 H100 矩阵](#h100-80-gb-sm90)的 `int8 W8A8` 列——同一个内核（int8 权重 + 内核内 per-token int8 激活量化）。

#### H100 (80 GB, SM90)

测于 NVIDIA H100 80GB HBM3（3352 GB/s 峰值带宽、989 TFLOP/s dense tensor core），torch 2.13.0+cu130 / triton 3.7.1 / python 3.14.7，2026-09-03 口径，数据来自 [`quant_gemm_h100_20260903d.json`](benchmark_logs/kernels/quant_gemm_h100_20260903d.json)，以 `RAPID_LLM_AUTOTUNE=0` 运行——即用户没有调优缓存时拿到的启发式 tile。shape 是两个 checkpoint 的四个真实投影（qwen3-4b：hidden 2560 / intermediate 9728；qwen3-30b-a3b：hidden 2048 / moe_intermediate 768）× 6 个 token 档（1/8/32/128/512/2048）= 48 个测试点 × 6 个 scheme。括号内为相对 bf16 的加速比（bf16 耗时 ÷ 该格式耗时），大于 1 即快于基线。

##### 中尺寸投影：qwen3-4b/qkv（N=6144, K=2560）

| M | bf16 | fp8 W8A16 | fp8 W8A8 | int8 W8A8 | int4 (awq) | nvfp4 |
|---|---|---|---|---|---|---|
| 1 | 21.6 µs | 28.2 (0.77×) | 22.0 (0.98×) | **20.2 (1.07×)** | 29.8 (0.72×) | 49.1 (0.44×) |
| 8 | 21.4 | 27.2 (0.79×) | 22.6 (0.95×) | **20.7 (1.03×)** | 29.4 (0.73×) | 48.7 (0.44×) |
| 32 | 21.7 | 28.4 (0.76×) | 23.4 (0.93×) | **21.3 (1.02×)** | 38.7 (0.56×) | 50.6 (0.43×) |
| 128 | 21.5 | 49.6 (0.43×) | 27.3 (0.79×) | 24.1 (0.89×) | 58.2 (0.37×) | 67.9 (0.32×) |
| 512 | 30.7 | 81.7 (0.38×) | 37.5 (0.82×) | **30.5 (1.01×)** | 137.5 (0.22×) | 219.2 (0.14×) |
| 2048 | 90.8 | 252.2 (0.36×) | 93.5 (0.97×) | **73.7 (1.23×)** | 512.2 (0.18×) | 728.4 (0.12×) |

##### 最大权重投影：qwen3-4b/gate_up（N=19456, K=2560，bf16 权重 ~100 MB）

| M | bf16 | fp8 W8A16 | fp8 W8A8 | int8 W8A8 | int4 (awq) | nvfp4 |
|---|---|---|---|---|---|---|
| 1 | 48.8 | 40.4 (1.21×) | 34.2 (1.43×) | **33.5 (1.46×)** | 61.3 (0.80×) | 117.4 (0.42×) |
| 8 | 49.0 | 40.0 (1.22×) | 35.0 (1.40×) | **34.1 (1.44×)** | 60.3 (0.81×) | 118.2 (0.41×) |
| 32 | 50.4 | 57.7 (0.87×) | **36.3 (1.39×)** | 36.7 (1.38×) | 112.4 (0.45×) | 122.8 (0.41×) |
| 128 | 50.2 | 142.5 (0.35×) | 50.1 (1.00×) | **44.5 (1.13×)** | 168.4 (0.30×) | 187.0 (0.27×) |
| 512 | 81.7 | 211.5 (0.39×) | 82.1 (0.99×) | **67.7 (1.21×)** | 412.4 (0.20×) | 647.4 (0.13×) |
| 2048 | 304.1 | 789.8 (0.39×) | 276.5 (1.10×) | **210.5 (1.45×)** | 1560.5 (0.19×) | 2254.2 (0.13×) |

##### 胜负统计与判据

- **48 个测试点里 cuBLAS bf16 在 35 个最快**；剩下 13 个有量化行反超，共 20 个 scheme 胜场（int8 W8A8 13 / fp8 W8A8 5 / fp8 W8A16 2），集中在 gate_up 全部 6 档、qwen3-4b/qkv 的 5 档与 30B 的两个 prefill 测试点。int8 W8A8 在 gate_up 六个档上跑 1.46× / 1.44× / 1.38× / 1.13× / 1.21× / 1.45×——**它的胜场活过了 prefill**。
- 门槛按格式分：**int8 W8A8 从 bf16 占峰值 HBM ~44% 起就赢**（qkv M=1 实测 43.5% → 1.07×；30B/qkv 的 38.5% 差一点没赢），fp8 W8A8 与 fp8 W8A16 只在最顶端（gate_up 的 60.9%）赢。int8 门槛更低的原因：scale 全在 epilogue，且 imma 没有 `BLOCK_M ≥ 64` 门槛；fp8 行要付 Triton 的 wgmma codegen 加一个每 token 激活量化 pass（JSON 的 `ablation` 行单独隔离了后者）。
- decode（M≤32）对 bf16 的几何均值差距：int8 W8A8 1.16×、fp8 W8A8 1.29×、**w4a16 1.65×**、w8a16 1.71×、nvfp4 2.83×——这是唯一可能有量化行 outright 赢的区间，因为 M=1 对整张权重矩阵只有 ~2 FLOP/byte，而 H100 的 ridge 在 ~295。int4 的 decode 列要软着读：同一 launch config 在 m=1 的 run-to-run 波动是 22–30 µs。
- prefill（M≥512）算术强度越过 ridge，cuBLAS 开始真正吃 tensor core（m=2048 各投影 390–766 TFLOP/s，中尺寸投影 ~73% 峰值）。epilogue-scale 修复把 fp8 W8A8 的差距拉到 1.34×、int8 W8A8 到 1.12×（两者 outright 赢 gate_up 与 qkv），weight-only 行仍被解包循环钉死：w4a16 4.45×、w8a16 2.70×、nvfp4 6.58×。
- **W4A16 现状**（对照上面 A10 那张初版内核表，那张表勿引用）：decode（M≤32）0.35–0.88×（M=1/8 档 0.50–0.88×，M=32 档掉到 0.35–0.86×；最高 30B/down 的 M=8 0.88×）、prefill（M≥512）0.17–0.47×。内核 docstring 记录了同 shape 家族（N=K=4096）的重写前后：m=1 从 33.9 → 23.2 µs（cuBLAS fp16 23.1 µs，即打平）、m=64 从 49.9 → 31.1 µs。它也是五个 dense 量化内核里唯一读 autotune store 的，`--tune` 后 m≥512 还能再快 13–25%。
- **NVFP4 48 个测试点全输**（包括那 13 个有别的量化行赢的），是结论不是 bug：e2m1 解包每权重元素约 10 条整数运算，比它省下的字节贵一个数量级——读作显存的价格。
- 与 A10 的关键差异：同一个 W8A16 格式，A10 上 decode 稳定赢 1.6–1.7×，H100 上 24 个 decode 测试点只赢 2 个（都在 gate_up，1.21–1.22×）、qkv 档输到 0.76–0.79×——A10 的 600 GB/s 让 bf16 自己就卡在带宽上，删字节直接省时间；H100 的 bf16 只占峰值 43.5%，内核没在等显存。反过来 H100 有 sm90 的 fp8 `wgmma`（`BLOCK_M ≥ 64` 才发射）与 int8 `imma`（`BLOCK_M=16` 起可用），所以真 W8A8 两行在 H100 才有胜机（int8 decode 赢 6/24、fp8 W8A8 赢 3/24，全在 bf16 带宽占比高的投影），而这两行在 A10（sm86 无原生 fp8 GEMM）上付的是纯开销。

#### 量化算子精度汇总

精度与设备无关（只取决于量化粒度与反量化路径）：

| 量化方案 | 相对误差 (vs fp32) | 权重内存节省 |
| ---------- | ------------------- | ------------- |
| fp8 blockwise (128×128) | < 0.04% | 2× |
| int8 per-channel | < 0.03% | 2× |
| int4 group-wise (AWQ/GPTQ) | < 5% | 4× |
| smoothquant W8A8 | < 2% | 2× |

复现：

```bash
# 内核精度测试（两台设备同一套用例）
python -m pytest tests/kernels/test_quantization.py tests/kernels/test_w4a16_accuracy.py -v

# 先把 w4a16 的 tile 配置调到本机（落盘 ~/.cache/rapid_llm/autotune/）
python -m scripts.verify.autotune_collect --ops w4a16_matmul \
    --extra-shape 1x4096x4096 --extra-shape 8x4096x4096 \
    --extra-shape 64x4096x4096 --extra-shape 512x4096x4096

# 性能基准
python -c "
import torch, triton
from rapid_llm.kernels.ops.quantization import w8a16_matmul, w4a16_matmul
M, N, K = 1, 4096, 4096
x = torch.randn(M, K, device='cuda', dtype=torch.float16)
qw = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
sc = torch.ones(32, 32, device='cuda')
print('w8a16', triton.testing.do_bench(lambda: w8a16_matmul(x, qw, sc, group_n=128, group_k=128)))

q4 = torch.randint(-2**31, 2**31 - 1, (N, K // 8), device='cuda', dtype=torch.int32)
s4 = torch.rand(N, K // 128, device='cuda') * 0.02
z4 = torch.randint(0, 16, (N, K // 128), device='cuda').float()
w = torch.randn(N, K, device='cuda', dtype=torch.float16)
print('fp16 ', triton.testing.do_bench(lambda: torch.nn.functional.linear(x, w)))
print('w4a16', triton.testing.do_bench(lambda: w4a16_matmul(x, q4, s4, z4, group_size=128)))
"

# H100 口径：全格式矩阵（48 个测试点 × 6 scheme，真实投影 shape）
RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py \
    --json docs/benchmark_logs/kernels/quant_gemm_h100_20260903d.json
RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py --tokens 1 32 2048   # 只测自己服务的宽度
python benchmarks/kernels/bench_quant_gemm.py --tune --dry-run                          # int4 tile 搜索（唯一读缓存的内核）
```

## 二 模型 e2e benchmark 汇总

两节都是**离线推理（offline inference）口径**：全部 prompt 一次性提交、跑完收工，没有 serving 层的请求排队与连续到达。端到端性能从两个互补视角评估：

1. rapid_llm 与 HF transformers 同口径对照，回答"比裸 transformers 快多少"；
2. rapid_llm 自己关/开 CUDA graph 对照，回答"graph 优化本身值多少"。

测试 1：rapid_llm 默认启用 CUDA graph（TextGenerator 和 VisionGenerator 的 use_cuda_graph 均为 True—多模态的 decode 步骤与纯文本结构相同，视觉 token 在 prefill 之后也只是一行普通的 KV cache）。所以两表的 rapid_llm 数字同源—只是 gen_len 与 TPOT 统计方式不同（整体摊销中位数 vs 逐步间隔均值），数字接近而不相等，各按原口径保留。

两节的测试矩阵相同：单卡 A10 22 GiB 放得下的全部 checkpoint，纯文本（四种架构 × bf16/FP8/AWQ）以 batch 并行口径测，多模态（llava / qwen3_vl）以逐请求串行口径测（表一末尾 batch=serial 的行）；单卡放不下的 8B b16 档用 `--tensor-parallel-size 2` 开双卡 TP 测（表一中 GPU=A10×2 的行，decode 走 eager——当时 TP 路径尚不能捕获 graph；连续批处理引擎的 TP-safe 捕获落地后该限制已解除，见下文 H100 补测）。当时未包含、现已在 2×H100 上补齐的：**Qwen2.5-0.5B-Instruct**（A10 本机无权重）、**Qwen3-30B-A3B 的 b16 checkpoint**（A10 双卡放不下；H100 单卡即可）、**Qwen3-4B-Thinking-2507**（A10 无权重）——见两节各自的 H100 补测小节。仍未包含：**Qwen-1_8B**（第一代 `qwen` model_type，不在支持列表，加载即被 registry 拒绝）、**Qwen3-Next-80B**（双卡放不下，需 4 卡级 TP）、**Qwen3-MoE-Tiny**（2 层 4 专家的玩具 checkpoint，fp32 存储 547 MB，数字仅证明 qwen3_moe 架构与 fused_moe kernel 在三层 dispatch 下端到端可用，不代表 MoE 吞吐量级）。

### rapid_llm vs HF transformers（examples/benchmark.py）

**测试环境**：单卡 NVIDIA A10 24GB（sm86，~600 GB/s），torch 2.11.0+cu129 / transformers 5.8.0 / Python 3.12（2026-08-31 重测）；TP2 行为 2×A10。**推理负载**：`examples/benchmark.py` 的 PROMPTS 扩到 batch 8（gen_len 128）或 batch 16（gen_len 256），贪心解码、两端同一 tokenizer 统计输出 token、两端自然 EOS 停止、`torch.cuda.synchronize` 计时、取中位数；rapid_llm 默认启用 CUDA graph，HF 侧 sdpa attention。**日志**：每行对应一份 `docs/benchmark_logs/models/<模型>_b<batch>_g<gen>_<日期>_<时间>.json`（如 `models/Qwen2.5-1.5B-Instruct_b8_g128_20260831_200147.json`，含完整 config 与环境 meta）。指标口径对齐 vLLM/SGLang serving benchmark：

- **TTFT**（首 token 时延，s）= 预填充延迟；
- **TPOT**（每输出 token 时延，ms）= `(latency - ttft) / (output_len - 1)`；
- **TPS**（每请求吞吐，tokens/s）= `1000 / TPOT`，单个请求 decode 阶段每秒生成的 token 数；
- **TGS**（token 生成速度，tokens/s）= `总输出 token / latency`（全 batch 聚合吞吐）；TP 并行行按 `TGS / 并行度` 折算每卡值；
- **TTFT / TPOT 加速比** = `transformers 指标 / rapid_llm 指标`（两项都是延迟，越低越好）；
- **TPS 加速比** = `rapid_llm TGS / transformers TGS`（吞吐越高越好）；
- 三列加速比都标在 rapid_llm 行（大于 1 即 rapid_llm 更快），单侧跑的组合无对照记 `—`。

多模态四行（batch=serial）由 `examples/benchmark_vision.py` 测得：rapid_llm 的多模态路径逐请求串行（processor 单请求），lite 侧 decode 走 CUDA graph 重放（视觉 token 在 prefill 后已是 KV cache 行，捕获的 decode 步与纯文本同构）；TTFT/TPOT 为单请求平均、TGS 为串行循环的聚合吞吐，与纯文本行的 batch 并行口径不同，不要直接比较。TP2 行的 rapid_llm 侧走 `ContinuousBatchingEngine`（唯一带 plan 广播的执行路径），transformers 侧 `device_map=auto` 把层均摊到同样的两张卡（模型并行），两端硬件一致。

| 模型 | GPU | batch | gen_len | 引擎 | TTFT (s) | TPOT (ms) | TPS (tok/s) | TGS (tok/s) | TTFT 加速比 | TPOT 加速比 | TPS 加速比 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen1.5-0.5B | A10 | 8 | 128 | rapid_llm | 0.0180 | 3.04 | 328.9 | 2535.3 | 1.19× | 6.47× | 6.23× |
| Qwen1.5-0.5B | A10 | 8 | 128 | transformers | 0.0215 | 19.65 | 50.9 | 406.8 | — | — | — |
| Qwen1.5-0.5B | A10 | 16 | 256 | rapid_llm | 0.0192 | 3.55 | 281.7 | 4424.8 | 1.24× | 5.64× | 5.54× |
| Qwen1.5-0.5B | A10 | 16 | 256 | transformers | 0.0238 | 20.02 | 50.0 | 798.4 | — | — | — |
| Qwen2.5-1.5B | A10 | 8 | 128 | rapid_llm | 0.0219 | 9.36 | 106.8 | 844.5 | 1.24× | 2.53× | 2.50× |
| Qwen2.5-1.5B | A10 | 8 | 128 | transformers | 0.0271 | 23.67 | 42.2 | 337.6 | — | — | — |
| Qwen2.5-1.5B | A10 | 16 | 256 | rapid_llm | 0.0228 | 8.69 | 115.1 | 1830.1 | 1.27× | 2.79× | 2.77× |
| Qwen2.5-1.5B | A10 | 16 | 256 | transformers | 0.0289 | 24.21 | 41.3 | 660.4 | — | — | — |
| Qwen2.5-1.5B-Instruct | A10 | 8 | 128 | rapid_llm | 0.0216 | 8.24 | 121.4 | 958.8 | 1.28× | 2.93× | 2.90× |
| Qwen2.5-1.5B-Instruct | A10 | 8 | 128 | transformers | 0.0277 | 24.14 | 41.4 | 331.1 | — | — | — |
| Qwen2.5-1.5B-Instruct | A10 | 16 | 256 | rapid_llm | 0.0225 | 8.51 | 117.5 | 1868.2 | 1.24× | 2.78× | 2.76× |
| Qwen2.5-1.5B-Instruct | A10 | 16 | 256 | transformers | 0.0278 | 23.62 | 42.3 | 677.0 | — | — | — |
| Qwen2.5-3B | A10 | 8 | 128 | rapid_llm | 0.0279 | 18.67 | 53.6 | 426.6 | 1.29× | 1.92× | 1.91× |
| Qwen2.5-3B | A10 | 8 | 128 | transformers | 0.0361 | 35.82 | 27.9 | 223.3 | — | — | — |
| Qwen2.5-3B | A10 | 16 | 256 | rapid_llm | 0.0364 | 19.23 | 52.0 | 828.9 | 1.29× | 1.83× | 1.83× |
| Qwen2.5-3B | A10 | 16 | 256 | transformers | 0.0468 | 35.18 | 28.4 | 454.1 | — | — | — |
| Qwen3-0.6B | A10 | 8 | 128 | rapid_llm | 0.0253 | 4.23 | 236.4 | 1820.3 | 1.25× | 6.84× | 6.59× |
| Qwen3-0.6B | A10 | 8 | 128 | transformers | 0.0317 | 28.94 | 34.6 | 276.2 | — | — | — |
| Qwen3-0.6B | A10 | 16 | 256 | rapid_llm | 0.0256 | 4.70 | 212.8 | 3346.7 | 1.29× | 6.10× | 6.00× |
| Qwen3-0.6B | A10 | 16 | 256 | transformers | 0.0329 | 28.65 | 34.9 | 558.2 | — | — | — |
| Qwen3-0.6B-FP8 | A10 | 8 | 128 | rapid_llm | 0.0293 | 4.09 | 244.5 | 1864.5 | 1.06× | 7.10× | 6.78× |
| Qwen3-0.6B-FP8 | A10 | 8 | 128 | transformers | 0.0311 | 29.08 | 34.4 | 274.9 | — | — | — |
| Qwen3-0.6B-FP8 | A10 | 16 | 256 | rapid_llm | 0.0291 | 4.53 | 220.8 | 3460.5 | 1.06× | 6.17× | 6.04× |
| Qwen3-0.6B-FP8 | A10 | 16 | 256 | transformers | 0.0308 | 27.92 | 35.8 | 572.8 | — | — | — |
| Qwen3-1.7B | A10 | 8 | 128 | rapid_llm | 0.0264 | 9.28 | 107.8 | 850.0 | 1.19× | 3.13× | 3.09× |
| Qwen3-1.7B | A10 | 8 | 128 | transformers | 0.0315 | 29.07 | 34.4 | 275.0 | — | — | — |
| Qwen3-1.7B | A10 | 16 | 256 | rapid_llm | 0.0270 | 9.77 | 102.4 | 1626.2 | 1.27× | 3.04× | 3.02× |
| Qwen3-1.7B | A10 | 16 | 256 | transformers | 0.0342 | 29.68 | 33.7 | 538.8 | — | — | — |
| Qwen3-MoE-Tiny | A10 | 8 | 128 | rapid_llm | 0.0059 | 0.93 | 1075.3 | 8281.3 | 1.07× | 4.20× | 4.05× |
| Qwen3-MoE-Tiny | A10 | 8 | 128 | transformers | 0.0063 | 3.90 | 256.4 | 2043.3 | — | — | — |
| Qwen3-MoE-Tiny | A10 | 16 | 256 | rapid_llm | 0.0068 | 0.98 | 1020.4 | 15934.9 | 1.04× | 4.59× | 4.50× |
| Qwen3-MoE-Tiny | A10 | 16 | 256 | transformers | 0.0071 | 4.51 | 221.7 | 3540.1 | — | — | — |
| Llama-3.2-3B-Instruct | A10 | 8 | 128 | rapid_llm | 0.0254 | 15.41 | 64.9 | 516.4 | 1.22× | 1.64× | 1.64× |
| Llama-3.2-3B-Instruct | A10 | 8 | 128 | transformers | 0.0309 | 25.33 | 39.5 | 315.3 | — | — | — |
| Llama-3.2-3B-Instruct | A10 | 16 | 256 | rapid_llm | 0.0514 | 15.96 | 62.7 | 994.1 | 1.08× | 1.74× | 1.73× |
| Llama-3.2-3B-Instruct | A10 | 16 | 256 | transformers | 0.0557 | 27.74 | 36.0 | 574.6 | — | — | — |
| Qwen3-8B | A10 | 8 | 128 | rapid_llm | 0.0561 | 36.79 | 27.2 | 216.6 | — | — | — |
| Qwen3-8B (TP2) | A10×2 | 16 | 128 | rapid_llm | 0.0618 | 41.61 | 24.0 | 383.1 | 1.65× | 1.24× | 1.24× |
| Qwen3-8B (TP2) | A10×2 | 16 | 128 | transformers | 0.1021 | 51.52 | 19.4 | 308.2 | — | — | — |
| Meta-Llama-3.1-8B-Instruct | A10 | 8 | 128 | rapid_llm | 0.0581 | 35.30 | 28.3 | 225.5 | — | — | — |
| Meta-Llama-3.1-8B-Instruct (TP2) | A10×2 | 16 | 128 | rapid_llm | 0.0684 | 31.99 | 31.3 | 495.7 | 1.94× | 1.46× | 1.47× |
| Meta-Llama-3.1-8B-Instruct (TP2) | A10×2 | 16 | 128 | transformers | 0.1327 | 46.82 | 21.4 | 336.9 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507-FP8 (TP2) | A10×2 | 8 | 128 | rapid_llm | 0.0829 | 84.03 | 11.9 | 95.2 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507-FP8 (TP2) | A10×2 | 16 | 128 | rapid_llm | 0.0838 | 84.22 | 11.9 | 190.0 | — | — | — |
| Qwen3-14B-AWQ | A10 | 8 | 128 | rapid_llm | 0.1499 | 43.49 | 23.0 | 180.5 | — | — | — |
| Qwen3-14B-AWQ | A10 | 16 | 128 | rapid_llm | 0.2724 | 42.93 | 23.3 | 357.8 | — | — | — |
| Qwen3-14B-AWQ | A10 | 16 | 256 | rapid_llm | 0.2808 | 45.01 | 22.2 | 348.4 | — | — | — |
| llava-1.5-7b-hf | A10 | serial | 128 | rapid_llm | 0.1599 | 31.72 | 31.5 | 29.3 | 1.22× | 1.15× | 1.15× |
| llava-1.5-7b-hf | A10 | serial | 128 | transformers | 0.1950 | 36.43 | 27.4 | 25.4 | — | — | — |
| Qwen3-VL-4B-Instruct | A10 | serial | 128 | rapid_llm | 0.1296 | 19.44 | 51.4 | 48.7 | 1.11× | 1.72× | 1.68× |
| Qwen3-VL-4B-Instruct | A10 | serial | 128 | transformers | 0.1442 | 33.47 | 29.9 | 29.0 | — | — | — |

结论（2026-08-31 重测，torch 2.11.0+cu129 / transformers 5.8.0 / Python 3.12，覆盖受支持的全部架构含多模态）：

- rapid_llm 的 **decode 全面更快** — TPOT 加速比在 **1.15×～7.1×** 之间，模型越大比值越低（0.6B 档 ~6-7×，3B 档收敛到 ~1.6-1.9×，多模态 7B 档 1.15×；模型越大 decode 越偏 compute-bound，两端都吃满算力）；多模态 4B 档（Qwen3-VL）拿到 **1.72×**—decode 步与纯文本同构，CUDA graph 的收益直接兑现；
- 8B 级 TP2 双卡档同样领先（Qwen3-8B 1.24×、Llama-3.1-8B 1.46×，两端都在同样的两张卡上），说明 TP 切分 + eager decode 在通信开销下仍保住优势；
- 聚合吞吐 TGS 同步放大，TPS 加速比 **1.15×～6.8×**（与 TPOT 同向、略低——TTFT 摊进总时延）。每组配置两端输出 token 数一致，工作量对等。
- **TTFT** 绝对值小（纯文本 6～50 ms），加速比 **1.04×～1.94×**，rapid_llm 普遍略优但 run-to-run 抖动明显，不逐行解读；多模态 TTFT（129～200 ms）含视觉塔前向，rapid_llm 优 1.11×～1.22×。原始日志见 `docs/benchmark_logs/models/*.json`（每份含完整 config）。
- 30B 级 MoE（Qwen3-30B-A3B-FP8，TP2 eager decode）：TPOT ~84 ms 与 batch 8/16 无关（~3B 激活参数 + top-8 专家权重读取，A10 带宽主导），batch 8→16 吞吐线性放大（95→190 tok/s）说明带宽还有余量；权重 29.06 GB 分两卡后每卡仍有 ~6 GB KV（104,528 token/卡）。transformers 侧无法对照（fp8 反量化为 bf16 需 ~60 GB，双卡 44 GB 放不下），同 14B-AWQ 一样记 rapid_llm 单侧。

> 本节表中未出现的组合：**8B 级 b16 单卡档**的 KV 预算（16×2048 token ≈ 4.8 GiB + 16 GiB 权重）超出 22 GiB—已用 `--tensor-parallel-size 2` 双卡 TP 补上（GPU=A10×2 行）；**8B 级 b8 档的 transformers 侧**因 transformers 5.8 的 `caching_allocator_warmup` 需要约双倍模型显存，单卡放不下（b16 双卡档已补测，b8 不再用双卡测以保持与 rapid_llm 单卡 graph 行的硬件口径一致）；**14B-AWQ 的 transformers 侧**因 AWQ 反量化需要 gptqmodel/autoawq（未安装）标为 rapid_llm 单侧。

复现：

```bash
# 全量复现（上表 14 个模型，含各量化路径与多模态的差异化参数）：
PYTHON=/home/honggao/projects/.venv/bin/python benchmarks/suites/run.py compare
# 单模型：
python examples/benchmark.py --model my_weight/Qwen2.5-1.5B-Instruct \
    --batch-size 8 --gen-len 128 --iters 2      # 结果打印并存入 docs/benchmark_logs/*.json
# FP8 checkpoint 的 transformers 基线：--hf-dtype auto（无原生 fp8 的卡上自动 dequant 为 bf16）
# transformers 无法加载的量化（AWQ 需 gptqmodel/autoawq）：--engine rapid_llm 单侧
# 8B 单卡 b8 档：--max-gpu-num-blocks 16384 收缩 KV 池（profile 默认值留给 graph 捕获的空间不足）
# 8B 双卡 TP2 b16 档（lite 走 ContinuousBatchingEngine eager，HF 走 device_map=auto）：
python examples/benchmark.py --model my_weight/Qwen3-8B \
    --batch-size 16 --gen-len 128 --iters 2 --tensor-parallel-size 2
# 多模态（llava / Qwen3-VL，逐请求串行口径，decode 走 CUDA graph）：
python examples/benchmark_vision.py --model my_weight/Qwen3-VL-4B-Instruct
# 30B MoE FP8（TP2 双卡，decode eager；transformers 侧放不下 60 GB bf16，单侧）：
python examples/benchmark.py --model my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --batch-size 16 --gen-len 128 --iters 2 --tensor-parallel-size 2 --engine rapid_llm
```

#### 2×H100 80GB 补测（2026-09-03）

同一套 `examples/benchmark.py` 口径（贪心、两端同一 tokenizer 统计输出 token、自然 EOS 停止、`torch.cuda.synchronize` 计时、取中位数），换到 2×H100 80GB（sm90，torch 2.13.0+cu130 / transformers 5.15.1 / triton 3.7.1 / Python 3.14）。覆盖 modelzoo 里**权重完整且架构受支持**的全部 checkpoint：Qwen2.5-0.5B-Instruct、Qwen3-4B-Thinking-2507（两者 A10 本机无权重），以及 Qwen3-30B-A3B-Instruct-2507（bf16，A10 双卡放不下）与它的 FP8 版。与 A10 表的两点口径差异：bf16 checkpoint 的 transformers 侧改用 `--hf-dtype bf16`（与 rapid_llm 加载的 dtype 一致；A10 套件用的是脚本默认 fp16），FP8 checkpoint 的 transformers 侧仍是单侧（原因换了，见下）。

三个加速比列全部是**同一行内对同一 checkpoint 的 HF transformers 基线**的比值（`HF / rapid_llm`，大于 1 即 rapid_llm 更快），不是跨卡型或跨档位的比较：TTFT 加速比 = HF TTFT / lite TTFT，TPOT 加速比 = HF TPOT / lite TPOT，TGS 加速比 = lite TGS / HF TGS。没有 HF 对照行的档位记 `—`。

| 模型 | GPU | batch | gen_len | 引擎 | TTFT (s) | TPOT (ms) | TPS (tok/s) | TGS (tok/s) | TTFT 加速比 | TPOT 加速比 | TPS 加速比 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | H100 | 8 | 128 | rapid_llm | 0.0137 | 1.25 | 800.0 | 5936.5 | 1.40× | 24.98× | 23.11× |
| Qwen2.5-0.5B-Instruct | H100 | 8 | 128 | transformers | 0.0193 | 31.23 | 32.0 | 256.9 | — | — | — |
| Qwen2.5-0.5B-Instruct | H100 | 16 | 256 | rapid_llm | 0.0140 | 1.34 | 746.3 | 11552.7 | 1.43× | 25.20× | 24.26× |
| Qwen2.5-0.5B-Instruct | H100 | 16 | 256 | transformers | 0.0200 | 33.65 | 29.7 | 476.2 | — | — | — |
| Qwen3-4B-Thinking-2507 | H100 | 8 | 128 | rapid_llm | 0.0228 | 4.86 | 205.8 | 1601.6 | 1.37× | 8.66× | 8.40× |
| Qwen3-4B-Thinking-2507 | H100 | 8 | 128 | transformers | 0.0311 | 42.06 | 23.8 | 190.6 | — | — | — |
| Qwen3-4B-Thinking-2507 | H100 | 16 | 256 | rapid_llm | 0.0253 | 5.07 | 197.2 | 3107.8 | 1.31× | 8.83× | 8.68× |
| Qwen3-4B-Thinking-2507 | H100 | 16 | 256 | transformers | 0.0330 | 44.74 | 22.4 | 358.0 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507 | H100 | 8 | 128 | rapid_llm | 0.0463 | 10.96 | 91.2 | 712.2 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507-FP8 | H100 | 8 | 128 | rapid_llm | 0.0493 | 10.16 | 98.4 | 764.3 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507 (TP2) | H100×2 | 8 | 128 | rapid_llm | 0.0627 | 9.97 | 100.3 | 770.8 | 1.43× | 9.90× | 9.50× |
| Qwen3-30B-A3B-Instruct-2507 (TP2) | H100×2 | 8 | 128 | transformers | 0.0897 | 98.71 | 10.1 | 81.1 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507 (TP2) | H100×2 | 16 | 128 | rapid_llm | 0.0636 | 11.64 | 85.9 | 1328.3 | 1.29× | 8.52× | 8.22× |
| Qwen3-30B-A3B-Instruct-2507 (TP2) | H100×2 | 16 | 128 | transformers | 0.0820 | 99.19 | 10.1 | 161.5 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507-FP8 (TP2) | H100×2 | 8 | 128 | rapid_llm | 0.0641 | 10.08 | 99.2 | 761.5 | — | — | — |
| Qwen3-30B-A3B-Instruct-2507-FP8 (TP2) | H100×2 | 16 | 128 | rapid_llm | 0.0674 | 11.23 | 89.0 | 1370.8 | — | — | — |

读法：

- **decode（TPOT）领先 8.5×～25.2×，吞吐（TGS）同量级**：0.5B 档 24.98×/25.20×，4B 档 8.66×/8.83×，30B-A3B bf16 TP2 档 9.90×/8.52×——模型越大比值越低（HF 侧也逐步吃上算力）。A10 同档只有 6.47×/5.64×（0.5B）：两端都换了卡，比值仍放大——rapid_llm 的 decode 步已压到接近权重带宽下限（0.5B：1.25 ms，对 0.92 GB 权重 / 3.35 TB/s 的 0.27 ms），而 HF eager decode 的每步固定开销基本不随卡型下降。
- **TTFT 只领先 1.29×～1.43×**，与 A10 表同量级：prefill 是 compute-bound 的大 GEMM，两端都走 cuBLAS，差距只在调度与 KV 分配开销上，与 decode 的 launch-bound 局面不同。
- **TGS 与 TPOT 比值接近但不相等**：TGS 的分母是整轮墙钟（含 TTFT 与采样），batch 越大、gen_len 越长，TTFT 的占比越小，两个比值越靠拢（b16/g256 档：0.5B 25.20× 对 24.26×）。
- **30B-A3B bf16 第一次有了 transformers 对照**：A10 双卡 44 GB 装不下 60 GB 权重，H100 上 `device_map=auto` 摊到两张卡即可跑（HF TPOT 98.71 ms，rapid_llm TP2 9.97 ms）。HF 侧跑 MoE 的 128 专家是 Python 循环，这是它 TPOT 的主因，不是硬件差距。
- **30B 级在单张 H100 上就能跑**：bf16 checkpoint 权重 56.87 GB、FP8 版 29.03 GB，所以多出两行 GPU=H100 的 TP1 档（A10 22 GiB 无此档位）；HF 侧的 allocator warmup 要 ~2× 权重，单卡放不下，所以这两行无对照（记 `—`）。TP2 买到的是 KV 容量而不是速度：bf16 从 13.3 万 token/卡 到 86.6 万 token/卡，TPOT 10.96 → 9.97 ms，与 [quantization.md](quantization.md) 的 30B-A3B 结论一致。TP2 行的 rapid_llm 侧走 `ContinuousBatchingEngine`，decode **走 graph**（TP-safe 捕获已落地），与 A10 表的 TP2 eager 口径不同。
- **FP8 checkpoint 的 transformers 侧仍是单侧，但原因换了**：A10 是反量化后的 ~60 GB bf16 放不下显存；H100 显存够，缺的是 transformers finegrained-fp8 kernel 的依赖（`kernels` 包，不在本项目依赖表里），加载即 ImportError。

> 环境注记：transformers 的 `device_map` 需要 `accelerate`（已在 `requirement.txt`，本次补装到 `.venv`）；缺它时 `examples/benchmark.py` 的 HF 侧直接 ValueError。

复现（`$RAPID_LLM_MODELZOO` 为权重根目录）：

```bash
# 小/中模型双引擎两档：
python examples/benchmark.py --model $RAPID_LLM_MODELZOO/Qwen/Qwen2___5-0___5B-Instruct \
    --batch-size 8 --gen-len 128 --iters 2 --hf-dtype bf16
python examples/benchmark.py --model $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --batch-size 16 --gen-len 256 --iters 2 --hf-dtype bf16
# 30B bf16 单卡 TP1（HF 侧的 allocator warmup 要 ~2× 权重，单卡放不下，故单侧）：
python examples/benchmark.py --model $RAPID_LLM_MODELZOO/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --batch-size 8 --gen-len 128 --iters 2 --engine rapid_llm
# 30B bf16 双卡 TP2 双引擎对照（HF 走 device_map=auto）：
python examples/benchmark.py --model $RAPID_LLM_MODELZOO/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --batch-size 16 --gen-len 128 --iters 2 --tensor-parallel-size 2 --hf-dtype bf16
# 30B FP8 同参数，换 --engine rapid_llm（HF 侧缺 fp8 kernel）：
python examples/benchmark.py --model $RAPID_LLM_MODELZOO/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --batch-size 16 --gen-len 128 --iters 2 --tensor-parallel-size 2 --engine rapid_llm
```

原始日志见 `docs/benchmark_logs/models/Qwen*_20260903_*.json`。

rapid_llm 流式输出实录（Qwen2.5-3B，仅演示效果，非并排对比录制）：

![rapid_llm 流式输出](images/qwen2.5-3b-output.gif)

#### 2×H100 80GB 复测（2026-09-08，short 128 / medium 4k / long 32k × 三方）

新基准框架（sglang 式重构后的 `benchmarks/`）首次全模型套件：`bench_models.py` 预检/计划/编排，`bench_offline_throughput.py` 三方引擎对比（rapid_llm / transformers / vllm 0.28 跨 venv），RandomDataset 精确 token 长度（vLLM 同款 decode→re-encode 校正，三档零漂移），greedy、batch 8、iters 2。rapid_llm 按生产默认开三件套（CUDA graph + prefix cache + chunked prefill），完整 engine args 落 JSON。TTFT/TPOT 列为逐请求 p50/p99（transformers 为 batch 级均值，标 `*`）；TPS 两端口径一致可直接比。各引擎原始 JSON 与扩展点明细见 [benchmark_logs/models/models_suite_h100_20260908_125429.md](benchmark_logs/models/models_suite_h100_20260908_125429.md)。

**Qwen2.5-0.5B-Instruct（bf16，TP1；long 档因 max_position_embeddings=32768 收缩为 31744 输入）**

| 场景 | 引擎 | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | TPS (tok/s) |
| --- | --- | ---: | ---: | ---: |
| short | rapid_llm | 43.4/43.4 | 1.57/1.57 | 2775.4 |
| short | transformers | 1014.0* | 15.58* | 171.0 |
| short | vllm | 20.7/31.6 | 2.93/3.04 | 1974.7 |
| medium | rapid_llm | 328.5/328.5 | 15.19/15.94 | 453.6 |
| medium | transformers | 1068.2* | 41.19* | 162.6 |
| medium | vllm | 19.2/22.8 | 1.91/2.07 | 3385.1 |
| long | rapid_llm | 3438.6/3438.6 | 15.59/22.56 | 276.2 |
| long | transformers | 2359.2* | 45.14* | 147.7 |
| long | vllm | 24.0/31.6 | 2.76/3.42 | 1754.3 |

DP2 扩展点（medium, 64 请求）：TP1 1342.1 / TP2 863.6（0.64×）/ DP2 2787.1（2.08×）

**Qwen3-4B-Thinking-2507（bf16，TP1）**

| 场景 | 引擎 | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | TPS (tok/s) |
| --- | --- | ---: | ---: | ---: |
| short | rapid_llm | 52.3/52.3 | 4.79/4.79 | 1274.5 |
| short | transformers | 970.0* | 28.48* | 138.2 |
| short | vllm | 24.6/40.1 | 4.04/4.04 | 1483.0 |
| medium | rapid_llm | 905.2/905.2 | 25.40/29.59 | 217.1 |
| medium | transformers | 1630.9* | 51.40* | 125.5 |
| medium | vllm | 53.9/68.4 | 5.60/5.92 | 1217.1 |
| long | rapid_llm | 21764.3/21764.3 | 30.18/100.27 | 69.5 |
| long | transformers | 10344.0* | 80.87* | 66.1 |
| long | vllm | 35.9/39.1 | 16.29/16.50 | 435.6 |

DP2 扩展点：TP1 580.7 / TP2 487.3（0.84×）/ DP2 1039.6（1.79×）。4B long 的 transformers 侧在套件内因 caching-allocator 碎片 OOM，以 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 补跑成功（表内即补跑值，原始 JSON 的 `retried` 字段有标注）。

**Qwen3-30B-A3B-Instruct-2507（bf16，61.1GB；long 档 KV+权重超单卡切 TP2，其余 TP1）**

| 场景 | 引擎 | TP | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | TPS (tok/s) |
| --- | --- | --- | ---: | ---: | ---: |
| short | rapid_llm | 1 | 88.1/88.1 | 9.95/9.95 | 645.5 |
| short | transformers | 1 | 1293.9* | 91.57* | 61.9 |
| short | vllm | 1 | 21.2/29.1 | 9.14/9.14 | 803.5 |
| medium | rapid_llm | 1 | 1325.4/1325.4 | 48.74/98.20 | 102.7 |
| medium | transformers | 1 | 2241.5* | 121.04* | 58.1 |
| medium | vllm | 1 | 21.9/33.7 | 8.82/8.95 | 825.0 |
| long | rapid_llm | 2 | 19001.5/19001.5 | 76.00/124.82 | 53.3 |
| long | transformers | 2 | OOM | OOM | — |
| long | vllm | 2 | 17.4/21.5 | 10.02/10.18 | 646.1 |

TP2+EP2 A/B（long）：53.3 → 33.4 tok/s（0.63×，batch 8 下 all-to-all 开销超过专家并行收益；EP 归属 `bench_expert_parallel` 特性 A/B）。

**Qwen3-30B-A3B-Instruct-2507-FP8（fp8，31.2GB，TP1；transformers 无原生 fp8 加载，计划内单侧）**

| 场景 | 引擎 | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | TPS (tok/s) |
| --- | --- | ---: | ---: | ---: |
| short | rapid_llm | 92.6/92.6 | 10.34/10.34 | 619.5 |
| short | vllm | 21.6/32.3 | 6.78/6.78 | 1019.1 |
| medium | rapid_llm | 1999.1/1999.1 | 52.54/103.71 | 88.9 |
| medium | vllm | 29.9/40.0 | 6.82/7.14 | 1038.1 |
| long | rapid_llm | 34202.8/34202.8 | 53.91/163.37 | 42.7 |
| long | vllm | 35.1/38.7 | 14.27/14.40 | 485.4 |

DP2 扩展点：TP1 245.9 / TP2 254.6（1.04×）/ DP2 518.5（2.11×）。

读法（详细分析见汇总 md）：

- **short 档 rapid_llm 0.5B 夺冠（1.41×）**：小上下文 decode graph 全程 replay（71 次），TPOT 1.57 ms 低于 vllm 的 2.93 ms；4B/30B short 档与 vllm 差距 0.80–0.86×。
- **medium/long 档 rapid_llm TPOT 恶化 3–8×，根因是 decode graph 按 `seq_len_buckets=(…,4096)` 封顶**：context 超 4096 即回退 eager（medium 档仅 replay 2 次、long 档 0 次），0.5B medium TPOT 15.19 ms 是 launch 开销主导的 eager decode——vllm 同负载 1.91 ms。这是本轮套件发现的首要引擎优化点（扩大/解耦 seq_len buckets 或 decode-only graph），本次按约定不改引擎仅记录。TPOT p99/p50 比值同源：rapid_llm long 档 1.6–3.3×，vllm 仅 1.02–1.24×。
- **TTFT 不可跨引擎直比**：vllm 逐请求 TTFT 在 chunked prefill 下是首 chunk 延迟（故 long 档 vllm "TTFT" 反而低于 short 档），rapid_llm 表内为全 prefill 完成的 1-token 轮批量口径；TPS/TPOT p50/p99 可比。
- **DP2 干净扩展 1.79×–2.11×；TP2 在单卡放得下的模型上全负收益（0.64×/0.84×/1.04×）**——TP 只当显存开关，与计划决策一致；30B bf16 long 切 TP2 后 rapid_llm/vllm 均可跑（KV 25.9GB 超单卡预算），transformers 则本质 OOM（权重/TP+KV/TP+MoE permute ≈ 75GB+ > 79GB，reserved-unallocated 仅 564MB，非碎片）。
- skip 清单（qwen3_5(_moe) 不支持、顶层 30B 不完整分片、compressed-tensors 无 loader、DeepSeek-V4 超预算、Wan2.2 非 LLM）与运行注记（引擎间显存排空、accelerate、expandable_segments）见汇总 md。

复现：

```bash
python benchmarks/suites/run.py models --dry-run   # 打印将执行的完整命令
python benchmarks/suites/run.py models             # 全套实跑（tee 到 docs/benchmark_logs/）
```

### eager vs CUDA graph（benchmarks/suites/run.py e2e）

**测试环境**：单卡 NVIDIA A10 22 GiB（sm86），torch 2.11.0+cu129 / triton 3.6.0 / Python 3.12（2026-08-31）。**推理负载**：batch 8、greedy、`max_gen_len=256`，`--mode both` 同时测 eager 与 CUDA graph；多模态为 8 请求串行口径（TTFT 取每请求首 token 均值、TPS 为串行循环聚合吞吐）。**日志**：`docs/benchmark_logs/engine/e2e_<模型>_b<batch>_g<gen>_<版本>.json`（如 `e2e_Qwen2.5-1.5B_b8_g128_v09_release.json`，含环境 meta）。一次覆盖全部四种受支持架构、三条优化路径与两个多模态模型：

| 模型 | 架构 / 优化 | TTFT (ms) | TPOT eager (ms) | TPOT graph (ms) | graph 加速 | TPS (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen1.5-0.5B | qwen2 / bf16 | 17.6 | 18.07 | 3.39 | 5.3x | 2319.9 |
| Qwen2.5-1.5B | qwen2 / bf16 | 20.9 | 20.47 | 8.54 | 2.4x | 931.7 |
| Qwen2.5-3B | qwen2 / bf16 | 26.4 | 25.85 | 16.68 | 1.55x | 478.4 |
| Qwen3-0.6B | qwen3 / bf16 | 23.0 | 24.00 | 4.61 | 5.2x | 1706.7 |
| Qwen3-0.6B-FP8 | qwen3 / fp8 | 26.2 | 25.94 | 4.49 | 5.8x | 1746.9 |
| Qwen3-1.7B | qwen3 / bf16 | 25.1 | 25.00 | 9.63 | 2.6x | 825.3 |
| Qwen3-8B | qwen3 / bf16 | 57.2 | 38.66 | 37.33 | 1.04x | 213.9 |
| Qwen3-14B-AWQ | qwen3 / w4a16 | 154.7 | 44.49 | 43.90 | 1.01x | 180.4 |
| Qwen3-MoE-Tiny | qwen3_moe / fused MoE | 4.7 | 3.42 | 1.19 | 2.9x | 6620.0 |
| Llama-3.2-3B-Instruct | llama / bf16 | 24.7 | 20.99 | 15.80 | 1.33x | 505.1 |
| Qwen3-VL-4B-Instruct | qwen3_vl / 多模态串行 | 130.7 | 30.30 | 19.46 | 1.56x | 50.3 |
| llava-1.5-7b-hf | llava / 多模态串行 | 169.6 | 30.34 | 29.78 | 1.02x | 32.9 |

统计口径与观察：

- 三项指标：TTFT 取 graph 档从提交到首 token 可见的墙钟（prefill 主导；多模态行为每请求均值），TPOT 取首 token 之后所有步间隔的均值，TPS 为聚合吞吐（`gen_tokens / 总时间`；多模态行为串行循环吞吐）；每档先 warmup 两轮再计时。
- **多模态的 decode 步可以走 graph**：视觉 token 在 prefill 时已写入 KV cache，decode 步与纯文本模型同构（`MultiModalCausalLM.forward` 在 `multi_modal_inputs=None` 时就是纯文本路径），所以 graph 捕获对 llava / qwen3_vl 一样成立—Qwen3-VL-4B 拿到 1.56x，而 llava 7B 只 1.02x，与纯文本的规模规律一致（模型越大算术时间占比越高）。
- graph 加速随规模衰减是结构性的：≤1.7B 的 decode 步只有几毫秒，kernel launch 开销占比高，重放拿到 2.4-5.8x；8B/14B 算术时间主导，加速收敛到 1.01-1.04x。Qwen3-8B 的 KV 池收缩到 16384 token 以进 22 GiB（`--max-gpu-num-blocks 16384`）。
- FP8 与 bf16 的 0.6B TPOT 几乎相同（4.49 vs 4.61 ms）：小模型 decode 是 launch-bound，权重带宽减半的收益体现不出来，FP8 的收益要到大模型才显形。

复现（套件脚本一次跑全矩阵，产出同口径 JSON；解释器要有能跑 CUDA 的 torch 构建—项目 `.venv` 若装了比驱动新的 cu 版本，脚本预检会拦下并提示换 `PYTHON=`）：

```bash
python benchmarks/suites/run.py e2e --out /tmp/e2e             # 全部模型
PYTHON=/home/honggao/projects/.venv/bin/python benchmarks/suites/run.py e2e
.venv/bin/python -m rapid_llm.benchmark.one_batch \
    --model my_weight/Qwen3-14B-AWQ --batch-sizes 8 --verify # 单模型（graph / eager 各一轮）
python examples/benchmark_vision.py --engine rapid_llm \
    --model my_weight/Qwen3-VL-4B-Instruct                   # 多模态（逐请求串行口径）
```

### DeepSeek 多层推理三方对比（含 MoE，examples/benchmark.py，rapid_llm vs transformers vs vLLM）

DeepSeek-V2/V3 的前若干层是 dense、之后才是 MoE（V2-Lite `first_k_dense_replace=1`：层 0 dense、层 1-26 MoE；V3 `first_k_dense_replace=3`：层 0-2 dense、层 3 起 MoE）。**单层 benchmark 只跑到 dense 层、完全跳过稀疏路由，没有意义**，故本节一律跑多层、覆盖到 MoE 层：三方（rapid_llm / HF transformers / vLLM）同一批 prompt、同一口径（贪心、`torch.cuda.synchronize` 计时、取中位数、离线一次性提交）对比 MLA + MoE 的实际 decode 性能。两个 checkpoint 取自共享权重盘：

- **DeepSeek-V2-Lite（完整 27 层 = 1 dense + 26 MoE）**：30 GB 权重单张 A10（22 GiB）放不下，走 **TP2 跨两卡**；不剪层是因为 vLLM 0.21.0 的 DeepSeek 加载器无法跳过 `num_hidden_layers` 剪掉的层（对多出来的 `layers.N..26` 直接 `KeyError`），要拿到 vLLM 三方就必须跑完整模型。完整 27 层覆盖全部 26 个 MoE 层，稀疏路由被充分 exercise。
- **DeepSeek-V3-4layers-MTP-BF16（4 层 = 3 dense + 1 MoE）**：官方剪裁 checkpoint，13 GB 单卡可放；完整 V3（61 层 / 256 专家 / 671B）两卡远放不下，故用这个 4 层 checkpoint、跑满 4 层覆盖到层 3 的 MoE；路由用 golden 门验证过的 regroup override（`n_group=2, topk_group=1, num_experts_per_tok=2`，把 8 专家 / 8 组重组成 2 组 × 4，恢复 noaux_tc 分组语义）。

batch=8、gen_len=128、iters=2、bf16。**每个框架在自己的 venv 下测**（双框架双环境口径）：rapid_llm 与 transformers 在 `rapid_llm/.venv`（torch 2.13.0+cu129、transformers 5.15），vLLM 在源码仓 venv（vllm 0.28.1rc1.dev、torch 2.13.0+cu129，PATH 需含其 bin——flashinfer JIT 要 ninja）。V2-Lite 行为历史共享 venv 口径（vLLM 0.21.0 / torch 2.11），V3 行为双 venv 重测值。**并行口径按模型而定**：V2-Lite TP2（两卡，rapid_llm 走 `ContinuousBatchingEngine`、transformers `device_map=auto`、vLLM `tensor_parallel_size=2`），V3-4layers 单卡；同一模型内三方并行一致、可直接比，跨模型（TP2 vs 单卡）不直接可比。

| 模型 | 层数（dense+MoE） | 并行 | 引擎 | TTFT (s) | TPOT (ms) | TPS (tok/s) | TGS (tok/s) | 每卡 TGS (tok/s) | TPOT 加速比 | TGS 加速比 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V2-Lite | 27（1+26） | TP2 | rapid_llm | 0.0660 | 25.343 | 39.5 | 311.76 | 155.9 | 5.89× | 5.81× |
| DeepSeek-V2-Lite | 27（1+26） | TP2 | transformers | 0.1197 | 149.287 | 6.7 | 53.68 | 26.8 | — | — |
| DeepSeek-V2-Lite | 27（1+26） | TP2 | vLLM | 0.0988 | 24.439 | 40.9 | 318.82 | 159.4 | 0.96× | 0.98× |
| DeepSeek-V3-4layers | 4（3+1） | 单卡 | rapid_llm | 0.0230 | 15.372 | 65.1 | 518.41 | 518.41 | 1.54× | 1.53× |
| DeepSeek-V3-4layers | 4（3+1） | 单卡 | transformers | 0.0248 | 23.621 | 42.3 | 338.55 | 338.55 | — | — |
| DeepSeek-V3-4layers | 4（3+1） | 单卡 | vLLM | 0.0330 | 13.579 | 73.6 | 582.64 | 582.64 | 0.88× | 0.89× |

（TPS = `1000 / TPOT` 每请求口径——V2/V3 的日志产生于 TPS 字段加入前的脚本版本，由 TPOT 折算，与同次 run 内实测等价；TGS = 总输出 token / latency 聚合口径；每卡 TGS：TP2 行按 TGS/2 折算，单卡行即 TGS 本身。加速比 = `对照指标 / rapid_llm 指标`，标在 rapid_llm 行；vLLM 行的 TPOT 加速比小于 1 即 vLLM 的 decode 更快，TGS 加速比同理是吞吐比。）

结果日志（`docs/benchmark_logs/models/`）：V2-Lite 的 rapid_llm / transformers 行在 `DeepSeek-V2-Lite_b8_g128_tp2_20260903_170619.json`，vLLM 行在 `DeepSeek-V2-Lite_b8_g128_tp2_20260903_021325.json`（三方同次运行的 vLLM 数字；lite 行取 k-tile 修复与管线落地后的最新值，对比见下方结论）；V3 三方分别为 `DeepSeek-V3-4layers-MTP-BF16_b8_g128_20260904_050014.json`（rapid_llm）、`..._050117.json`（transformers）、`..._052340.json`（vLLM），双 venv 重测。

结论（四项指标 TTFT / TPOT / TPS / TGS 齐报）：

- **rapid_llm 的 decode 全面快过 transformers**：V2-Lite 27 层 TP2 TPOT **5.89×**、TGS 5.81×；V3 4 层（双 venv 重测）TPOT **1.54×**、TGS 1.53×（MLA 吸收 + 分页 KV，与主表规律一致）；TTFT 也领先（V2 1.81×、V3 1.08×）。
- **vLLM 的稳态 decode 仅快 ~4%，两步修复后基本追平**：V2-Lite 27 层 TP2 上 vLLM 的 TPOT 是 rapid_llm 的 **0.96×**（24.4 vs 25.3ms）、TGS 高 2%。TP CUDA graph（61.9→30.6ms）之后又落了两步：(1) **MoE grouped-GEMM 的 fp16 k-tile 提到整条 cache line 宽**（`_launch_config` 启发式：带宽主导的 M≤512 用 128、大 M 回落 64）——32 个 fp16 权重只有 64B，每行读半条 cache line，实测带宽效率 61%→81%、M 全程 -12%~-28%，TPOT 30.6→25.3ms；(2) **`RAPID_LLM_PIPELINE=1`（O2 launch/harvest 管线，默认关）** 把 plan/readback 的 host 串行移出关键路径（decode 输入从 device 侧 token 网格 gather、索引走 pinned 异步上传），稳态 24.6ms/step（phase 计时口径，与 vLLM 持平）；此时 profile 确认 GPU kernel 已饱和（MoE 达 A10 实测带宽的 ~81%，专家权重读取是下限）。V3 4 层单卡上差距 **0.88×**（4 层里仅 1 层 MoE，k-tile 修复的影响被 dense 层稀释）。
- **但 rapid_llm 的 TTFT（prefill）反超 vLLM**：V2-Lite **1.50×**（0.0660 vs 0.0988 s）、V3 **1.43×**（0.0230 vs 0.0330 s），prefill 路径更轻。即 rapid_llm 首 token 更快、且全面快过 transformers；vLLM 在完整 MoE 栈的稳态吞吐略高，各有胜负。

三处口径限制（如实记录）：

- **V2-Lite 必须跑完整模型才能三方**：vLLM 0.21.0 的 DeepSeek 加载器不跳过 `num_hidden_layers` 剪掉的层（`KeyError`），剪层只剩 rapid_llm + transformers 两方；完整 27 层单卡放不下需 TP2，故 V2 走 TP2 完整栈、V3 用物理 4 层 checkpoint 单卡。
- **TP2 三方须分进程跑**：rapid_llm TP2 的 follower 进程（rank 1）在 `engine.shutdown()` 后不释放显存，同一 invocation 里接着跑 transformers 会 OOM，故 V2 三方拆成 `--engine rapid_llm / transformers / vllm` 三次独立进程（V3 单卡无此问题，一次 `--engine all`）。
- **vLLM 环境**：本机共享 venv 的 vLLM 为 editable 安装，源码一度漂到 0.23.1rc1 而预编译 `_C` 扩展停在 0.21.0（缺 `get_cuda_view_from_cpu_tensor` 算子），构造 `LLM` 直接崩；0.23.1rc1 源码又要求 torch 2.13（cu130），本机驱动 550 跑不了。把 vLLM 源码 checkout 回 v0.21.0 与预编译二进制 / torch 2.11 对齐后才测得上表（测完已恢复共享仓原 HEAD）。

复现：

```bash
# 三方需把 .venv/bin 与 cuda bin 放进 PATH（vLLM 的 flashinfer JIT 要 ninja）：
export PATH=/home/honggao/projects/.venv/bin:/usr/local/cuda-12.9/bin:$PATH
# V2-Lite 完整 27 层 TP2 —— 每个引擎单独进程（TP2 follower 不释放显存，同进程连跑会 OOM）：
for eng in rapid_llm transformers vllm; do
  PYTHONPATH=. /home/honggao/projects/.venv/bin/python examples/benchmark.py \
    --model /data/shared/llm_weights/DeepSeek-V2-Lite \
    --batch-size 8 --gen-len 128 --iters 2 --engine $eng --hf-dtype bf16 \
    --tensor-parallel-size 2 --vllm-gpu-mem-util 0.8
done
# 可选：rapid_llm 一次运行前加 RAPID_LLM_PIPELINE=1 启用 O2 launch/harvest 管线
# （稳态 24.6ms/step phase 口径；它会延迟一步处理 stop，是部署选项而非默认）。
# V3 四层三方（双 venv 分跑，单卡）：rapid_llm + transformers 在 rapid_llm venv：
for eng in rapid_llm transformers; do
  PYTHONPATH=. /home/honggao/projects/rapid_llm/.venv/bin/python examples/benchmark.py \
    --model /data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16 \
    --batch-size 8 --gen-len 128 --iters 2 --engine $eng --hf-dtype bf16 \
    --hf-overrides '{"n_group":2,"topk_group":1,"num_experts_per_tok":2}'
done
# vLLM 在源码仓 venv（PATH 需含其 bin，flashinfer JIT 要 ninja；gpu_mem_util 0.7 给 MLA workspace 留空间）：
export PATH=/home/honggao/projects/open_source/vllm/.venv/bin:$PATH
PYTHONPATH=. /home/honggao/projects/open_source/vllm/.venv/bin/python examples/benchmark.py \
  --model /data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16 \
  --batch-size 8 --gen-len 128 --iters 2 --engine vllm --hf-dtype bf16 \
  --hf-overrides '{"n_group":2,"topk_group":1,"num_experts_per_tok":2}' \
  --vllm-gpu-mem-util 0.7
```

### DeepSeek-V4-Flash-6layers（DSpark weight-only fp8/MXFP4，TP2，rapid_llm + transformers 实测）

DeepSeek-V4-Flash 官方剪裁的真实权重 checkpoint（22 GB，DSpark 推理格式，非随机初始化），6 层覆盖 V4-Flash 的全部前向算子：层 0-5 的 attention 按 compress_ratios `[0,0,4,128,4,128]` 排布为 SWA、SWA、CSA、HCA、CSA、HCA（滑动窗、压缩注意力、带 hyper-connection 的注意力各两种），`num_hash_layers=3` 使前 3 层 MoE 走 hash 路由（tid2eid 查表）、后 3 层走 score 路由（含 e_score_correction_bias）。量化全部是 weight-only：线性层 fp8 e4m3 权重 + 128×128 e8m0 block scale（`w8a16` kernel 内 dequant），专家权重 MXFP4 e2m1（byte-packed，偶数 K 低 nibble）+ 32 通道 e8m0 scale（`fused_moe` kernel 内解码），运行时激活保持 bf16。22 GB 权重单卡放不下，TP2 每卡 13.74 GiB。

**vLLM 在本机 A10（SM86）上无法服务 DeepseekV4，任意层数都不行**，证据链：

- V4 的两条 attention 后端（FlashMLA / FlashInfer）都经 `DeepseekV4Indexer` → `SparseAttnIndexer`，其 `__init__`（`vllm/model_executor/layers/sparse_attn_indexer.py:778`）在 CUDA 平台上 `not has_deep_gemm()` 即 raise，无 fallback——indexer 的 `fp8_fp4_paged_mqa_logits` 是 DeepGEMM kernel 直包；
- `vllm/platforms/cuda.py::support_deep_gemm` 白名单只有 SM90 / SM100 家族 / SM120 家族，DeepGEMM 的 cmake 架构集合（9.0a / 10.0x / 12.0x）与 SM86 交集为空——这是 kernel 支持矩阵限制，不是层数或配置问题（剪到 4 层、改 `num_hidden_layers` 都绕不开 indexer）；
- 源码仓 vendored 的 `deep_gemm._C` 扩展还是旧 torch ABI 编译（引用 torch 2.13 已删除的 `materialize_cow_storage` 符号），pypi `deep_gemm` 1.0.0 sdist 本机构建亦失败（缺 cutlass 子模块）。

V3 不受影响（MLA 有 Triton 路径，不依赖 DeepGEMM）。V4 的性能对比因此是 rapid_llm + transformers 两方：transformers 侧先把 DSpark checkpoint 离线反量化成 bf16、按 transformers 模块树重命名后落盘（`tests/layer/convert_v4_hf.py`，自包含键映射 + fp8/MXFP4 dequant + 逐专家 w1/w3 fuse 成 `gate_up_proj`，探针断言 / meta 扫描 / 重开核对三重自验证；产物 `/data/shared/llm_weights/DeepSeek-V4-Flash-6layers-hf-bf16-v2`，12 分片 75.6 GiB），GPU 上即以 bf16 原生跑；内存里逐 key 转换 + fp32 CPU 的组合只保留给精度 oracle（下节），性能数字不再依赖它。转换有一个 transformers 5.15 的默认行为要显式绕开：`save_pretrained` 默认 `save_original_format=True`，会把 state_dict 反向转换回 checkpoint 原始键（DSpark 形态），必须传 `save_original_format=False` 才能落出 transformers 原生键的产物。精度对比的参考实现同用 transformers 5.15 的 eager `DeepseekV4ForCausalLM`（下节）。

环境与负载（两侧同口径，均在 rapid_llm venv：torch 2.13.0+cu129 / transformers 5.15.1 / Python 3.13 / CUDA 12.9；2×A10 22 GiB，sm86，PCIe host bridge 互联；64 核 CPU / 369 GB 内存）：batch=8、gen_len=128、iters=2、bf16 激活、贪心解码、`torch.cuda.synchronize` 计时、取中位数；两侧 decode 都走 eager（lite 侧 `--no-cuda-graph`：V4 每层滑窗/压缩器状态是 Python 侧张量重绑定，CUDA graph 只重放 kernel 不重放属性绑定，捕获即失效；transformers 侧本身就是 eager）。**两侧执行模型不同，数字不构成纯 kernel 对照**：

- **rapid_llm**：全 GPU TP2，fp8/MXFP4 weight-only kernel 内 dequant，每卡 13.74 GiB；
- **transformers**：bf16 全量反量化权重；attention / hyper-connection / 路由 / 共享专家在单卡（cuda:0），routed experts 走 CPU 异构——每层 routed 专家 ~26 GB bf16（`gate_up_proj` [256,8192,4096] + `down_proj` [256,4096,2048]），22 GiB 卡放不下单层，且层内 hyper-connection 融合（attn 输出、hc 参数、mlp 输出在单表达式混合）不容 layer 内跨卡切分，故 monkeypatch `DeepseekV4SparseMoeBlock.forward` 让 routed 激活（每 token 只有 top-6 行）过 PCIe 到 CPU 算完搬回；加载走 `--hf-direct-load`（meta-init + safetensors assign，绕过 from_pretrained 的 DSpark 转换路径）。

| 引擎 | 执行模型 | TTFT (s) | TPOT (ms) | TPS (tok/s) | TGS (tok/s) | 每卡 TGS (tok/s) | TPOT 加速比 | TGS 加速比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapid_llm | 2×A10 TP2 全 GPU，fp8/MXFP4 kernel 内 dequant，eager | 0.1333 | 52.520 | 19.0 | 150.59 | 75.3 | 29.8× | 29.5× |
| transformers | 1×A10 + CPU：attn/路由/共享专家 GPU，routed experts bf16 CPU 异构，eager | 2.0281 | 1565.764 | 0.64 | 5.10 | 5.10 | — | — |

（TPS = 1000/TPOT 每请求口径；TGS = 总输出 token / latency 聚合口径；每卡 TGS：TP2 行按 TGS/2 折算，transformers 侧实际只有 1 张 GPU 参与算力，5.10 即每卡值。加速比 = transformers 指标 / rapid_llm 指标，含 CPU offload 代价（TTFT 加速比 15.2×、TPOT 29.8×、TGS 29.5×），不反映纯 kernel 差距。）

V4-Flash 每层都是 256 专家 top-6 路由的 MoE（moe_intermediate 2048、hidden 4096），且 CSA/HCA 层每步还要维护 indexer（index_topk 512）与 compressor 的前缀状态——单层算子密度远高于 V2/V3 的 MoE 层。rapid_llm eager decode 下 6 层的逐层 Python 开销叠加，TPOT 52.5 ms 与 V2-Lite 27 层 eager 时期的 61.9 ms 同量级，主要构成是每层的路由、专家 GEMM（fp8/MXFP4 weight-only dequant）与滑窗状态维护；这是 V4 在 rapid_llm 的首次端到端吞吐记录，优化（graph 兼容的滑窗状态重构）留待后续。transformers 侧的 TPOT 1565.8 ms 主要耗在每个 decode 步把 routed 激活搬去 CPU、在 CPU 上做 bf16 专家 GEMM（256 选 6 的 grouped GEMM 无 GPU 加速）再搬回，PCIe 往返 + CPU 算力共同拉长步时；TTFT 2.03 s 同构成（prefill 每 token 同样过 CPU 专家栈）。

复现（日志：lite 侧 `docs/benchmark_logs/models/DeepSeek-V4-Flash-6layers_b8_g128_tp2_20260904_045910.json`，transformers 侧 `models/DeepSeek-V4-Flash-6layers-hf-bf16-v2_b8_g128_tp2_20260904_162952.json`，均含完整 config 与指标）：

```bash
cd /home/honggao/projects/rapid_llm

# 0) 一次性：DSpark checkpoint -> transformers bf16 落盘（CPU-only，约 2.5 min，产物 75.6 GiB；
#    默认读 my_weight/DeepSeek-V4-Flash-6layers、写 my_weight/DeepSeek-V4-Flash-6layers-hf-bf16-v2，
#    RAPID_LLM_TEST_DSV4_DIR / RAPID_LLM_TEST_DSV4_HF_DIR 可指到别处的源 checkpoint 与产物）：
.venv/bin/python -m tests.layer.convert_v4_hf

# 1) rapid_llm TP2（--no-cuda-graph：V4 的滑窗缓存每步重绑 Python 侧张量，
#    graph 回放不了属性重绑定，decode 走 eager）：
PYTHONPATH=. .venv/bin/python examples/benchmark.py \
  --model /data/shared/llm_weights/DeepSeek-V4-Flash-6layers \
  --batch-size 8 --gen-len 128 --iters 2 --engine rapid_llm \
  --tensor-parallel-size 2 --hf-dtype bf16 --no-cuda-graph

# 2) transformers 侧（--hf-direct-load：键已匹配模块树，meta-init + safetensors
#    assign，跳过 from_pretrained 的 DSpark 转换路径）：
PYTHONPATH=. .venv/bin/python examples/benchmark.py \
  --model /data/shared/llm_weights/DeepSeek-V4-Flash-6layers-hf-bf16-v2 \
  --batch-size 8 --gen-len 128 --iters 2 --engine transformers \
  --tensor-parallel-size 2 --hf-dtype bf16 --hf-direct-load
```

### 精度差异：V3-4layers 三方对比 & V4-Flash-6layers（rapid_llm vs transformers）

两套精度口径都回答同一问题：rapid_llm 的模型结构与参考实现（transformers / vLLM）是否逐 token 等价。指标：贪心序列逐步 token 一致率（agreement，@n 表示首分歧步）、prefill 逐位置 top-1 一致率、逐步 top-5 id 一致率。

**V3-4layers 三方**（真实文本 prompt 三种长度、贪心 32 步；rapid_llm 与 transformers 单卡 bf16，vLLM TP1 bf16 源码仓 venv，同 golden 门 override；日志：`docs/benchmark_logs/accuracy/accuracy_v3_parity_20260904_034741.json`、`accuracy_v3_vllm_20260904_043841.json`）：

| prompt 长度 | prefill 逐位置 top1（lite~HF） | lite~HF | lite~vLLM | vLLM~HF |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.000 | 0.469 @13 | 0.281 @8 | 0.250 @8 |
| 130 | 0.992 | 32/32 全对 | 32/32 全对 | 32/32 全对 |
| 514 | 0.971 | 0.219 @7 | 0.219 @7 | 32/32 全对 |

分叉步的逐 top5 解剖（`tests/layer/deepseek.py v3 three-way`）说明残部分歧是 bf16 数值噪声而非结构差异：

- seq 130 三方 32 步完全一致（含 MoE 层完整路由）——路由与 MLA 路径结构等价的直接证据；
- seq 16 的两处分叉步上，vLLM 自己的 top1 与 top2 logprob 完全相等（step 8：17117 与 48301 均 -3.1553；step 13：260 与 10466 均 -1.1548），三方各自的选择就是平局 tie-break 差异；
- seq 514 是 rapid_llm 在第 7 步分叉（vLLM 与 HF 一致选 7294，双方 margin 0.25；rapid_llm 选 6791）。该 prompt 是 130-token 文本重复 4 次，prefill logits 的 max_abs_diff（14.06，vs 短 prompt 的 0.25/0.5）集中在长序列后段：4 层浅模型输出分布温和、MLA 吸收路径的 bf16 噪声随 prefill 长度累积，MoE top-2 专家的边界 token 翻转后被贪心序列放大。

**V4-Flash-6layers**（rapid_llm bf16 TP2 vs transformers fp32 CPU 参考；输入为同种子随机 token id、三种 prefill 长度、贪心 32 步，两边共享同一确定性序列，排除分词差异噪声；日志：`docs/benchmark_logs/accuracy/accuracy_v4_lite_20260904_042455.json`、`accuracy_v4_hf_20260904_043200.json`）。参考实现将 DSpark 权重在内存中转换为 HF 命名并 dequant（fp8 块、MXFP4 nibble 解包）后以 fp32 计算，是最强 oracle：

| prompt 长度 | greedy 一致 | top-5 id 一致率 | shared logprob max-drift |
| ---: | ---: | ---: | ---: |
| 64 | 30/32（首分歧 @30） | 0.950 | 0.279 |
| 256 | 32/32 全对 | 0.981 | 0.486 |
| 1024 | 12/32（首分歧 @12） | 0.406 | 4.568 |

逐步 margin 核对（compare 落盘的两侧 per-step top5 JSON）：96 步里真正独立的分歧只有 2 处，其余全部是首分歧后上下文分叉的雪崩——

- seq 1024 step 12：rapid_llm 的 top1/top2 logprob 完全相等（margin 0.0000，argmax tie-break 取了索引小的 token），HF 侧 margin 仅 0.0709，且 HF 选的 token 就是 rapid_llm 分布的 rank-1；
- seq 64 step 30：双方 margin 0.125 / 0.037，互相落在对方 top-2，logprob 差 0.04；
- 即所有独立分歧都发生在 margin < 0.13 的近平局步、对方选择都在自己 top-2 内。bf16 权重 + bf16 计算相对 fp32 参考的数值差在这个量级属预期，非结构差异；256-token prefill 下 32 步全对进一步排除了路由 / attention / 量化路径的结构性偏差。

这两条性质（首 token 共享决策 + 独立分歧均为近平局）与各长度的匹配步数下限一起固化为 pytest 精度门 `tests/golden/test_deepseek_v4_flash_parity.py`（gpu+slow 档）：门不锁逐 token 相等，缺 checkpoint / 双卡 / ~200 GB 空闲内存时显式 xfail（UNVERIFIED）而非静默跳过。

复现：

```bash
# V3 三方（前两者在 rapid_llm venv 单卡；vLLM 在 vllm 源码仓 venv）：
python -m tests.layer.deepseek v3 parity
/home/honggao/projects/open_source/vllm/.venv/bin/python -m tests.layer.deepseek v3 vllm
python -m tests.layer.deepseek v3 three-way \
    docs/benchmark_logs/accuracy/accuracy_v3_parity_<ts>.json docs/benchmark_logs/accuracy/accuracy_v3_vllm_<ts>.json
# V4 精度对比（rapid_llm TP2 双卡；transformers 跑在 CPU，需 ~200 GB 内存做 fp32 转换）：
python -m tests.layer.deepseek v4 lite
python -m tests.layer.deepseek v4 hf
python -m tests.layer.deepseek v4 compare \
    docs/benchmark_logs/accuracy/accuracy_v4_lite_<ts>.json docs/benchmark_logs/accuracy/accuracy_v4_hf_<ts>.json
# 同两侧共用的 pytest 精度门（单文件直跑；需要双卡 + ~200 GB 空闲内存）：
pytest tests/golden/test_deepseek_v4_flash_parity.py
```

## 三 性能优化历史记录

### 迭代式优化记录

输入提示词：

```bash
prompts: List[str] = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

    Hi everyone,

    I just """,
    # Few shot prompt (providing a few examples before asking model to complete more);
    "Roosevelt was the first president of the United States, he has",
]
```

1，针对 decode 阶段使用 cuda graph 优化后，单次 decode 阶段时间为 `8.2402` ms，使用之前为 `17.2241` ms，性能提升 2x 倍，这个结果跟 vllm 应用 cuda graph 后的性能提升倍数几乎一致。

```bash
INFO: After apply cuda graph, Decode inference time: 8.2402 ms
INFO: Before apply cuda graph, Decode inference time: 17.2241 ms
```

2，在前面的基础上，继续优化，使用 flashattention 替代原有的标准 attention。

> flashattention1 对训练模型帮助更大，在提示词很短时，其速度提升效果有限。推理时的 decode 阶段应该用 flash-decoding。

```bash
INFO: input tokens shape is  torch.Size([8, 115])
# 使用 flashattention 前
INFO:rapid_llm.generate:Batch inference time: 3152.0476 ms
INFO:rapid_llm.generate:Tokens per second: 97.71 tokens/s
# 使用 flashattention1 后
INFO:rapid_llm.generate:Batch inference time: 2681.3823 ms
INFO:rapid_llm.generate:Tokens per second: 114.87 tokens/s
```

3，继续优化, 将 `flashattention` 升级到 `flashattention2`, 减少一定计算量。

```bash
INFO:rapid_llm.generate:Batch inference time: 2103.0737 ms
INFO:rapid_llm.generate:Tokens per second: 146.45 tokens/s
```

4，再次优化，decode 阶段的推理使用 `flashdecoding`，提升 decode 阶段的 attention 计算并行度，充分发挥 GPU 算力。

```bash
INFO:rapid_llm.generate:Decode stage Batch inference time: 1641.4178 ms
INFO:rapid_llm.generate:Decode stage tokens per second : 187.64 tokens/s
```

5，继续再次优化，支持 kv cache 高效的动态管理（类似 tokenattention），解决了 kv cache 显存浪费和分配低效的问题。

```bash
INFO:rapid_llm.generate:Decode stage Batch inference time: 1413.9111 ms
INFO:rapid_llm.generate:Decode stage tokens per second : 217.84 tokens/s
```

6，一个简单的优化, 使用 `GQA_KV_heads_index` 替代 `repeat_kv` 函数。

7，一个常见且简单的优化, kv 线性层融合。

8，一个常用的优化，算子融合：残差连接的 skip 操作和 `rmsnorm` 算子融合，形成新的 `skip_rmsnorm` 算子。

9，重构并优化 `MHA` 模块，优化 `context_attention` 和 `token_attention` 内核支持 `Nopad attention` 和 `kv cache` 动态分配和管理：

- token_attention 支持直接传入 kv_cache 索引和序列实际长度 seq_len, 减少了 kv cache 在 `MHA` 模块中的 `concat` 和 `view` 操作，并实现了 `Nopad` token_attention。
- 将每次 prefill/decode 过程动态分配实际 prompts 长度的 kv cache 索引个数，而不是在模型推理之前一次性分配连续的 `(max(promptes_len) + max_gen_len) * batch_size` 个 tokens 的 kv cache 空间。

10，引擎侧消除 decode 循环中的 GPU→CPU 同步。原实现每步执行 `bool(hit_stop[i])×batch + all() + .item()` 共 9 次同步，另外 `decode_alloc_kv_cache` 每步做一次 40960 元素的 `torch.nonzero` + 2 次 `.item()`。CPU 一旦读 GPU 张量就必须等前面的 kernel 全部完成，launch 流水线被反复清空，这是 eager decode 里 TPOT 比 GPU 真实计算时间高一倍的主因。

- 新增 [`StopCriteria`](rapid_llm/engine/stop_criteria.py)：结束标志常驻 GPU，用词表大小的布尔查找表判 EOS，全批一次张量运算完成，每 8 步才做一次 `all()` 轮询。
- 新增 [`_DecodeSession`](rapid_llm/engine/llm_engine.py)：把 per-request 状态封装出来，主机侧仅在轮询边界读回一次采样结果。
- 给 [`KVCacheManager`](rapid_llm/executor/kv_cache_manager.py) 加 bump 分配器：`generate()` 内 KV 缓存是纯追加分配，只用一个 int 游标记录写入位置，任一部分释放后自动回退到原全表搜索。

11，消除 O(n²) 流式解码。原实现每步都 `tokenizer.decode(tokens[prompt_len:cur])` 整段解码再与已输出内容做差，256 token 时累计 ~0.8 ms/step。新增 [`IncrementalDetokenizer`](rapid_llm/engine/detokenizer.py) 用滑动窗口只解码 `[prefix_offset:]` 和 `[prefix_offset:read_offset]` 两小段，仍能正确处理 SentencePiece 的前导空格（`▁` 需要上下文）与跨 token 的多字节 UTF-8（结尾遇 `\ufffd` 时先攥住不吐）。摊销后每步常数代价。

12，向量化 repetition penalty。原实现按 batch 逐行 `torch.unique + clone + index_put`，约 4·batch 次 kernel 启动 + 一次全量 clone。新增 [`GeneratedSpan`](rapid_llm/engine/sampler.py) 数据类和 padding-safe scatter：把 batch 已生成 token 一次 scatter 到 `[batch, vocab+1]` 布尔表的哨兵列（避免 padding 位置用 False 覆盖真实命中），再两次 `torch.where`。共 3 个 kernel、无 clone。

以上 10-12 项主机侧优化的前后对比（NVIDIA A10 23 GB / Qwen2.5-0.5B / batch=8 / max_gen_len=256 / greedy；TTFT 为首 token 墙钟时间、TPOT 为稳态每 token 延迟、TPS 为 batch 聚合吞吐，口径对齐 vLLM）：

| 配置                    | TTFT (ms) | TPOT (ms) | TPS (token/s) |
|------------------------|-----------|-----------|---------------|
| eager（优化前）          | 15.0      | 15.04     | 532           |
| eager（10-12 项优化后）  | 13.7      | 13.55     | 590           |

eager 路径的 GPU 计算本身没有变化，TPOT 从 15.04 ms 降到 13.55 ms 几乎全部来自主机侧开销的削减：每步 GPU→CPU 同步从 9 次降到 0.125 次、流式解码从 O(n²) 降到摊销 O(1)、penalty 从约 4·batch 次 kernel 降到 3 次。

13，[`TextGenerator`](rapid_llm/engine/generator.py) 默认开启 CUDA Graph 捕获（多模态显式关闭）。KV 显存预算里为 graph 捕获预留 workspace（[`estimate_capture_workspace`](rapid_llm/executor/cuda_graph.py)），并把捕获 batch 上界钳到请求表容量，修复 `0.9 gpu-util + graph capture` 场景下的 OOM。

与 HuggingFace transformers 的对比（NVIDIA A10 23 GB / **Qwen2.5-1.5B-Instruct** fp16 / batch=8 / max_gen_len=256 / greedy，指标口径同上）。HF 侧由 [`rapid_llm.benchmark.offline_throughput --engine transformers`](rapid_llm/benchmark/offline_throughput.py) 测量：左 padding、不套 chat template、`min_new_tokens` 强制跑满 256 步、sdpa attention：

| 引擎                             | TTFT (ms) | TPOT (ms) | TPS (token/s) | 生成总时间 (s) |
|----------------------------------|-----------|-----------|---------------|---------------|
| transformers 5.15（sdpa）        | 27.6      | 24.24     | 330           | 6.21          |
| rapid_llm eager                 | 15.8      | 16.86     | 475           | 4.31          |
| **rapid_llm graph（10-13 项）** | **16.4**  | **9.05**  | **881**       | **2.32**      |

相对 transformers：eager 路径 TPS 1.44x、TPOT 1.44x、TTFT 1.75x；graph 路径 TPS 2.67x、TPOT 2.68x。1.5B fp16 权重约 3.09 GB，A10 带宽下限约 5.2 ms/token（3.09 GB / 600 GB/s）；graph 模式 TPOT 9.05 ms 中主机侧开销已被 CUDA Graph 消除，与下限之间剩余的约 3.9 ms 是 kernel 级优化空间（decode attention、小 GEMM 效率）。

> 0.5B 上的历史对照（同口径）：graph 优化前 TPOT 5.54 ms / TPS 1433，10-13 项后 TPOT 3.77 ms / TPS 2096（3.94x），已逼近 0.5B fp16 权重带宽下限 3.46 ms（1260 MB / 600 GB/s）。

精度验证（[`scripts/verify/golden_tokens.py`](scripts/verify/golden_tokens.py)）：8 个 greedy 用例覆盖单条 / 等长 batch / 混合长度 batch × 有无 repetition penalty，优化前后逐字节完全一致。

14，配置 / 权重加载 / 模型注册三个模块重构，参照 vLLM 的分层：

- **配置**。删除按架构手写的 `model_config.py` dataclass 与它的 HF 字段别名表，schema / 解析 / 默认值全部交给 `AutoConfig`（[`models/config.py`](rapid_llm/models/config.py)）。`ModelConfig` 只补两件 HF config 给不了的东西：运行时旋钮 `max_seq_len`，以及 `num_kv_heads` / `head_dim` / `rope_theta` 的归一化。复用的是配置体系，**不引入** `modeling_*.py`—文本模型仍全跑自己的 Triton 内核。这也修好了一个真 bug：transformers 5.x 把 `rope_theta` / `mrope_section` 收进 `rope_parameters`，旧别名表不认识，Qwen3-VL 的 `mrope_section` 会静默丢失并退化成普通 RoPE。
- **权重加载**。删除 `tools/convert_weights.py` 与 `rapid-llm-convert` 入口，不再需要离线产物。流程与 vLLM 一致：meta 设备上构造空模型 → 就地分配 fp16 参数 → 从 safetensors 流式 `copy_` 到位。只做名字 / 结构重映射（[`models/weights.py`](rapid_llm/models/weights.py)）：K/V 写进 `kv_proj_weight` 的上下两半，MoE 逐专家矩阵堆叠进 `gate_up_proj` / `down_proj`，FP8 block 量化在目标设备上反量化。拷贝循环按**元素个数**统计覆盖率，漏写 / 半写 / 重写都会报错—这是 `strict=True` 的 `load_state_dict` 看不到的（fused 参数只写一半仍然“存在”）。
- **注册**。[`registry.py`](rapid_llm/models/registry.py) 从 181 行降到 81 行：每个条目只剩 `model_type -> (实现类路径, 是否多模态)`，每架构一个 config loader 工厂、`load_config` / `build_model` / `read_model_type` 全部取消。新增一个模型 = 一行表项 + 一个类。

加载耗时（A10 / 页缓存已预热 / 取 3 次最小值）：

| 模型                    | 旧：转换一次 + 加载 `.pth` | 新：直读 safetensors | 硬盘占用变化 |
|------------------------|--------------------------|------------------------|-------------|
| Qwen2.5-0.5B           | 5.75 s + 0.26 s          | **0.22 s**             | −988 MB     |
| Qwen2.5-1.5B-Instruct  | — + 0.61 s               | **0.60 s**             | −3.09 GB    |

稳态推理吞吐不变（同机器各跑 3 次，Qwen2.5-0.5B / batch=8 / greedy）：graph 路径 TPS 2111 / 2119 / 2120（重构前）vs 2117 / 2104 / 2116（重构后），eager 路径两边同处 565–606 的噪声带内。真正省下的是部署路径：0.5B 从“转换 6.0 s + 多占 988 MB”变成“直接 0.22 s 加载”，30B-A3B-FP8 则不再需要那份 61 GB 的 `.pth` 副本。

权重加载的精度验证分三层：

- [`tests/models/test_weight_mapping.py`](tests/models/test_weight_mapping.py)：逐个 key 形状的映射单测 + 覆盖率记账（漏 key / 半写 fused / shape 不对 / 映射到不存在的参数，均必须报错）。
- [`tests/models/test_weight_parity.py`](tests/models/test_weight_parity.py)：6 个架构各随机初始化一个 tiny HF 模型存成真 safetensors，跑完整加载路径后逐参数逐元素对比—k/v 互换、专家下标错位、gate/up 颠倒这些“形状全对但值错位”的 bug 只有这层能抓。
- [`tests/models/test_checkpoint_index.py`](tests/models/test_checkpoint_index.py)：拿真实发布 checkpoint 的 `model.safetensors.index.json`（本地验证过 llava-1.5-7b-hf 686 key、Qwen3-VL-4B 713 key、Qwen3-30B-A3B-FP8 37491 key），在 meta 设备上不读一字节权重就验证“每个 key 都有参数接 / 每个参数都有 key 写”。

顺手抓出的两个旧 bug，都改变了输出，所以单独记一笔：

1. **Qwen3-VL 的 RoPE base 错了 500 倍**。transformers 5.x 只在 `rope_parameters` 里写 `rope_theta`，而多模态路径的旧配置是 `LlamaConfig.from_dict(config.text_config.to_dict())`，读不到顶层 `rope_theta` 就退到 dataclass 默认值 **10000.0**，而 checkpoint 声明的是 **5,000,000**。在 201 token 的纯文本 prompt 上与 HF `Qwen3VLForConditionalGeneration` 对照 logits：

   | rope_theta | 与 HF 的平均 cosine | 最小 cosine | top-1 一致率 |
   | ------------ | --------------------- | -------------- | -------------- |
   | 10000（旧，默认值） | 0.928 | **−0.195** | 99.50% |
   | 5e6（新，读配置） | **0.99973** | **0.982** | **100%** |

   最小 cosine 为负意味着部分位置的 logits 向量完全反了方向。回归用例：[`test_nested_rope_theta_is_not_lost`](tests/config/test_config.py)、[`test_qwen3_vl_language_model_gets_mrope_and_the_right_base`](tests/models/test_weight_parity.py)。
2. **`inv_freq` 被降成 fp16**。旧 loader 最后一句 `model.half()` 连非持久化 buffer 一起转了，RoPE 的 `inv_freq` 静默变成 fp16。它以 `position × inv_freq` 参与相位计算，误差随位置线性放大：Qwen2.5-0.5B 在 position 1024 处相位误差 0.086 rad，LLaVA-1.5（theta=1e4）在 4096 处 0.99 rad。新 loader 不再动 buffer，`inv_freq` 保持 fp32。

因为这两项修正，golden 基线已重录。作为反向验证：强行把 `inv_freq` 改回 fp16 后，新加载路径在 Qwen2.5-0.5B 上与旧 golden 基线 8 个用例逐字节完全一致，说明三个模块的重构本身是 bit-exact 的，输出差异全部来自上面两个修正。重录 diff 里可以看到多条原本陷入重复循环的输出变成正常叙述。

复现命令：

```bash
# rapid_llm 端到端指标（graph / eager 两档；--model 切换模型）
python -m rapid_llm.benchmark.one_batch --model my_weight/Qwen2.5-1.5B-Instruct \
    --batch-sizes 8 --output-len 256 --verify

# HF transformers 基线（同指标口径，同一后端框架换 arm）
python -m rapid_llm.benchmark.offline_throughput --model my_weight/Qwen2.5-1.5B-Instruct \
    --engine transformers --greedy --num-prompts 8 --random-output-len 256

# 精度对照
python -m scripts.verify.golden_tokens --save /tmp/golden.json          # 优化前录制
python -m scripts.verify.golden_tokens --check /tmp/golden.json         # 优化后比对
python -m scripts.verify.golden_tokens --check /tmp/golden.json --cuda-graph
```

### 历史吞吐对比总表（旧脚本，仅供参考）

> 数据来源：本表数字来自本文档下方各模型章节的**历史记录**（由仓库作者早前用
> 旧版 `benchmark.py` 在 3090 上跑出），**并非本次实测**。环境：趋动云 B1.small（3090 的 1/4 卡）/ B1.big（3090 整卡），当时的软件栈未记录；负载：各章节列出的 prompt 集（变长）× 各档 max_gen_len，单次运行。**该批数字无 JSON 日志留存**（结果仅以文本输出形式记在各章节）。旧脚本存在方法学问题：
> transformers 被强制忽略 EOS（`eos_token_id=None`）跑满长度，而 rapid_llm 会提前
> 停止，两端工作量并不一致；且仅有单次运行、只统计吞吐、无 TTFT/TPOT。因此这些倍数
> 仅作趋势参考，请以上方实测表为准。

| 模型 | GPU | batch_size | seq_len¹ | max_gen_len | rapid_llm (tokens/s) | transformers (tokens/s) | 吞吐加速比 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 3090 的 1/4 卡 (B1.small) | 16 | 变长 | 1900 | 411.04 | 104.70 | 3.93× |
| Llama-3.2-3B-Instruct | 3090 整卡 (B1.big) | 8 | 变长 | 1900 | 458.97 | 134.37 | 3.42× |
| Llama-3.2-3B-Instruct | 3090 整卡 (B1.big) | 12 | 变长 | 1900 | 730.45 | 183.95 | 3.97× |
| Qwen2.5-3B-Instruct | 未标注 | 2 | 变长 | 2000 | 98.71 | 69.83 | 1.41× |
| Qwen2.5-3B-Instruct | 未标注 | 4 | 变长 | 256 | 182.28 | 133.33 | 1.37× |
| Qwen2.5-3B-Instruct | 未标注 | 12 | 变长 | 1900 | 581.20 | 172.19 | 3.38× |
| Qwen2.5-3B-Instruct | 未标注 | 16 | 变长 | 512 | 724.38 | 504.73 | 1.44× |
| Qwen2.5-3B-Instruct | 未标注 | 16 | 变长 | 1900 | 735.73 | 215.62 | 3.41× |

> ¹ 这些基准使用的是「多条不同长度提示词」组成的 batch，未固定或记录单一 prompt 长度，
> 故 seq_len 记为「变长」；Qwen2.5-3B 章节未标注 GPU 型号。

#### Llama-3.2-1B-Instruct 性能测试

趋动云 `B1.small` 等同于 `3090` 的 `1/4` 之一卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，rapid_llm 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 16 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 67.8760 s
Transformers inference time: 131.8708 s
rapid_llm throughput: 411.04 tokens/s
Transformers throughput: 104.70 tokens/s
rapid_llm per token latency: 2.432831 ms/token
Transformers per token latency: 9.551007 ms/token
```

#### Llama-3.2-3B-Instruct 性能测试

/gemini/code/rapid_llm/my_weight/Llama-3.2-1B-Instruct

趋动云 `B1.big` 等同于 `3090` 卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，rapid_llm 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 8 的提示词：

```bash
prompts: List[str] = [
        "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "A Complete Introduction to the History of the American Civil War",
        "Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
    ]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 32.0826 s
Transformers inference time: 51.2225 s
rapid_llm throughput: 458.97 tokens/s
Transformers throughput: 134.37 tokens/s
rapid_llm per token latency: 2.178783 ms/token
Transformers per token latency: 7.441883 ms/token
```

batch_size = 12 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 31.3463 s
Transformers inference time: 69.1433 s
rapid_llm throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
rapid_llm per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

#### Qwen2.5-3B-Instruct 性能测试

`batch_size = 2` 时的提示词

```bash
prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code ",
        "How to learn cuda programming, give me some code example.",
    ]
```

`max_gen_len = 2000` 时, benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 34.9293 s
Transformers inference time: 31.6787 s
rapid_llm throughput: 98.71 tokens/s
Transformers throughput: 69.83 tokens/s
rapid_llm per token latency: 10.130305 ms/token
Transformers per token latency: 14.321302 ms/token
```

`batch_size = 4` 时的提示词

```bash
    prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code.",
        "How to learn cuda programming, give me some code example.",
        "How to learn rust, give me some code examples.",
        "How to learn java, give me some code example.",
    ]
```

`max_gen_len = 256` 时, benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 5.5739 s
Transformers inference time: 7.6803 s
rapid_llm throughput: 182.28 tokens/s
Transformers throughput: 133.33 tokens/s
rapid_llm per token latency: 5.486118 ms/token
Transformers per token latency: 7.500309 ms/token
```

`batch_size = 12` 时的提示词

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 26.8804 s
Transformers inference time: 63.2376 s
rapid_llm throughput: 581.20 tokens/s
Transformers throughput: 172.19 tokens/s
rapid_llm per token latency: 1.720564 ms/token
Transformers per token latency: 5.807474 ms/token
```

`batch_size = 16` 时的提示词

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 512` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 11.3434 s
Transformers inference time: 14.9981 s
rapid_llm throughput: 724.38 tokens/s
Transformers throughput: 504.73 tokens/s
rapid_llm per token latency: 1.380484 ms/token
Transformers per token latency: 1.981256 ms/token
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 38.4323 s
Transformers inference time: 70.3268 s
rapid_llm inference output tokens number: 28276
Transformers inference output tokens number: 15164
rapid_llm throughput: 735.73 tokens/s
Transformers throughput: 215.62 tokens/s
rapid_llm per token latency: 1.359186 ms/token
Transformers per token latency: 4.637745 ms/token
```
