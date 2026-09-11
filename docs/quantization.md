# 量化支持

rapid_llm 支持多种权重量化方案，架构与 [sglang](https://github.com/sgl-project/sglang) 对齐，便于扩展。

## 支持的方案

| Scheme | Config Class | Weight | Activation | Scale Granularity | Use Case |
| -------- | ------------- | -------- | ------------ | ------------------- | ---------- |
| **fp8** | `Fp8Config` | fp8-e4m3 | fp16 | 128×128 block | Qwen/DeepSeek FP8 checkpoints |
| **w8a8_fp8** | `W8A8Fp8Config` | fp8-e4m3 | fp8-e4m3 (dynamic) | per-channel / per-token | True W8A8 runtime (`--quantization fp8`) |
| **blockwise_int8** | `BlockInt8Config` | int8 | fp16 | per-channel / group-wise | Runtime int8 (`--quantization int8`) |
| **w8a8_int8** | `W8A8Int8Config` | int8 | int8 (dynamic) | per-channel / per-token | SmoothQuant (`--quantization smoothquant`) |
| **awq** | `AWQConfig` | int4 | fp16 | group-wise (128) | Pre-quantised AWQ checkpoints |
| **gptq** | `GPTQConfig` | int4 | fp16 | group-wise (128) | Pre-quantised GPTQ checkpoints |
| **fp8 KV cache** | `Fp8KVCacheMethod` | — | — | per-tensor | `--kv-cache-dtype fp8` halves KV memory |

## 快速上手

### FP8 checkpoint（Qwen3-30B-A3B-FP8）

FP8 checkpoint 会从 `config.json` 自动检测：

```bash
python -m rapid_llm.cli --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

### 运行时 INT8 量化

加载时把 fp16 checkpoint 量化为 int8：

```bash
python -m rapid_llm.cli --model-dir my_weight/Qwen3-0.6B --quantization int8
```

### 真 W8A8 FP8（权重不反量化）

```bash
python -m rapid_llm.cli --model-dir my_weight/Qwen3-0.6B --quantization fp8
```

### FP8 KV Cache（decode 显存减半）

```bash
python -m rapid_llm.cli --model-dir my_weight/Qwen3-0.6B --kv-cache-dtype fp8
```

### NVFP4 仅权重 4-bit

```bash
python -m rapid_llm.cli --model-dir my_weight/Qwen3-4B --quantization nvfp4
```

本文所有方案中权重最小的（比 bf16 低 2.85×），但在 H100 上测过的每个 shape 都**比 bf16 慢**——选它之前先读 [NVFP4 仅权重 FP4](#nvfp4-仅权重-fp4) 一节。

## 架构

```
rapid_llm/modules/quantization/
├── __init__.py            # BASE_QUANTIZATION_METHODS 注册表 + 工厂函数
├── base_config.py         # QuantizeMethodBase / LinearMethodBase / FusedMoEMethodBase / QuantizationConfig 抽象基类
├── fp8.py                 # Fp8Config + Fp8LinearMethod + Fp8MoEMethod
├── w8a8_fp8.py            # W8A8Fp8Config + W8A8Fp8LinearMethod + W8A8Fp8MoEMethod（A8 专家）
├── w8a8_int8.py           # W8A8Int8Config + W8A8Int8LinearMethod + W8A8Int8MoEMethod
├── blockwise_int8.py      # BlockInt8Config + BlockInt8LinearMethod + BlockInt8MoEMethod
├── nvfp4.py               # NVFP4Config + NVFP4LinearMethod（仅权重，仅 dense）
├── awq.py                 # AWQConfig + AWQLinearMethod + AWQMoEMethod
├── gptq.py                # GPTQConfig + GPTQLinearMethod + GPTQMoEMethod
├── unquant.py             # UnquantizedConfig（fp16 默认）
├── kv_cache.py            # BaseKVCacheMethod + Fp8KVCacheMethod
├── parameter.py           # RawParameter（loader 不得转型为 fp16）
└── utils.py               # 量化辅助函数 + checkpoint 布局适配器（AWQ/GPTQ）
```

### 与 sglang 的对齐表

| rapid_llm | sglang equivalent | Notes |
| ------------ | ------------------- | ------- |
| `QuantizationConfig` | `QuantizationConfig` | ABC with `get_quant_method(layer, prefix)` |
| `LinearMethodBase` | `LinearMethodBase` | `create_weights` + `apply` |
| `FusedMoEMethodBase` | `FusedMoEMethodBase` | 堆叠专家（stacked expert）策略 |
| `Fp8Config` | `Fp8Config` | 仅权重 fp8（block-wise scales） |
| `W8A8Fp8Config` | `W8A8Fp8Config` | 真 W8A8 fp8（per-token 激活量化） |
| `W8A8Int8Config` | `W8A8Int8Config` | SmoothQuant W8A8 |
| `BlockInt8Config` | `BlockInt8Config` | 仅权重 int8 |
| `AWQConfig` | `AWQConfig` | Int4 AWQ checkpoint |
| `GPTQConfig` | `GPTQConfig` | Int4 GPTQ checkpoint |
| `BASE_QUANTIZATION_METHODS` | `BASE_QUANTIZATION_METHODS` | `{name: ConfigClass}` 注册表 |

### 注册表与配置流转

```python
from rapid_llm.modules.quantization import (
    BASE_QUANTIZATION_METHODS,
    get_quantization_config,
    get_quant_config_from_hf,
    for_runtime_scheme,
)

# checkpoint 自动检测：config.json → Config 类 → from_config()
quant = get_quant_config_from_hf(hf_config)  # Fp8Config / AWQConfig / None

# 运行时量化：--quantization int8
quant = for_runtime_scheme("int8")  # BlockInt8Config.per_channel()

# 层向自己的配置索要对应的 method：
method = quant.get_quant_method(layer, prefix)  # Fp8LinearMethod / ...
```

## 性能基准测试

单卡 NVIDIA A10（24 GB，sm86），decode batch size 4，max_gen_len=64，greedy；测量于 2026-08-23（该次运行的 JSON 早于环境日志功能——按同一周的 e2e 日志推断，A10 主机当时的软件栈为 torch 2.11.0+cu129 / triton 3.6.0 / python 3.12）。基线：HuggingFace transformers fp16（eager，相同 prompts）。

### Qwen3-0.6B（dense，28 层，hidden=1024）

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS | vs HF Speedup |
| -------- | ----------- | ------------- | ----------- | ----- | --------------- |
| HF fp16 (baseline) | 1.17 GB | — | 28.19 | 141.7 | 1.0× |
| lite fp16 | 1.40 GB | 147,875 tok | 4.14 | 918.8 | 6.5× |
| lite int8 | 0.99 GB | 141,549 tok | 4.16 | 904.1 | 6.4× |
| lite int8-blockwise | 1.00 GB | 138,385 tok | 4.44 | 849.4 | 6.0× |
| lite fp8（W8A8） | 0.99 GB | 139,153 tok | 8.35 | 448.1 | 3.2× |
| lite smoothquant（W8A8） | 0.99 GB | 135,642 tok | 3.70 | 983.8 | 6.9× |

> Model Mem 仅指模型权重；KV Capacity 为最大缓存 token 数（分页池占满剩余 GPU 显存）。
> 基准日志：该次运行早于 JSON 环境日志功能落地，**无独立 JSON 留存**（数字仅见于本表）；后续各轮的完整日志见 [`docs/benchmark_logs/`](../docs/benchmark_logs/)。
> 复现（同口径重跑，A10 或任意设备）：
>
> ```bash
> python benchmarks/engine/run.py quant --model-dir <Qwen3-0.6B> \
>     --schemes fp16 int8 int8-blockwise fp8 smoothquant \
>     --batch 4 --max-gen 64 --cuda-graph --json out.json   # HF 基线行去掉 --skip-hf
> ```
>
> 上表是 0.6B 模型在 A10 上的结果。Qwen3-4B 与 Qwen3-30B-A3B 在 2×H100 上的完整矩阵——每种方案 × TP/DP × CUDA graph × KV dtype，离线与在线，附两套精度参照——见 [`quant_matrix_20260901.md`](benchmark_logs/quantization/quant_matrix_20260901.md)。它的头条结论（在 H100 的 4B 上没有任何量化方案在速度上胜过 bf16）已被 0903 的三轮 dense GEMM 修复推翻：int8 W8A8 现在 48 个 kernel 级测试点中胜 13 个、fp8 W8A8 胜 5 个、fp8 W8A16 胜 2 个（见[第三次 tile 重扫](#dense-量化-gemm-的第三次-tile-重扫h100)、[第四轮 epilogue 化](#in-loop-scale-的消除epilogue-化与-wgmma-流水线h100-第四轮)与[第五轮 launch 配置维度](#launch-配置的设备与-dtype-维度h100-第五轮)，后者另给 int8 per-channel 换了专属 tile 表，kernel 级再提 ~1.18×），量化矩阵表待重跑后更新。

### Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, 2×H100)

30B 级 MoE checkpoint 的权重以 fp8-e4m3（uint8 存储）+ 128×128 block scales 直接驻留显存，kernel 在 `tl.dot` 前反量化（W8A16）。真 W8A8 路径（`--quantization fp8`，`w8a8_fp8` scheme）用 per-channel weight scale，与本 checkpoint 的 block-scale 布局不匹配——声明 `activation_scheme: dynamic` 的 A8 路径属于那条 runtime scheme，不在此 checkpoint 上启用。checkpoint 的 `modules_to_not_convert`（`lm_head`、每层两个 norm、router `mlp.gate`，共 145 项）由 `Fp8Config.ignored` 接住，保持 bf16 不量化。

按层分发的量化算子（`Fp8Config.get_quant_method`）：

| 层 | Quant Method | Kernel | 权重格式 |
| ---- | -------------- | -------- | --------- |
| `self_attn.qkv_proj` / `o_proj` | `Fp8LinearMethod` | `w8a16_matmul` | fp8-e4m3 + 128×128 block scales |
| `mlp.experts`（128 专家的 gate_up / down） | `Fp8MoEMethod` | `fused_moe` `QUANT_MODE=1` | fp8-e4m3 + block scales（sm89+ 单条硬件 `cvt` 加宽；旧设备 bit-trick 折 256× 进 `DEQUANT_SCALE`） |
| `mlp.gate`（router）/ `lm_head` | `UnquantizedLinearMethod` | cuBLAS 线性层 | bf16 |

两条 kernel 路径都接受 fp16 或 bf16 激活（checkpoint 的 `torch_dtype: bfloat16` 走 bf16）：反量化后的操作数统一对齐激活 dtype 再进 tensor core。TP2 下每卡存半个副本（权重 14.53 GB + KV 各半）；连续批处理引擎的 TP-safe graph 捕获让 TP2 的 decode 同样走 graph（设计见 [tensor_parallel.md](tensor_parallel.md)）——早期"NCCL 集合通信不能进 CUDA graph 捕获"的 eager 限制属于旧引擎路径，已被推翻。

2×H100 80GB (sm90)，batch 4，max_gen_len=64，greedy，max_seq_len 1024。golden 基线为本 checkpoint 的 eager/TP1/KV-auto 配置（`scripts/verify/golden_tokens.py` 录制，control row 复现 1.000）：

| Config | Model Mem | KV Capacity | TTFT (ms) | TPOT (ms) | TPS | golden prefix |
|--------|-----------|-------------|-----------|-----------|-----|---------------|
| tp1+graph | 29.03 GB | 435,879 tok | 66.1 | 13.16 | 285.9 | 1.000 |
| tp1+eager | 29.03 GB | 452,263 tok | 62.5 | 63.35 | 63.2 | 1.000 |
| tp2+graph | 14.53 GB | 1,171,324 tok | 77.5 | 12.76 | 290.3 | 0.638 |
| tp2+eager | 14.53 GB | 1,204,092 tok | 76.1 | 78.50 | 51.0 | 0.638 |
| tp1+kv-fp8+graph | 29.03 GB | 871,790 tok | 67.3 | 14.05 | 268.7 | 0.697 |
| tp1+kv-fp8+eager | 29.03 GB | 904,558 tok | 68.9 | 68.97 | 58.0 | 0.697 |
| tp2+kv-fp8+graph | 14.53 GB | 2,342,680 tok | 82.8 | 13.64 | 271.6 | 0.751 |
| tp2+kv-fp8+eager | 14.53 GB | 2,408,216 tok | 83.1 | 85.52 | 46.8 | 0.751 |
| tp1+dp2+graph | 29.03 GB/卡 | 435,879 tok/卡 | — | — | 559.2 | 0.872 |

读法：

- **CUDA graph 是这个尺寸的主要杠杆**：TPOT 13.16 vs 63.35 ms（4.8×）。48 层 MoE 每 token 的 kernel 数在 eager 下 launch-bound，graph 把整步折叠成一次 replay；TP2 同理（12.76 vs 78.50 ms）。
- **TP2 买容量不买速度**：290 vs 286 TPS 持平，KV 容量 ×2.7（1.17M token）——MoE 权重按专家维切分后每卡读取量减半，抵消了集合通信开销。
- **KV fp8 买容量只付 ~7%**：容量翻倍（436K→872K，TP2 下到 2.34M token），TPOT 13.16→14.05 ms。
- **DP2 近线性扩展**：559 TPS ≈ 2×285.9×0.98，每 replica 独立持有 graph。
- **精度**：TP1 下 graph 与 eager 数值等价（golden 1.000，26/26 exact）。TP2 的 0.638 与 KV fp8 的 0.697 都是 greedy 混沌对首个分叉 token 的放大（prefix 计首差前的长度）：TP2 的分叉来自 all-reduce 顺序，KV fp8 的来自 KV 舍入——量级与 0.6B 档 [quant_matrix_20260901.md](benchmark_logs/quantization/quant_matrix_20260901.md) §4 的误差谱一致。
- 同 checkpoint 在 A10×2（22 GiB、TP2 eager 旧口径）TPOT 82.77 ms / TPS 48.3——H100 单卡 graph 是它的 5.9×。

> Model Mem 为全 replica 权重总量（rank 0 分片 × TP）；KV Capacity 为每卡容量（KV 按 TP 切分后同一数字即 replica 的 token 容量）。
> 复现：`python benchmarks/engine/run.py quant --model-dir <Qwen3-30B-A3B-Instruct-2507-FP8> --schemes fp16 --kv-cache-dtype auto fp8 --tp 1 2 --cuda-graph --no-cuda-graph --skip-hf --json docs/benchmark_logs/quantization/quant_Qwen3-30B-A3B-FP8_20260901.json`；DP 行另跑 `--tp 1 --dp 2 --cuda-graph`；golden 基线：`python -m scripts.verify.golden_tokens --save tests/golden/data/Qwen3-30B-A3B-Instruct-2507-FP8.json --model-dir <...>`。
> e2e 指标见 [`benchmark_models.md`](benchmark_models.md)；量化 kernel 精度回归：`python -m pytest tests/kernels/test_fused_moe.py -k fp8`（fp16 与 bf16 激活各一例，对 fp32 反量化参考）

### 未覆盖的 FP8 checkpoint

- **Qwen3.8-27B-FP8**（`model_type: qwen3_5`，`Qwen3_5ForConditionalGeneration`）：64 层中 48 层是 linear attention（gated-delta-net：conv kernel 4、16 key heads × 128 dim、48 value heads × 128 dim）、每 4 层插一层 full attention，另带 vision tower。`rapid_llm/models/` 支持到 qwen3_moe / qwen3_vl，尚无 qwen3_5——需要 linear attention 的 chunked-scan 内核与混合层调度，属新模型实现而非量化路径问题（其 fp8 格式与 30B-A3B 完全一致：e4m3 + 128×128 block scales + dynamic activation）。
- **Qwen3-VL-235B-A22B-Instruct-FP8**：本地副本不完整——index 要求 24 个 shard 仅存在 3 个（22/23/24），且无 config.json（27 GB ≄ ~235 GB），物理上无法加载。

### 性能说明

- **lite fp16 vs HF**：6.5× 提速来自 CUDA graphs + 融合 kernel + 分页 KV
- **int8 per-channel（W8A16）**：吞吐与 fp16 持平，节省 ~0.4 GB 权重显存
- **int8-blockwise（W8A16）**：group-wise scale 粒度更细；scale 读取更多，因此略慢
- **smoothquant（W8A8 int8）**：最快的方案——两个操作数都是 int8，吃满 int8 tensor core（6.9×）
- **fp8 W8A8**：A10（sm86 无原生 fp8 GEMM）上 per-token 激活量化开销大；在 H100/sm90 上表现更好
- **INT4 MoE（AWQ/GPTQ）**：fused_moe kernel 支持带 group-wise scales+zeros 的 int4 权重，byte 布局（2 nibble/uint8）+ 寄存器 nibble 分离的双 dot kernel；见[下文 INT4 byte 布局与双 dot kernel](#int4-byte-布局与双-dot-kernel)
- **KV cache fp8**：未反映在上表中（与权重量化正交）；把 KV cache 占用减半，可支持约 2× 更长的序列

| Scheme | Token Match vs HF fp16 | Expected |
| -------- | ---------------------- | ---------- |
| lite fp16 | ~25% | Normal — different attention kernel numerics cause divergence after first mismatch |
| int8 per-channel | ~23% | Within fp16 divergence range |
| fp8 W8A8 | ~5% | e4m3's 3 mantissa bits cause earlier divergence |

NVIDIA ModelOpt / TensorRT-LLM 布局，实现于 `rapid_llm/kernels/ops/quantization/nvfp4.py`，以 `native/linear_nvfp4` 派发：

- 权重为 fp4-e2m1，每个 `uint8` 字节存两个值（低 nibble = 偶数下标）；
- 每 16 个连续 `k` 元素一个 fp8-e4m3 block scale，因此 `BLOCK_K` 取 16 的倍数即可让每个 k-tile 都落在同一个 scale 内；
- 每个 tensor 一个 fp32 `weight_global_scale`；
- `w = e2m1(nibble) * dequant_e4m3(block_scale) * global_scale`。

TP 切分要求每个分片都是 32 的倍数（2 值/字节 × 16 元素 block），由 `NVFP4Config.shard_is_aligned` 强制执行，因此任何分片都不会切开一个字节或一个 block scale。MoE 专家**未实现**：`get_quant_method` 遇到 fused-MoE 层会直接报错，而不是悄悄回退到 bf16 专家。

### 成本与收益

sm90（H100）没有 fp4 张量核 MMA，所以它在构造上就是 weight-only：nibble 在寄存器里解包，`tl.dot` 仍以 bf16 运行。Triton 确有 `tl.dot_scaled`（microscaling API，操作数可为 uint8 打包的 fp4），但在没有原生 microscaling 硬件的架构（含 sm90）上它走软件模拟——先把 fp4 upcast 成 bf16 再 dot，与本内核的解包路径等价，拿不到 fp4 MMA 收益；原生 fp4 MMA 要到 Blackwell（sm100）。省下的是字节数，而在 H100 上字节并不是 decode 的瓶颈。

`qwen3-4b/qkv`（N=6144，K=2560），测于 NVIDIA H100 80GB HBM3（torch 2.13.0+cu130 / triton 3.7.1 / python 3.14.7），数据来自 [`quant_gemm_h100_20260903d.json`](benchmark_logs/kernels/quant_gemm_h100_20260903d.json)（以 `RAPID_LLM_AUTOTUNE=0` 运行，即用户没有调优缓存时拿到的启发式 tile）：

| M | bf16 | fp8 W8A8 | int4 (awq) | nvfp4 |
|---|---|---|---|---|
| 1 | 21.6 µs | 22.0 µs | 29.8 µs | **49.1 µs** |
| 128 | 21.5 µs | 27.3 µs | 58.2 µs | **67.9 µs** |
| 2048 | 90.8 µs | 93.5 µs | 512.2 µs | **728.4 µs** |

同一 H100 环境下 Qwen3-4B-Thinking-2507 的端到端结果（batch 4，64 个新 token，greedy）：权重 2.63 GB，对 bf16 的 7.49 GB；TPOT 13.66 ms，对 bf16 的 4.77 ms。

**请把它读作显存结果，而不是速度结果。** NVFP4 一行搬运的权重字节少 ~3.6×，decode 仍慢 2.3×、prefill 慢 8.0×——解包 nibble、用位运算展开 e2m1、再乘两个 scale 的 ALU 开销，超过了 H100 的 HBM3 省下的时间。NVFP4 适合 checkpoint 别处放不下的场合；放得下时，它就是错误的选择。

int4/AWQ 是一个有启发性的对照，但故事不同：它读的权重字节只有四分之一，decode 却落后 bf16 1.4×（29.8 vs 21.6 µs）——它以峰值 HBM 的 ~8% 读取 7.9 MB，是解包受限，而 bf16 以 43% 流式读取 31 MB 是带宽受限，两个不同极限没有相遇；到 M=2048 差距拉开到 5.6×。历史注记：0901 轮曾量得 m=1 22.2 µs（距 cuBLAS 2% 以内），但 w4a16 的 launcher docstring 记录了同一 launch 配置 run-to-run 在 22–30 µs 间波动，且 0902 起的五个独立轮次全部稳定在 29.6–29.9 µs——那次读数落在波动带下缘。`--tune` 找到的启发式修复（`GROUP_M` 分档）仍然是真实的，见[下文 w4a16 的 tile 启发式缺陷](#w4a16-中的第二个-tile-启发式缺陷)。

精度是另一项成本：对 bf16 基线的 greedy prefix 一致率在 Qwen3-4B 上为 0.233，fp8 为 0.617，int8 为 0.822。fp4 的三个尾数状态里有两个是非规格化数（subnormal），而 16 元素的 block 是一个相当粗糙的共享指数单位。

## FP8 W8A8 融合 MoE

`W8A8Fp8MoEMethod` 与 `W8A8Int8MoEMethod` 在量化专家权重的同时也量化**激活**（入口 `fused_moe_w8a8_fp8` / `fused_moe_w8a8_int8`）：GEMM1 之前 per-token，silu 输出在 GEMM2 之前 per-row（低于 32 行时两者都融进 GEMM kernel 内部，见 `_INLINE_A_QUANT_MAX_ROWS`），全程不做 host 同步，因此 MoE 层仍可被 graph 捕获。在此之前，`W8A8Fp8MoEMethod.apply` 与 `Fp8MoEMethod.apply` 是同一个函数，`W8A8Int8MoEMethod.apply` 调的是 weight-only 的 `fused_moe`——激活始终是 bf16，W8A8 只是个标签，不是一条路径。

Qwen3-30B-A3B 几何（E=128，top_k=8，hidden 2048，moe_intermediate 768），测于 NVIDIA H100 80GB HBM3（torch 2.13.0+cu130 / triton 3.7.1 / python 3.14.7），数据来自 [`fused_moe_h100_20260902_int4byte.json`](benchmark_logs/kernels/fused_moe_h100_20260902_int4byte.json)（以 `RAPID_LLM_AUTOTUNE=0` 运行，即用户没有调优缓存时拿到的启发式 tile）；int4 列是 byte 布局 + 双 dot kernel（见[下文](#int4-byte-布局与双-dot-kernel)），其余列与 [`fused_moe_h100_20260902_fp8cvt.json`](benchmark_logs/kernels/fused_moe_h100_20260902_fp8cvt.json) 一致（t1 档存在 ~8% 的整机漂移，launch-bound 档的格式间差异无意义）。基线与激活 dtype 为 bf16——即该 checkpoint 实际服务的精度（`torch_dtype: bfloat16`）；fp8 W8A16 的 e4m3 加宽在 sm89+ 上走单条硬件 `cvt`（kernel 开关 `FP8_CVT`，修正因子 256 随之消失）。前一天的 fp16 基线测量保留在 [`fused_moe_h100_20260901.json`](benchmark_logs/kernels/fused_moe_h100_20260901.json)，同日早间的 [`fused_moe_h100_20260902.json`](benchmark_logs/kernels/fused_moe_h100_20260902.json) 是修复中途的快照（其 t1 行与自身消融行矛盾，勿引用）：

| tokens | bf16 | fp8 W8A16 | fp8 W8A8 | **int8 W8A8** | int8 W8A16 | int4 |
|---|---|---|---|---|---|---|
| 1 | 108.2 µs | 113.1 µs | 120.7 µs | 117.5 µs | 111.8 µs | 113.2 µs |
| 8 | 186.1 µs | 118.7 µs | 123.8 µs | 132.2 µs | **114.9 µs** | 130.0 µs |
| 64 | 415.9 µs | 240.0 µs | 236.3 µs | **234.4 µs** | 234.1 µs | 242.2 µs |
| 512 | 469.7 µs | 355.0 µs | 280.4 µs | **275.9 µs** | 313.0 µs | 403.6 µs |
| 4096 | 1062.4 µs | 1320.8 µs | 868.9 µs | **673.1 µs** | 1148.0 µs | 1701.9 µs |

decode 与 prefill 在这里是两个不同的操作，不平均成一个加速比。1 token 时所有格式挤在 bf16 的 ±12% 内（108-121 µs）：这层是五个背靠背的 kernel（align、GEMM1、silu、GEMM2、sum），launch 而不是字节决定下限。W8A8 的激活量化已不再增加 launch——低于 32 行时量化融进 GEMM 内部（`_INLINE_A_QUANT_MAX_ROWS`），silu 输出在 store 时量化（`QUANT_OUT`）——代价是每个 GEMM program 重新推导 amax，t1 残留约 10%，t8 起被字节优势吞没（int8 weight-only 快 62%，W8A8 fp8 快 50%，int4 快 43%）。64 token 起 slot 数超过行块、每个专家载入被摊销，所有量化行都胜过 bf16：int8 W8A8（234.4 µs，1.77×）是此档下限，int8 W8A16 与 W8A8 fp8 落在其 1% 内，int4（242.2 µs，1.72×）咬住 8-bit 行——byte 布局的 dense 加载拿到流量收益，寄存器 nibble 分离只比 8-bit 格式的单次加宽贵一点。512 token 起 weight-only 收益反转、W8A8 接管：int8 W8A8（275.9 µs，1.70×）是该档最快一行；4096 时 weight-only 全线倒退（fp8 慢 24%、int8 慢 8%、int4 慢 60%）——GEMM 一旦 compute-bound，逐 row-block 反量化权重 tile 就不再摊销——而 W8A8 不付这笔账：int8 W8A8 以 673.1 µs（1.58×，459 TFLOP/s）成为全表任意 shape 的最快行，比 fp8 W8A8（868.9 µs，1.22×）快 22%——int8 imma 从 `BLOCK_M=16` 就能用 tensor core（fp8 要 64 才发 wgmma），且 int32 累加精确、无 `K_PROMOTE` 式精度税。A8 的收益在 MMA 里，所以恰好出现在 weight-only 收益消失的地方。fp8 W8A16 的 24% 残值已是 `FP8_CVT` 之后的数字（换入硬件 `cvt` 值 5.3%：1404.0→1329.7 µs）；再往下是逐 row-block 重读权重 tile 的结构性成本——`BLOCK_M=256` 试图消它反而全面变慢（0.26-0.90×，shared memory 逼 `num_stages=2`、accumulator 寄存器压力压坍 occupancy，而 `GROUP_M=8` 的 L2 分组已吸收大部分重读）。

一个表里看不出的告诫：Triton 只有在 `BLOCK_M >= 64` 时才发射 Hopper 的 fp8 `wgmma`，而 `_launch_config` 的 fp8 W8A8 分档要到 4096 token 才到这个行块（512 的 tier-1 tile 是 `BLOCK_M=32`）。低于它的两个 e4m3 操作数加宽成 fp16 走 `mma.sync`，所以除 t4096 外的 fp8-A8 行都没测到 fp8 tensor core——t512 的领先来自字节与跳过的 bit-trick，不是 MMA。int8-A8 没有这个门槛（imma 从 `BLOCK_M=16` 起可用）。

复现（环境与负载见本节开头；日志即上表引用的四份 JSON）：

```bash
RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_fused_moe.py \
    --json docs/benchmark_logs/kernels/fused_moe_h100_<date>.json   # 全格式矩阵
python benchmarks/kernels/bench_fused_moe.py --tune               # tile 扫描（写 ConfigStore，不入库）
python -m pytest tests/kernels/test_fused_moe.py                  # 格式正确性门
```

### 这批数字发现的 tile 启发式缺陷

`_launch_config` 原来返回 `BLOCK_K = 128 if quant_mode else 32`。对内存事务而言这是对的（fp16 tile 32 个元素就能填满一次，字节 tile 需要 128 个），对这个层却是错的：这里的专家 GEMM 宽 768、对 2048 的 hidden size，窄 k-tile 只会把循环次数放大。用 `benchmarks/kernels/bench_fused_moe.py --tune` 对 17 配置空间做 tile 扫描（当时是 fp16 基线），发现任何 token 数下未量化行的优胜配置 `BLOCK_K` 都**不低于** 64；基准里保留的窄 tile 消融行在 bf16 上复现同样的形状——t8 时慢 26.3%、64 时 30.8%、512 时 26.8%、4096 时 12.9%。

这个缺陷只会压低*未量化基线*，这正是没有任何测试抓到它的原因，也是这个 kernel 上过去的量化对比都读起来比实际更好的原因——在当时的 fp16 基线上，512 token 时它就是「W8A16 fp8 看起来赢 18%」与实际的「输 11%」之间的差别。现在所有模式都用 128。基准测试保留窄 tile 作为消融行，让修复保持被度量：两行（基线与窄 tile）收敛就意味着它复发了。

同一次扫描持久化到 `ConfigStore` 后，15 个 store key 中 13 个相对启发式有提升（最大：fp16 在 M512 档，2502.8 → 1694.1 µs，+32.3%）。搜索按 `TuneKey` 进行而不是按 token 数，因为 `bucket_m` 会把 M 向上取整到 (16, 32, 64, 128, 256, 512) 的下一档——t1 与 t8 共享一个条目，按 token 数搜索会让它们互相覆盖。注意未量化模式现在按*激活 dtype* 键入（`bf16` 与 `fp16` 是两个条目），这批 fp16 扫描条目不会覆盖 bf16 路径——切到 bf16 基线后请重跑 `--tune`。换新设备同理；持久化的缓存不入库。

## INT4 byte 布局与双 dot kernel

fused MoE 的 int4 权重存储从 int32 8-nibble 打包换成了 vLLM 的 uint8 byte 布局（`[E, N, K//2]`：byte b 的低 nibble 覆盖 k=2b，高 nibble 覆盖 k=2b+1）。checkpoint 布局不变——GPTQ/AWQ 的 int32 词在加载后由 `repack_int4_experts`（`rapid_llm/kernels/ops/quantization/int4_repack.py`）一次性转换，调用点是 vLLM 同名的 `process_weights_after_loading` 钩子（`GptqMoEMethod`/`AwqMoEMethod` 实现，`Model.load_weights` 尾部统一触发；`MoE.quantize_` 的在线量化路径同样收口于此）。

动机是一次测量的结论：**byte 布局本身不解决问题，vLLM 的复制寻址 idiom 在 Triton 上同样无法向量化**。他们的 kernel（`fused_moe_kernel_gptq_awq`）让逻辑 k 读 byte k//2——每个 byte 出现在它两个 nibble 的行里——非仿射索引使 Triton 的合并分析失效，编译成 128 条标量 `ld.global.b8`。逐字提取该 kernel 到本机同几何实测 13-18 ms/GEMM；vLLM 的生产 int4 走的是 Marlin CUDA kernel，Triton 版只是 fallback。

rapid_llm 的 kernel 走另一条路：B 按 `[BLOCK_K//2, BLOCK_N]` **仿射 dense** 加载（向量化、软件流水线保持 `cp.async`），两个 nibble 平面在寄存器分离（`(b & 0xF)` 与 `(b >> 4) & 0xF`，直转 compute_type——[-15,15] 的小整数在 bf16 精确），A 侧以 `tl.split(tl.reshape(a, (BLOCK_M, BLOCK_K//2, 2)))` 拆出偶/奇 k 列，两个半 K dot 之和等价于原全 K dot。`EVEN_K`（K 整除 BLOCK_K 时）免掉 masked load——逐元素谓词同样会把加载拆成标量字节；Qwen3-30B-A3B 的两个 GEMM（K=2048/768）在 BLOCK_K=128 下都满足。

t4096（最难的档）的演进：int32 格式 1.92 ms → byte 布局 + vLLM 复制寻址 **7.35 ms**（倒退 3.8×，即上面那个 idiom）→ dense 加载 + 双 dot 3.31 ms → `EVEN_K` + nibble 直转 **1.70 ms**。对照 int8 同档 1.15 ms、bf16 1.06 ms：0.54× → 0.62×，中间档（t8/t64/t512）从 1.02-1.11× 提到 1.16-1.72×，t4096 绝对值也首次低于 int32 格式（1916.4 µs）。演进各步的测量环境为 NVIDIA H100 80GB HBM3（torch 2.13.0+cu130 / triton 3.7.1），Qwen3-30B-A3B 几何（E=128，top_k=8，hidden 2048，i=768），token 档 1～4096；终态数字落在 [`fused_moe_h100_20260902_int4byte.json`](benchmark_logs/kernels/fused_moe_h100_20260902_int4byte.json) 的 int4 列（中间步骤的数字来自当轮会话测量，未单独成档）。复现：`RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_fused_moe.py --json out.json`，int4 正确性门 `python -m pytest tests/kernels/test_fused_moe.py -k int4`。

tile 重扫（12 候选 × 5 token 档）确认现有表仍最优（tier 0 16×128、其余 64×128）：BLOCK_K=256（每 k 迭代 4 个半 K dot，寄存器压力）与 BLOCK_N=256（两个 (BLOCK_K, BLOCK_N) compute_type 平面驻留寄存器）都慢 1.6-2×。t4096 残留的 0.62× 是结构性成本：每 row-block 重读权重 tile 时寄存器 nibble 分离的 ALU 随重读次数线性放大，比 8-bit 格式的单次加宽贵——Triton 上 int4 weight-only 的通病，vLLM 的解法是换 Marlin CUDA kernel，不是换寻址。

## w4a16 中的第二个 tile 启发式缺陷

dense GEMM 存在同类问题，而且其中只有一个能通过缓存修复。五个量化 kernel 里，**`w4a16_matmul` 是唯一会查 `ConfigStore` 的**——fp8 W8A8、fp8/int8 W8A16 与 NVFP4 都无条件计算 launch 配置，因此 `bench_quant_gemm.py --tune` 对它们如实报告「无消费者」，而不是写入没有任何 kernel 会读的条目。（`v0.5` 的 changelog 声称 autotune 覆盖「量化 GEMM」；对 dense 路径而言，那只是五个 kernel 中的一个。其余四个 kernel 的 fallback 在 0903 按 H100 扫描重定过，见[下文](#dense-量化-gemm-的第三次-tile-重扫h100)。）

就在这一个 kernel 上，扫出来的不是按 shape 的调优机会，而是一个启发式缺陷。`m <= 32` 分支原来用 `GROUP_M=1, num_stages=2`，而 `GROUP_M=8, num_stages=4`——*同一个* 16×64 tile——在**全部 16 个** `m <= 32` store key（两个 Qwen3 几何 × 四个投影 × M16/M32 两档）上赢 9.0–41.5%，且 tile 固定不变，这两个旋钮是仅有的变量。`GROUP_M=1` 什么也不分组，于是相邻 program 按行主序遍历网格，在 L2 里共享不到任何权重 tile；更深的流水线随后盖住了 16 行 tile 盖不住的 nibble 解包。因为收益是一致的，修复应该放进 kernel 的 fallback，而不是按 shape 键入的缓存——现在它随每一个没跑过调优的设备一起生效，也正是它把上面的 M=1 int4 行从 34.0 µs 移到 22.2 µs。

修复之后，按 shape 的调优仍有大量空间：32 个 key 中 29 个相对修正后的启发式有提升，幅度 9.7–46.0%。只有三个报告「启发式已是最优」——`qwen3-4b/qkv`、`qwen3-4b/gate_up` 与 `qwen3-30b-a3b/qkv` 的 M16 key，那里 16×64 已经是对的。其他地方的赢家在 decode 时*更窄*（M16/M32 上为 16×32 或 64×32），prefill 时宽得多（M512 上从 128×64 到 128×256）。这种散布是三分支 fallback 覆盖不了的，也正是这一个 dense kernel 值得留缓存的原因。

一个结构性告诫：共享的 bucket 条目是按桶内 token 数的*总量*选出的，所以桶内某个宽度可能回退，而条目整体仍是净赢。抽查 `qwen3-30b-a3b/qkv` 与 `qwen3-4b/qkv` 的 M512 条目，两个宽度在两个 key 上都有提升（t512 +0.7% / +12.2%，t2048 +25.5% / +24.3%），因此这里没有观察到回退——但只做 decode 的部署仍应把 `--tokens` 收窄到它实际服务的宽度，而不是继承一个 prefill 加权的条目。

这个修复在暴露它的那个配置上值 1.32× 端到端：Qwen3-4B int4 + decode graph 在 tp1 从 419.0 → 551.3 TPS（TPOT 9.28 → 6.97 ms），tp2 471.9 → 583.1，dp2 816.8 → 1057.6；greedy 匹配率如 tile 变更所应有的那样保持 0.157 不变。eager 行只动了 2-4%，这是有用的对照：没有 graph 时 launch 开销主导 decode，更好的 tile 无从显现。重测的行见 [`quant_matrix_20260901.md`](benchmark_logs/quantization/quant_matrix_20260901.md) §2。

## dense 量化 GEMM 的第三次 tile 重扫（H100）

w4a16 的缺陷修完后，剩下三个可改 fallback 的 kernel——`fp8_matmul`（fp8 W8A8）、`_smoothquant_matmul`（int8 W8A8）、`_w8a16_matmul`（fp8/int8 W8A16）——的 tile 表仍然是 A10 时代的数据（`w8a16` 的 docstring 原话就写着 "measured on an A10"）。对 H100 这张卡它们错在同一个地方：**`BLOCK_N` 128/256 让 grid 饥饿**。N=6144 的 qkv 投影用 256 宽的 N tile 只有 24 个列块，对 132 个 SM——即使 M 方向有块也填不满芯片。窄 tile 的代价（更多 k 循环迭代、每块更少的计算）在 decode 档几乎不存在，因为那时尚未 compute-bound。

扫描覆盖三个 kernel × 两个 Qwen3 几何的五个代表性投影（4b qkv/o/down/gate_up + 30b o/gate_up，N 从 1536 到 19456）× 五个 M 档（1/8/64/512/2048），tile 空间 BLOCK_M 按档 16–256、BLOCK_N 64/128/256、BLOCK_K 128、warps 4/8、stages 3/4/5、GROUP_M 1/8——fp8/int8 各约 3000 个候选、w8a16 1360 个（脚本直调 kernel，绕开 launcher 的 `_launch_config`）。测量环境：NVIDIA H100 80GB HBM3（torch 2.13.0+cu130 / triton 3.7.1），脚本为会话内的临时 sweep 脚本（直调 kernel 逐候选 `do_bench`，未入库）；**全量扫描数据当时写在会话 scratch（`/tmp/sweep_dense.json`、`/tmp/sweep_w16.json`），未归档、已不可复得**——可复得的是结论：每个 kernel 的每档 fallback 取「档内全部候选对全部 shape 无回归、几何均值最优」的配置，已编码进三个 `_launch_config` 的 docstring，终态效果由下方 `bench_quant_gemm.py` 全路径 A/B 验证（复现：`RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py --json out.json`）。

新 fallback 的 kernel 级收益（扫描内新旧行块对比的几何均值）：fp8 W8A8 解码 1.14×/中档 1.07×/prefill 1.21×；int8 W8A8 解码 1.06×/中档 1.03×/prefill 1.25×；w8a16 fp8 解码 1.06×/中档 1.11×/prefill 1.20×。w8a16 的 decode 档 winner 分散在 8 个不同配置上（9 个测试点），说明这一档真实需要按 shape 调优，fallback 只能取无回归面最大的那个；`BLOCK_N=64` 在它的 N=19456 gate_up 上倒退 38%，所以 w8a16 的 decode 档保留 128。

随本次重扫一起落地的还有两个 kernel 层优化：

- **`FP8_CVT` 移植到 dense w8a16**：sm89+ 上 e4m3 加宽改用单条硬件 `cvt`（fused MoE kernel 已有的开关，见上节），bit-trick 的五条整数指令与 256× 修正因子消失；旧设备走原路。
- **`SINGLE_SCALE` 提出循环**：权重 scale 为 per-channel/per-row（`group_k >= k`，fp8 W8A8 与 Qwen block-scale 路径的默认）时，scale 的 k 地址在循环内不变，把它提出 k 循环后循环体只剩权重 tile 加载与 `tl.dot`——原本每个 k 步都驮着一次 `[BLOCK_N]` scale 加载及其地址算术。w8a16 的 HAS_ZEROS（GPTQ bits=8）分支同样受益。

三 kernel 的 fallback 改动 + 两个 kernel 优化一起，在 `bench_quant_gemm.py` 全路径（含激活量化 pass）上量得 ([`quant_gemm_h100_20260903.json`](benchmark_logs/kernels/quant_gemm_h100_20260903.json)，对 [`..._20260902.json`](benchmark_logs/kernels/quant_gemm_h100_20260902.json)；同表 bf16/awq/nvfp4 行 1.00× 持平，证明测量无系统性漂移)：

| scheme | m≤8 geo | m=32–128 geo | m≥512 geo | 备注 |
|---|---|---|---|---|
| int8 W8A8 | 0.86× | 0.92× | 0.81× | vs bf16 胜场 5→**12**/48 |
| fp8 W8A8 | 0.92× | 0.94× | 0.77× | prefill 从 2.0–2.6× 落后缩到 1.4–2.0× |
| fp8 W8A16 | 0.96× | 1.00× | 0.68× | prefill 从 3.8–5.1× 落后缩到 2.1–3.2× |

（表内数字是 new/old 的时长几何均值，<1 为更快。）新胜场全部来自 int8 W8A8：4b/gate_up 全六档（1.13–1.46×）、4b/qkv 的 m1/m8/m32/m2048（1.02–1.25×）、30b qkv 与 o 的 m2048（1.06×/1.03×）。没有一个此前胜过 bf16 的测试点回退。mid 档的取舍如实记录：w8a16 用 30b o 投影 m64 的 -14% 换其余四投影 +2–54%（一层总时间为正）；fp8 W8A8 用 4b/qkv m64 的 -4% 换其余 +11–21%。

nvfp4 与 awq 未动、仍全败（nvfp4 的 Triton 解包在 H100 上结构性劣势，见上文）；int4 的 autotune 消费者地位不变，本轮未重扫。

## in-loop scale 的消除：epilogue 化与 wgmma 流水线（H100 第四轮）

第三次重扫后 fp8 W8A8 的 prefill 仍落后 1.4–2.0×，这轮把原因追到了 kernel 的循环体里。PTX 检查确认 M64 档确实发射了 `wgmma.f32.e4m3.e4m3`（不是被 fallback 到 mma.sync），但吞吐只到 24% 峰值；排除 `tl.trans`（K-major 加载无差别）后，剩余嫌疑只剩循环体内这一行：

```python
accumulator += tl.dot(a, tl.trans(b)) * b_scale[None, :]
```

每个 k 步把 wgmma 产出的 `[BLOCK_M, BLOCK_N]` fp32 tile 乘一次 scale 再加回累加器——这个同步的向量乘法插在异步 wgmma 链中间，把流水线串行化了。int8 W8A8（`_smoothquant_matmul`）的 scale 全在 epilogue，同形状下到 52% 峰值，是现成的对照。

**修复**：per-row scale（`group_k >= k`，即 `SINGLE_SCALE`，也是 `w8a8_fp8`/`blockwise_int8` 运行时量化的默认布局）在数学上分配律成立——先累加再整体乘一次与每块乘再累加等价，仅 fp32 舍入顺序不同（全部被现有测试容差覆盖，173 个 kernel 级测试全绿）：

```python
accumulator = tl.dot(a, tl.trans(b), acc=accumulator)   # 纯累加，wgmma 链端到端异步
# epilogue:
result *= b_scale[None, :]
```

同一改动进 `_w8a16_matmul`。两个配套发现也来自实测而非推断：

- **fallback 必须按 `single_scale` 分叉**。128×128 tile 只有在 epilogue 路径才成立：block-scale（`SINGLE_SCALE=False`，如 Qwen FP8 checkpoint 的 128×128 布局）仍走 in-loop 路径，那个路径下 128 行累加器直接寄存器溢出——中间版曾把 M128 fallback 无条件交给两个路径，bench 里 w8a16 fp8 行（block-scale）因此回退 2.1–2.5×。现在 `_launch_config(num_tokens, single_scale)` 在 prefill 档返回 `BLOCK_M = 128 if single_scale else 64`，fp8 同样分叉。
- **`acc=` 形式不是全档免费**。w8a16 的 int8 + SINGLE_SCALE 路径（e2e 的 `--quantization int8`，per-channel）在 decode 档实测回退 3–11%：`tl.dot(a, b, acc=)` 在小 M 档编译出的调度比「dot 后乘再加」更差，而 prefill 档提升 6–12%。所以 w8a16 加了 `EPILOGUE_SCALE` 开关，仅 `m > 128` 且 single_scale 时走 epilogue 形式——decode 回退归零，prefill 提升保留。fp8 W8A8 无此现象（decode 档同为改善），不需要开关。

全路径 bench（含激活量化 pass，[`quant_gemm_h100_20260903d.json`](benchmark_logs/kernels/quant_gemm_h100_20260903d.json) 对 in-loop 基线 [`..._20260903.json`](benchmark_logs/kernels/quant_gemm_h100_20260903.json)；bf16/awq/nvfp4/int8-smoothquant 控制行全 1.00× 持平）:

| scheme | m≤8 geo | m=32–128 geo | m=512 geo | m=2048 geo | 备注 |
|---|---|---|---|---|---|
| fp8 W8A8 | 0.990 | 0.968 | 0.897 | **0.738** | 单个测试点最大 1.45×（4b/qkv、30b/o m2048） |
| fp8 W8A16（block-scale 行） | 1.000 | 1.000 | 1.005 | 1.004 | in-loop 路径未变，控制行 |
| w8a16 int8 per-channel（SINGLE_SCALE） | — | — | — | 1.06–1.13× | kernel 级 A/B 脚本实测，bench 未覆盖此布局 |

（表内数字是 new/old 的时长几何均值，<1 为更快。）至此 48 个测试点上共 20 个 scheme 胜场（int8 13、fp8 W8A8 5、fp8 W8A16 2），落在 13 个测试点里——gate_up 全六档、qkv 五档、30b qkv/o 的 m2048 各一；bf16 在其余 35 个测试点上仍是最快行。胜场的带宽门槛分层：int8 W8A8 从 ~44% 的 bf16 带宽占比起赢（qkv m1 1.07×；38.5% 处 0.96× 差一点），fp8 两格式只在 60.9%（gate_up）赢——int8 的 scale 全在 epilogue 且 imma 无 `BLOCK_M >= 64` 门槛，fp8 还驮着 Triton wgmma codegen 与激活量化 pass。

复现（环境：H100 80GB，torch 2.13.0+cu130 / triton 3.7.1；负载：两个 Qwen3 几何的五个真实投影 × 6 个 token 档，`RAPID_LLM_AUTOTUNE=0`）：

```bash
RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py \
    --json docs/benchmark_logs/kernels/quant_gemm_h100_<date>.json
python -m pytest tests/kernels/test_quantization.py   # epilogue 化的数值等价门（173 个用例）
```

### e2e 验证（第四轮修复的端到端效应）

kernel 级的收益要在 e2e 上兑现。第四轮代码对 modelzoo 全部可跑权重复跑完整矩阵（每模型 × b1/s1024、b32/s512、b8/s2048、b8/s8192 四个 batch/seq 组合，TPOT 比值 new/old，<1 为更快；数据 [`e2e_matrix_20260903c/`](benchmark_logs/quantization/e2e_matrix_20260903c/) 对 in-loop 基线 [`e2e_matrix_20260903/`](benchmark_logs/quantization/e2e_matrix_20260903/)）：

| 模型 | 走本轮改动路径的行 | 走未动路径的行（控制） | 判读 |
|---|---|---|---|
| Qwen2.5-0.5B（dense） | fp8 0.978–0.998、int8 0.999–1.027 | 0.992–1.003 | 持平：1.3 ms 步长 launch 主导，kernel 改善被淹没 |
| Qwen3-4B（dense） | **fp8 0.963–0.999**、int8 0.999–1.009 | 0.998–1.014 | fp8 b32 +3.7%、b1 +2.3%；int8 无回退（EPILOGUE_SCALE 分档的 e2e 验证） |
| Qwen3-30B-A3B-FP8（MoE checkpoint） | checkpoint 原生 W8A16（in-loop 路径）0.996–1.002 | bf16/kvfp8 0.948–1.002 | 持平：in-loop 路径本轮未动，此行即控制行 |
| Qwen3-30B-A3B bf16（MoE runtime 量化） | fp8 0.992–1.001、int8 0.995–0.999 | 0.997–1.000 | 持平：MoE expert GEMM 主导 TPOT，dense 投影占比小 |

三个细节值得单独记录：4B fp8 的 golden prefix 从 0.617 升到 0.686——纯累加与逐步乘加的 fp32 舍入顺序不同，greedy 混沌下方向不定（0.5B fp8 同时从 0.346 降到 0.307），不构成精度承诺；30B-FP8 b8/s8192 的 kvfp8 行 +5.1% 是全矩阵唯一超过 ±2.5% 的控制行，同组合 bf16 行 0.999 排除测量漂移，属 KV 分页池布局的运行间差异；0.5B int8 b8/s2048 的 1.027 与同模型 bf16 控制行的 0.995 同宽，在 launch-bound 噪声带内。

## launch 配置的设备与 dtype 维度（H100 第五轮）

前四轮的 fallback 只按行数选 tile，三个维度从未检验过：设备（表是 H100 实测，A10 用户吃到的是 H100 的窄 tile）、dtype（w8a16 的表只用 fp8 权重扫过，int8 权重一直吃 fp8 的表——launcher docstring 当时的原话是 "int8 weights share the kernel's load/dot structure so the geometry carries over"，这是个假设）、shape（N 宽度从未进入选表逻辑）。三个维度逐一用测量裁决：

- **dtype：假设被推翻，int8 拿到自己的表。** 用与 launcher 相同的 EPILOGUE_SCALE 分档补扫 int8 per-channel（1360 个候选，五个投影 × 五档）：fp8 的表让 int8 在 decode 档慢 7–57%（geo 1.30）——fp8 表 decode 档为 N=19456 保留的 `BLOCK_N=128` 对 int8 是错误宽度，int8 全档想要 64，m=2048 还想要 `BLOCK_M=256`（128 行 tile 在该档慢至 31%）。int8 专属表（四档：16×64w8s5 / 32×64w4s3 / 128×64w8s3 / 256×64w8s3）在 launcher 级 A/B 上 25 个测试点中 23 个胜出 1.07–1.46×（几何均值 1.18×）；仅 m512 档 down/gate_up 两个测试点回退 6–8%（N64 vs N128 的档内取舍，其余三投影同档 +16–41%）。条件是 `single_scale and not is_fp8`，即 e2e `--quantization int8` 的真实路径；int8 block-scale（GPTQ bits=8）与 fp8 共表——它共享 in-loop 路径且 zero-point 加载无专项扫描，docstring 如实注明。
- **设备：sm90 门槛，A10 恢复实测旧表。** 三个 8-bit kernel 的 launcher 加 `sm_version` 门槛（缓存查询，`has_native_fp8` 同源）：sm90+ 用 H100 表，pre-Hopper 回到被第三次重扫替换掉的 A10 表（sm86 实测，三个 kernel 当年同一张）。`EPILOGUE_SCALE` 同步 gate 到 sm90——A10 保持其 tile 表被测量时的 in-loop kernel 形态，而非未经实测的 epilogue 组合。
- **shape：检验后否定。** 五个投影（N=1536–19456）按窄（≤2560）/宽（≥6144）分组重析：三个 8-bit kernel 每档两组的 geomean-best 都是同一配置（fp8/w8a8）或差在噪声内（w8a16）——`n` 不是选表的有效输入，这个否定结论写进了 launcher docstring，避免后人重走一遍。

控制行验证（fp8 W8A8、w8a16 fp8 block-scale、smoothquant——本轮未动路径）见 [`quant_gemm_h100_20260903e.json`](benchmark_logs/kernels/quant_gemm_h100_20260903e.json) 对 [`..._20260903d.json`](benchmark_logs/kernels/quant_gemm_h100_20260903d.json)。测量环境：NVIDIA H100 80GB HBM3（torch 2.13.0+cu130 / triton 3.7.1）；负载：int8 per-channel 补扫 1360 候选（五投影 × 五档，与 launcher 同 EPILOGUE_SCALE 分档），launcher 级 A/B 25 个测试点。复现：kernel 级 `RAPID_LLM_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py --json out.json`；e2e 级 `python benchmarks/engine/run.py quant --model-dir <ckpt> --schemes int8 --cuda-graph --no-cuda-graph`（int8 per-channel 即 `--quantization int8` 路径）。

### 量化为什么常常比 bf16 慢：roofline 判断

epilogue 化把 fp8 W8A8 的 prefill 从 2.0–2.6× 落后拉到 1.05–1.5×。剩下的缺口里，哪些还能追、哪些是结构性追不动的，靠一个 roofline 框架来分：量化能否赢 bf16，不取决于权重压缩了多少倍，取决于这次 GEMM 此刻卡在带宽还是算力上。判据是 bf16 基线自己的瓶颈，不是压缩比——本节早先一版拿压缩比预测，得出过方向完全相反的结论（nvfp4 字节少 3.6× 所以 decode 最快），实际它在 48 个内核测试点里全部垫底。

**decode（M 小）卡在带宽。** 每个权重元素只参与少量乘加，算术强度低，时间花在从 HBM 搬权重上。量化把权重字节砍半（int8/fp8）或砍到 1/4（int4），直接缩短搬运时间——这是量化唯一稳定兑现收益的区间。赢多少看 bf16 行的带宽占比：gate_up 占 60.9% 时赢得明显，down 只占 10.4% 时内核根本没在等显存，删字节省不下时间，反量化反倒成了纯增量成本。

**prefill（M 大）卡在算力。** 算术强度高，时间花在 tensor-core 乘加上。此时省字节没用，能不能赢只看量化后的 MMA 吞吐是否真的高过 bf16。三类算子在这里命运不同：

- **真 W8A8（fp8-e4m3 / int8，激活也量化）** 是唯一在 prefill 算术上*可能*赢的一类：两个操作数都进 tensor core，fp8/int8 的 MMA 峰值是 bf16 的 2×（H100：fp8/int8 ~1979 TFLOPS 对 bf16 ~989）。兑现程度分两档——int8 W8A8 兑现了，imma 从 `BLOCK_M=16` 就能用 tensor core、int32 累加精确、scale 全在 epilogue，prefill 到 1.58×（MoE t4096，459 TFLOP/s）；fp8 W8A8 只兑现一半，残余落后 1.05–1.5×。`ablation: fp8_matmul only` 行把这块缺口分成两份：Triton 的 fp8 wgmma 代码生成本身在 0.7–1.05× cuBLAS bf16 之间波动（shape 依赖），激活量化 pass 再加 5–17%。前者是编译器层的边界——cuBLAS 的 fp8 kernel 是 CUTLASS 级手写调优的，Triton 生成的 codegen 够不着，要兑现理论 2× 得换 deep_gemm 一类专用后端（§5：行已注册、未安装）；后者是 W8A8 必须先量化激活的额外 memory-bound pass。两份都不是 kernel 写法能治的。
- **W8A16 / W4A16（weight-only，激活仍 fp16/bf16）** 在 prefill 结构性无胜机。GEMM 前要把量化权重反量化成 fp16 再走 `tl.dot`，计算速率就是 fp16 rate、和 bf16 同档，不是 int8/int4 rate——算力上限没抬高，反而多付 unpack/widen 的 ALU。它们只在 decode（省带宽）赢；prefill 要赢得换 Marlin 类把反量化和 MMA 完全融合的 kernel，本仓库的 Triton 实现不是这条路线。

两类 e2e 现象也落在同一个框架里：

- **小模型（0.5B）全方案落后 bf16**：decode 步长 ~1.3ms，GEMM 只占小头，kernel launch 与量化 pass 的固定成本主导——launch-bound，还没轮到带宽就先卡在 launch 上。这是模型尺寸的结构性结果，不是 kernel 问题。
- **MoE（30B-A3B）b8 档量化全胜**（int8 +20.4%、smoothquant +17.6%、fp8 +16.0%）：expert GEMM 在 b8 时每个专家分到的 token 少，权重读取主导，落回 bandwidth-bound 区间，压缩直接兑现。

收口一句：量化赢的地方都是 bandwidth-bound（decode 的大权重占比投影、MoE b8），输的地方都是 compute-bound（prefill）或 launch-bound（小模型）。

## Tensor 并行 × CUDA Graph

decode graph 过去在 `tp_world_size > 1` 时一律拒绝。现在它们会被捕获，但要通过 `ModelRunner.enable_cuda_graph` 里的三道闸门——TP 下错误的 graph 不是抛异常，而是在集合通信里挂死：

| 闸门 | 检查内容 | 失败时 |
|---|---|---|
| `RAPID_LLM_TP_CUDA_GRAPH=0` | 总开关（kill-switch） | 转为 eager，并给出警告 |
| grid 一致性 | 对 `(len(batch_sizes), len(seq_len_buckets), hash(grid))` 做 all-reduce | **每个** rank 丢弃自己的 graph |
| 数值 | 按 `(bs, bucket)` 对比 graph 与 eager 输出，atol 1e-2 | 丢弃 graph，回退 eager |

捕获还需要 `NCCL_GRAPH_MIXING_SUPPORT=1`（在 `init_parallel` 里设置），因为 prefill 保持 eager 而 decode 走 replay，同一个通信域上混合了已捕获与未捕获的集合通信；另需 `warmup_collectives()` 一轮，确保没有通信域在捕获区域内被惰性创建。

测于 2× NVIDIA H100 80GB HBM3（torch 2.13.0+cu130 / triton 3.7.1，2026-09-01），Qwen3-4B-Thinking-2507，`fp8+tp2+graph`：77 次 replay，每 rank 权重 2.06 GB（tp1 为 4.11 GB），KV 容量 955,832 token（对 465,750）——省下的权重显存变成了缓存。这个模型的吞吐*低于* tp1（622 vs 664 tok/s）：4B 时每步 all-reduce 的代价超过第二张卡算力的收益。这里的 TP 是容量特性，不是速度特性。负载：batch 8，max_gen_len=256，greedy；日志：[`quant_Qwen3-4B-Thinking-2507_h100_20260901.json`](benchmark_logs/quantization/) 同批次的 quant 矩阵运行（quant_matrix_20260901.md §3）。复现：

```bash
python benchmarks/engine/run.py quant --model-dir <Qwen3-4B-Thinking-2507> \
    --schemes fp8 --tp 2 --cuda-graph --skip-hf --json out.json
python -m pytest tests/distributed/test_tp_cuda_graph.py   # TP×graph×quant 交叉验证门
```

### 哪个引擎能跑 TP

`LLM`（因此 `TextGenerator`）不能：它的 generate 循环只广播采样出的 token，从不广播步计划，follower rank 会永远等待。它现在对 `tensor_parallel_size > 1` 直接报错，而不是默默在单卡上跑——那是它过去的行为，也正是让某行标注 `tp2` 的基准实际测成 tp1 的原因。请改用 `ContinuousBatchingEngine.from_pretrained(...)`、`rapid-llm serve` 或 `rapid-llm batch`，它们的 executor 会广播每一步的计划。

## 精度

Qwen3-0.6B 的 token 级精度对比（A10，greedy decode，与[性能基准测试](#性能基准测试)表相同的 prompt 集与同一次 2026-08-23 运行；该次运行无独立 JSON 留存，复现命令见性能节）：

| 方案 | 对 HF fp16 的 token 匹配率 | 解读 |
|---|---|---|
| lite fp16 | ~25% | 正常——attention kernel 数值不同，首个分叉之后即发散 |
| int8 per-channel | ~23% | 在 fp16 的发散范围内 |
| fp8 W8A8 | ~5% | e4m3 只有 3 个尾数位，发散更早 |

> **logits 级精度**（在 greedy argmax 之前逐 token 测量）一致率高得多：int8 相对误差 <0.03%，fp8 <0.04%——确认量化没有损害模型质量，只是 greedy decode 放大了舍入差异。

## FP8 KV Cache：为什么 `k_scale = v_scale = 1.0`

`Fp8KVCacheMethod` 附带固定为 1 的 scale。这看起来像遗漏，但我们用测量代替争论：`scripts/debug/quant_kv_error.py` 在 Qwen3-4B-Thinking-2507（36 层；NVIDIA H100 80GB HBM3，torch 2.13.0+cu130 / python 3.14.7）上运行，完整日志见 [`kv_fp8_error_qwen3-4b_20260901.json`](benchmark_logs/quantization/kv_fp8_error_qwen3-4b_20260901.json)。

| 测量项 | 结果 | 解读 |
|---|---|---|
| 任意层观测到的最大 \|K\|/\|V\| | 294.0（`layers.0…attn.k`） | 低于 e4m3 的 448 |
| 在 448 处被截断的值 | **0 / 47,112,192** | 单位 scale 没截掉任何值 |
| 写入的非有限值 | 0 | — |
| scale 1.0 时的平均相对 RMS 误差 | 2.66e-02 | 3 个尾数位的代价 |
| 用 per-call *oracle* scale 的平均相对 RMS | 2.59e-02 | **好 1.030×** |
| oracle scale 下最好的单层 | 1.185× | 只是一层，不是整个模型 |
| 最大的 subnormal 占比 | 56.7%（`layers.0…attn.v`） | scale *可能*起作用的地方 |
| greedy token 一致率，`auto` vs `fp8_e4m3` | **0.3164**（162/512） | 真实发散 |
| 同上，`auto` vs `auto`（对照） | 1.0000 | 发散来自 fp8，不是噪声 |
| GSM8K 精确匹配，500 题 | 0.1920 → 0.1640（**−2.8 pp**） | 1.96·se = 4.74 pp → **未能判定** |

三条发现并不指向同一个方向，而且是有意保持这种状态：

1. **没有任何值被截断。** per-tensor scale 恰好在两处改变 e4m3 的结果：高于 448 处会截断，以及接近 2⁻⁶ 处值进入 subnormal 并丢失尾数位。e4m3 是*浮点*格式——3 个尾数位意味着每个 binade 内约 6% 的相对间隔——所以乘一个常数大多只是沿指数轴平移数值，并不改变相对误差。这与 int8 恰好相反：int8 的步长是绝对的，标定无条件值得做。
2. **发散是真实的。** 相对 1.0000 的同 dtype 对照，0.3164 的 token 一致率排除了基准噪声：fp8 KV 确实改变了模型说什么。greedy decoding 是混沌的，一个翻转的 argmax 会重写补全的剩余部分——首个分叉落在 128 个 token 中的第 22 / 11 / 74 / 33 个。
3. **标定不是解法。** 这里的 oracle scale 是*逐调用*算出的 `amax/448`，即任何静态 per-tensor 标定的乐观上界——没有哪个离线 scale 能胜过拿着数据现选的 scale。它在各层均值上买来 1.030×。决策用的是均值而不是最好的层：每一层都喂同一条残差流，到达 logits 的是聚合误差。单层 1.185× 只说明该层的激活离 2 的幂更远，而不是标定有用。

**结论：** 误差来自格式，而不是 scale。因此离线标定（`scripts/calibrate_kv_scale.py`，per-layer per-tensor）**没有**实现——它只能削掉 2.7% 误差中的 ~3%。我们*没有*声称 fp8 KV cache 是免费的：它可测地改变了 token，GSM8K 一侧低了 2.8 pp。在 500 题的样本量下这在噪声以内（非配对 1.96·se = 4.74 pp），所以诚实的说法是任务效应**在这个样本量下未测出**，而不是它为零。V 张量中 56.7% 的 subnormal 占比是低于 1.0 的 scale *可能*仍有收益的唯一一处，也是未来某个模型真出现任务回退时的首选排查点。

同一次运行的吞吐注记：fp8 KV decode 1952 tok/s，bf16 KV 为 2297 tok/s。读路径受反量化限制而不是带宽限制（见 `benchmarks/kernels/bench_paged_decode.py`），所以字节数减半并不会让时间减半——fp8 KV 买的是**容量**（约 2× 的缓存 token），不是速度。

复现：

```bash
python -m scripts.debug.quant_kv_error --model-dir $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --max-gen-len 128 --gsm8k 500 --json docs/benchmark_logs/quantization/kv_fp8_error.json
```

## 运行基准测试

```bash
# 单模型 + 指定方案
python benchmarks/engine/run.py quant --model-dir /data/shared/llm_weights/Qwen3-0.6B \
    --schemes fp16 int8 fp8

# 全部代表性模型（plan 子集）
python benchmarks/engine/run.py quant --all

# 输出 JSON 供 CI 跟踪
python benchmarks/engine/run.py quant --model-dir ... --json results.json
```

## 视觉-语言模型支持

TP 与量化都延伸到多模态（VLM）checkpoint。视觉塔以原生 dtype 保持整卡复制不动；只有语言模型的投影会被切分或量化。

| 模型 | 权重 | TP | INT8 | vl-chat |
|-------|---------|----|----|--------|
| Qwen3-VL-4B-Instruct | BF16 | ✓ | ✓ | ✓ |
| LLaVA-1.5-7B (HF) | FP16 | ✓ | ✓ | ✓ |
