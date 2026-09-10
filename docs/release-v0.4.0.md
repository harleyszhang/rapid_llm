# Release v0.4.0 — 可信基线 (Trustworthy Baseline)

**Date:** 2026-08-23  
**Branch:** `add_awq_gptq`  
**Theme:** 量化模块架构重构 + Golden 门禁强化 + TP 采样 RNG 同步

## Summary

v0.4.0 建立 rapid_llm 的可信基线：量化子系统从 `models/quantization` 迁移到 `modules/quantization`，对齐 sglang 架构、新增 AWQ/GPTQ/W8A8 的 MoE Method 支持（含 INT4 packed weights）；golden 回归门禁升级为显式 UNVERIFIED 状态（不再静默 skip）；TP 模式下的非 greedy 采样 RNG 同步通过 rank0 broadcast 解决。

## 1. Feature: 量化模块架构重构 (sglang aligned)

**重构内容：**

| Before (v0.3.x) | After (v0.4.0) |
| ------------------ | ---------------- |
| `rapid_llm/models/quantization/` (分散式) | `rapid_llm/modules/quantization/` (集中式) |
| 无 MoE 量化 Method | `AWQMoEMethod`, `GPTQMoEMethod`, `W8A8Fp8MoEMethod`, `W8A8Int8MoEMethod` |
| `GPTQLinearMethod` 继承自 `AWQLinearMethod` | 独立实现，各有自己的 `create_weights`/`apply` |
| 无 INT4 MoE 支持 | `fused_moe` kernel 支持 int4 packed weight nibble extraction |

**重构意义：**

1. **可读性** — 每个量化方案一个文件，Config + LinearMethod + MoEMethod 三元组统一。
2. **可插拔** — `BASE_QUANTIZATION_METHODS` 注册表 + `get_quant_method(layer, prefix)` 委托。
3. **可扩展** — 新增方案仅需一个文件 + 注册即完成，无需修改模型代码。

**新架构目录：**

```
rapid_llm/modules/quantization/
├── __init__.py            # 注册表 + 工厂函数
├── base_config.py         # ABC: QuantizationConfig / LinearMethodBase / FusedMoEMethodBase
├── fp8.py                 # Fp8Config + Fp8LinearMethod + Fp8MoEMethod
├── w8a8_fp8.py            # W8A8Fp8Config + W8A8Fp8LinearMethod + W8A8Fp8MoEMethod
├── w8a8_int8.py           # W8A8Int8Config + W8A8Int8LinearMethod + W8A8Int8MoEMethod
├── blockwise_int8.py      # BlockInt8Config + BlockInt8LinearMethod + BlockInt8MoEMethod
├── awq.py                 # AWQConfig + AWQLinearMethod + AWQMoEMethod
├── gptq.py                # GPTQConfig + GPTQLinearMethod + GPTQMoEMethod
├── unquant.py             # UnquantizedConfig (fp16 默认路径)
├── kv_cache.py            # Fp8KVCacheMethod
├── parameter.py           # RawParameter (loader 不可 cast)
└── utils.py               # 工具函数 + checkpoint 适配器
```

## 2. Feature: INT4 MoE (fused_moe kernel 扩展)

`fused_moe` Triton kernel 新增 `QUANT_MODE=3` (INT4) 路径：

- int32 packed weight 加载 → 8 nibble 位抽取 (`(word >> shift) & 0xF`)
- group-wise scale 乘法 + optional zero-point 减法
- 与 fp16/fp8/int8 路径共享 tile config 和对齐逻辑

**使用方式：**

```python
from rapid_llm.kernels.fused_moe import fused_moe

out = fused_moe(
    x, w1_packed_int32, w2_packed_int32, topk_weights, topk_ids,
    w1_scale=s1, w2_scale=s2,
    w1_zeros=z1, w2_zeros=z2,
    group_n=1, group_k=128,
)
```

## 3. Fix: Golden 门禁去静默 skip

**修复前：** 无 GPU 或无权重时 golden tests 被 `pytest.mark.skip` 静默标记 → CI dashboard 显示全绿（误导性）。

**修复后：**

- Golden 测试使用 `xfail(reason="UNVERIFIED: no CUDA device", run=False)` → CI 显示为 xfail（黄色/橙色），**不会**被误认为已验证。
- 设置 `RAPID_LLM_GOLDEN_STRICT=1` 时升级为 `strict=True` → hard FAIL。
- `cases.py` 新增 `CB_CASES`（连续批处理路径）和 `QUANT_CASES` + `QUANT_SCHEMES`（量化路径覆盖）。
- `scripts/golden_tokens.py` 支持 `--batch-save --models` 多模型批量重录和 `--all-schemes` 全量化方案录制。

**验证方式：**

```bash
# 无 GPU CI 上的表现
RAPID_LLM_GOLDEN_STRICT=1 pytest tests/golden/ -v
# 输出: XFAIL (UNVERIFIED: no CUDA device) — 非绿色

# 多模型批量重录
python scripts/golden_tokens.py --batch-save \
    --models my_weight/Qwen2.5-0.5B my_weight/Qwen3-0.6B --all-schemes
```

## 4. Fix: TP 采样 RNG 不同步

**修复前：** TP 模式下每个 rank 独立运行 `torch.multinomial`，各自使用本地 RNG 状态 → 非 greedy 采样时 rank 间 diverge → 下一步 input_ids 不一致 → 模型输出错误。

**修复后：** Rank 0 采样后通过 `broadcast_tp(next_token)` 广播给所有 TP rank：

```python
# rapid_llm/distributed/parallel_state.py
def broadcast_tp(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from TP-local src to all TP ranks."""
    ...

# rapid_llm/engine/llm_engine.py (decode loop)
next_token = engine.sampler.sample(logits, params, generated).reshape(-1)
if get_tp_world_size() > 1:
    next_token = broadcast_tp(next_token)
```

**影响范围：** `llm_engine.py`（offline batch）和 `continuous_engine.py`（online batch）的两个采样点均已修复。

**修复前后对比 (TP=2, Qwen3-0.6B, temperature=0.7, seed 每 rank 不同)：**

| 条件 | Rank 0 输出 | Rank 1 输出 | 一致性 |
|------|-------------|-------------|--------|
| 修复前 (无 broadcast) | `a concept that is debated, and the meaning of life can be found in the process` | `the question that has very among but the same is life can be found in the appli` | **DIVERGE** |
| 修复后 (broadcast ON) | `a concept that is often discussed in philosophical circles, and it has been use` | `a concept that is often discussed in philosophical circles, and it has been use` | **AGREE** |

> 验证脚本: `scripts/verify_tp_sampling.py`  
> 对比日志: [`docs/benchmark_logs/engine/tp2_sampling_fix_comparison.json`](benchmark_logs/engine/tp2_sampling_fix_comparison.json)

## 5. Chore: CI 与工程治理

| Item | 变更 |
| ------ | ------ |
| `.github/workflows` | 适配 `modules/quantization` 新路径 |
| `tools/pre_commit/check_hardcoded_paths.py` | exempt `benchmarks/` 和 `scripts/`（机器相关脚本） |
| `pyproject.toml` | 版本号 `0.3.0` → `0.4.0` |
| 删除 `docs/models_shape/`, `docs/benchmarking/` | 迁移至 `docs/benchmark_logs/` |

## 6. Benchmark 结果

### Qwen3-0.6B (A10, batch=4, seq_len=25, gen_len=64, greedy)

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS | vs HF fp16 |
| -------- | ----------- | ------------- | ----------- | ----- | ------------ |
| HF fp16 (baseline) | 1.17 GB | — | 28.19 | 141.7 | 1.0× |
| lite fp16 | 1.40 GB | 147,875 tok | 4.14 | 918.8 | **6.5×** |
| lite int8 | 0.99 GB | 141,549 tok | 4.16 | 904.1 | **6.4×** |
| lite int8-blockwise | 1.00 GB | 138,385 tok | 4.44 | 849.4 | **6.0×** |
| lite fp8 (W8A8) | 0.99 GB | 139,153 tok | 8.35 | 448.1 | **3.2×** |
| lite smoothquant (W8A8) | 0.99 GB | 135,642 tok | 3.70 | 983.8 | **6.9×** |

> Benchmark 日志: [`docs/benchmark_logs/quantization/quant_Qwen3-0.6B_all_20260823.json`](benchmark_logs/quantization/quant_Qwen3-0.6B_all_20260823.json)

### Qwen3-VL-4B-Instruct (A10, batch=4, seq_len=25, gen_len=64, greedy)

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS |
| -------- | ----------- | ------------- | ----------- | ----- |
| lite fp16 | 8.99 GB | 73,676 tok | 23.36 | 170.7 |
| lite int8 | 5.61 GB | 93,559 tok | 27.47 | 145.3 |
| lite int8-blockwise | 5.71 GB | 92,748 tok | 27.97 | 142.7 |
| lite fp8 (W8A8) | 5.61 GB | 93,345 tok | 59.25 | 67.4 |
| lite smoothquant (W8A8) | 5.61 GB | 93,559 tok | 34.00 | 117.5 |

> Benchmark 日志: [`docs/benchmark_logs/quantization/quant_Qwen3-VL-4B_all_20260823.json`](benchmark_logs/quantization/quant_Qwen3-VL-4B_all_20260823.json)

### 可视化

![Qwen3-0.6B quantization](images/quantization_benchmark.gif)

![Qwen3-VL-4B quantization](images/Qwen3-VL-4B-Instruct_quantization_benchmark.gif)

## 7. 测试结果

```
$ pytest tests/ -q
394 passed, 3 skipped, 48 deselected    (CPU subset)
475 passed, 3 skipped                   (full suite with GPU)
4 passed                                (golden gate, with weights)
```

## Upgrade

```bash
git fetch origin add_awq_gptq
git checkout v0.4.0
uv pip install -e .
```

量化 import 路径变更：

```python
# Before (v0.3.x)
from rapid_llm.models.quantization import ...

# After (v0.4.0)
from rapid_llm.modules.quantization import ...
```
