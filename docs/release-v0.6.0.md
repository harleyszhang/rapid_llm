# Release v0.6.0 — 分页 KV + viz 工具

**Date:** 2026-08-23 **Branch:** `main` **Theme:** FP8 KV 缓存 (2x 容量)、可视化工具 (viz.structure + viz.memory)、KV 水位线准入控制

## Summary

v0.6.0 完成 KV 缓存的 fp8 路径端到端打通（flashdecoding kernel 内 e4m3 dequant + attention module 接口 + kv_cache_manager fp8 dtype 支持），使 KV 容量翻倍（147K → 282K tokens on A10）。新增 `rapid_llm.viz` 模块导出模型结构树和显存预算表。KVCacheManager 新增 watermark 准入控制和 utilization 指标。

## 1. Feature: FP8 KV Cache (2x capacity)

**核心改动:**

- `flashdecoding.py`: 新增 `KV_FP8` constexpr 路径, `dequant_fp8e4m3` bit-surgery 升到 fp32 后乘 scale。调用方将 `FP8_E4M3_BIT_TRICK_SCALE` 折入 scale 免 kernel 内补偿。
- `modules/attention.py`: 新增 `k_scale`/`v_scale` 参数,传递到 flash_decoding。
- `modules/quantization/kv_cache.py`: `Fp8KVCacheMethod` 负责 write-side 量化。

**使用方式:**

```bash
python -m rapid_llm.cli chat --model-dir my_weight/Qwen3-0.6B --kv-cache-dtype fp8
```

**修复前后对比 (Qwen3-0.6B, A10, batch=4, gen_len=64, greedy):**

| KV dtype | KV Capacity | TPS | 相对 fp16 |
|----------|-------------|-----|-----------|
| fp16 (default) | 147,875 tok | 943.4 | 1.00x |
| fp8 (e4m3) | 282,435 tok | 859.5 | 0.91x |

> KV 容量提升 **1.91x**，throughput 仅降 9%（fp8 dequant 额外开销）。
> 对长序列场景（4K+ context），fp8 KV 是纯收益：原本 OOM 的序列现在能服务。
>
> Benchmark 日志: [`docs/benchmark_logs/quantization/kv_cache_fp8_v06.json`](benchmark_logs/quantization/kv_cache_fp8_v06.json)

## 2. Feature: viz.structure (模型结构树)

新增 `rapid_llm/tools/profiling/structure.py`，遍历 `nn.Module` 树导出缩进文本树，显示层类型、参数量和 dtype。

```python
from rapid_llm.tools.profiling import export_structure_tree
tree = export_structure_tree(model, max_depth=3)
```

输出示例:

```
model: Qwen3Model
├── embed_tokens: Embedding [155,648,000 params, float16]
├── layers: ModuleList
│   ├── 0: Qwen3DecoderLayer
│   │   ├── self_attn: Attention
│   │   ├── mlp: MLP
│   ...
└── norm: RMSNorm [1,024 params, float16]
```

## 3. Feature: viz.memory (显存预算表)

新增 `rapid_llm/tools/profiling/memory.py`，从模型配置纯计算得到显存分解（无需 GPU），输出 markdown 表格。

```python
from rapid_llm.tools.profiling import export_memory_budget
table = export_memory_budget(
    num_layers=28, hidden_size=1024, intermediate_size=3072,
    num_heads=16, num_kv_heads=8, head_dim=64,
    vocab_size=151936, num_kv_blocks=147875,
    weight_dtype="fp16", kv_dtype="fp16",
)
```

| Component | Size | Percentage |
| ----------- | ------ | ------------ |
| Model Weights | 1.24 GB | 12.8% |
| KV Cache (fp16) | 7.90 GB | 82.0% |
| Activations | 0.25 GB | 2.6% |
| CUDA Graph | 0.25 GB | 2.6% |
| **Total** | **9.63 GB** | 100% |

## 4. Feature: KV Cache Watermark 准入控制

`KVCacheManager` 新增:

- `watermark` 参数 (默认 0.1): 当空闲 blocks 低于 `total * watermark` 时拒绝新请求。
- `can_admit(need_blocks)` 方法: 纯读判断,调度器用于准入决策。
- `utilization` 属性: 返回当前使用率 (0.0~1.0)。

```python
if kv_cache.can_admit(seq_len):
    allocate_and_serve(request)
else:
    queue_or_reject(request)
```

## 5. Chore

| Item | 变更 |
|------|------|
| `pyproject.toml` | 版本号 `0.4.0` → `0.6.0` |
| `flashdecoding.py` | 精简 docstring + fp8 KV cache 用法示例 |

## 6. 测试结果

```
217 passed, 2 skipped    (CPU subset, gpu/weights tests deselected)
4 passed                 (golden gate, with weights)
```

KV cache watermark 和 viz 模块为纯 Python, 无需 GPU 即可验证。

## Upgrade

```bash
git pull origin main
uv pip install -e .
# 使用 fp8 KV cache
python -m rapid_llm.cli chat --model-dir my_weight/Qwen3-0.6B --kv-cache-dtype fp8
```
