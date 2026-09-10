# BoolQ（benchmarks/eval/boolq/）

是否型阅读理解（3270 条 validation），5-shot，生成 `True` / `False` 精确匹配。口径对齐
sglang 的 `benchmark/boolq/bench_sglang.py`，两处差异：

1. 判分前 strip——sglang 直接拿生成文本与 `"True"` 比较，而 tokenizer 在 `Answer:`
   之后通常先吐一个空格，原样比较会踩空；
2. 数据取 SuperGLUE 官方 zip（`BoolQ/val.jsonl`）。sglang README 链的 GCS 源已 403；
   zip 里的答案字段叫 `label`，HF parquet 版叫 `answer`，`common.py` 两个名字都认。

| 脚本 | 引擎 | 说明 |
| --- | --- | --- |
| `bench_rapid_vllm.py` | rapid_llm | CUDA graph 默认开启（优化模式） |
| `bench_hf.py` | transformers | HF baseline |

## 跑

```bash
python -m benchmarks.eval.boolq.bench_rapid_vllm \
    --model-dir Qwen2.5-0.5B --num-questions 200
python -m benchmarks.eval.boolq.bench_hf \
    --model-dir Qwen2.5-0.5B --num-questions 200
```

首次运行下载 SuperGLUE zip（4 MB）到 `~/.cache/rapid_llm/evals/boolq/`。结果追加到
`docs/benchmark_logs/accuracy/accuracy_boolq_{engine}_<时间戳>.json`。

## 口径

- 提示词：train split 前 5 条作示例，格式 `Question: {question}{passage}\nAnswer:`；
- 解码：greedy，每题最多 6 个 token（sglang 是 5 个 + `\n` 停止，同一预算）；
- 判分：strip 后等于 `True` / `False`；其他计无效（invalid_rate 单独报）。

实测数字见 [`docs/eval_models.md`](../../../docs/eval_models.md)。
