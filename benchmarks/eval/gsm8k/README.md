# GSM8K（benchmarks/eval/gsm8k/）

小学数学应用题，5-shot，取 completion 最后一个整数与参考答案做精确匹配。口径复用
回归套件 [`tests/evals/gsm8k.py`](../../../tests/evals/gsm8k.py)（同一份
`build_prompts` / `score`），与 vLLM `tests/evals/gsm8k` 的口径也一致，三边数字可直接对比。

三个脚本，同一口径不同引擎：

| 脚本 | 引擎 | 说明 |
| --- | --- | --- |
| `bench_rapid_vllm.py` | rapid_llm | 直接调 `tests.evals.gsm8k.evaluate_gsm8k`；CUDA graph 默认开启（优化模式） |
| `bench_hf.py` | transformers | HF baseline；left-padding 静态 batch，greedy |
| `bench_vllm.py` | vllm | 需要装有 vllm 的环境（rapid_llm 开发环境不带） |

## 跑

```bash
# rapid_llm（优化模式：CUDA graph 开启）
python -m benchmarks.eval.gsm8k.bench_rapid_vllm \
    --model-dir Qwen2.5-0.5B --num-questions 200

# instruct 模型加 chat 模板
python -m benchmarks.eval.gsm8k.bench_rapid_vllm \
    --model-dir Qwen2.5-1.5B-Instruct --num-questions 200 --chat-template

# HF baseline
python -m benchmarks.eval.gsm8k.bench_hf \
    --model-dir Qwen2.5-0.5B --num-questions 200
```

`--model-dir` 接受路径或名字（依次在当前路径、`RAPID_LLM_MODELZOO`（默认
`/data/shared/llm_weights`）、`my_weight/` 下查找）。每次运行把完整结果追加到
`docs/benchmark_logs/accuracy/accuracy_gsm8k_{engine}_<时间戳>.json`。

## 口径

- 提示词：train split 前 5 条作示例，接 `Question: ...\nAnswer:`；测试题取 test split 前 N 条；
- 解码：greedy（`temperature=0`），关 `repetition_penalty` 与 `stop_on_repeat`；
- 停止：截到 `Question` / `Assistant:` / `<|separator|>`；
- 判分：最后一个整数精确匹配；解析不出数字计无效（invalid_rate 单独报）。

实测数字与 HF / vLLM 的对比见 [`docs/eval_models.md`](../../../docs/eval_models.md)。
