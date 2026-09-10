# HellaSwag（benchmarks/eval/hellaswag/）

常识续写四选一（10042 条 validation），20-shot，按选项的续写对数似然打分。口径对齐
sglang 的 `benchmark/hellaswag/bench_sglang.py`，一处差异：打分题目从第 21 条开始——
sglang 从第 0 条开始，它的前 20 道题同时就是 few-shot 示例，答案原文躺在提示词里。

两个引擎都报两种判分：

- `acc`：选项总 log-prob 取 argmax（lm-eval-harness 口径）；
- `acc_len`：选项平均 log-prob 取 argmax（sglang `sgl.select` 默认的
  token_length_normalized 口径）。

| 脚本 | 引擎 | 说明 |
| --- | --- | --- |
| `bench_rapid_vllm.py` | rapid_llm | 引擎 `prompt_logprobs` 打分；CUDA graph 默认开启 |
| `bench_hf.py` | transformers | HF baseline，一次 forward 出全 prompt 的 logprob |

关键实现：两侧都把 (prefix, ending) 分别 tokenize 后按 token 拼接（lm-eval-harness
的做法），边界不经过整串重编码，所以两个引擎给每个选项打分用的是完全相同的 token
序列——对比不受 BPE 边界漂移影响。

## 跑

```bash
python -m benchmarks.eval.hellaswag.bench_rapid_vllm \
    --model-dir Qwen2.5-0.5B --num-questions 200
python -m benchmarks.eval.hellaswag.bench_hf \
    --model-dir Qwen2.5-0.5B --num-questions 200
```

`--batch-size` 默认 8：打分要把 prefill 的全网格 logits（batch × prompt_len × vocab）
物化出来，是显存瓶颈而不是速度瓶颈；大模型跑不下时调小。首次运行下载
`hellaswag_val.jsonl`（~30 MB）到 `~/.cache/rapid_llm/evals/hellaswag/`。结果追加到
`docs/benchmark_logs/accuracy/accuracy_hellaswag_{engine}_<时间戳>.json`。

实测数字见 [`docs/eval_models.md`](../../../docs/eval_models.md)。
