---
name: validate-trimmed-model
description: Validate rapid_llm's support for an oversized model by trimming it to fit the GPUs — the cut script, the layer-count rules that keep full operator coverage, zero-missing-key loading, per-operator execution evidence, and the two-engine greedy parity and performance comparison on the identical trimmed checkpoint. Use when a checkpoint does not fit in memory, when validating DeepSeek-scale support on small hardware, or when producing an engine-vs-engine comparison on a trimmed model.
---

# Validating a Trimmed Model

A full-size model (671B-class) does not fit the available GPUs. Cut it
to N layers to get a checkpoint that does — then use it to prove the
framework supports the architecture correctly and to compare against a
reference engine (vLLM / SGLang / transformers) on the *identical*
small model. **Numbers from a trimmed model compare engines; they never
represent full-size performance.**

## The cut script

Input: source weights directory, kept layer count N, output directory.
Output: embedding + first N layers + final norm + lm_head, with
`num_hidden_layers=N` patched into the config and every other field
untouched. DeepSeek trimming and conversion kits already live under
`tests/layer/` (`deepseek.py`, `dspark_to_hf.py`, `convert_v4_hf.py`) —
extend them before writing a new one.

## Choosing N

The kept layers must cover **every operator type the model uses**.
Known trap: some models run dense layers first and start MoE only
later (DeepSeek's first 3 layers have no experts) — N must clear the
dense prefix or the MoE path never executes. Whether auxiliary modules
(MTP heads and the like) are kept: decide, and write the decision and
its reason into the report.

## The verification chain

1. **Weight loading**: real config, real weights (confirm the path
   with the user if unsure). The load report must show
   `missing keys = 0` and `unexpected keys = 0` (the `add-model`
   skill's weight-coverage machinery produces this report). Randomly
   initialized substitutes are an automatic fail.
2. **Operator coverage**: list the model's operators (attention
   variant, MoE routing, norms, RoPE, quantization kernels), then
   prove that each one executes at least once in an e2e run — the
   dispatcher's `explain()` lists the selected KernelSpec row per op,
   a torch.profiler trace shows the launched kernels.
3. **Accuracy parity**: the same trimmed checkpoint loaded in
   rapid_llm and in the reference engine; same prompt set, greedy
   decoding, token-by-token comparison. Bar: ≥ 99% agreement, with a
   per-sample analysis of every disagreement. Never widen a tolerance
   to make the comparison pass — a difference gets a cause first
   (`locate-numeric-divergence` is the narrowing procedure;
   `tests/golden/test_deepseek_trimmed_parity.py` is the committed
   precedent).
4. **Performance comparison**: same hardware, same load, both engines
   — TTFT / TPOT / throughput / memory, with the differences
   explained.

## Out of scope

- No claim that trimmed numbers represent full-size performance.
- No weight surgery, no fine-tuning.
- No tolerance widening to pass a comparison.

## Deliverables

Evidence per `model-benchmark-and-report`, trimmed-model specifics included:

1. Environment: versions/commits of both engines, CUDA/driver,
   hardware, the model and N, launch parameters.
2. Load: prompt set, input/output lengths, concurrency, sampling
   parameters.
3. Full reproduction commands, including the cut script's usage.
4. Logs: the weight-load report and the operator-coverage evidence.
5. Metrics: TTFT, TPOT, TPS, TGS when parallel, throughput, latency
   percentiles, error rate, memory/GPU utilization.
6. Comparison summary: rapid_llm vs reference engine, conclusions and
   anomalies explained.
