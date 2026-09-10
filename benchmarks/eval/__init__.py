"""Accuracy benches, organised one directory per dataset (sglang's layout).

Each dataset directory holds the arms that run it — ``bench_rapid_vllm.py``
(the rapid_llm engine, CUDA graphs on by default) and ``bench_hf.py`` (the
transformers baseline) — plus a ``common.py`` defining that dataset's prompts
and scoring once, so every arm scores identically. Numbers land in
``docs/benchmark_logs/accuracy/`` and the tables live in
``docs/eval_models.md``.

* :mod:`benchmarks.eval.gsm8k` — 5-shot math, last-integer exact match;
  reuses the ``tests/evals`` prompts and scorer, plus a vllm comparison arm
* :mod:`benchmarks.eval.boolq` — 5-shot yes/no reading comprehension
* :mod:`benchmarks.eval.hellaswag` — 20-shot four-way continuation
  likelihood (both the lm-eval and the sglang rulings)

The DeepSeek trimmed-checkpoint parity suite (V3/V4 reference-vs-lite) moved
to ``tests/layer/``.
"""
