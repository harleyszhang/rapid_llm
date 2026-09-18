---
name: optimize-model-performance
description: Profile-driven performance work on an already-supported rapid_llm model — a fixed baseline, a bottleneck list that comes only from profiling, one change per commit, and the two gates (5% performance bar, no-regression accuracy bar) every kept change must pass. Use when speeding up a supported model, cutting TTFT/TPOT, planning an optimization PR, or reviewing a speedup claim that has no before/after data.
---

# Optimizing a Supported Model

The model is already correct — greedy parity with transformers at or
above 99%, per the `add-model` verification ladder. Two rules outrank
every idea: **no optimization may cost accuracy**, and **one round
changes one thing, measured immediately after**. Run the phases in
order; do not skip ahead.

## Phase 0 — Fix the baseline

- Measure the un-optimized state with `benchmarks/engine/run.py`: TTFT,
  TPOT, throughput, peak memory, GPU utilization.
- Pin the environment: commit, hardware, workload parameters. Every
  later "faster" claim traces back to this baseline; a claim that
  cannot is not a result (evidence rules live in `model-benchmark-and-report`).

## Phase 1 — Profile, don't guess

Optimization candidates come from profiling only, never from intuition:

- **nsys timeline**: bubbles, kernel-launch overhead, sync points.
- **Per-layer breakdown**: which layers hold an abnormal share of step
  time.
- **Roofline reading**: prefill reads compute utilization, decode reads
  memory-bandwidth utilization — name the bound before treating it
  (`kernel-microbenchmark` is the per-kernel version of this loop).
- **External comparison**: put each metric next to vLLM/SGLang numbers
  for the same model; the metric with the largest gap is the first
  priority.
- **Feature cross-matrix**: CUDA Graph / TP / speculative decoding
  toggled independently, looking for a combination that degrades.

Output of the phase: a bottleneck list ranked by expected gain.

## Phase 2 — One change per commit

Candidate classes are a closed enum: operator fusion, CUDA-graph
coverage, communication-compute overlap, KV-cache layout, kernel
selection/replacement, scheduler overhead. Each candidate lands as its
own commit — a commit mixing two optimizations is reverted on sight,
because an untestable delta cannot be kept or reverted on evidence.
Kernel work itself follows `triton-kernel-writing`, measurement follows
`kernel-microbenchmark`.

## Phase 3 — Two gates per change

- **Performance gate**: before/after measurement on the phase-0
  baseline; a gain under 5% reverts — the complexity is not worth the
  number.
- **Accuracy gate**: replay the onboarding parity set (`tests/golden/`,
  baselines recorded with `scripts/golden_tokens.py`); greedy agreement
  must not drop.

**Stop condition**: two consecutive rounds both gain under 5%, or
profiling shows the path is already at the roofline.

## Out of scope

- No drive-by refactoring of performance-irrelevant code — noticed
  problems get logged, not fixed here (`refactor-module` is the place).
- No quantization work (that is `add-quant-method`).
- No new dependency without a written justification.

## Deliverables

1. Profiling report: timeline data, the ranked bottleneck list, and the
   ranking rationale.
2. Optimization ledger: per change — what, why, before/after numbers,
   kept or reverted and why.
3. Final comparison: baseline vs optimized, and how the gap to
   vLLM/SGLang closed, metric by metric.
4. Docs updated per `model-benchmark-and-report`; the model's optimization
   record (e.g. `docs/model_optimize_benchmarks.md`) states each kept
   change's measured gain.
5. Git: new branch, one commit per change with the measured gain in the
   message, pushed when the gates are green.
