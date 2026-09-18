---
name: add-scheduler-feature
description: Build rapid_llm scheduling capability in dependency order — continuous batching, then chunked prefill, then prefix caching, then minimal PD separation — each behind its own dedicated workload and the golden test (feature on/off must be greedy token-identical). Use when adding or extending a scheduler feature, when TTFT/TPOT regresses after a scheduling change, or when reviewing a scheduler PR.
---

# Adding Scheduler Features

Four features in dependency order; each is accepted before the next
starts — never build them in parallel, because each one reshapes the
scheduling loop the next one stands on. First confirm the current
state: continuous batching already exists (`make bench-continuous`,
`docs/continuous_batching.md`); if reality differs from the assumption
below, correct the assumption before writing code.

## The golden test (every stage, no exceptions)

Feature on vs feature off, same prompt set, greedy decoding: output
must be token-identical. **Scheduling changes performance, never
results.** A stage that fails this test is not done, whatever its
throughput number says. Every stage also cross-checks × {TP, CUDA
Graph} with no accuracy or performance regression.

## Stage 1 — Continuous batching

- Iteration-level scheduling: requests join/leave the batch at each
  decode step; a finished request frees its resources immediately,
  without waiting for the rest of the batch.
- Config: `max_num_seqs`; policy FCFS.
- Dedicated workload: mixed long and short requests.
- Bar: throughput gain vs static batching (measured), no abnormal
  per-request latency degradation.

## Stage 2 — Chunked prefill

- Long prompts prefill in chunks; decode steps may interleave between
  chunks.
- Config: `max_num_batched_tokens` (chunk size).
- Dedicated workload: long prompts mixed with concurrent decode.
- Bar: TPOT jitter under the mixed load drops vs stage 1 (show the
  data). TTFT is allowed to rise — report the magnitude and the
  reason. Reporting only the metrics that improved is concealment.

## Stage 3 — Prefix caching

- Block-level prefix reuse, hit by block hash.
- Config: `enable_prefix_caching`, block size (default 16).
- Dedicated workload: many requests sharing a system prefix — an
  ordinary workload cannot produce a hit rate.
- Bar: hit rate, TTFT drop in the shared-prefix scenario, and correct
  behavior with chunked prefill enabled at the same time.

## Stage 4 — PD separation (minimal viable)

- Prefill instances and decode instances separated; the KV transfer
  channel works end to end.
- Scope stops at: a 1P1D topology running end to end, correct output,
  measurable TTFT/TPOT.
- Explicitly not in this stage: multi-P multi-D topologies, instance
  orchestration, load balancing, fault tolerance.

## Out of scope (all stages)

No priority scheduling, no preemption policy, no model-compute
changes, no scheduler features beyond these four.

## Deliverables (one set per stage)

1. One-page design note: where the scheduling loop changed, the key
   data structures, and the interaction with existing features.
2. Test report: the dedicated workload plus golden-test data, evidence
   per `model-benchmark-and-report`.
3. README: what the feature does, its config knobs and defaults, when
   to use it.
4. Git: one commit set per stage, messages stating the verified
   result; push after the last stage's gates pass.
