---
name: add-speculative-decoding
description: Add speculative decoding to rapid_llm with pluggable draft sources — the propose/verify/arbitrate separation, the three integration points (scheduler budget and rollback, single-pass batched verification, rejection sampling with the bonus token), and the four high-risk coupling points (KV rollback, batch-state divergence, CUDA Graph, sampler RNG). Losslessness is the hard bar. Use when adding a draft model or an external draft library, when acceptance accounting or KV rollback misbehaves, or when reviewing a speculative-decoding change.
---

# Adding Speculative Decoding

Draft tokens come from a pluggable source; the target model verifies
and the sampler arbitrates. **Losslessness is the hard bar** — a faster
engine that changes the output is a broken engine.

Names below are by responsibility — scheduler (request scheduling and
token budget), model runner (forward), sampler (output sampling). Find
the real modules in the tree first; if you cannot map one, stop and
ask — do not create a parallel module with the same name.

## Stage 0 — Interface alignment (no implementation code)

For each draft source, confirm and write down, one interface document
per source, reviewed before coding:

1. Type: draft model / Medusa / EAGLE / n-gram.
2. Input: token ids only, or hidden states / KV as well.
3. Output: k candidate tokens — does it include draft probabilities?
4. Call cadence: synchronous per step? Batch shapes?
5. Version constraints, error semantics, timeout handling.

Two sources may differ (one wants hidden states, one wants only token
ids) — stage 0 exists to expose that difference, because it shapes the
unified interface.

## Architecture — propose / verify / arbitrate

The external library only **proposes** candidates. The framework
**verifies** (target-model forward) and **arbitrates** (rejection
sampling). Swap the draft library without touching the framework;
change the framework without touching the library.

- One `Proposer` interface: `propose(...) -> k candidates + draft
  probabilities`; a proposer registry keyed by name, selected by
  config. Adding a proposer = a new module + one registration.
- Per-library differences live in that library's adapter; the main
  flow only ever sees the unified interface.

## The three integration points

1. **Scheduler — draft budget and rollback.** With speculation on,
   every request reserves k+1 token slots per step (k candidates + 1
   bonus); KV-block allocation and token accounting follow that
   budget. After verification accepts a ≤ k tokens, unused slots are
   reclaimed — token budget, KV blocks and the request position cursor
   roll back together, as one consistent operation. Speculation acts
   on decode only; prefill chunks never participate.
2. **Model runner — single batched verification.** The k candidates
   go into one forward (length k+1 per request), yielding k+1 logits
   positions. Never k serial verification passes. Candidate positions
   use a chain mask (tree candidates are out of scope). Requests in a
   batch may have different k — state the padding or varlen policy.
3. **Sampler — rejection sampling and the bonus token.** Position by
   position: accept and keep the draft token; at the first rejection,
   sample from the corrected distribution. When all k are accepted,
   sample the bonus token from the target model's (k+1)-th
   distribution — the bonus is part of the speedup; dropping it throws
   away one free token per fully-accepted step.

## The four high-risk coupling points

Each needs a design note and dedicated tests:

1. **KV-cache rollback** — rejected tokens' KV slots are reclaimed
   correctly; cover accept-all / accept-some / accept-none.
2. **Batch-state divergence** — requests accept different counts in
   the same step, and the batch changes mid-step: scheduler and runner
   must keep token accounting consistent.
3. **CUDA Graph** — variable-length speculation vs fixed graph
   shapes: resolve explicitly (multi-shape graphs or eager fallback),
   with a conclusion and measured data. Compare `supports_cuda_graph`
   in `add-model` for the model-side opt-out precedent.
4. **Sampler RNG** — rejection-sampling randomness is state-managed
   so results are reproducible.

## Acceptance bars

1. **Losslessness, per proposer**: greedy — speculation on vs off is
   token-identical; sampled — same-seed token-identical (preferred),
   or a distribution check: ≥100k tokens at the same temperature,
   chi-square p > 0.05, method and raw data reported.
2. **Speedup** (workload fixed and stated — acceptance rate is a
   function of the workload): a high-acceptance load (e.g. code
   continuation) shows positive TPOT/throughput gain; a low-acceptance
   load (high-temperature open generation) degrades ≤ 5% and the
   adaptive cutoff (disable speculation below an acceptance threshold)
   triggers correctly.
3. The four coupling points are test-covered, including 0% and 100%
   acceptance edges.
4. Cross-matrix, cell by cell (each cell = greedy-identical + no perf
   regression): speculation × {CUDA Graph, TP, quantization,
   comm-compute overlap, chunked prefill, prefix caching}; negative
   interactions recorded with the cause.
5. Soak: ≥ 30 minutes at high concurrency — no state corruption,
   memory curve flat.
6. Library isolation: every draft source has its own mock; unit tests
   never need the real library; an upstream interface change fails a
   test immediately.
7. Extensibility demo: adding a proposer (a mock is fine) touches only
   its file plus one registration.

## Deliverables

1. One interface document per draft source.
2. Design note: the three-way separation, the three integration
   points, the four coupling points.
3. Proposer support matrix: library × type × interface version ×
   status.
4. Test and benchmark report per `model-benchmark-and-report`, plus
   speculation-specific metrics: acceptance rate and average accepted
   tokens per step, on/off and across acceptance regimes.
5. README: principle, switches and tuning knobs, with a recorded
   on/off speed comparison.
6. Git: new branch — core / each proposer integration / runner
   integration / cross-validation and docs as separate commits, each
   green alone.
