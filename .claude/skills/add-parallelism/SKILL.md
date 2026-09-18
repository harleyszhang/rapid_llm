---
name: add-parallelism
description: Staged build-out of distributed parallelism in rapid_llm — TP, DP, EP, PCP and DCP, each landing behind its own N-rank-vs-single-rank parity anchor before mixed-parallel composition and feature cross-testing. Use when adding a parallel mode, raising a TP/EP degree, enabling long-context context parallelism, debugging a multi-rank numeric mismatch, or reviewing a parallelism PR.
---

# Adding Parallelism (TP / DP / EP / PCP / DCP)

Each stage ships independently and is accepted independently — a later
stage builds on the previous one's working collectives, so stages never
run in parallel. Target-model constraints: EP needs a MoE model;
PCP/DCP exist for long-context serving.

## Align the terms first

Definitions below govern; "etc." does not extend the list.

- **TP (tensor parallel)** — shard one layer's weights and compute
  across ranks. Megatron-style column/row sharding usually needs a
  reduce, but all-reduce is not definitional: the primitive follows
  from the sharding, the parallel composition and the topology.
- **DP (data parallel)** — request-level parallelism across full
  replicas; no intra-layer sharding, no inference-time communication.
- **EP (expert parallel)** — MoE experts live on different ranks;
  routed tokens move by all-to-all and results return the same way.
- **PCP (prefill context parallel)** — shard the input along the
  sequence dim during prefill; attention results or KV need merging.
- **DCP (decode context parallel)** — shard the KV cache along the
  sequence dim during decode; attention gathers across ranks, taking
  single-card KV pressure down for long contexts.

## The stages

1. **TP (infrastructure, first)**: attention/MLP sharding,
   embedding/sampler, NCCL wiring. Correctness anchor: TP=1 and TP=N
   produce identical output within tolerance. The repo already works
   this muscle — `scripts/dsv2_tp2_parity_probe.py` and the TP rows in
   `benchmarks/suites/run.py`; a TP mismatch has its own rung in
   `locate-numeric-divergence` (TP=1 vs TP=2 splits the search in
   half).
2. **DP**: request routing, multi-replica scheduling, load balancing.
   Write down the boundary with the scheduler before coding — who owns
   admission, who owns rebalance.
3. **EP (needs a MoE model)**: expert placement, all-to-all, routing
   and load balance.
4. **PCP / DCP (long context)**: sequence-dim sharding, KV-cache
   distribution, cross-rank attention.
5. **Mixed parallelism**: enumerate the composition matrix explicitly
   (e.g. TP×DP, TP×EP, TP+PCP, TP+PCP+DCP — confirm the list before
   testing) and fill every cell: functionally correct + accuracy within
   tolerance + performance acceptable, or the cell says "unsupported"
   with its reason. "Theoretically supported" is not a cell.
6. **Cross with existing features**: 〈parallel mode〉 × {CUDA Graph,
   speculative decoding, scheduler policies}. Full-on combinations show
   no accuracy/performance problem; negative interactions get recorded
   with the cause.

## Test bar

- **Correctness**: every mode, N ranks vs one rank — logits-level and
  end-to-end generation, both.
- **Performance**: scaling efficiency measured (e.g. TP=2 throughput ≥
  single card × threshold), communication overhead reported as its own
  line, TGS (= TPS / parallel degree) included whenever parallelism is
  on.
- **Accuracy**: downstream-task spot check with tolerances stated.

## Deliverables

1. Evidence per `model-benchmark-and-report`, with the environment naming
   card count and interconnect topology (unmeasured topology is marked
   "not measured", never inferred).
2. Docs: per mode — usage, combination constraints, launch commands;
   README gains a real recorded run.
3. Git: new branch, at least one commit per mode, cross-validation in
   its own commit; every commit passes tests alone.

## Failure modes to expect

| Symptom | First suspicion |
| --- | --- |
| N-rank output diverges from single-rank | shard splits a scale group or a collective is missing — `shard_is_aligned`, then the rung-2 layer diff |
| Throughput flat from 1→2 ranks | communication exposed, not overlapped — nsys before touching code (see `add-overlap`) |
| A composition cell passes alone, fails mixed | KV/token accounting assumed one mode's budget — check the scheduler boundary written in stage 2 |
