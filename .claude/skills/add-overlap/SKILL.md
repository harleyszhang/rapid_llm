---
name: add-overlap
description: Build the three overlap levels in rapid_llm — L1 CPU-GPU (async scheduling, pinned H2D, parallel sampling), L2 communication-compute (micro-batch dual-stream pipelining), L3 in-kernel pipelines — with nsys bubble analysis as the core acceptance evidence, never end-to-end metrics alone. Use when hiding CPU or communication cost from the critical path, when an overlap claim lacks before/after profiles, or when reviewing an overlap change.
---

# Adding Overlap (L1 / L2 / L3)

Goal: remove CPU cost, communication cost and memory-movement cost from
the critical path. The three levels below are the whole scope — a new
level is a separate proposal. Each level gets its own switch, its own
tests, its own benchmark; levels compose and the combination gains are
measured separately.

## The three levels

- **L1 — CPU-GPU overlap (engine/scheduler layer)**: asynchronous
  scheduling (the GPU executes step n while the CPU prepares step
  n+1), pinned-memory non-blocking H2D copies, sampling/detokenize
  running in parallel with next-step preparation. vLLM's model runner
  v2 is the reference design.
- **L2 — communication-compute overlap (distributed layer)**:
  communication rides the interconnect (NVLink/IB), computation rides
  the SMs — different hardware resources, so they can run in
  parallel. Split the batch along the request dim into two
  micro-batches and interleave them on two streams (A communicates
  while B computes), taking per-layer cost from `comm + compute` down
  to `max(comm, compute)`. Design points:
  - decode splits along requests; a prefill sequence is never split
    (causality);
  - split ratio configurable, automatically degrading to no-split when
    the batch is too small;
  - compute/comm dual streams with cross-stream events — every sync
    point drawn, layer by layer, into the design doc;
  - EP pipelines its three phases: dispatch → expert GEMM → combine;
  - the two micro-batches own independent activation buffers, and the
    memory overhead is measured, not estimated;
  - when communication is longer than computation it cannot be fully
    hidden — the residual exposed time is measured and reported;
  - **at least one optimization of your own** (copying a known
    two-batch-overlap implementation does not count): state the
    problem, the idea, the expected gain, and the measured gain. The
    repo's existing TBO/SBO work under `docs/benchmark_logs/overlap/`
    is the baseline to beat, not to clone.
- **L3 — in-kernel pipeline (kernel layer)**: tile-granular movement
  overlapped with compute (cp.async/TMA, double buffering, multi-stage
  pipelines). Kernel-side work follows `triton-kernel-writing` and is
  measured per `kernel-microbenchmark`.

## Architecture constraints

- L1/L2 touch the engine and scheduler: overlap logic stays decoupled
  from scheduling logic — the interface is reviewed on its own,
  overlap state is never scattered across the scheduler, and a module
  dependency graph is part of the delivery.
- L3 is a standalone kernel and must not depend on engine/scheduler
  state.
- Correctness first: each level on vs off is token-identical (or
  within stated tolerance); L1/L2 document their synchronization
  design explicitly (stream sync points, buffer lifetimes); data races
  are disqualifying.
- The CUDA Graph interaction is documented on its own: captured graphs
  pin their input buffers, which fights asynchronous writes — say how
  that conflict is resolved.

## Performance bars (per level, measurable or it does not count)

- L1: single-step CPU cost fully hidden behind GPU execution; TPOT
  improves ≥ threshold over baseline.
- L2: nsys-measured exposed communication time drops ≥ threshold;
  scaling efficiency improves.
- L3: kernel latency drops ≥ threshold over the non-pipelined baseline
  (memory-bound kernels report achieved bandwidth).

## nsys bubble analysis — the core evidence, required per level

- Capture one profile with the level on and one with it off; compare
  GPU timelines: idle gaps, exposed communication, H2D gaps — each
  quantified, before and after.
- The report shows key timeline screenshots plus the gap-duration
  numbers proving the bubble was eliminated or compressed. **An
  end-to-end metric alone never proves overlap worked.**
- Residual bubbles get attributed (sync overhead / uneven split /
  compute too short) and the attribution goes into the report.

## Cross-matrix and stability

- {L1, L2, L3} × {CUDA Graph, speculative decoding, TP, EP,
  quantization}, cell by cell — no "theoretically compatible".
- Correctness: per level on/off, logits-level and end-to-end
  generation, both.
- Stability: sustained high concurrency — no race, no deadlock, no
  memory leak.

## Deliverables

1. Evidence per `model-benchmark-and-report`: environment includes the
   interconnect topology; artifacts include the nsys profiles and the
   timeline analysis; L2 adds exposed-communication time; every level
   adds before/after GPU bubble durations.
2. Docs: the three-level design (principle, sync design, module
   dependency graph), the L2 own-optimization note, and usage.
3. Git: new branch — L1/L2/L3 as independent commits, cross-validation
   separately; every commit green alone.
