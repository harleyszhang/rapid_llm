---
name: kernel-microbenchmark
description: Build, run and read rapid_llm GPU kernel microbenchmarks — Triton/torch kernels, cold-L2 timing, correctness gates before numbers, KV-cache pool fixtures, host-time measurement for the block allocator, SOL sanity checks, and feeding results back into autotune and dispatch.
---

# Kernel Microbenchmark

A measurement in this repo touches three tiers: the kernels themselves in
`rapid_llm/kernels/*.py`, the `KernelSpec` rows that declare what a backend can
serve in `rapid_llm/kernels/backends/*.py`, and `dispatch()` in
`rapid_llm/kernels/ops/`. The reason to benchmark is to give the third tier a
basis for preferring one row over another, so a table whose rows do not name
registry entries cannot close that loop, however precise its numbers are.

## Workflow

1. Check correctness before timing, against a reference that is not the kernel
   under test. Keep tolerances explicit and named (`_RTOL`, `_ATOL`), and match
   what `tests/kernels/` already uses for the same kernel.
2. Give every code path its own check. A dtype variant (fp8, int8) runs
   different code inside the kernel, so an fp8 row verified only through its
   fp16 sibling is an unverified row.
3. Time only the operation. Pool construction, index building, random input
   generation and `.contiguous()` copies stay outside the timed region unless
   they are the subject.
4. Score against the theoretical best for the operation, not against what the
   implementation does: bytes assume each input is read exactly once from global
   memory, FLOPs assume no redundant work. A kernel's own split-K partial
   buffers do not enter the formula.
5. Report metadata sufficient to reproduce: GPU, dtype, shapes, commit, and the
   env vars that change which kernel runs. `microbench.metadata()` emits this.
6. Treat an explanation as a hypothesis until an artifact backs it — an
   ablation row, a profiler trace, or a second construction of the same input.
   When a result overturns the reason a fixture exists, rewrite the reason;
   a fixture justified by a disproven claim gets deleted by the next reader.

For an experiment, a short
`Question / Change / Correctness / Result / Observation / Next` note is enough.

## Harness

Use `benchmarks/kernels/microbench.py`; do not re-implement timing in a
benchmark script.

- `Work(flops=..., moved=...)` carries the theoretical cost; `Row(impl, case,
  us, work)` derives TFLOP/s and GB/s from it; `report(rows)` prints throughput
  first and latency second, because throughput is what survives a change of
  shape.
- `report()` flags any row above 100% of peak as a violation rather than a win,
  and prints the check order: unit factors, then the FLOP/byte formula, then
  work the kernel skipped, then a working set that stayed in L2, then a baseline
  doing a different operation.
- Memory peak is derived from CUDA properties. Tensor-core peak comes from a
  table keyed on device name; an absent entry prints achieved TFLOP/s with no
  percentage instead of a fraction of a guessed peak. Add the device rather than
  guessing.
- Label each row with the `KernelSpec.name` it measures, keep that string in one
  `_IMPL` constant, and assert `sel.spec.name == _IMPL` in the script's dispatch
  check. Two registered rows for one op otherwise leave the table naming one
  kernel while dispatch runs another.
- Print `sel.explain()` next to the table. The decision chain records which
  layout tags and dtypes were required, so a filtered-out backend is visible in
  the log instead of being inferred from a missing row.

## Choosing the timer

Three functions, and the choice changes the number by more than most
optimisations do.

| Use | When | Instrument |
| --- | --- | --- |
| `bench(fn)` | `fn` is idempotent | `triton.testing.do_bench`, median, L2 flushed before each replay and outside the timed events |
| `bench_stateful(fn, reset)` | each call mutates state (block alloc, ref release, eviction) | CUDA events, all intervals enqueued before any is read; `reset` must stay on the device |
| `bench_host(fn, reset)` | cost is the host waiting, not GPU work | `perf_counter` around `fn` plus a trailing synchronise |

Rules behind the table:

- Cold L2 is the default because a few MB of L2 hold a negligible slice of a
  multi-GB KV pool. A warm-L2 number is the optimistic bound and must be
  labelled as one.
- **Do not synchronise per iteration.** Draining the queue every iteration pulls
  Python and launch overhead into the window and imposes a floor — measured at
  roughly 100 µs on an A10, where the same 2 MiB working set times 26.6 µs cold
  and 100.5 µs warm-but-synchronised. Anything shorter than that floor is
  reported as the floor. `AutotuneSearcher._benchmark` has exactly this shape
  (`rapid_llm/kernels/autotune/searcher.py:99`), so its search is not
  discriminating at decode sizes; use the harness when comparing configs there.
- A CUDA-event timer around a host-blocking call reports it as nearly free: it
  measures the device timeline, and a function that stalls the launch queue for
  250 µs while issuing 3 µs of kernels shows 3 µs. `bench_host` is for those.
- `bench_host` includes its own synchronise, so print `bench_host(lambda: None)`
  as a floor row whenever results approach it.
- The first call is untimed on purpose: it forces the Triton JIT and any
  `autotune.get_best_config()` search to finish outside the measurement.

## KV cache kernels

Attention decode, the KV scatter and the block allocator all read state that a
freshly built tensor does not reproduce. Build inputs with
`benchmarks/kernels/kv_pool.py::paged_pool`, not with `torch.randn`, and never
`.contiguous()` a production input — that copy is the measurement mistake, so it
belongs in its own labelled row.

Four properties the fixture reproduces, and what each is worth on an A10 at the
Qwen3/Llama-3 decode geometry (8 KV heads, 128 dim, fp16):

- **Fragmented row table.** Sequence rows scattered across the pool, as the
  allocator hands them out. Worth 2-4% against a contiguous layout: at 2 KiB per
  cache row a random gather runs near streaming speed. Paging is not where a
  decode regression hides, and this row is the bound that says so.
- **Combined per-layer buffer with strided K/V views.** What
  `rapid_llm/modules/attention.py` passes. A split-allocation variant, which
  halves the row stride, never came out ahead — equal at three of four shapes,
  8% slower at one. Each side is already 16 cache lines, so no line's payload
  changes. Keep the row as a guard for smaller geometries (MQA, 64-dim heads),
  where a row approaches a single line.
- **Pool at least 8x L2.** Not because a smaller pool would be served from cache
  — `do_bench` flushes L2 and a decode step reads each row once — but because it
  forces the attended working set large enough that the kernel is
  bandwidth-bound rather than launch-bound. Capacity beyond that is worth ~2%.
- **fp8 through the real quantizer**, e4m3 bytes in a uint8 container with
  caller-side scales, not a cast.

Read the dtype rows as separate bottlenecks, not as a dtype column:

- On the **read** side, fp8 halves the traffic but takes only 6-10% off the
  time, so %bw falls from ~67% to ~37%. The path is dequant-bound. Read as a
  dtype variant, that lower GB/s looks like a regression when it is a different
  limit.
- On the **write** side, fp8 halves the prefill time at constant %bw: a scatter
  never decodes the bytes it copies.
- Verify an fp8 row against the *same* bytes widened by torch
  (`view(torch.float8_e4m3fn).to(torch.float16)`), which isolates the kernel's
  dequant from fp8 rounding. Comparing against an fp16 pool confounds the two.

Two more constraints specific to these kernels:

- **Separate prefill from decode; they are different operations on one kernel,
  not two points on a curve.** The KV scatter at prefill is a bandwidth kernel
  (70-76% of peak). At decode every case collapses to 4-5 µs whether it writes 1
  row or 64: it is launch-bound, %bw is meaningless in those rows, and the only
  way to make it cheaper is fewer launches, not fewer bytes.
- **Free pools between cases** (`del pool; torch.cuda.empty_cache()`), or a
  later case measures a fragmented caching allocator on top of its kernel.
- Verify a scatter by reading rows back **through the destination index**, and do
  it on the fragmented layout. A stride bug in the combined buffer shows up as K
  landing in the V half, which a sequential index list masks. Comparing the whole
  pool costs more than the benchmark.

## Allocator and other host-side state

The largest number found on the decode path is not in a kernel.
`KVCacheManager.alloc_kvcache_index` measures 24 µs on its bump-pointer fast
path and 265-275 µs once the bump cursor is invalidated — an **11x** step that
one finished request imposes on every later decode step, next to a 4 µs scatter.
It is host time spent on `nonzero(...).item()`, invisible to a CUDA-event timer.

- Time it with `bench_host`, and print the no-op floor beside it.
- Reach each state through the public API, and make the reset *complete*.
  `free_all()` restores the bump cursor, so a reset that only calls it measures
  the fast path in all three rows and reports a 1x spread.
- State the state in the case label (`bump`, `run_search`, `fragmented`). An
  allocator row without one is not reproducible.
- Assert the invariant per state — distinct free rows returned — since a
  fast-but-wrong allocator is a correctness bug, not a benchmark result.

## Feeding autotune and dispatch

- `dispatch()` ranks on `(perf, -preference_score, -priority, name)`. Install
  measurements with
  `from rapid_llm.kernels.ops.dispatch import set_perf_provider` — it is not
  re-exported from `rapid_llm.kernels.ops`. The provider returns **milliseconds**
  while the harness reports **microseconds**; divide by 1000.
- `AutotuneSearcher` persists `latency_us` (`config_store.py`), a third unit in
  the same loop. Convert deliberately at each boundary.
- An unverified row is excluded by ranking anyway: `GoldenRecord(verified=...)`
  gates a spec, and `max_abs_diff` is where a benchmark's correctness gate
  belongs, so run the check before publishing a latency.
- `LayoutRequirement(required=("kv:paged",))` filters by set algebra. Pass the
  tags the benchmark's inputs actually satisfy, and let a missing tag surface in
  `explain()` rather than silently selecting a different row.

## In-repo examples

- `benchmarks/kernels/bench_paged_decode.py` — bandwidth-bound read path, four
  constructions of one input, fp8 verified separately.
- `benchmarks/kernels/bench_kv_write.py` — scatter across prefill/decode shapes
  plus the allocator's three states through `bench_host`.
- `benchmarks/kernels/kv_pool.py` — the pool fixture and the reasons each of its
  properties survives.

Adapt the operation, cases, work formula and tolerances rather than copying a
script unchanged; the numbers quoted above are A10 results and are the thing
most likely to be stale on another device.
