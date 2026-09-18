---
name: add-op-backend
description: Add a backend implementation to an existing op in rapid_llm's kernel layer — one op interface with many implementations under it, the KernelSpec row with truthful conditions, dispatch ranking and fallback behavior, and the three integration paths (Triton, CUDA C++ extension, external library). Use when adding an implementation for an existing op, integrating FlashInfer/FlashAttention/DeepGEMM-style external kernels, wiring a CUDA extension, or debugging why dispatch picked the wrong row.
---

# Adding an Op Backend

Three tiers decide what runs: the kernels in `rapid_llm/kernels/ops/`,
the `KernelSpec` rows declaring what a backend can serve in
`rapid_llm/kernels/backends/`, and `dispatch()` in
`rapid_llm/kernels/ops/`. One op has one function signature; many
implementations hang under it. **Adding an implementation = a new
implementation file + one KernelSpec row** — business code never grows
an if-else, because the selection rules live in the registry, not in
call sites.

## The KernelSpec row

A row states what the implementation needs to be eligible: priority
(higher wins), and conditions — hardware (`sm`), dtype, layout tags,
and required libraries. Dispatch walks rows from highest priority down
and picks the first whose conditions all hold; `sel.explain()` records
why each row was skipped, so a filtered-out backend is visible in the
log instead of inferred from a missing row (see
`kernel-microbenchmark` for why tables must name registry rows).

Conditions must be truthful: a row that overclaims eligibility gets
selected and then fails or, worse, silently computes the wrong thing.
A row that underclaims never runs.

## The three integration paths

All three register identically; none is special to dispatch.

1. **Triton (default for new implementations)** — follow
   `triton-kernel-writing`: tile policy via `resolve_tiles`, int64
   addressing, a fallback or `CapabilityRequirement` below the device
   floor. First-call JIT compilation is expected behavior.
2. **CUDA C++ extension** — build as an extension module, then register
   like any other backend. This path must stay proven end to end:
   one example op that went write → compile → register → selected by
   dispatch. New business kernels in CUDA are not required; a working
   example is.
3. **External library (optional dependency)** — FlashAttention /
   FlashInfer / FlashMLA for attention families, DeepGEMM for GEMM,
   DeepEP for MoE communication. When the library is absent, the row
   is filtered out — a missing library must never crash the process,
   and the skip reason shows up in `explain()`. Version pinning and
   install scripts are a separate task; this path covers registration
   and degradation only.

The pure-PyTorch reference implementation is always registered and
always eligible: it is the correctness baseline every other row is
checked against.

## Verification bar

1. **Extensibility demo**: adding an implementation touches only its
   file plus one registration call.
2. **Dispatch correctness (unit-tested)**: on a condition miss
   (library/hardware/dtype/shape), dispatch falls to the next-priority
   row; when every row misses, the error names the op and lists each
   row's skip reason.
3. **CUDA path**: the example extension op is selected by dispatch and
   produces correct results.
4. **Numerical parity**: every implementation of every touched op
   matches the PyTorch reference (`allclose`, tolerance per dtype),
   parametrized over the full (op × implementation) matrix — tests
   follow `write-test`.
5. **Degradation**: uninstall each external library in turn; the full
   suite still passes.
6. **Performance record**: per op, a microbenchmark per implementation
   (method per `kernel-microbenchmark`), plus a note of which row
   dispatch actually selects in production shapes and why.

## Out of scope

- No changes to model forward logic — only the op call sites get
  retargeted.
- No AOT precompilation of Triton kernels (separate project if
  needed).
- No version locking or install tooling for external libraries.

## Deliverables

1. Design note where the op family is new: interface, registry,
   dispatch flow, and one concrete priority chain (e.g. MLA attention:
   FlashMLA → FlashInfer → Triton → PyTorch).
2. Support matrix: op × implementation × conditions × status.
3. Test report and microbenchmark data per `model-benchmark-and-report`.
4. README section: how to add an implementation — one complete
   registration example each for Triton, CUDA extension and external
   library.
5. Git: registry/dispatch infrastructure, per-op migration, per-library
   integration, and the CUDA example as separate commits.
