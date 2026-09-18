---
name: refactor-module
description: Behavior-preserving refactors and comment governance in rapid_llm — the three goals that never mix (structure, comments, optionally measured performance), the byte-identical equivalence bar, per-module batch commits, and the dtype-policy auto-ification special case. Use when restructuring a module, deleting dead code, doing a comment pass, cleaning up hardcoded dtypes, or reviewing a refactor PR that smuggled in behavior changes.
---

# Refactoring a Module

Fill in the problem statement before touching code (mixed
responsibilities / duplicated logic / dead code / a measured hotspot).
If the problem is unclear, read the code, summarize, confirm — then
start.

## Three goals, never mixed

1. **Structure**: split responsibilities, merge duplicated logic,
   delete dead code, converge the API. **Behavior must not change.**
2. **Comments**: the module docstring follows `rules/comment-style.md`
   — one-line responsibility, mechanism, `Usage:` block; inline
   comments explain constraints and reasons, never restate the code.
3. **Performance (optional)** — only when the problem statement names
   a hotspot: profile first, then change, measuring each step.

"More accurate" is not a goal of a refactor — the accuracy requirement
is *unchanged*. An accuracy problem found mid-refactor gets logged and
filed separately; fixing numerics inside a refactor is forbidden.

## Out of scope

- No external interface-semantics changes (parameters, return
  behavior); each unavoidable one is listed with its reason.
- No changes to other modules; problems found elsewhere are logged,
  not fixed.
- No "optimization" without a baseline.

## Comment governance

**Delete**: comments that restate the code; comments that no longer
match the implementation; commented-out dead code; decorative comments
(dividers, "constructor", "main flow").

**Keep and rewrite**: why-comments (non-obvious constraints,
trade-offs, pitfalls, workaround reasons); public API docstrings
(purpose, parameters, returns, exceptions — not implementation
details); TODO/FIXME only with a trigger condition or an owner,
otherwise delete; paper/algorithm references and license headers stay
untouched.

## Acceptance

1. **Behavior unchanged** (structure and comment goals): same input →
   byte-identical output tensors, and identical greedy token sequences
   on inference paths; the existing suite stays green. The dtype
   special case has its own equivalence bar — see below.
2. **Structure**: every deletion/merge answers "why"; API changes ship
   with a before/after table.
3. **Comments**: total volume visibly down; every survivor earns its
   place; deleted comments are classified (restated-code / outdated /
   wrong).
4. **Performance (only if attempted)**: before/after measurement with
   environment, workload, command and raw data; a no-gain change
   reverts — "theoretically faster" is never a reason to keep it.

## Process

- Commit in per-module batches, each independently reviewable;
  structure, comments and performance in separate commits.
- Run the suite after each batch; green before the next batch starts.
- Performance-sensitive paths get a before/after benchmark even for
  pure-structure batches.
- Regression coverage follows `write-test`; whether a case should
  exist at all follows the `unit-test-admission` rule.

## Special case — dtype policy auto-ification

A recurring refactor with its own extra bars, since it *does* change a
default: make the framework take its compute dtype from the
checkpoint's config (`torch_dtype`) by default — `auto`, aligned with
vLLM/SGLang — with an explicit user override winning when given.
Equivalence bar for this case only: on an fp16 checkpoint `auto` must
reproduce the old behavior byte-identically — that arm is the
regression guard; on a bf16 checkpoint the compute dtype legitimately
changes, so that arm is validated by the per-kernel numeric checks and
the benchmark below, not by byte-identity against the old fp16 path.

- Kernel dtypes derive from input tensors; never assume fp16. When a
  bf16 template instantiation or dispatch arm is missing, add it —
  deleting the hardcode without adding the path trades a silent lie
  for a runtime error.
- Delete framework-level fp16 hardcodes: `torch.float16` literals,
  `.half()` casts, fp16 defaults in signatures. Audit with
  `grep -rn 'torch\.float16\|\.half()' rapid_llm/`.
- Numerical-stability exceptions (fp32 softmax/accumulation) stay, but
  each carries a comment saying why; anything uncertain is listed for
  confirmation, never silently handled.
- Audit the dtype source of weights, activations and KV cache so
  "configured bf16, one stage still fp16" cannot happen.
- Test matrix: {auto, explicit} × {fp16 checkpoint, bf16 checkpoint};
  per-kernel numeric checks against a reference for both dtypes;
  same-load benchmark before/after with no significant TTFT/TPOT/TPS
  regression.

## Cross-references

- `rules/comment-style.md` — the authoritative comment/docstring rules
- `locate-numeric-divergence` — when the equivalence check fails
- `model-benchmark-and-report` — evidence format for the performance bar
- `write-test` / `unit-test-admission` — regression coverage decisions
