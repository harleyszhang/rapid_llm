---
name: write-test
description: How to write tests in rapid_llm — tier and marker selection, file placement and the conftest collection policy, what to assert against, and the golden recording workflow. Use when adding or moving tests, when a test skips on a machine that should run it, when a new test module fails collection on CPU, or when recording or refreshing a golden baseline.
---

# Writing Tests

First read the `unit-test-admission` rule (`.claude/rules/unit-test-admission.md`)
— it decides *whether* a case should exist. This skill covers where the
case lives and how it runs.

## Tiers and markers

Tiers are selected by marker; every target auto-skips what the machine
lacks and reports why. Gating lives in `tests/conftest.py`, not in the
targets.

| Tier | Marker / path | Needs | Make target |
| --- | --- | --- | --- |
| CPU | `not gpu and not weights` | nothing | `make test-cpu` |
| Kernel | `gpu` (auto in `tests/kernels/`) | CUDA + triton | `make test-gpu` |
| Weights | `weights` | a checkpoint | `make test-weights MODEL_DIR=...` |
| Golden | `tests/golden/` | CUDA + checkpoint | `make test-golden` |
| Eval | `tests/evals/` | checkpoint named by each config | `make test-eval` |
| Serving | engine + entrypoints | nothing (fake/stub engines on CPU) | `make test-serving` |

`make test-fast` drops `slow`; CI runs the CPU tier with
`--cov-fail-under=48` as a darkness check, and the self-hosted GPU job
runs everything. When in doubt about collection safety, run
`make test-cpu` — it is the floor CI enforces.

## Where a file goes, and the collection policy

Put the case in the subsystem's existing file or directory. New
`tests/<dir>/` trees are the exception, not the pattern.

Two automatic rules in `tests/conftest.py` decide what can even be
imported on a machine without CUDA:

- Everything under `tests/kernels/` gets the `gpu` marker by location, so
  new kernel tests inherit the requirement automatically.
- A module that imports the Triton runtime at module scope must be listed
  in `_GPU_RUNTIME_FILES`, or `pytest_ignore_collect` cannot save CPU
  collection — the marker is evaluated only after the import runs. Keep
  CPU-importable unit tests in neighbouring directories instead of
  excluding whole trees.

Golden tests never silently skip: without CUDA or a checkpoint they
report as `UNVERIFIED` (an xfail, run=False — yellow in CI, not green).
`RAPID_LLM_GOLDEN_STRICT=1` turns that into a hard failure instead.

Fixtures and process-wide policy you get for free:

- autouse seeding (`torch.manual_seed(0)`) — unseeded random inputs make
  failures non-reproducible.
- `model_dir` session fixture — resolves the checkpoint and skips the
  test with the actual problem when unusable.
- `RAPID_LLM_FROZEN_RANK=0` set at conftest *import*: a developer's frozen
  dispatch records must never flip the suite, and a module-scoped engine
  fixture would otherwise cache a dispatch before the function-scoped
  autouse fixture runs. Do not re-enable frozen ranking outside a test
  that opts in via `monkeypatch`.

## What to assert against

- Torch references live in `tests/reference.py` (`varlen_causal_attention`,
  `paged_decode_attention`, `rmsnorm`, `skip_rmsnorm`, `swiglu`,
  `fused_moe_reference`, `nvfp4_dequant`, `rope_half_split`). Import
  them; do not copy one into a test file.
- Tolerances are named module-level constants (`_RTOL`, `_ATOL`) matching
  the same kernel's existing cases — no scattered magic numbers. Not the
  kernel under test is the rule: a reference implemented with the same
  helper proves nothing.
- Parametrize boundary shapes and pin negative contracts. The
  `tests/kernels/test_kv_cache_ops.py` pattern — `decode-single-token`,
  `ragged-count`, plus cases asserting *untouched rows stay untouched*
  and *destination order is not assumed monotonic* — is the shape to
  imitate.
- Construct inputs that can expose the guarded failure mode (boundary
  alignment, ragged counts, hard numerics). Sampling only "normal"
  random shapes tests the least interesting path.

## Recording a golden baseline

`tests/golden/cases.py` is the single source of truth for cases —
the recording script and the pytest side both read it, so a case added
for a script is immediately replayable by the test.

```bash
# record one checkpoint (or: make golden-update MODEL_DIR=...)
.venv/bin/python scripts/golden_tokens.py --save tests/golden/data/Qwen2.5-0.5B.json

# a quantized path gets its own file
.venv/bin/python scripts/golden_tokens.py --save tests/golden/data/Qwen3-0.6B_int8.json \
    --model-dir my_weight/Qwen3-0.6B --quantization int8
```

The diff of a re-recorded baseline is *the model's output* — read it
before committing. An unexpected change there is the regression the tier
exists to catch, not noise to accept. `test_token_parity.py` and
`test_logprob_parity.py` replay the committed JSON; the naming supports
both checkpoint spellings (`Qwen2.5-0.5B` locally, `Qwen2___5-0___5B`
under a shared root), so a byte-identical copy still runs the gate.

## Quick checks before calling it done

1. `pytest tests/<path> -m "not gpu"` on the new file alone.
2. `make test-cpu` — proves the collection policy still holds on a
   machine without CUDA.
3. `make lint` — ruff owns style, including docstring code blocks.
