---
name: benchmark-and-report
description: Run engine-level benchmarks in rapid_llm and write the numbers into docs under the repository's evidence rules — device-separated tables, reproduction commands, JSON logs, and known regressions. Use when measuring a feature end to end, producing or reviewing a performance claim, updating benchmark docs or release notes, or re-recording the README GIFs.
---

# Benchmark and Report

Two layers, two skills. Kernel microbenchmarks belong to
`kernel-microbenchmark` (do not re-implement its timing harness here).
This skill covers engine- and model-level measurement and — the part that
outlives the run — how the numbers enter the docs.

## Run

- `benchmarks/engine/run.py` — the unified CLI: `scheduler`
  (`continuous`, plus the scheduler commands), `optimizations` (feature
  A/B matrix), `quant`, `cpu`.
- `make bench-continuous` — the continuous-vs-static table printed in
  `docs/continuous_batching.md`.
- `benchmarks/suites/run.py` — multi-model matrices: per-model
  `compare_configs` / `tp_configs` sweeps (TP shapes, e2e pool sizes,
  vision flags) with shlex-quoted subprocesses; `benchmarks/suites/qk_norm.py`
  is the QK-RMSNorm A/B.
- README GIFs are recordings, not design tools:
  `scripts/gen_<feature>_gif.py` (e.g. `make serving-gif`). Re-record
  after the behavior shown changed, not before.

Every run states its own configuration: model, device, dtype/scheme,
input and output lengths, batch/concurrency, and the optimization
switches in play. A number missing its configuration is not a result.

## Store the raw data

Logs land under `docs/benchmark_logs/<domain>/<name>_<YYYYMMDD>/` —
`kernels/`, `engine/features_YYYYMMDD/`, `quantization/e2e_matrix_*`,
`qk_norm/`, `serving/`, `models/`, `overlap/`, `accuracy/`. Keep the
JSON as produced, including the environment record; a table whose source
JSON carries no environment cannot be re-read six months later, and the
docs deliberately mark such historical numbers as "do not cite".

## Write the docs

The repository's evidence rules, from `docs/README.md`:

- Reproduction command, environment, raw-data path and limitations stay
  with the conclusion — always.
- Performance claims name device, model, precision, input/output lengths,
  concurrency and enabled optimizations.
- Never generalize a single run into a universal speedup.

`docs/optimization_features.md` is the pattern for a feature section:
what it does, a figure, the reproduction command, the result table, and
which release/hardware the numbers belong to. Known negative results are
part of the section, not an omission — the docs already publish drop-in
cases (e.g. optimizations with no gain on a given shape) so the next
reader does not re-discover them.

`docs/benchmark_models.md` is the pattern for comparative matrices:

- Devices get separate sections because the same format can give
  opposite verdicts across them (the W8A16 story: A10 decode wins, H100
  decode loses). Absolute numbers never compare across tables; only
  relative positions within one table.
- Each table names its stance: date, software stack, shapes/workload,
  instrument (`triton.testing.do_bench`, engine loop, ...), and, when
  superseded, a "do not cite" note pointing at the replacement data.
- Verdicts come with their mechanism — the roofline reading (bandwidth
  vs compute bound, ridge point) — not just a ratio.

## Checklist before the claim ships

1. Correctness gate passed on the exact build being measured (an
   unverified row never serves — see `kernel-microbenchmark` for the
   kernel analogue).
2. Raw JSON committed under `docs/benchmark_logs/`, environment included.
3. Table states device, model, scheme, lengths, concurrency, switches.
4. Reproduction command in the doc runs as written.
5. Negative or flat results included with the same prominence as wins.
6. If a release note or `optimization_features.md` claims it, the claim
   matches the JSON; if the JSON moved, the doc moved with it.
