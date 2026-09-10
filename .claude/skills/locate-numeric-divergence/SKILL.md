---
name: locate-numeric-divergence
description: Localize where rapid_llm's numerics leave a reference — the ladder from whole-model golden checks to per-layer diff to single-layer mirror to component hooks, plus the capture/replay diagnosis for CUDA-graph-only corruption. Use when outputs are wrong or garbled, when a TP, quantized or optimized path stops matching eager or HF, when golden parity tests fail, or when a GPU run misbehaves only under capture.
---

# Locating a Numeric Divergence

The fix is usually cheap once the divergence is a single named function.
The work is narrowing: whole model → one layer → one component. Follow
the ladder; do not start by reading kernels.

## Step 0 — Pin the symptom to a configuration

Ask which runs disagree before asking where. The answer splits the search
in half:

- eager vs CUDA graph → capture-side state, not math (see replay probes
  below).
- TP=1 vs TP=2 → shard alignment or a collective (`shard_is_aligned`,
  o_proj/MoE collectives).
- fp16 vs quantized → the quant method, not the model.
- prefill vs decode → a path-specific kernel (the two are separate
  kernels on purpose).
- CPU vs GPU → dtype promotion on a fallback path.

## The ladder

### 1. Whole model against a committed baseline

`tests/golden/` replays recorded outputs: `test_token_parity.py` (not one
token may move), `test_logprob_parity.py` (logprob drift),
`test_deepseek_trimmed_parity.py` (trimmed checkpoint vs HF). These run
under the golden gate — an `UNVERIFIED` result means the gate did not
actually run (no CUDA / no checkpoint), which is exactly the case where a
regression slips through, so check the reason before trusting a pass.

When the reference is an HF checkpoint rather than a recording, the
agreement statistics come from
`scripts/dsv2_tp2_parity_probe.py`: greedy tokens vs a teacher-forced HF
reference, per step — how many argmax matches outright, how many are
ties, how far logprobs drift. Its numbers are what the committed golden
thresholds were calibrated from.

### 2. First layer out of band

`scripts/dsv2_layer_diff.py` runs whole-model prefill on both engines
with a forward hook on every decoder layer and diffs each layer's
residual-stream output. The first row outside the band names the layer
to hand to rung 3. Both sides are real engines at full width — this rung
answers *which layer*, never *why*.

### 3. Single layer, mirrored weights

`scripts/layer_harness.py` runs one layer in isolation; the usable trick
is `--weights mirror`: it copies transformers' same-layer random weights
into the rapid_llm layer and compares token by token — no full
checkpoint needed, so a 671B model's layer is a single-card object.

```bash
# numeric parity against the HF layer
.venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B \
    --layer 3 --weights mirror --tolerance 2e-2

# timing and dispatch only, no weights beyond config.json
.venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B --layer 0

# real weights, decode-shaped load
.venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B \
    --layer 3 --weights checkpoint --batch 4 --seq-len 512 --decode-steps 32
```

`--list-keys` prints the layer's checkpoint keys without reading tensors
— the fastest check when you suspect a translation rule.

Mirror-parity failures localize along known lines: rotary convention
(`tests/reference.py::rope_half_split`), norm placement, attention
scaling, GQA head expansion. Fix on the layer before touching anything
whole-model.

### 4. Component level

When the layer diff is nonzero, split the layer: attention vs MLP vs MoE,
each against its own reference. `scripts/debug_v4_layer_parity.py` is the
pattern — it imports the test fixtures from
`tests/models/test_deepseek_v4.py` and diffs component by component; a
new family should grow its own copy rather than a new framework.

### 5. Capture/replay probes

For "eager is right, graph replay is wrong" (or repeated replays drift),
`scripts/probe_layer_capture.py` replays one decoder layer under capture
and reads the output magnitude across replays:

| Observation across replays | Diagnosis |
| --- | --- |
| magnitude grows linearly | an in-place accumulate (`index_add_` / all-reduce) re-applies to its own result each replay |
| constant but wrong value | stale buffer read — captured pointer outlived its source |
| first replay right, later wrong | per-step Python state mutated during forward; the model must opt out of graph (`supports_cuda_graph=False`) |

Garbled stdout (replacement characters `\ufffd`) from
`scripts/cli_smoke.sh` is the same family: a stale CUDA-graph pointer,
not a sampling bug.

`scripts/debug_v3_moe.py` shows the hook shape for device asserts inside
MoE — hook layer inputs and router outputs, and the assert's own message
tells you which expert ran out of bounds.

## Discipline

- Scripts under `scripts/` in the list above are development scaffolds,
  not part of the test suite. They may take `CKPT` paths as literals and
  are expected to be edited while investigating.
- When the root cause lands, promote the minimal reproduction into a
  regression test (the bug-regression class of the `unit-test-admission`
  rule) and delete the scaffold if nothing references it. A probe without
  a bug is dead weight.
- Tolerance discipline: compare like against like (same dtype, same
  device, same shapes). A parity diff run across dtypes measures the
  dtype, not the bug.
