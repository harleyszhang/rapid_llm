---
name: add-quant-method
description: Step-by-step for adding a quantisation format to rapid_llm — the Config + LinearMethod + MoEMethod triple, registry wiring (checkpoint detection, runtime scheme, aliases), packed-weight loading rules, kernel hand-off, and the correctness-then-roofline verification bar. Use when adding a weight/activation format, wiring a new checkpoint family (ModelOpt, compressed-tensors), or reviewing a quant-method change.
---

# Adding a Quantisation Method

A format is three classes and three wiring decisions. The strategy layer
(`rapid_llm/modules/quantization/`) is torch-level and testable without
kernels; the kernels it calls follow the `triton-kernel-writing` skill
and register through `kernel-conventions`. Read `docs/quantization.md`
first — its architecture section is the map, and its roofline section is
the bar the format will be judged by.

## Decide the three doors

1. **Checkpoint detection** — `config.json` carries
   `quantization_config`: `get_quant_config_from_hf` must route its
   `quant_method` string to your config class.
2. **Runtime scheme** — quantising an fp16 checkpoint on load via
   `--quantization <name>`: add to `RUNTIME_SCHEMES` in
   `rapid_llm/modules/quantization/__init__.py`.
3. **Alias or not** — checkpoints in the wild reuse names. The registry
   already maps `int8` → `BlockInt8Config`, `smoothquant` →
   `W8A8Int8Config`, and NVIDIA ModelOpt's `modelopt_fp4` → `NVFP4Config`.
   An alias is a row in `BASE_QUANTIZATION_METHODS`, not a new class.

## The triple

### XConfig(QuantizationConfig)

`rapid_llm/modules/quantization/base_config.py` defines the contract;
`awq.py` is the shortest complete sample, `fp8.py` adds a device floor,
`mxfp4.py` shows a repack step. Required:

- `get_name`, `get_supported_act_dtypes`, `get_min_capability` — the
  latter feeds device filtering; a kernel below its device minimum must
  fall back, not fail (fp8 → torch quantiser below sm89).
- `from_config` — parse the checkpoint's `quantization_config` dict.
- `get_quant_method(layer, prefix)` — dispatch per module type
  (linear → your LinearMethod, MoE block → your MoEMethod, `None` to
  leave a layer unquantised; honour `ignored` prefixes).
- `storage_dtype`, `is_packed`, `is_int4` — the loader reads these to
  decide shape and packing.
- `scale_shape(out, in)` / `shard_is_aligned(size)` — scale grids and TP
  shard alignment; a shard that splits a scale group is not servable.

### XLinearMethod(LinearMethodBase)

- `create_weights` — allocate packed weight + scales with the right grid
  (`group_size` for int4, 128×128 blocks for fp8, 16-element blocks for
  nvfp4) through `RawParameter`: **a load-time loader must never cast to
  fp16**.
- `apply` — dequantise on the fly against the kernel's expectations, or
  hand off to the true-W8A8 kernel.
- `quantize_from_fp16` (runtime schemes) — reuse `utils.py` helpers
  (`quantize_fp8_per_channel`, `quantize_int8_groupwise`) when they fit.
- `process_weights_after_loading` — any repack that must not happen on
  every forward (interleaving, transposing, merging scale pairs).

### XMoEMethod(FusedMoEMethodBase)

Same shape for expert stacks: `create_weights(block)` returns a
`dict[str, nn.Parameter]`, and `apply(block, x, topk_weights, topk_ids)`
runs the expert contract the router expects. SKIP when the format is
dense-only (NVFP4 currently is).

## Weight loading

- Packed checkpoints arrive in vendor layouts; `adapt_packed_checkpoint`
  (called from `load_weights`) rewrites them to the canonical layout —
  extend it rather than teaching each loader the vendor's quirk.
- Per-block dequant scales are stored under the `weight_scale_inv`
  suffix convention; follow it so tooling and docs agree.
- Translation tests: `tests/models/test_weight_mapping.py` covers
  per-layer loaders and coverage accounting; a new scheme means new rows
  there for the scale keys.

## Kernel hand-off

The method's job ends at tensor shapes and layouts; the kernel's starts
there. Do not write Triton inline in the strategy layer:

1. Implement the kernel per `triton-kernel-writing` (device-tiered tile
   from `resolve_tiles`, explicit int64 addressing, fallback path).
2. Register the KernelSpec row per `kernel-conventions`, with truthful
   dtype/layout tags so dispatch can filter it.
3. Measure per `kernel-microbenchmark`.

## Verification bar

1. **Strategy layer**: `tests/models/test_quant_methods.py` — registry
   lookups, weight creation shapes for every scheme, rejection paths. No
   kernels here; get this green first.
2. **Kernel accuracy**: mirror the existing accuracy suites
   (`tests/kernels/test_quantization.py`, `test_w4a16_accuracy.py`,
   `test_fp8_kv_accuracy.py`) — relative error against an fp32 reference,
   dtypes named explicitly.
3. **Golden**: record a baseline for the quantized path with
   `scripts/golden_tokens.py --quantization <scheme>` (see `write-test`).
4. **Performance**: the format earns a line in `docs/quantization.md`
   only with a `kernel-microbenchmark` table plus the roofline reading.
   What the bar expects, from the measurements already in the repo:
   - decode (M≤32) is bandwidth-bound — wins come from moving fewer
     bytes; prefill (M≥512) crosses the ridge — wins come from the MMA
     path, not from smaller storage (W8A16 wins 1.6–1.7× on A10 decode
     but loses prefill and loses H100 decode).
   - Report per device: the same format can give opposite verdicts on
     A10 (sm86) and H100 (sm90), and absolute numbers never compare
     across the two tables.
   - A negative result is a result: NVFP4 weight-only loses on every
     tested shape and the docs say so. Publish the loss with its reason
     (e.g. e2m1 unpack cost per weight) instead of omitting the row.

## Finish

- Registry rows + tests + docs (`docs/quantization.md` table and, if the
  format has a verdict, its own section) land in the same PR — a format
  that only some of the docs know about is already stale.
- If the scheme participates in serving, check the CLI surface
  (`--quantization`, `--kv-cache-dtype` in `rapid_llm/cli.py`) and the
  checkpoint-detection path end to end with a real checkpoint before
  calling it done.
