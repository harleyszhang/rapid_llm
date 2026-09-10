---
name: add-model
description: Step-by-step tutorial for adding a new architecture to rapid_llm — ModelSpec registration, a CausalLM subclass over the shared decoder skeleton, HF checkpoint key translation, and the verification ladder from weight-mapping tests to the CLI smoke run. Use when a checkpoint's model_type is rejected by the registry, when porting a new model family, or when extending an existing family (a new variant, a multimodal pairing).
---

# Tutorial: Adding a New Model

Goal: make a checkpoint whose `config.json` names an unknown `model_type`
load, generate correctly, and stay verifiable. Running an already-supported
architecture on one more checkpoint needs none of this — point
`--model-dir` at it and go.

One invariant holds everything together: the model class says how to
*translate* checkpoint keys, and `load_weights` refuses to finish until
every parameter has been covered. A missed rule fails loudly at load time,
not silently at generation time.

## Step 1 — Register the architecture

`ModelRegistry._SPECS` in `rapid_llm/models/registry.py` is the single
source of truth; add a `ModelSpec` row:

```python
ModelSpec("myarch", "rapid_llm.models.myarch:MyArchModel"),
```

- `model_type` must match the string in `config.json` (lookup is
  case-insensitive).
- `implementation` is `"module.path:ClassName"`, imported lazily on first
  use — keep the module import-clean.
- `is_multimodal=True` for vision-language models (see Step 3b).
- `supports_cuda_graph=False` when forward mutates Python-side state per
  step. DeepSeek-V4's per-layer rolling caches replay incorrectly, which is
  why its row opts out; everything else defaults to captured decode.

`ModelRegistry.resolve` lists every supported type in its error, so a
wrong string is self-diagnosing.

## Step 2 — Implement the class

Subclass `CausalLM` (`rapid_llm/models/base.py`) and let the shared
skeleton do the stacking: `__init__` builds embedding, layers, norm and
LM head from the existing `modules/` building blocks. Override only where
the architecture differs:

- **Class attrs**: `qkv_bias` (True for Qwen2), `use_qk_norm` (Qwen3),
  `rotary_class`, `hf_prefix` (strip prefix of checkpoint keys, default
  `"model."`), `packed_modules_mapping` (Step 3).
- **Build hooks**: `_build_attention` / `_build_mlp` /
  `_build_decoder_layer` — the seams where a family's quirks live (MLA for
  DeepSeek, MoE MLP for Qwen3-MoE).
- **forward**: signature is fixed by the engine —
  `(input_ids, position_ids, atten_info, inputs_embeds, logits_positions)`.
  `atten_info.is_prefill` decides prefill vs decode, so a single-token
  prompt is still a prefill.

`rapid_llm/models/llama.py` is the minimal reference; read
`rapid_llm/models/qwen2.py` next for how small the deltas usually are.

## Step 3 — Teach the key translator

`CausalLM.translate_weight_key` strips `hf_prefix` and calls
`weights.translate_text_key`, which fuses what the skeleton fused:

- `packed_modules_mapping` (class attr) rules gate/up → `gate_up_proj`,
  q/k/v → `qkv_proj`, per-expert → stacked MoE experts. This table is
  production: tests import it directly, so a rule added here is a rule
  tested.
- Layernorms are folded by suffix in `weights._FLATTENED`
  (`q_norm.weight` → `q_norm_weight`); MLA's latent norms are the same
  mechanism.
- **Coverage is enforced**: `weights._verify_coverage` raises unless every
  parameter was filled. A renamed key or a missing rule surfaces here —
  read the error, it names the parameter.

Add rows to the translation tests alongside: `tests/models/test_weight_mapping.py`
drives the production table (key → `(name, shard_id)` pairs, GQA shard
boundaries), and `test_weight_parity.py` / `test_checkpoint_index.py`
cover loading from real key shapes.

Quantized checkpoints: `quantize_from_fp16` path is `quantize_()` on the
model; packed layouts go through `adapt_packed_checkpoint` in
`load_weights`. See the `add-quant-method` skill for the quant side.

### Step 3b — Multimodal models

Subclass `MultiModalCausalLM` (`rapid_llm/models/interfaces.py`) instead:
build `self.language_model` (a `CausalLM`), implement `encode_vision` and
`placeholder_token_ids`, and set `weight_prefixes` — `(checkpoint prefix,
rapid_llm prefix)` pairs with `LANGUAGE_MODEL_PREFIX` handing the rest to
the text model's own translation. The placeholder scatter is
`merge_multimodal_embeddings`, which raises on a count mismatch instead of
padding silently. `qwen3_vl.py` is the reference; LLaVA shows the
`USER: <image>` template variant.

## Step 4 — Verification ladder

Climb in this order; each rung is cheaper than the one below it and
catches a different class of error.

1. `pytest tests/models/test_weight_mapping.py -k <new>` — translation
   rules are pure functions; get them green first.
2. **Single layer against transformers** — no full checkpoint needed:
   `.venv/bin/python scripts/layer_harness.py --model-dir <ckpt> --layer 3 --weights mirror --tolerance 2e-2`
   mirrors the HF layer's random weights and diffs token by token. This is
   the fastest place to catch rotary conventions (`rope_half_split`),
   RMSNorm placement, and attention scaling.
3. `scripts/verify_models.py --model-dir <ckpt> --kind text` — full
   greedy generation, checks non-empty output and a clean stop reason.
4. `bash scripts/cli_smoke.sh <name>` — the real CLI (`chat` / `vl-chat`
   auto-detected); rejects empty output and replacement characters
   (garbled text is the signature of a stale CUDA-graph pointer, not a
   sampling issue).
5. Golden + accuracy: record a baseline with `scripts/golden_tokens.py`
   and run the eval tier — workflows are in the `write-test` skill.
6. Numbers for docs: `benchmark-and-report` skill.

## Common failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `load_weights` coverage error | key rename missing | add the rule / suffix to the translator |
| Mirror diff diverges on layer 0 | rotary or norm convention | compare against `tests/reference.py::rope_half_split` |
| Garbled CLI output after graph capture | per-step Python state replayed | set `supports_cuda_graph=False` on the spec |
| Vision embeddings count mismatch | processor/config disagree | check `patch_size` and feature-selection strategy |
| Registry rejects an alias type | model_type misspelling or family shares one class | add a second `ModelSpec` row pointing at the same class (the `deepseek_v2`/`deepseek_v3` precedent) |
