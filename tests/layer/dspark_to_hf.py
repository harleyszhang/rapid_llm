"""DSpark V4 checkpoint -> transformers DeepseekV4ForCausalLM, host memory.

The reference sides that consume the DSpark-format Flash checkpoint — the
fp32 CPU oracle in :mod:`tests.layer.deepseek` and the bf16-on-disk
variant in :mod:`tests.layer.convert_v4_hf` — share this loader.
transformers has no reader for the DSpark layout, so :func:`load_dspark_hf`
renames keys through :func:`rapid_llm.models.deepseek_v4.adapt_dspark_key`
plus the few leaves where transformers' module tree differs
(:data:`_LAYER_RENAMES`), widens the fp8 linears with
:func:`rapid_llm.executor.weight_utils.dequant_block_fp8`, rebuilds the
MXFP4 expert stacks one layer at a time (:func:`dequant_mxfp4`) so the
dequantised intermediates never double up in host memory, and re-creates
the non-persistent RoPE tables the checkpoint never carries.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.models.deepseek_v4 import adapt_dspark_key

_EXPERT_WEIGHT = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")

#: Key leaves where the two HF dialects disagree; everything else survives
#: :func:`adapt_dspark_key` verbatim (checked against transformers' module
#: tree — q/kv norms, compressor, indexer, hyper-connections share names).
_TOP_RENAMES = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "norm_weight": "model.norm.weight",
    "hc_head.hc_fn": "model.hc_head.hc_fn",
    "hc_head.hc_base": "model.hc_head.hc_base",
    "hc_head.hc_scale": "model.hc_head.hc_scale",
}
_LAYER_RENAMES = {
    "input_layernorm_weight": "input_layernorm.weight",
    "post_attention_layernorm_weight": "post_attention_layernorm.weight",
    "mlp.gate_weight": "mlp.gate.weight",
    "mlp.gate_e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "mlp.gate_tid2eid": "mlp.gate.tid2eid",
    "self_attn.compressor.indexer.weights_proj.weight": (
        "self_attn.compressor.indexer.scorer.weights_proj.weight"
    ),
}


def hf_key(dspark_key: str) -> str:
    """DSpark checkpoint key -> transformers ``DeepseekV4ForCausalLM`` name."""
    lite = adapt_dspark_key(dspark_key)
    if lite.startswith("layers."):
        layer, _, leaf = lite[len("layers.") :].partition(".")
        return f"model.layers.{layer}.{_LAYER_RENAMES.get(leaf, leaf)}"
    return _TOP_RENAMES.get(lite, lite)


def load_dspark_hf(
    model,
    checkpoint: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Fill a (meta) ``DeepseekV4ForCausalLM`` straight from the DSpark files.

    Every floating tensor is dequantised/cast up to *dtype* on the way in
    (the fp32 oracle runs fp32 end to end; the on-disk conversion casts to
    bf16); integer tensors keep their stored dtype.
    """
    from safetensors.torch import safe_open

    from rapid_llm.executor.weight_utils import dequant_block_fp8
    from rapid_llm.modules.quantization.mxfp4 import dequant_mxfp4

    ckpt = Path(checkpoint)
    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    filled: set[str] = set()

    def assign(hf_key: str, tensor: torch.Tensor) -> None:
        """Swap the parameter wholesale — meta tensors reject ``.data =``."""
        if hf_key not in targets:
            raise KeyError(f"converted key has no HF parameter: {hf_key}")
        module_path, _, leaf = hf_key.rpartition(".")
        module = model.get_submodule(module_path)
        tensor = tensor.to(dtype) if tensor.is_floating_point() else tensor
        if leaf in dict(module.named_parameters(recurse=False)):
            setattr(module, leaf, nn.Parameter(tensor, requires_grad=False))
        else:
            module.register_buffer(leaf, tensor)
        filled.add(hf_key)

    t0 = time.time()
    for fname in sorted(set(weight_map.values())):
        with safe_open(ckpt / fname, "pt") as f:
            for key in f.keys():  # noqa: SIM118
                if key.endswith(".scale"):
                    continue  # consumed with its .weight twin below
                if _EXPERT_WEIGHT.match(key):
                    continue  # experts are rebuilt layer-by-layer below
                tensor = f.get_tensor(key)
                if (
                    key.endswith(".weight")
                    and (twin := key.removesuffix(".weight") + ".scale") in weight_map
                ):
                    tensor = dequant_block_fp8(tensor, f.get_tensor(twin), dtype)
                assign(hf_key(key), tensor)

    # Expert stacks: build each layer's [E, 2*inter, hidden] gate_up and
    # [E, hidden, inter] down one layer at a time so the dequantised
    # intermediates never double up in host memory. safetensors handles stay
    # open across experts — one file typically carries many.
    handles: dict[str, object] = {}

    def open_file(fname: str):
        if fname not in handles:
            handles[fname] = safe_open(ckpt / fname, "pt")
        return handles[fname]

    for layer, block in enumerate(model.model.layers):
        experts = block.mlp.experts
        e_total, gu_shape = experts.gate_up_proj.shape[0], experts.gate_up_proj.shape[1:]
        inter = gu_shape[0] // 2
        gate_up = torch.empty(e_total, *gu_shape, dtype=dtype)
        down = torch.empty(e_total, *experts.down_proj.shape[1:], dtype=dtype)
        for e in range(e_total):
            parts = {}
            for nm in ("w1", "w2", "w3"):
                wk = f"layers.{layer}.ffn.experts.{e}.{nm}.weight"
                sf = open_file(weight_map[wk])
                parts[nm] = (
                    sf.get_tensor(wk),
                    sf.get_tensor(wk.removesuffix(".weight") + ".scale"),
                )
            # HF's gate_up packs gate (w1) first, up (w3) second.
            gate_up[e, :inter] = dequant_mxfp4(*parts["w1"])
            gate_up[e, inter:] = dequant_mxfp4(*parts["w3"])
            down[e] = dequant_mxfp4(*parts["w2"])
        assign(f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up)
        assign(f"model.layers.{layer}.mlp.experts.down_proj", down)
        print(f"  experts layer {layer} dequantised ({time.time() - t0:.0f}s)", flush=True)
    handles.clear()

    # Rope frequency tables are non-persistent buffers the checkpoint never
    # carries; rebuild every rotary module with real tensors.
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    for name, module in list(model.named_modules()):
        if isinstance(module, DeepseekV4RotaryEmbedding):
            parent_path, _, leaf = name.rpartition(".")
            parent = model if not parent_path else model.get_submodule(parent_path)
            setattr(parent, leaf, DeepseekV4RotaryEmbedding(model.config))

    non_persistent: set[str] = set()
    for name, module in model.named_modules():
        for buf in module._non_persistent_buffers_set:
            non_persistent.add(f"{name}.{buf}" if name else buf)
    missing = (set(targets) - filled) - non_persistent
    if missing:
        raise KeyError(f"HF parameters never filled: {sorted(missing)[:5]}")
