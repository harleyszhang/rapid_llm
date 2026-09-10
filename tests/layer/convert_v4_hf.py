"""Standalone DSpark V4 checkpoint -> transformers bf16 safetensors on disk.

Self-contained on purpose: every DSpark -> transformers key mapping lives in
:data:`_TOP` / :data:`_LAYER` / :func:`_expert_parts` below (merged from
``rapid_llm.models.deepseek_v4``'s rename tables plus the transformers-dialect
fixups), so the result cannot silently depend on which rename table an import
resolves to. fp8 e4m3 linears dequantise with their e8m0 128x128 block scales,
MXFP4 routed experts are rebuilt one layer at a time into the fused
``gate_up_proj``/``down_proj`` the transformers module tree holds, and the run
verifies itself three ways: probe asserts on the table, a meta/missing sweep
over the filled model, and a reopen of the written shards.

    rapid_llm venv (CPU only, no GPU needed):
        python -X pycache_prefix=/tmp/pyc_v4conv -m tests.layer.convert_v4_hf
Output: /data/shared/llm_weights/DeepSeek-V4-Flash-6layers-hf-bf16-v2
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CKPT = "/data/shared/llm_weights/DeepSeek-V4-Flash-6layers"
OUT = Path("/data/shared/llm_weights/DeepSeek-V4-Flash-6layers-hf-bf16-v2")
DTYPE = torch.bfloat16

#: Keys outside the decoder stack.
_TOP = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "embed.weight": "model.embed_tokens.weight",
    "norm_weight": "model.norm.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
    "hc_head.hc_fn": "model.hc_head.hc_fn",
    "hc_head.hc_base": "model.hc_head.hc_base",
    "hc_head.hc_scale": "model.hc_head.hc_scale",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}

#: Layer-local leaves, keyed after stripping ``layers.N.``.
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.attn_sink": "self_attn.sinks",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wkv.weight": "self_attn.kv_proj.weight",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    "attn.indexer.wq_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.weights_proj.weight": "self_attn.compressor.indexer.scorer.weights_proj.weight",
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
}

_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|scale)$")
_LAYER_PREFIX = re.compile(r"^layers\.(\d+)\.(.+)$")

#: E2M1 lookup for one nibble (sign bit set -> negative half of the table).
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def hf_key(dspark_key: str) -> str:
    if dspark_key in _TOP:
        return _TOP[dspark_key]
    m = _LAYER_PREFIX.match(dspark_key)
    if m is None:
        raise KeyError(f"no mapping for top-level key: {dspark_key}")
    leaf = m.group(2)
    if leaf not in _LAYER:
        raise KeyError(f"no mapping for layer leaf: {dspark_key}")
    return f"model.layers.{m.group(1)}.{_LAYER[leaf]}"


def e8m0_to_fp32(t: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, t.view(torch.uint8).to(torch.int32).float() - 127.0)


def dequant_fp8(w, scale, dtype) -> torch.Tensor:
    """fp8 e4m3 [out, in] with an e8m0 128x128 block scale."""
    w = w.to(torch.float32)
    s = e8m0_to_fp32(scale).repeat_interleave(128, 0).repeat_interleave(128, 1)
    return (w * s[: w.shape[0], : w.shape[1]]).to(dtype)


def dequant_mxfp4(packed, scale, dtype) -> torch.Tensor:
    """byte-packed MXFP4 [out, in/2] (even K low nibble) + e8m0 32-wide scale."""
    b = packed.view(torch.uint8).to(torch.int64)
    vals = torch.empty(b.shape[0], b.shape[1] * 2, dtype=torch.float32)
    vals[:, 0::2] = _E2M1[b & 0xF]
    vals[:, 1::2] = _E2M1[b >> 4]
    s = e8m0_to_fp32(scale).repeat_interleave(32, 1)
    return (vals * s[:, : vals.shape[1]]).to(dtype)


def assign(model, targets, filled, hf_name: str, tensor: torch.Tensor) -> None:
    if hf_name not in targets:
        raise KeyError(f"converted key has no transformers parameter: {hf_name}")
    module_path, _, leaf = hf_name.rpartition(".")
    module = model.get_submodule(module_path)
    if leaf in dict(module.named_parameters(recurse=False)):
        setattr(module, leaf, nn.Parameter(tensor, requires_grad=False))
    else:
        module.register_buffer(leaf, tensor)
    filled.add(hf_name)


def main() -> int:
    from safetensors import safe_open
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from rapid_llm.models.config import ModelConfig

    # 1) table self-check: the exact failures the earlier run hid
    assert hf_key("layers.0.attn.attn_sink") == "model.layers.0.self_attn.sinks"
    assert hf_key("head.weight") == "lm_head.weight"
    assert hf_key("layers.0.attn.wq_a.weight") == "model.layers.0.self_attn.q_a_proj.weight"
    assert hf_key("layers.0.ffn.gate.tid2eid") == "model.layers.0.mlp.gate.tid2eid"
    assert (
        float(e8m0_to_fp32(torch.tensor([127], dtype=torch.uint8).view(torch.float8_e8m0fnu))[0])
        == 1.0
    )

    config = ModelConfig.from_pretrained(CKPT, max_seq_len=2048).hf_config
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(config)
    targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    ckpt = Path(CKPT)
    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    filled: set[str] = set()

    def assign_key(key: str, tensor: torch.Tensor) -> None:
        if tensor.is_floating_point():
            twin = key.removesuffix(".weight") + ".scale"
            if key.endswith(".weight") and twin in weight_map:
                tensor = dequant_fp8(tensor, safe.get_tensor(twin), DTYPE)
            else:
                tensor = tensor.to(DTYPE)
        assign(model, targets, filled, hf_key(key), tensor)

    t0 = time.time()
    for fname in sorted(set(weight_map.values())):
        with safe_open(ckpt / fname, "pt") as safe:
            for key in safe.keys():  # noqa: SIM118
                if _EXPERT.match(key) or key.endswith(".scale"):
                    continue
                assign_key(key, safe.get_tensor(key))
    print(f"linears + buffers done ({time.time() - t0:.0f}s)", flush=True)

    # Routed experts: rebuild each layer's fused gate_up/down one layer at a
    # time so the dequantised intermediates never double up in host memory.
    handles: dict[str, object] = {}

    def open_file(fname: str):
        if fname not in handles:
            handles[fname] = safe_open(ckpt / fname, "pt")
        return handles[fname]

    for layer, block in enumerate(model.model.layers):
        experts = block.mlp.experts
        e_total = experts.gate_up_proj.shape[0]
        inter = experts.gate_up_proj.shape[1] // 2
        gate_up = torch.empty(e_total, *experts.gate_up_proj.shape[1:], dtype=DTYPE)
        down = torch.empty(e_total, *experts.down_proj.shape[1:], dtype=DTYPE)
        for e in range(e_total):
            parts = {}
            for nm in ("w1", "w2", "w3"):
                wk = f"layers.{layer}.ffn.experts.{e}.{nm}.weight"
                sf = open_file(weight_map[wk])
                parts[nm] = (
                    sf.get_tensor(wk),
                    sf.get_tensor(wk.removesuffix(".weight") + ".scale"),
                )
            gate_up[e, :inter] = dequant_mxfp4(*parts["w1"], DTYPE)
            gate_up[e, inter:] = dequant_mxfp4(*parts["w3"], DTYPE)
            down[e] = dequant_mxfp4(*parts["w2"], DTYPE)
        assign(model, targets, filled, f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up)
        assign(model, targets, filled, f"model.layers.{layer}.mlp.experts.down_proj", down)
        print(f"  experts layer {layer} dequantised ({time.time() - t0:.0f}s)", flush=True)
    handles.clear()

    # 2) model sweep: everything filled, nothing meta, sane dtypes
    non_persistent: set[str] = set()
    for name, module in model.named_modules():
        for buf in module._non_persistent_buffers_set:
            non_persistent.add(f"{name}.{buf}" if name else buf)
    missing = (set(targets) - filled) - non_persistent
    assert not missing, f"never filled: {sorted(missing)[:8]}"
    state = model.state_dict()
    assert set(state) == filled, sorted(set(state) ^ filled)[:8]
    for k, v in state.items():
        assert not v.is_meta, f"meta tensor survived: {k}"
    print(f"model sweep ok: {len(state)} keys, all real ({time.time() - t0:.0f}s)", flush=True)

    if getattr(model.config, "quantization_config", None) is not None:
        delattr(model.config, "quantization_config")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("model-*.safetensors"):
        old.unlink()
    model.save_pretrained(
        OUT, safe_serialization=True, max_shard_size="12GB", save_original_format=False
    )
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = ckpt / f
        if src.exists():
            shutil.copy(src, OUT / f)

    # 3) reopen the written index and require the exact model key set
    idx = json.loads((OUT / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(idx) == set(state), (
        f"written keys differ: extra {sorted(set(idx) - set(state))[:4]} "
        f"missing {sorted(set(state) - set(idx))[:4]}"
    )
    with safe_open(OUT / sorted(set(idx.values()))[0], "pt") as f:
        sample = sorted(f.keys())[:3]
    print(f"shard sample keys: {sample}")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"OK: {OUT} {size / 2**30:.1f} GiB ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
