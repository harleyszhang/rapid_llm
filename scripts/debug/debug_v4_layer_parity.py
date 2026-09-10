"""Component-level parity diff for one V4 decoder layer (lite vs transformers 5.15)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tests" / "models"))

import torch
from test_deepseek_v4 import _BODY, _loaded_pair
from transformers.cache_utils import DynamicCache


def diff(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    d = (a.float() - b.float()).abs()
    print(f"  {label:<28} max {d.max().item():.5f}  mean {d.mean().item():.6f}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    hf_model, model, config = _loaded_pair(tmp)

    batch, seq_len = 2, 7
    torch.manual_seed(6)
    hidden = torch.randn(
        batch, seq_len, _BODY["hc_mult"], _BODY["hidden_size"], device="cuda", dtype=config.dtype
    )
    input_ids = torch.randint(0, _BODY["vocab_size"], (batch, seq_len), device="cuda")
    pos = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous().cuda()
    pe = {
        lt: model.rotary_emb(hidden[..., 0, :], pos, lt) for lt in ("main", "compress")
    }
    valid = torch.ones(batch, seq_len, dtype=torch.bool, device="cuda")

    li = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model.reset_v4_caches()
    cache = DynamicCache(config=config.hf_config)
    layer = model.layers[li]
    layer_h = hf_model.model.layers[li]  # fixture keeps strict-fp32 modules fp32

    with torch.no_grad():
        print(f"layer {li} ({_BODY['layer_types'][li]}/{_BODY['mlp_layer_types'][li]}):")

        # 1) mHC pre
        post, comb, collapsed = layer.attn_hc(hidden.clone())
        post_h, comb_h, collapsed_h = layer_h.attn_hc(hidden.clone())
        diff(post, post_h, "attn_hc post")
        diff(comb, comb_h, "attn_hc comb")
        diff(collapsed, collapsed_h, "attn_hc collapsed")

        # 2) norm
        normed = layer._norm(collapsed.clone(), layer.input_layernorm_weight)
        normed_h = layer_h.input_layernorm(collapsed_h.clone())
        diff(normed, normed_h, "input_layernorm")

        # 3) attention
        attn = layer.self_attn(normed.clone(), pos, pe["main"], valid)
        attn_h, _ = layer_h.self_attn(
            normed_h.clone(),
            input_ids=input_ids,
            position_embeddings=pe,
            position_ids=pos,
            attention_mask=None,
            past_key_values=cache,
        )
        diff(attn, attn_h, "self_attn out")

        # 4) full layer
        actual = layer(hidden.clone(), pos, pe["main"], input_ids, valid)
        cache2 = DynamicCache(config=config.hf_config)
        model.reset_v4_caches()
        expected = layer_h(
            hidden.clone(),
            input_ids=input_ids,
            position_embeddings=pe,
            position_ids=pos,
            attention_mask=None,
            past_key_values=cache2,
        )
        diff(actual, expected, "layer out")


if __name__ == "__main__":
    main()
