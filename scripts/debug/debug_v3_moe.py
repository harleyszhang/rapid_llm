"""Localise the V3 MoE device assert: hook layer inputs and router outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CKPT = "/data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16"
HF_OVERRIDES = {"n_group": 2, "topk_group": 1, "num_experts_per_tok": 2}


def main() -> None:
    from rapid_llm.executor.attention_metadata import AttentionMetadata
    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.config import ModelConfig
    from rapid_llm.models.registry import ModelRegistry

    config = ModelConfig.from_pretrained(CKPT, max_seq_len=2048, hf_overrides=HF_OVERRIDES)
    model = ModelRegistry.resolve("deepseek_v3").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(CKPT, "cuda", dequantize_fp8=False))
    model.to("cuda").eval()

    for i, layer in enumerate(model.layers):
        layer.register_forward_pre_hook(
            lambda m, inp, idx=i: print(
                f"layer {idx} in finite:",
                bool(inp[0].isfinite().all()),
                "absmax:",
                float(inp[0].abs().max()),
            )
            if idx >= 2
            else None
        )

    moe_layer = model.layers[3].mlp
    orig_route = moe_layer._route

    def traced_route(x):
        w, i = orig_route(x)
        torch.cuda.synchronize()
        print(
            f"route: ids min {int(i.min())} max {int(i.max())} shape {tuple(i.shape)} "
            f"dtype {i.dtype} | weights finite {bool(w.isfinite().all())}"
        )
        print("route: logits finite:",
              bool(torch.nn.functional.linear(x.float(), moe_layer.gate_weight.float()).isfinite().all()))
        print("route: bias finite:", bool(moe_layer.gate_e_score_correction_bias.isfinite().all()))
        return w, i

    moe_layer._route = traced_route

    seq_len = 16
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(10, 120000, (1, seq_len), generator=g).cuda()
    cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
    meta = AttentionMetadata(
        kv_buffer=[
            torch.zeros(seq_len, 1, cache_dim, dtype=config.dtype, device="cuda")
            for _ in range(config.num_layers)
        ],
        cur_select_index=torch.arange(seq_len, dtype=torch.int32, device="cuda"),
        b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        max_actual_seq_len=seq_len,
    )
    pos = torch.arange(seq_len, device="cuda").unsqueeze(0)
    with torch.no_grad():
        logits = model(ids, pos, meta)
    torch.cuda.synchronize()
    print("prefill logits finite:", bool(logits.isfinite().all()), "absmax:", float(logits.abs().max()))


if __name__ == "__main__":
    main()
