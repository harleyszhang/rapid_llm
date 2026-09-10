"""TP=2 sharding of DeepSeek-V4: stacked-expert narrow loaders and head splits.

The family's parallel decisions in one two-rank pass: the 3D stacked expert
tensors narrow along their contracted dims, the grouped ``o_a_proj`` keeps
whole groups per rank (feeding the row-parallel ``o_b_proj`` a contiguous
slice), the MQA ``kv_proj`` and the compressors replicate, and the router
stays fp32 while the vocab head serves local logits. Prefill plus one decode
step run under NCCL; the concatenated vocab shards must match the
single-process forward within bf16 tolerance — the two runs tile their GEMMs
and reduce their partial sums in different orders.

Usage:
    pytest tests/distributed/test_tp2_v4.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

#: Same trimmed body as ``tests/models/test_deepseek_v4.py`` — the point here
#: is the sharding, not the architecture coverage, so the two files pin the
#: identical config and stay in sync by hand.
_BODY = {
    "model_type": "deepseek_v4",
    "vocab_size": 512,
    "hidden_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 6,
    "layer_types": [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    * 2,
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    "mlp_layer_types": ["hash_moe", "moe", "moe", "hash_moe", "moe", "moe"],
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "q_lora_rank": 64,
    "o_groups": 2,
    "o_lora_rank": 32,
    "partial_rotary_factor": 0.5,
    "sliding_window": 16,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 4,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 4,
    "swiglu_limit": 7.0,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


def _v4_payload(rank: int) -> dict:
    """Build, shard-load, and run the trimmed V4 on this rank's device.

    The checkpoint is rebuilt from the same seed on every rank, so the loaders
    see identical source tensors; ``load_weights``' coverage check then verifies
    every narrowed parameter absorbed its slice before any collective runs.
    """
    from transformers import DeepseekV4ForCausalLM

    from rapid_llm.executor.attention_metadata import AttentionMetadata
    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.config import ModelConfig
    from rapid_llm.models.registry import ModelRegistry

    tmp = Path(tempfile.mkdtemp())
    (tmp / "config.json").write_text(json.dumps(_BODY))
    config = ModelConfig.from_pretrained(tmp, max_seq_len=128)
    torch.manual_seed(0)
    hf_model = DeepseekV4ForCausalLM(config.hf_config).eval()
    # Inject router-table signal exactly like the parity fixture: a silent
    # loader drop would otherwise pass unnoticed.
    with torch.no_grad():
        for layer in hf_model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "e_score_correction_bias"):
                gate.e_score_correction_bias.normal_(0.0, 0.25)
            if hasattr(gate, "tid2eid"):
                gate.tid2eid.copy_(torch.randint(0, _BODY["n_routed_experts"], gate.tid2eid.shape))
    state = {key: value.detach().clone() for key, value in hf_model.state_dict().items()}
    save_file(state, str(tmp / "model.safetensors"), metadata={"format": "pt"})

    model = ModelRegistry.resolve("deepseek_v4").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(tmp, dequant_dtype=config.dtype))
    model.cuda().eval()

    torch.manual_seed(1)
    batch, seq_len = 2, 11
    ids = torch.randint(0, _BODY["vocab_size"], (batch, seq_len), device="cuda")

    def meta(prefill: bool, length: int) -> AttentionMetadata:
        m = AttentionMetadata()
        m.is_prefill = prefill
        m.b_seq_len = torch.full((batch,), length, dtype=torch.long)
        return m

    with torch.no_grad():
        pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1).contiguous()
        prefill = model(ids, pos, meta(True, seq_len))[:, -1, :].float().cpu()
        # One decode step over the cache the prefill just built — the second
        # half of what the sliding/compressor state machine has to carry.
        pos2 = torch.full((batch, 1), seq_len, device="cuda")
        decode = model(ids[:, :1], pos2, meta(False, 1))[:, -1, :].float().cpu()
    return {"prefill": prefill, "decode": decode}


@needs_gpus(2)
def test_tp2_sharding_matches_single_process():
    """The concatenated vocab shards equal the unsharded forward."""
    tp2 = run_on_tp_ranks(_v4_payload, tp_size=2)
    tp1 = run_on_tp_ranks(_v4_payload, tp_size=1)

    for stage in ("prefill", "decode"):
        joined = torch.cat([tp2[0][stage], tp2[1][stage]], dim=-1)
        torch.testing.assert_close(joined, tp1[0][stage], atol=6e-2, rtol=6e-2)
