"""Smoke test: the real DSpark DeepSeek-V4-Flash-6layers checkpoint over TP-2 A10s.

The six-layer Flash carries routed experts on every layer (hash and top-k
alike), ~24 GiB of MXFP4 stacks in total, so a single 22 GiB card cannot hold
it — the smoke run rides the tensor-parallel grid the engine would use.

Checks the full wiring this task built: DSpark key adaptation, fp8 weight-only
projections, MXFP4 experts, the hash/top-k routers, and all three attention
types over their compressors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CKPT = "/data/shared/llm_weights/DeepSeek-V4-Flash-6layers"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tests" / "distributed"))

from tp_harness import needs_gpus, run_on_tp_ranks

from rapid_llm.executor.attention_metadata import AttentionMetadata
from rapid_llm.executor.loader import materialise_parameters
from rapid_llm.executor.weight_utils import hf_weights_iterator
from rapid_llm.models.config import ModelConfig
from rapid_llm.models.registry import ModelRegistry


def _v4_payload(rank: int) -> dict:
    config = ModelConfig.from_pretrained(CKPT, max_seq_len=2048)
    assert config.quant is not None, "a DSpark checkpoint must resolve a quant config"

    model = ModelRegistry.resolve("deepseek_v4").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(
        hf_weights_iterator(
            CKPT, "cuda", dequantize_fp8=config.quant is None, dequant_dtype=config.dtype
        )
    )
    model.to("cuda").eval()
    peak = torch.cuda.max_memory_allocated() / 2**30

    torch.manual_seed(0)
    ids = torch.randint(10, 120000, (1, 16), device="cuda")
    meta = AttentionMetadata()
    meta.is_prefill = True
    meta.b_seq_len = torch.full((1,), 16, dtype=torch.long)
    pos = torch.arange(16, device="cuda").unsqueeze(0)

    with torch.no_grad():
        logits = model(ids, pos, meta)
    # Vocabulary-parallel head: each rank projects its vocab slice.
    assert logits.shape == (1, 16, config.vocab_size // 2), logits.shape
    assert torch.isfinite(logits).all(), "non-finite prefill logits"

    meta = AttentionMetadata()
    meta.is_prefill = False
    meta.b_seq_len = torch.full((1,), 1, dtype=torch.long)
    pos = torch.full((1, 1), 16, device="cuda")
    with torch.no_grad():
        logits2 = model(ids[:, :1], pos, meta)
    assert torch.isfinite(logits2).all(), "non-finite decode logits"

    return {
        "peak_gib": peak,
        "prefill_mean": logits.float().mean().item(),
        "prefill_std": logits.float().std().item(),
        "prefill_top5": logits[0, -1].topk(5).values.float().cpu().tolist(),
        "decode_mean": logits2.float().mean().item(),
        "decode_std": logits2.float().std().item(),
    }


@needs_gpus(2)
def test_smoke() -> None:
    results = run_on_tp_ranks(_v4_payload, tp_size=2)
    for rank, r in enumerate(results):
        print(
            f"rank {rank}: peak {r['peak_gib']:.2f} GiB | "
            f"prefill mean {r['prefill_mean']:.4f} std {r['prefill_std']:.4f} "
            f"top5 {[f'{v:.3f}' for v in r['prefill_top5']]} | "
            f"decode mean {r['decode_mean']:.4f} std {r['decode_std']:.4f}"
        )
    # The head is vocabulary-parallel: each rank's statistics describe a
    # different half of the vocab, so cross-rank agreement is not expected
    # here — the finiteness asserts inside the payload are the real gate.
    for r in results:
        assert r["prefill_std"] > 0 and r["decode_std"] > 0, "degenerate logits"
    print("SMOKE OK")


if __name__ == "__main__":
    test_smoke()
