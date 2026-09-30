"""DeepSeek-V4 numerical alignment against HuggingFace transformers 5.8.

V4 ships no public weights, so the parity reference is transformers'
``DeepseekV4ForCausalLM`` over a randomly-initialised trimmed checkpoint —
exactly how a vendor integration would smoke-test the family. The trimmed
config exercises every structural variant at test size: all three layer
types (SWA / CSA / HCA), both MoE families (``hash_moe`` and ``moe``),
the mHC stream stack, the interleaved partial RoPE and the bounded SwiGLU.

HF initialises ``e_score_correction_bias`` and ``tid2eid`` to zeros — the
fixture injects random values before checkpointing so the routed paths have
discriminating signal (a zeros bias selects experts 0..k-1, which any
bug that ignores the bias entirely would still pass).

Usage:
    pytest tests/models/test_deepseek_v4.py
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from rapid_llm.models.config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Trimmed V4 at test size. Six layers = two passes over the three attention
#: types; two of them route through the hash router, four through top-k.
#: ``sliding_window=16`` keeps the rolling window exercised at prompt scale.
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
    # Must stay >= head_dim * partial_rotary_factor (32): the indexer's
    # heads take the trailing-rope slice, so a smaller table cannot reach all
    # of the indexer's channels — the reference raises on it too.
    "index_head_dim": 32,
    "index_topk": 4,
    "swiglu_limit": 7.0,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


def _write_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_BODY))


def _loaded_pair(tmp_path, seed: int = 0):
    """Build the HF reference, checkpoint it, load the same weights in rapid_llm.

    Returns:
        ``(hf_model, lite_model, config)`` with both sides on CUDA in eval
        mode; ``load_weights``'s coverage check verified the mapping on the
        way in.
    """
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.registry import ModelRegistry

    _write_config(tmp_path)
    config = ModelConfig.from_pretrained(tmp_path, max_seq_len=128)
    torch.manual_seed(seed)
    hf_model = DeepseekV4ForCausalLM(config.hf_config).eval()
    # The reference inits both router tables to zeros; inject signal so a
    # loader or forward that silently drops them cannot pass parity.
    with torch.no_grad():
        for layer in hf_model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "e_score_correction_bias"):
                gate.e_score_correction_bias.normal_(0.0, 0.25)
            if hasattr(gate, "tid2eid"):
                gate.tid2eid.copy_(
                    torch.randint(0, _BODY["n_routed_experts"], gate.tid2eid.shape, generator=None)
                )
    state = {key: value.detach().clone() for key, value in hf_model.state_dict().items()}
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})

    model = ModelRegistry.resolve("deepseek_v4").load_class()(config)
    materialise_parameters(model, DEVICE, dtype=config.dtype)
    model.load_weights(hf_weights_iterator(tmp_path, dequant_dtype=config.dtype))
    # Buffers were built on the CPU (materialise_parameters only moves
    # parameters); the engine's loader finishes with ``model.to(device)``.
    model.to(DEVICE)
    # Both sides on CUDA at the parity dtype, so per-test ``.to(...).to(DEVICE)``
    # calls degenerate to no-ops and every comparison runs like-for-like.
    # ``.to(dtype)`` would also narrow the reference's fp32 *buffers* — the
    # rope tables and the router's ``e_score_correction_bias`` — while the
    # lite side keeps both fp32 by design (non-persistent buffers and
    # ``RawParameter`` storage). Restore them so the parity comparison
    # measures the two code paths, not the two precisions.
    fp32_buffers = [
        (submodule, name, buf.clone())
        for submodule in hf_model.modules()
        if isinstance(submodule, type(hf_model.model.rotary_emb))
        for name, buf in submodule.named_buffers()
        if buf.is_floating_point()
    ]
    for layer in hf_model.model.layers:
        bias = getattr(layer.mlp.gate, "e_score_correction_bias", None)
        if bias is not None:
            fp32_buffers.append((layer.mlp.gate, "e_score_correction_bias", bias.clone()))
    # The mHC modules sit on the reference's strict-fp32 list too
    # (``_keep_in_fp32_modules_strict``: they stay fp32 even at a bf16
    # deployment dtype), matching lite's ``RawParameter`` storage — save
    # them alongside the buffers so ``.to(config.dtype)`` narrows neither
    # side's hyper-connections.
    strict_fp32_params = [
        (hf_model.model.hc_head, name, p.detach().clone())
        for name, p in hf_model.model.hc_head.named_parameters()
    ]
    for layer in hf_model.model.layers:
        for site in ("attn_hc", "ffn_hc"):
            hc = getattr(layer, site)
            strict_fp32_params.extend(
                (hc, name, p.detach().clone()) for name, p in hc.named_parameters()
            )
    hf_model = hf_model.to(config.dtype).to(DEVICE)
    for module, name, buf in fp32_buffers:
        module.register_buffer(name, buf.to(DEVICE))
    for module, name, param in strict_fp32_params:
        module._parameters[name].data = param.to(DEVICE)
    return hf_model, model.eval(), config


def _lite_metadata(batch: int, seq_len: int, *, prefill: bool):
    """A minimal AttentionMetadata carrying just the V4-relevant fields."""
    from rapid_llm.executor.attention_metadata import AttentionMetadata

    meta = AttentionMetadata()
    meta.is_prefill = prefill
    meta.b_seq_len = torch.full((batch,), seq_len, dtype=torch.long)
    return meta


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """One checkpoint build shared by the module's tests."""
    return _loaded_pair(tmp_path_factory.mktemp("v4"))


def test_rotary_tables_match(pair):
    """Both theta tables and the interleaved layout against the reference."""
    hf_model, model, _ = pair
    hidden = torch.randn(2, 7, _BODY["hidden_size"], device=DEVICE, dtype=model.config.dtype)
    pos = torch.arange(7).unsqueeze(0).expand(2, -1).contiguous().to(DEVICE)
    with torch.no_grad():
        for layer_type in ("main", "compress"):
            cos_h, sin_h = hf_model.model.rotary_emb(
                hidden, position_ids=pos, layer_type=layer_type
            )
            cos_l, sin_l = model.rotary_emb(hidden, pos, layer_type)
            torch.testing.assert_close(cos_l.float(), cos_h.float())
            torch.testing.assert_close(sin_l.float(), sin_h.float())


def test_hyper_connection_matches(pair):
    """mHC mapping: Sinkhorn comb, pre/post logits, and the stream collapse."""
    hf_model, model, _ = pair
    hc = model.layers[0].attn_hc
    # The fixture keeps the reference's mHC parameters fp32 (its strict-fp32
    # list) — the same storage lite uses — so no per-test cast happens here.
    hc_h = hf_model.model.layers[0].attn_hc
    streams = torch.randn(
        2, 5, _BODY["hc_mult"], _BODY["hidden_size"], device=DEVICE, dtype=model.config.dtype
    )
    with torch.no_grad():
        post, comb, collapsed = hc(streams)
        post_h, comb_h, collapsed_h = hc_h(streams.clone())
    torch.testing.assert_close(post.float(), post_h.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(comb.float(), comb_h.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(collapsed.float(), collapsed_h.float(), atol=2e-2, rtol=2e-2)


def test_router_sqrtsoftplus_semantics(pair):
    """Selection follows ``scores + bias``; weights are the gathered raw scores."""
    hf_model, model, _ = pair
    moe = model.layers[2].mlp  # a top-k ("moe") layer
    gate_h = hf_model.model.layers[2].mlp.gate  # fixture dtype: weight bf16, bias fp32

    torch.manual_seed(3)
    x = torch.randn(11, _BODY["hidden_size"], dtype=moe.gate_weight.dtype, device=DEVICE)
    with torch.no_grad():
        weights, ids = moe._route(x)
        _, weights_h, ids_h = gate_h(x.clone())

    assert torch.equal(ids, ids_h)
    # The block returns its fp32 router weights cast to the activation
    # dtype; compare against the reference after the same cast so the
    # tolerance measures the routing math, not bf16 rounding.
    torch.testing.assert_close(
        weights.float(), weights_h.to(weights.dtype).float(), atol=1e-3, rtol=1e-3
    )


def test_hash_router_reads_the_table(pair):
    """``hash_moe`` layers pick experts through ``tid2eid`` — not the logits."""
    hf_model, model, _ = pair
    moe = model.layers[0].mlp
    gate_h = hf_model.model.layers[0].mlp.gate  # fixture dtype: weight bf16, table int64

    torch.manual_seed(4)
    x = torch.randn(2, 6, _BODY["hidden_size"], dtype=moe.gate_weight.dtype, device=DEVICE)
    input_ids = torch.randint(0, _BODY["vocab_size"], (2, 6), device=DEVICE)
    with torch.no_grad():
        weights, ids = moe._route(x.reshape(-1, _BODY["hidden_size"]), input_ids)
        _, weights_h, ids_h = gate_h(x.clone(), input_ids)

    assert torch.equal(ids, ids_h)
    torch.testing.assert_close(
        weights.float(), weights_h.to(weights.dtype).float(), atol=1e-3, rtol=1e-3
    )


def test_moe_layer_matches_transformers(pair):
    """Whole MoE forward — routing, bias selection, clamped SwiGLU, shared."""
    hf_model, model, _ = pair
    for layer_index in (0, 2):  # hash layer and top-k layer
        moe = model.layers[layer_index].mlp
        moe_h = hf_model.model.layers[layer_index].mlp  # fixture already at parity dtype

        torch.manual_seed(5 + layer_index)
        x = torch.randn(2, 6, _BODY["hidden_size"], dtype=moe.gate_weight.dtype, device=DEVICE)
        input_ids = torch.randint(0, _BODY["vocab_size"], (2, 6), device=DEVICE)
        with torch.no_grad():
            actual = moe(x.clone(), input_ids=input_ids)
            expected = moe_h(x.clone(), input_ids=input_ids)
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_decoder_layer_matches_transformers(pair):
    """One full block — attention over the sliding/compressed KV plus the mHC mix."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask

    hf_model, model, config = pair
    batch, seq_len = 2, 7
    torch.manual_seed(6)
    hidden = torch.randn(
        batch, seq_len, _BODY["hc_mult"], _BODY["hidden_size"], device=DEVICE, dtype=config.dtype
    )
    input_ids = torch.randint(0, _BODY["vocab_size"], (batch, seq_len), device=DEVICE)
    pos = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous().to(DEVICE)
    # transformers 5.15 passes both rope tables as a layer-type dict; lite's
    # layer takes the "main" pair alone (the compressor builds its own).
    position_embeddings = {
        layer_type: model.rotary_emb(hidden[..., 0, :], pos, layer_type)
        for layer_type in ("main", "compress")
    }
    main_pe = position_embeddings["main"]
    valid = torch.ones(batch, seq_len, dtype=torch.bool, device=DEVICE)

    for layer_index in range(6):
        model.reset_v4_caches()
        cache = DynamicCache(config=config.hf_config)
        layer = model.layers[layer_index]
        layer_h = hf_model.model.layers[layer_index]  # fixture dtype set; a
        # per-layer ``.to(dtype)`` would re-narrow the fp32 mHC parameters.
        # A bare layer call skips the model-level mask build: without the
        # sliding-window causal mask the reference attends over the full
        # sequence, while lite's kernel always applies causality itself.
        causal_mask = create_sliding_window_causal_mask(
            config=config.hf_config,
            inputs_embeds=torch.zeros(
                batch, seq_len, _BODY["hidden_size"], device=DEVICE, dtype=config.dtype
            ),
            attention_mask=None,
            past_key_values=cache,
            position_ids=pos,
        )
        with torch.no_grad():
            actual = layer(hidden.clone(), pos, main_pe, input_ids, valid)
            expected = layer_h(
                hidden.clone(),
                input_ids=input_ids,
                position_embeddings=position_embeddings,
                position_ids=pos,
                attention_mask=causal_mask,
                past_key_values=cache,
            )
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=6e-2,
            rtol=6e-2,
        )


def test_end_to_end_greedy_parity(pair):
    """Greedy tokens from prefill + 12 decode steps match the reference."""
    from transformers.cache_utils import DynamicCache

    hf_model, model, config = pair
    torch.manual_seed(7)
    batch, prompt_len = 2, 9
    input_ids = torch.randint(0, _BODY["vocab_size"], (batch, prompt_len), device=DEVICE)
    steps = 12

    # --- reference: manual greedy loop over the HF model ------------------ #
    tokens_h = [input_ids]
    cache = DynamicCache(config=config.hf_config)
    with torch.no_grad():
        for step in range(steps):
            ids = tokens_h[-1]
            past_len = 0 if step == 0 else prompt_len + step - 1
            pos = (
                torch.arange(past_len, past_len + ids.shape[1], device=DEVICE)
                .unsqueeze(0)
                .expand(batch, -1)
                .contiguous()
            )
            out = hf_model(
                input_ids=ids,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            )
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_h.append(next_tok)
    reference = torch.cat(tokens_h[1:], dim=1)

    # --- rapid_llm: prefill then decode through the model runner API ------ #
    tokens_l = [input_ids]
    with torch.no_grad():
        meta = _lite_metadata(batch, prompt_len, prefill=True)
        pos = torch.arange(prompt_len, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()
        logits = model(input_ids, pos, meta)
        for step in range(steps):
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_l.append(next_tok)
            meta = _lite_metadata(batch, 1, prefill=False)
            pos = torch.full((batch, 1), prompt_len + step, device=DEVICE)
            logits = model(next_tok, pos, meta)
    actual = torch.cat(tokens_l[1:], dim=1)

    assert torch.equal(actual, reference), (
        f"greedy divergence:\nlite:    {actual[0].tolist()}\nref:     {reference[0].tolist()}"
    )


def test_incremental_equals_full_forward(pair):
    """Chunked prefill + decode must equal one whole-sequence forward.

    The comparison runs per layer on the stream stack, not on logits: an
    8-token prefill followed by single-token decodes exercises the same
    compressor windows (8 = 2× csa rate, 1× hca rate) as the whole-sequence
    pass, so any state-carry bug — a lost overlap slot, a mis-shifted
    window position — shows up as an O(1) divergence at the first layer that
    owns it. What remains is bf16 non-associativity (the chunked and whole
    GEMMs tile differently), which the per-layer tolerance below covers
    with an order of magnitude of headroom and which grows only slowly
    with depth; the layer-0 sliding path is bit-exact, pinning the KV state
    thread precisely.
    """
    _, model, _ = pair
    torch.manual_seed(8)
    batch, total_len = 2, 13
    input_ids = torch.randint(0, _BODY["vocab_size"], (batch, total_len), device=DEVICE)

    def run_layers(chunk_ids: torch.Tensor, pos: torch.Tensor) -> list[torch.Tensor]:
        """One pass over the stream stack; returns each layer's output."""
        hidden = model.get_input_embeddings(chunk_ids)
        pe = model.rotary_emb(hidden, pos, "main")
        streams = hidden.unsqueeze(2).expand(-1, -1, model.hc_mult, -1).contiguous()
        valid = torch.ones(batch, chunk_ids.shape[1], dtype=torch.bool, device=DEVICE)
        outs = []
        for layer in model.layers:
            streams = layer(streams, pos, pe, chunk_ids, valid)
            outs.append(streams.clone())
        return outs

    with torch.no_grad():
        # Whole sequence at once.
        model.reset_v4_caches()
        pos = torch.arange(total_len, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()
        full_outs = run_layers(input_ids, pos)

        # 8-token prefill, then one token at a time.
        model.reset_v4_caches()
        chunks = [input_ids[:, :8]]
        chunks.extend(input_ids[:, 8 + i : 8 + i + 1] for i in range(total_len - 8))
        offset = 0
        step_outs = [[] for _ in model.layers]
        for chunk in chunks:
            pos = (
                torch.arange(offset, offset + chunk.shape[1], device=DEVICE)
                .unsqueeze(0)
                .expand(batch, -1)
                .contiguous()
            )
            layer_outs = run_layers(chunk, pos)
            for li, out in enumerate(layer_outs):
                step_outs[li].append(out)
            offset += chunk.shape[1]

    for li, full in enumerate(full_outs):
        incremental = torch.cat(step_outs[li], dim=1)
        if li == 0:  # the sliding layer carries no compressor: bit-exact
            assert torch.equal(incremental, full)
        else:
            torch.testing.assert_close(incremental.float(), full.float(), atol=0.15, rtol=0.15)
