"""Locate where a TP=2 DeepSeek-V2-Lite run leaves its HF reference, layer by layer.

The whole-model prefill runs on both engines with a forward hook on every
decoder layer: the lite side on a two-rank grid (each rank's layer output is
the full hidden — the o_proj/MoE collectives land inside the layer), the HF
side spread over both cards with ``device_map="auto"``. Every layer's
residual-stream output is diffed; the first row out of the band names the
layer to hand to the single-layer harness.

Development scaffold for the golden gate; not part of the test suite.

Usage:
    .venv/bin/python scripts/dsv2_layer_diff.py
"""

from __future__ import annotations

import gc

import torch

_MODEL = "my_weight/DeepSeek-V2-Lite"
_PROMPT = (
    "Explain in plain language why the sky is blue. Cover how sunlight "
    "scatters off air molecules, why shorter wavelengths scatter more, and "
    "what that means for the colors we see at sunrise and sunset."
)
_SEQ_CAP = 64


def _capture(sink: list[torch.Tensor], *, lite: bool):
    """Hook that records one decoder layer's residual-stream output on CPU."""

    def hook(_module, _args, output):
        tensor = output[0] + output[1] if lite else output[0]
        sink.append(tensor.detach().float().cpu())

    return hook


def _probe_layer(mod, sink: dict, *, hf: bool = False) -> list:
    """Capture one MoE block's input, output, and router decision.

    The router is re-computed in the pre-hook, on the live device tensor.
    transformers 5.x routes through ``DeepseekV2TopkRouter``, whose fp32
    upcast GEMM matches ``_route``'s exactly — so on the HF side the probe
    just calls the gate and takes its routing; only lite re-computes.
    """
    import torch.nn.functional as F

    def pre(_m, args):
        x = args[0]
        sink["x"] = x.detach().float().cpu()
        if hf:
            logits, weights, ids = _m.gate(x)
            probs = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            logits = F.linear(x.float(), _m.gate_weight.float())
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            weights, ids = torch.topk(probs, _m.top_k, dim=-1)
        sink["probs"] = probs.detach().float().cpu()
        sink["ids"] = ids.cpu()
        sink["weights"] = weights.cpu()

    def post(_m, _a, out):
        # Store only: a forward hook's non-None return REPLACES the module's
        # output — returning the stashed CPU copy once swapped a live bf16
        # tensor for it and detonated the very next residual add.
        sink.setdefault("y", out.detach().float().cpu())

    def shared(_m, _a, out):
        sink["shared"] = out.detach().float().cpu()

    return [
        mod.register_forward_pre_hook(pre),
        mod.register_forward_hook(post),
        mod.shared_experts.register_forward_hook(shared),
    ]


def _lite_payload(rank: int):
    """Rank's share: load the model, prefill the prompt, return layer outputs."""
    from rapid_llm.distributed import parallel_state as ps

    ps.init_parallel(global_rank=rank, tp_size=2, dp_size=1)
    try:
        from rapid_llm.executor.loader import DefaultModelLoader
        from rapid_llm.models.config import ModelConfig
        from rapid_llm.models.registry import ModelRegistry
        from rapid_llm.tools.accuracy.divergence import PrefillCache

        device = f"cuda:{rank}"
        config = ModelConfig.from_pretrained(_MODEL, _SEQ_CAP)
        model = DefaultModelLoader().load_model(
            config, ModelRegistry.resolve(config.model_type).load_class(), _MODEL, device
        )
        from transformers import AutoTokenizer

        ids = AutoTokenizer.from_pretrained(_MODEL)(_PROMPT, return_tensors="pt").input_ids
        ids = ids[:, :_SEQ_CAP].to(device)
        seq_len = ids.shape[1]
        cache = PrefillCache(
            config.num_layers,
            1,
            seq_len,
            kv_row=config.kv_cache_row,
            dtype=config.kv_cache_torch_dtype,
            device=device,
        ).begin_prefill()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(1, seq_len)
        captured: list[torch.Tensor] = []
        handles = [
            layer.register_forward_hook(_capture(captured, lite=True)) for layer in model.layers
        ]
        moe_probe: dict = {}
        handles += _probe_layer(model.layers[3].mlp, moe_probe)
        try:
            with torch.no_grad():
                logits = model(
                    ids,
                    position_ids,
                    cache,
                    logits_positions=torch.full((1,), seq_len - 1, dtype=torch.long, device=device),
                )
        finally:
            for handle in handles:
                handle.remove()
        if rank == 0:  # every rank holds the same post-collective outputs
            # numpy, not torch: a tensor crosses the queue through shared-memory
            # fds that die with the worker before the parent can attach.
            return {
                "layers": [t.numpy() for t in captured],
                "logits": logits.detach().float().cpu().numpy(),
                "moe": {
                    key: value.numpy() if torch.is_tensor(value) else value
                    for key, value in moe_probe.items()
                },
            }
        return None
    finally:
        from rapid_llm.distributed import parallel_state as ps

        ps.destroy_parallel()


def _hf_layers() -> tuple[list[torch.Tensor], torch.Tensor, list[int], dict]:
    """The same prefill through transformers, spread over both cards."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        _MODEL,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=False,
    ).eval()
    try:
        ids = AutoTokenizer.from_pretrained(_MODEL)(_PROMPT, return_tensors="pt").input_ids
        ids = ids[:, :_SEQ_CAP]
        captured: list[torch.Tensor] = []
        handles = [
            layer.register_forward_hook(_capture(captured, lite=False))
            for layer in model.model.layers
        ]
        moe_probe_hf: dict = {}
        handles += _probe_layer(model.model.layers[3].mlp, moe_probe_hf, hf=True)
        try:
            with torch.no_grad():
                logits = model(ids).logits[:, -1]
        finally:
            for handle in handles:
                handle.remove()
        return captured, logits.detach().float().cpu(), ids[0].tolist(), moe_probe_hf
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tests.distributed.tp_harness import run_on_tp_ranks

    lite = run_on_tp_ranks(_lite_payload, tp_size=2, timeout=600)[0]
    hf_layers, hf_logits, prompt_ids, hf_moe = _hf_layers()

    print(f"{len(prompt_ids)} prompt tokens; {len(hf_layers)} HF layers")
    first = None
    for index, (mine, theirs) in enumerate(zip(lite["layers"], hf_layers, strict=True)):
        mine = torch.from_numpy(mine)
        delta = (mine - theirs).abs()
        scale = theirs.abs().max().item()
        rel = delta.max().item() / scale if scale > 0 else float("inf")
        marker = ""
        if (not torch.isfinite(delta).all() or rel > 5e-2) and first is None:
            first = index
            marker = "  <-- first out of band"
        # Where the worst error lives: per-position max over the hidden axis,
        # for the layers around the first jump, to tell an isolated token flip
        # (one hot position, rest quiet) from a systematic block error.
        if index in (2, 3, 4, 12):
            per_pos = delta.max(dim=-1).values
            hot = (per_pos > 0.5).nonzero().flatten().tolist()
            print(
                f"    layer {index}: positions above 0.5 abs diff: {hot} "
                f"(of {per_pos.numel()}); quiet p50={per_pos.median():.2e}"
            )
        print(
            f"  layer {index:>3}  max_abs={delta.max().item():.3e}"
            f" mean_abs={delta.mean().item():.3e} rel={rel:.3e}{marker}"
        )
    lite_logits = torch.from_numpy(lite["logits"])
    if lite_logits.shape == hf_logits.shape:
        logits_gap = (lite_logits - hf_logits).abs().max().item()
        agree = int(lite_logits.argmax()) == int(hf_logits.argmax())
        print(f"  logits max_abs={logits_gap:.3e} argmax agree={agree}")
    else:
        # TP=2 shards lm_head along the vocab: rank 0's logits are its local
        # half. The full-vocab argmax lives behind the engine's all-gather.
        print(
            f"  logits {tuple(lite_logits.shape)} vs {tuple(hf_logits.shape)}"
            " — vocab-sharded, skipped"
        )
    print(f"  first out of band: {first}")

    # Layer-3 MoE probe: does the BOS row route to different experts? If the
    # input matches row-for-row but the expert sets part ways at row 0, the
    # router — not the experts — decided the 8.0 spike.
    def rows(t):
        """Flatten [batch?, seq, ...] to [seq, ...]; lite may pre-flatten."""
        return t.reshape(-1, t.shape[-1])

    moe = lite["moe"]
    x_l, x_h = rows(torch.from_numpy(moe["x"])), rows(hf_moe["x"])
    dx = (x_l - x_h).abs().max(dim=-1).values
    print(f"\n  MoE x diff: BOS={dx[0]:.3e} other max={dx[1:].max():.3e} p50={dx[1:].median():.3e}")
    ids_l, ids_h = rows(torch.from_numpy(moe["ids"])), rows(hf_moe["ids"])
    flipped = [i for i in range(ids_l.shape[0]) if set(ids_l[i].tolist()) != set(ids_h[i].tolist())]
    print(f"  expert-set mismatch rows: {flipped}")
    print(f"  BOS ids lite={sorted(ids_l[0].tolist())} hf={sorted(ids_h[0].tolist())}")
    probs_l, probs_h = rows(torch.from_numpy(moe["probs"])), rows(hf_moe["probs"])
    for name, probs in (("lite", probs_l), ("hf", probs_h)):
        top8 = probs[0].topk(8)
        print(f"  BOS {name} top8 probs: {[f'{v:.4f}' for v in top8.values.tolist()]}")
        print(f"    rank6-7 gap: {top8.values[5] - top8.values[6]:.3e}")
    y_l, y_h = rows(torch.from_numpy(moe["y"])), rows(hf_moe["y"])
    dy = (y_l - y_h).abs().max(dim=-1).values
    print(f"  MoE y diff: BOS={dy[0]:.3e} other max={dy[1:].max():.3e} p50={dy[1:].median():.3e}")

    # Split the block output into its shared-expert and routed halves. Both
    # sides expose ``shared_experts``; routed = y - shared falls out by
    # subtraction, bisecting the 8.0 spike between the dense and the
    # gathered-expert path without re-running anything.
    shared_l = rows(torch.from_numpy(moe["shared"]))
    shared_h = rows(hf_moe["shared"])
    ds = (shared_l - shared_h).abs().max(dim=-1).values
    print(f"  MoE shared diff: BOS={ds[0]:.3e} other max={ds[1:].max():.3e}")
    dr = ((y_l - shared_l) - (y_h - shared_h)).abs().max(dim=-1).values
    print(f"  MoE routed diff: BOS={dr[0]:.3e} other max={dr[1:].max():.3e}")

    # Per-row relative error settles whether BOS is a real hot spot or an
    # arithmetic one: bf16's ULP grows with magnitude, so the largest-norm
    # row shows the largest *absolute* gap at the same relative error.
    norms = y_h.norm(dim=-1)
    rel_rows = dy / norms
    print(f"  y norms: BOS={norms[0]:.1f} other p50={norms[1:].median():.1f}")
    print(
        f"  MoE y rel: BOS={rel_rows[0]:.3e} other p50={rel_rows[1:].median():.3e} "
        f"max={rel_rows[1:].max():.3e}"
    )

    # Up close at the worst coordinate: a discrete jump prints as two clean
    # values (a power of two against zero); smooth amplification prints as
    # two nearby magnitudes.
    coord = (y_l - y_h).abs()[0].argmax()
    print(f"  BOS y worst coord {int(coord)}: lite={y_l[0, coord]:.4f} hf={y_h[0, coord]:.4f}")
    coord_x = (x_l - x_h).abs()[0].argmax()
    print(
        f"  BOS x worst coord {int(coord_x)}: lite={x_l[0, coord_x]:.6f} hf={x_h[0, coord_x]:.6f}"
    )


if __name__ == "__main__":
    main()
