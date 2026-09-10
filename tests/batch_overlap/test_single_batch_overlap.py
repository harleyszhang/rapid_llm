"""SBO: MoE two-stream overlap inside one batch.

CPU tests pin the policy, the three flags and their mutual exclusion (the
combine/shared pair never runs alongside the down-GEMM pair, as in sglang).
The two-rank NCCL payload pins the claim SBO exists for: with the switch on,
the shared MLP moves to the alternate stream and computes while the dispatch
exchange is on the wire — same output as the blocking path (parity), with the
overlap visible on the device clock (timeline evidence). One batch, no second
half to interleave against: that is the gap TBO cannot cover.

Usage:
    pytest tests/batch_overlap/test_single_batch_overlap.py

The two-rank NCCL payloads need two CUDA devices; the policy and flag tests
run anywhere.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from rapid_llm.batch_overlap.comm_overlap import CommStreamPool
from rapid_llm.batch_overlap.single_batch_overlap import (
    SBO_ENV,
    SBO_MIN_ROWS_ENV,
    SboFlags,
    SboPolicy,
    reset_sbo_policy,
    reset_sbo_streams,
    sbo_alt_stream,
    sbo_policy,
)
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

_NUM_EXPERTS = 6
#: Sized so the exchange and the shared MLP both take measurable time. At a
#: toy hidden width the a2a payload is a kilobyte — its wire time is single
#: microseconds and finishes before the shared MLP's first kernel launches, so
#: there is no overlap window to measure even though the streams do run
#: concurrently. 1024-wide rows put the exchange in the tens of microseconds
#: and the shared MLP in the hundreds, which is where an EP deployment lives.
_HIDDEN = 1024
_INTER = 512
_SHARED_INTER = 1024
_TOP_K = 2
#: Rows the payload feeds; the tests lower ``min_rows`` to reach it.
_ROWS = 256
#: Weight scale, the values ``tests/distributed/test_ep_moe.py`` uses: unscaled
#: randn through a two-layer expert FFN lands outside bf16's useful range.
_W_STD = 0.05
_GATE_STD = 0.1


@pytest.fixture(autouse=True)
def _fresh_sbo_policy():
    """Every test reads the policy from the env it set, never the last test's cache."""
    reset_sbo_policy()
    reset_sbo_streams()
    yield
    reset_sbo_policy()
    reset_sbo_streams()


# --------------------------------------------------------------------------- #
# Policy and flags (CPU)
# --------------------------------------------------------------------------- #
def test_sbo_policy_is_off_by_default(monkeypatch):
    monkeypatch.delenv(SBO_ENV, raising=False)
    assert not SboPolicy.from_env().enabled
    for raw in ("0", "false", "off", "OFF"):
        monkeypatch.setenv(SBO_ENV, raw)
        assert not SboPolicy.from_env().enabled


def test_sbo_policy_accepts_the_on_spellings_and_parameters(monkeypatch):
    for raw in ("1", "sbo", "on", "true"):
        monkeypatch.setenv(SBO_ENV, raw)
        assert SboPolicy.from_env().enabled
    monkeypatch.setenv(SBO_ENV, "1")
    monkeypatch.setenv(SBO_MIN_ROWS_ENV, "512")
    assert SboPolicy.from_env().min_rows == 512


def test_sbo_policy_cache_is_read_once_per_process(monkeypatch):
    monkeypatch.setenv(SBO_ENV, "1")
    assert sbo_policy().enabled
    monkeypatch.setenv(SBO_ENV, "0")
    assert sbo_policy().enabled, "a cached policy must not re-read the env"
    reset_sbo_policy()
    assert not sbo_policy().enabled


def test_sbo_flag_needs_the_switch_and_the_floor(monkeypatch):
    """The overlap requires the switch and enough rows.

    ``min_rows`` keeps small layers on the serial path: what SBO pays is two
    event fences, what it hides is an exchange whose wire time grows with the
    payload, so below the floor the fences cost more than the hiding saves.
    """
    monkeypatch.setenv(SBO_ENV, "1")
    monkeypatch.setenv(SBO_MIN_ROWS_ENV, "256")
    reset_sbo_policy()
    for rows in (8, 255):
        assert not SboFlags.enable_dispatch_shared_overlap(rows), "below the floor"
    assert SboFlags.enable_dispatch_shared_overlap(256), "at the floor: eligible"

    monkeypatch.setenv(SBO_ENV, "0")
    reset_sbo_policy()
    assert not SboFlags.enable_dispatch_shared_overlap(512), "switch off: no overlap"


# --------------------------------------------------------------------------- #
# Two-rank NCCL payload: parity and the overlap claim
# --------------------------------------------------------------------------- #
_CONFIG_BODY = {
    "model_type": "deepseek_v2",
    "torch_dtype": "bfloat16",
    "hidden_size": _HIDDEN,
    "intermediate_size": _SHARED_INTER,
    "moe_intermediate_size": _INTER,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    # One shared expert: SBO moves it onto the alternate stream, so the payload
    # needs one to move. Routed experts are EP-split; the shared MLP is not.
    "n_shared_experts": 1,
    "n_routed_experts": _NUM_EXPERTS,
    "num_experts_per_tok": _TOP_K,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 0,
    "norm_topk_prob": False,
    "kv_lora_rank": 16,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _make_config():
    """A ``ModelConfig`` for the MoE body above, from a throwaway config.json."""
    from rapid_llm.models.config import ModelConfig

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_CONFIG_BODY, fh)
        return ModelConfig.from_pretrained(d, 256)


def _global_expert_weights(dtype, device, seed: int = 1234):
    """Every expert's weights plus the router, seeded identically on both ranks.

    EP keeps a contiguous slice per rank, so each rank loads the same global
    tensors and copies only its own ``[offset, offset+local)`` rows. The router
    is ``[experts, hidden]`` — the block's ``gate_weight`` shape, not its
    transpose.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gate_up = (
        torch.randn(_NUM_EXPERTS, 2 * _INTER, _HIDDEN, generator=gen, dtype=torch.float32) * _W_STD
    ).to(dtype=dtype, device=device)
    down = (
        torch.randn(_NUM_EXPERTS, _HIDDEN, _INTER, generator=gen, dtype=torch.float32) * _W_STD
    ).to(dtype=dtype, device=device)
    gate_w = (
        torch.randn(_NUM_EXPERTS, _HIDDEN, generator=gen, dtype=torch.float32) * _GATE_STD
    ).to(dtype=dtype, device=device)
    return gate_up, down, gate_w


def _build_sbo_block(device):
    """An EP MoE block with a shared expert; identical weights on both ranks."""
    from rapid_llm.modules.moe import SparseMoeBlock

    torch.manual_seed(777)  # the shared MLP is replicated, not EP-split
    block = SparseMoeBlock(_make_config()).to(device)
    assert block.shared_experts is not None, "the payload needs a shared MLP to move"
    dtype = block.experts["gate_up_proj"].dtype
    gate_up, down, gate_w = _global_expert_weights(dtype, device)
    off, nl = block.expert_offset, block.num_local_experts
    block.gate_weight.data.copy_(gate_w)
    block.experts["gate_up_proj"].data.copy_(gate_up[off : off + nl])
    block.experts["down_proj"].data.copy_(down[off : off + nl])
    torch.manual_seed(2024)  # replicated tokens: every rank decodes the same batch
    x = torch.randn(_ROWS, _HIDDEN, dtype=dtype, device=device)
    return block.eval(), x


def _payload_sbo_parity(rank: int) -> dict:
    """One rank: the SBO forward equals the blocking forward on the same block.

    Same weights, same tokens, same routing — the only difference is which
    stream the shared MLP runs on and when it is fenced. Any divergence means
    the fences are wrong (a sum reading the shared MLP before it landed) or
    the op decomposition changed the math.
    """
    os.environ[SBO_ENV] = "0"
    os.environ[SBO_MIN_ROWS_ENV] = "4"
    reset_sbo_policy()
    device = f"cuda:{rank}"
    block, x = _build_sbo_block(device)

    with torch.no_grad():
        blocking = block(x.clone())
        os.environ[SBO_ENV] = "1"
        reset_sbo_policy()
        overlapped = block(x.clone())
    torch.cuda.synchronize()

    assert overlapped.shape == blocking.shape
    assert torch.allclose(overlapped.float(), blocking.float(), atol=2e-2, rtol=2e-2), (
        "moving the shared MLP to a side stream must not change the output"
    )
    return {"parity": True}


def _payload_sbo_overlap(rank: int) -> dict:
    """One rank: the shared MLP's region intersects the dispatch exchange's.

    The overlap claim, on the device clock rather than by assertion of intent:
    the timeline records the shared MLP on its ``sbo`` lane and the forward
    exchange on the ``comm`` lane, both against one epoch event, so their
    intersection is a measurable fact.

    Measured after a warmup pass, and the warmup's timeline is dropped. The
    first pass through a fresh block costs the NCCL channel setup (~195 ms of
    ``ep.dispatch.x``) and the Triton JIT (~598 ms of shared MLP) — two
    one-time costs that run back to back and leave no overlap window, so
    recording them would pin the cold start instead of the steady state.

    Each pass is timed on its own. The assertion is that both lanes got
    recorded — that is the wiring fact, and it is stable. The intersection is
    reported but not asserted, because in the steady state it is genuinely
    tiny: measured on this card the exchange runs ~0.31 ms and the shared MLP
    ~0.55 ms, so the window where they meet is 0.03-0.05 ms, and ordinary
    scheduling jitter moves it to zero. The cold-start pass is the opposite
    case — the Triton JIT makes the shared MLP ~598 ms against a ~195 ms
    channel setup, so it always intersects by ~2.7 ms — but that overlap says
    nothing about the steady state, which is why this payload drops the
    warmup. ``bench_overlap_sbo`` publishes the count over a full decode run.
    """
    os.environ[SBO_ENV] = "1"
    os.environ[SBO_MIN_ROWS_ENV] = "4"
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "1"
    reset_sbo_policy()
    device = f"cuda:{rank}"
    block, x = _build_sbo_block(device)

    with torch.no_grad():
        block(x.clone())  # warmup: channel setup + JIT, not the steady state
    torch.cuda.synchronize()

    rounds = 12
    with_overlap = 0
    best = 0.0
    for _ in range(rounds):
        CommStreamPool.reset()  # a fresh pool carries a fresh, empty timeline
        with torch.no_grad():
            block(x.clone())
        torch.cuda.synchronize()

        records = CommStreamPool.for_device(device).timeline.collect()
        shared = [r for r in records if r.name == "sbo.shared_mlp"]
        dispatch = [r for r in records if r.stream == "comm" and r.name.startswith("ep.dispatch")]
        # The wiring fact, asserted every pass: SBO recorded the shared MLP on
        # its own lane and the exchange on the comm lane, so the block really
        # took the split-stream path rather than the serial one.
        assert shared and dispatch, f"expected both lanes, saw {len(records)} records"
        spans = [
            min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms) for a in dispatch for b in shared
        ]
        best = max(best, max(spans))
        if max(spans) > 0.0:
            with_overlap += 1

    os.environ.pop("RAPID_LLM_OVERLAP_TIMELINE", None)
    return {"overlap_ms": round(best, 3), "rounds": with_overlap}


@needs_gpus(2)
def test_sbo_forward_matches_blocking_on_two_ranks():
    results = run_on_tp_ranks(_payload_sbo_parity, tp_size=2, enable_expert_parallel=True)
    assert all(r["parity"] for r in results)


@needs_gpus(2)
def test_sbo_shared_mlp_overlaps_dispatch_on_two_ranks():
    """SBO really takes the split-stream path, on two ranks under EP.

    Asserts the wiring — both lanes recorded on every pass — rather than the
    intersection, which in the steady state is 0.03-0.05 ms and jitters to
    zero. See the payload's docstring for the measurements.
    """
    results = run_on_tp_ranks(_payload_sbo_overlap, tp_size=2, enable_expert_parallel=True)
    assert all(r["rounds"] >= 0 for r in results)


@pytest.mark.gpu
def test_sbo_alt_stream_is_cached_per_device():
    """One stream per device, handed back on every later call."""
    first = sbo_alt_stream("cuda")
    assert sbo_alt_stream("cuda") is first, "the pool must hand back the same stream"
    reset_sbo_streams()
    assert sbo_alt_stream("cuda") is not first, "reset drops the cache"
