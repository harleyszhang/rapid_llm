"""Trimmed-stack parity: rapid_llm vs transformers on DeepSeek checkpoints.

Two gates the full-stack TP=2 gate cannot cover:

* ``num_hidden_layers`` trimmed to 1 — the whole point of ``hf_overrides``.
  A one-layer model still loads and runs through every production path
  (registry, weight loader, engine, sampler), and layer-local arithmetic
  mistakes that the full stack averages away show up at full size here.
* DeepSeek-V3's biased ``noaux_tc`` routing end to end: the V3-4layers
  checkpoint (4 backbone layers, sigmoid scoring, fp32 correction bias,
  grouped top-k) exercises the router semantics the V2-Lite gate never
  touches, and its MTP layer keys past ``num_hidden_layers`` exercise the
  loader's drop rule on a real checkpoint.

  The checkpoint shrinks the routed set to 8 experts but keeps ``n_group=8``,
  leaving one expert per group — a geometry whose group score (top-2 within
  a group) no grouped router can compute, in vLLM and transformers exactly
  as here. The gate regroups to 2 groups of 4 through the same ``hf_overrides``
  on both sides, which restores the full semantics on the same weights: the
  bias picks the surviving group and its experts, the original sigmoid scores
  weigh them, renormalise and the 2.5 routed scale apply.

Drift budgets follow the TP=2 gate's design: teacher-forced logprob drift
(mean/max nats) plus a greedy match-rate floor, sized for bf16 noise.

V2-Lite trims to 1 layer — the dense-MLP path plus MLA through every
production path. V3-4layers runs its full 4 backbone layers with the
regrouped router (see the module docstring) so the biased grouped routing
is exercised end to end alongside the MTP-key drop.

Usage:
    pytest tests/golden/test_deepseek_trimmed_parity.py

Needs one CUDA device and the checkpoints (override with
``RAPID_LLM_TEST_DSV2_DIR`` / ``RAPID_LLM_TEST_DSV3_DIR``).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from rapid_llm.engine.llm import LLM
from rapid_llm.engine.sampler import SamplingParams
from tests.conftest import REPO_ROOT, checkpoint_problem

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Both checkpoints live in the shared lab store; ``my_weight/`` carries
# symlinks to them, which keeps these defaults relative (the
# no-hardcoded-path hook) while pointing at the same weights as the TP=2 gate.
_DSV2 = os.environ.get("RAPID_LLM_TEST_DSV2_DIR", "my_weight/DeepSeek-V2-Lite")
_DSV3 = os.environ.get("RAPID_LLM_TEST_DSV3_DIR", "my_weight/DeepSeek-V3-4layers-MTP-BF16")


def _resolve(path: str) -> Path:
    """Relative defaults are repo-rooted, like the TP=2 gate's."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea.",
    "List three prime numbers.",
    "Explain what a GPU is in one sentence.",
]
_MAX_GEN = 16
_MAX_SEQ_LEN = 512
_KV_BLOCKS = 4096

# Drift budgets: the TP=2 gate's full-stack calibration observed mean/max
# prompt drift 0.046/0.298 and step drift 0.221/1.967 nats. A trimmed stack
# averages over fewer layers, but the first MoE layer (V3) sits closer to the
# output than any layer of a 27-layer stack, so these keep the TP=2 gate's
# generous sizing: a systematic error (a wrong absorption, a mis-built bias,
# a broken group mask) pushes through them, rounding does not.
_PROMPT_MEAN_DRIFT = 0.12
_PROMPT_MAX_DRIFT = 0.7
_STEP_MEAN_DRIFT = 0.5
_STEP_MAX_DRIFT = 4.0
_MIN_MATCH_RATE = 0.7
_TIE_GAP = 0.1


def _checkpoint_gate(path: Path) -> Path:
    """The no-silent-skip policy every golden gate shares."""
    problem = checkpoint_problem(path)
    if problem:
        if os.environ.get("RAPID_LLM_GOLDEN_STRICT", "") == "1":
            pytest.fail(f"GOLDEN GATE FAIL: {problem}", pytrace=False)
        pytest.xfail(f"UNVERIFIED: {problem}")
    return path


@pytest.fixture(scope="module")
def v2lite_dir() -> Path:
    return _checkpoint_gate(_resolve(_DSV2))


@pytest.fixture(scope="module")
def v3_dir() -> Path:
    return _checkpoint_gate(_resolve(_DSV3))


# --------------------------------------------------------------------------- #
# Shared machinery: run rapid_llm, run transformers, compare
# --------------------------------------------------------------------------- #
def _greedy_runs(checkpoints_dir: Path, hf_overrides: dict[str, object] | None) -> dict[str, Any]:
    """Greedy generations with per-step logprobs through the one-shot LLM API."""
    llm = LLM(
        model=str(checkpoints_dir),
        device="cuda:0",
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_BLOCKS,
        use_cuda_graph=False,
        hf_overrides=hf_overrides,
    )
    try:
        params = SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN, logprobs=2)
        runs = []
        for prompt, output in zip(_PROMPTS, llm.generate(_PROMPTS, params), strict=True):
            records = output.outputs[0].logprobs or []
            runs.append(
                {
                    "prompt_ids": llm.tokenizer.encode(prompt, add_special_tokens=True),
                    "tokens": [r.token_id for r in records],
                    "logprobs": [r.logprob for r in records],
                }
            )
        return {"runs": runs, "num_layers": llm.model_runner.config.num_layers}
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _reference_rows(checkpoints_dir: Path, hf_overrides: dict[str, object] | None, runs) -> list:
    """``[positions, vocab]`` log-softmax rows from transformers, per prompt.

    The same override is applied to the reference's config, so both sides run
    the same trimmed stack. Unexpected keys (the layers the trim removed, the
    MTP layer past ``num_hidden_layers``) are exactly what transformers
    ignores with a warning and what rapid_llm's loader drops on purpose.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(checkpoints_dir), trust_remote_code=False)
    for field, value in (hf_overrides or {}).items():
        setattr(config, field, value)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoints_dir),
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    ).eval()
    try:
        refs = []
        with torch.no_grad():
            for run in runs:
                ids = torch.tensor([run["prompt_ids"] + run["tokens"]], device="cuda:0")
                logits = model(ids).logits.float().cpu()
                refs.append(torch.log_softmax(logits, dim=-1)[0])
        return refs
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _budget(drifts: list[tuple[float, str]], mean_limit: float, max_limit: float, what: str):
    """Assert a set of same-token deviations is bf16 noise, not a wrong number."""
    assert drifts, f"no {what} positions were compared"
    worst, where = max(drifts)
    mean = sum(drift for drift, _ in drifts) / len(drifts)
    assert worst <= max_limit, f"{where} is {worst:.4f} nats from the reference"
    assert mean <= mean_limit, (
        f"{len(drifts)} {what} positions average {mean:.4f} nats from the reference, "
        f"worst {worst:.4f} at {where}: that is a bias, not rounding"
    )


def _assert_parity(name: str, runs, reference) -> None:
    """The three assertions every gate makes over one (runs, reference) pair."""
    for prompt, run in zip(_PROMPTS, runs, strict=True):
        assert len(run["tokens"]) == _MAX_GEN, f"{name} {prompt!r}: short generation"

    prompt_drifts = []
    for run, ref in zip(runs, reference, strict=True):
        # Step k's reported logprob is the score of the drawn token at the
        # position that predicted it: prompt end + k.
        for step, (token, reported) in enumerate(zip(run["tokens"], run["logprobs"], strict=True)):
            theirs = float(ref[len(run["prompt_ids"]) - 1 + step, token])
            prompt_drifts.append(
                (
                    abs(reported - theirs),
                    f"{name} step {step} ({reported:+.4f} vs {theirs:+.4f})",
                )
            )
    _budget(prompt_drifts, _STEP_MEAN_DRIFT, _STEP_MAX_DRIFT, f"{name} decode step")

    matches = steps = 0
    for run, ref in zip(runs, reference, strict=True):
        for step, token in enumerate(run["tokens"]):
            row = ref[len(run["prompt_ids"]) - 1 + step]
            top2 = row.topk(2)
            gap = float(top2.values[0] - top2.values[1])
            if token == int(top2.indices[0]) or gap < _TIE_GAP:
                matches += 1
            steps += 1
    rate = matches / steps
    assert rate >= _MIN_MATCH_RATE, (
        f"{name}: only {matches}/{steps} greedy steps ({rate:.0%}) match the reference; "
        f"the floor is {_MIN_MATCH_RATE:.0%} — this is divergence, not rounding"
    )


# --------------------------------------------------------------------------- #
# V2-Lite with num_hidden_layers=1: dense MLP + MLA through every real path
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def v2_single_layer(v2lite_dir: Path) -> dict[str, Any]:
    return _greedy_runs(v2lite_dir, {"num_hidden_layers": 1})


@pytest.fixture(scope="module")
def v2_single_layer_ref(v2lite_dir: Path, v2_single_layer) -> list:
    return _reference_rows(v2lite_dir, {"num_hidden_layers": 1}, v2_single_layer["runs"])


def test_v2_trim_builds_one_layer(v2_single_layer):
    """The override must actually reach the model, not just the config parse."""
    assert v2_single_layer["num_layers"] == 1


def test_v2_single_layer_matches_transformers(v2_single_layer, v2_single_layer_ref):
    """One layer of MLA + dense SwiGLU against transformers, greedy end to end."""
    _assert_parity("V2-Lite 1-layer", v2_single_layer["runs"], v2_single_layer_ref)


# --------------------------------------------------------------------------- #
# V3-4layers: noaux_tc routing end to end (bias, sigmoid, grouped top-k)
# --------------------------------------------------------------------------- #
_V3_ROUTING_OVERRIDES = {"n_group": 2, "topk_group": 1, "num_experts_per_tok": 2}


@pytest.fixture(scope="module")
def v3_full(v3_dir: Path) -> dict[str, Any]:
    return _greedy_runs(v3_dir, _V3_ROUTING_OVERRIDES)


@pytest.fixture(scope="module")
def v3_full_ref(v3_dir: Path, v3_full) -> list:
    return _reference_rows(v3_dir, _V3_ROUTING_OVERRIDES, v3_full["runs"])


def test_v3_stacks_four_layers_and_drops_mtp_keys(v3_full):
    """4 backbone layers load; the MTP layer's keys past the stack are dropped."""
    assert v3_full["num_layers"] == 4


def test_v3_noaux_tc_matches_transformers(v3_full, v3_full_ref):
    """V3's biased grouped routing against transformers, greedy end to end."""
    _assert_parity("V3-4layers", v3_full["runs"], v3_full_ref)
