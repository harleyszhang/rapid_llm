"""DeepSeek-V4-Flash-6layers must track its fp32 oracle across a TP=2 grid.

The checkpoint is the official trim — six layers, DSpark storage: fp8 e4m3
linears with 128x128 e8m0 block scales, MXFP4 routed experts, hash routing on
the first three MoE layers and score routing on the last three, SWA/CSA/HCA
attention (compress_ratios ``[0,0,4,128,4,128]``) — so this gate covers the
whole V4-Flash forward stack in weight-only-quantised form.

The two arms share a seed, not a runtime: rapid_llm loads the DSpark files
directly over the TP=2 harness (``tests.layer.deepseek.v4_lite_payload``),
while transformers runs the in-memory DSpark->HF conversion as an fp32 CPU
oracle (``tests.layer.dspark_to_hf``). fp8/MXFP4 weights and bf16 activations
cannot be token-exact against an fp32 reference, so the gate pins the
structural claim instead: the first token of every prompt is a shared
decision, any divergence sits at a near-tie (both top1-top2 margins under
``_TIE_MARGIN``, each side's pick inside the other's top-2), and the matching
prefixes stay long enough that a wrong path could not hide behind the noise.

Calibration (2026-09-04, the lab box: 2x A10, 64 cores; see
``docs/benchmark_models.md``, accuracy section): 30/32, 32/32 and 12/32
matching steps for the 64/256/1024-token prompts, with the two independent
divergences at margins 0.125/0.037 and 0.000/0.071.

Usage:
    pytest tests/golden/test_deepseek_v4_flash_parity.py

Needs the V4-Flash-6layers checkpoint (override with
``RAPID_LLM_TEST_DSV4_DIR``), two CUDA devices, and ~200 GiB of free RAM for
the fp32 oracle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.conftest import REPO_ROOT, checkpoint_problem
from tests.layer.deepseek import GREEDY_STEPS, V4_PREFILL_LENS

# No ``weights`` mark: that mark binds a test to the shared ``model_dir``
# fixture (the small dense default), and this file gates on its own checkpoint
# via ``v4_dir`` below — the same no-silent-skip policy, one directory over.
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: The checkpoint under test; the repository's ``my_weight/`` tree symlinks
#: the shared lab store in, like the other DeepSeek gates.
_DSV4 = "my_weight/DeepSeek-V4-Flash-6layers"

#: Free RAM the fp32 oracle needs: the logical weights are ~160 GiB in fp32
#: (20.5e9 stored MXFP4 elements — half a byte each — doubling to ~41e9
#: logical, four bytes each, plus the linears), and one layer's transient
#: build stacks on top; 200 GiB over-provisions rather than OOMing mid-run.
_ORACLE_MIN_FREE_RAM = 200 * 2**30

#: Matching greedy steps required per prompt length. Measured 30/32, 32/32
#: and 12/32 on 2026-09-04; the floors keep headroom under them, because a
#: wrong attention/quantisation path moves every length at once while bf16
#: noise moves a near-tie by a step or two.
_MATCH_FLOORS = {64: 24, 256: 26, 1024: 8}

#: Top1-top2 logprob margin under which a divergent step counts as a coin
#: flip: both measured independent divergences sat at 0.125 or below, while
#: confident decisions on this stack differ by nats.
_TIE_MARGIN = 0.5


def _mem_available() -> int | None:
    """Free RAM in bytes from /proc/meminfo, or ``None`` where unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _dsv4_problem(path: Path) -> str | None:
    """Why this machine cannot run the gate, or ``None`` if it can."""
    problem = checkpoint_problem(path)
    if problem:
        return problem
    if torch.cuda.device_count() < 2:
        return "TP=2 needs two CUDA devices"
    free = _mem_available()
    if free is not None and free < _ORACLE_MIN_FREE_RAM:
        return (
            f"the fp32 oracle needs ~{_ORACLE_MIN_FREE_RAM // 2**30} GiB of free RAM, "
            f"found {free // 2**30} GiB"
        )
    return None


@pytest.fixture(scope="module")
def v4_dir() -> Path:
    """The checkpoint under test, under the golden gate's no-silent-skip policy."""
    path = Path(os.environ.get("RAPID_LLM_TEST_DSV4_DIR", _DSV4))
    if not path.is_absolute():
        path = REPO_ROOT / path
    problem = _dsv4_problem(path)
    if problem:
        if os.environ.get("RAPID_LLM_GOLDEN_STRICT", "") == "1":
            pytest.fail(f"GOLDEN GATE FAIL: {problem}", pytrace=False)
        pytest.xfail(f"UNVERIFIED: {problem}")
    return path


@pytest.fixture(scope="module")
def lite(v4_dir: Path) -> list[dict[str, Any]]:
    """rapid_llm's arm: DSpark storage read directly, fp8/MXFP4 kernels, TP=2.

    A generous timeout: it covers a 22 GB checkpoint load plus the three
    prompts on a box that may be shared, where the harness default (300 s)
    would fire on load time alone.
    """
    from tests.distributed.tp_harness import run_on_tp_ranks
    from tests.layer.deepseek import v4_lite_payload

    return run_on_tp_ranks(v4_lite_payload, tp_size=2, timeout=1800)[0]["prompts"]


@pytest.fixture(scope="module")
def oracle(v4_dir: Path) -> list[dict[str, Any]]:
    """transformers' arm: the fp32 CPU oracle over the same seeded prompts."""
    from tests.layer.deepseek import v4_oracle_records

    return v4_oracle_records(v4_dir)


def test_both_arms_record_every_step(lite, oracle):
    """Guards the comparisons below, which assert nothing over an empty record set."""
    for arm, records in (("lite", lite), ("oracle", oracle)):
        assert [p["seq_len"] for p in records] == V4_PREFILL_LENS, arm
        for prompt in records:
            assert len(prompt["greedy_tokens"]) == GREEDY_STEPS, (arm, prompt["seq_len"])
            for step in prompt["steps"]:
                ids = [token for token, _ in step["top5"]]
                assert len(ids) == len(set(ids)) == 5, (arm, prompt["seq_len"])


def test_the_first_token_is_a_shared_decision(lite, oracle):
    """Step 0 reads only the prompt: quantisation noise must not move its argmax.

    Every prompt measured a matching first token — the shared decision the
    rest of the walk is anchored to. A miss here is a wrong prefill
    (attention geometry, dequantisation, routing), not rounding.
    """
    for lp, op in zip(lite, oracle, strict=True):
        assert lp["greedy_tokens"][0] == op["greedy_tokens"][0], (
            f"seq {lp['seq_len']}: the first token diverges "
            f"({lp['greedy_tokens'][0]} vs {op['greedy_tokens'][0]}) — that is a "
            f"wrong prefill, not rounding"
        )


def test_divergences_are_ties(lite, oracle):
    """A divergence may only be a step both stacks could call either way.

    At the first differing step the two arms consumed identical context, so
    their distributions are comparable: both top1-top2 margins must sit under
    ``_TIE_MARGIN`` and each side's token must be inside the other's top-2 —
    the shape both measured independent divergences had. Past that step the
    contexts genuinely fork, and nothing further is comparable.
    """
    for lp, op in zip(lite, oracle, strict=True):
        a, b = lp["greedy_tokens"], op["greedy_tokens"]
        where = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
        if where is None:
            continue
        lite_top = [token for token, _ in lp["steps"][where]["top5"]]
        oracle_top = [token for token, _ in op["steps"][where]["top5"]]
        lite_margin = lp["steps"][where]["top5"][0][1] - lp["steps"][where]["top5"][1][1]
        oracle_margin = op["steps"][where]["top5"][0][1] - op["steps"][where]["top5"][1][1]
        assert max(lite_margin, oracle_margin) <= _TIE_MARGIN, (
            f"seq {lp['seq_len']} step {where}: divergence at margins "
            f"{lite_margin:.4f}/{oracle_margin:.4f} — a confident step, so one side "
            f"is wrong, not rounding"
        )
        assert b[where] in lite_top[:2], (
            f"seq {lp['seq_len']} step {where}: the oracle picked {b[where]}, outside "
            f"lite's top-2 {lite_top[:2]} — that is divergence, not a tie"
        )
        assert a[where] in oracle_top[:2], (
            f"seq {lp['seq_len']} step {where}: lite picked {a[where]}, outside the "
            f"oracle's top-2 {oracle_top[:2]} — that is divergence, not a tie"
        )


def test_matching_prefixes_stay_long_enough(lite, oracle):
    """The per-length floors under the measured matching steps.

    The floors keep headroom under the calibration (30/32, 32/32, 12/32): a
    shorter match on every length at once is a wrong path showing through, a
    step or two is bf16 noise near a tie.
    """
    for lp, op in zip(lite, oracle, strict=True):
        matched = sum(x == y for x, y in zip(lp["greedy_tokens"], op["greedy_tokens"], strict=True))
        floor = _MATCH_FLOORS[lp["seq_len"]]
        assert matched >= floor, (
            f"seq {lp['seq_len']}: only {matched}/{GREEDY_STEPS} steps match the oracle, "
            f"below the floor {floor}/{GREEDY_STEPS} — divergence, not rounding"
        )
