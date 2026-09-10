"""``grouped_topk`` routing semantics against vLLM's reference behaviour.

DeepSeek's two grouped families route differently from the plain greedy
top-k: ``group_limited_greedy`` (V2) first picks which expert groups a token
may draw from, and ``noaux_tc`` (V2.5+/V3) additionally lets an fp32
correction bias choose the experts while the *original* sigmoid scores weigh
them. Each rule is pinned by a hand-built example where the rule — not a
tolerance — decides the outcome, and the whole function is checked against a
naive per-token transcription of vLLM's implementation across parameter
combinations.

Usage:
    pytest tests/models/test_grouped_topk.py            # rules + CPU grid
    pytest tests/models/test_grouped_topk.py -m gpu     # the device section
"""

from __future__ import annotations

import itertools

import pytest
import torch

from rapid_llm.modules.moe import grouped_topk
from tests.registry import import_needs_cuda

import_needs_cuda()


def _naive_grouped_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A per-token transcription of vLLM's ``grouped_topk``.

    Written the way the vLLM source reads — ``topk`` values as weights when
    there is no bias, gathered originals when there is one — so agreement
    with the vectorised single-path version also proves that the merge of
    the two branches is exact.
    """
    scores = (
        torch.softmax(router_logits, dim=-1)
        if scoring_func == "softmax"
        else torch.sigmoid(router_logits)
    )
    originals = scores.clone()
    n_experts = scores.shape[-1]
    per_group = n_experts // num_expert_group
    all_weights, all_ids = [], []
    for row in range(scores.shape[0]):
        if e_score_correction_bias is not None:
            biased = scores[row] + e_score_correction_bias
            grouped = biased.view(num_expert_group, per_group)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        else:
            grouped = scores[row].view(num_expert_group, per_group)
            group_scores = grouped.max(dim=-1).values
        alive = torch.zeros(n_experts, dtype=torch.bool, device=scores.device)
        for group in torch.topk(group_scores, k=topk_group).indices.tolist():
            alive[group * per_group : (group + 1) * per_group] = True
        if e_score_correction_bias is not None:
            ids = torch.topk(biased.masked_fill(~alive, float("-inf")), k=top_k).indices
            weights = originals[row, ids]
        else:
            values, ids = torch.topk(scores[row].masked_fill(~alive, float("-inf")), k=top_k)
            weights = values
        if renormalize:
            weights = weights / weights.sum()
        weights = weights * routed_scaling_factor
        all_weights.append(weights)
        all_ids.append(ids)
    return torch.stack(all_weights), torch.stack(all_ids)


def _sorted_pairs(weights: torch.Tensor, ids: torch.Tensor) -> list[tuple[int, float]]:
    """A token's ``(id, weight)`` choices in id order, so top-k order never matters."""
    return sorted(zip(ids.tolist(), weights.tolist(), strict=True))


def _assert_same_routing(
    weights: torch.Tensor,
    ids: torch.Tensor,
    ref_weights: torch.Tensor,
    ref_ids: torch.Tensor,
    context: str,
) -> None:
    """Selection exactly, weights to fp32 reduction-order tolerance.

    The batched reductions and the per-row ones round their fp32 sums in
    different orders, so weights agree to ~1 ulp (``rel=1e-6`` absorbs it);
    the expert ids are integers and must match exactly.
    """
    assert weights.dtype == torch.float32
    for row in range(ids.shape[0]):
        got = _sorted_pairs(weights[row], ids[row])
        want = _sorted_pairs(ref_weights[row], ref_ids[row])
        assert [id_ for id_, _ in got] == [id_ for id_, _ in want], (
            f"{context}, row {row}: experts {got} vs reference {want}"
        )
        assert [w for _, w in got] == pytest.approx([w for _, w in want], rel=1e-6), (
            f"{context}, row {row}: weights {got} vs reference {want}"
        )


# --------------------------------------------------------------------------- #
# The noaux_tc rules, each pinned by an example the rule alone decides
# --------------------------------------------------------------------------- #
def test_bias_selects_experts_but_original_scores_weight_them():
    """A heavy bias must route to its expert while leaving the weight tiny.

    The bias exists to move *selection* without moving the output magnitude;
    if the biased score leaked into the weight, every routed output would
    carry the bias as gain.
    """
    logits = torch.tensor([[-5.0, -4.0, +5.0, +4.0]])  # sigmoid: e2 strongest
    bias = torch.tensor([10.0, 0.0, 0.0, 0.0])  # e0 bought the win
    weights, ids = grouped_topk(
        logits,
        top_k=1,
        renormalize=False,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    assert ids.tolist() == [[0]], "the bias must decide which expert runs"
    assert weights[0, 0] == pytest.approx(torch.sigmoid(logits[0, 0]).item()), (
        "the weight must be the *original* score of the biased-in expert"
    )


def test_group_score_sums_its_two_best_when_biased():
    """With a bias, a group is the sum of its two best — one outlier can't win it.

    Group 0 owns the globally strongest biased expert; group 1 is uniformly
    strong and wins on the sum. Without the bias the group score is the plain
    max, and the outlier's group wins instead.
    """
    logits = torch.tensor([[+4.0, -3.0, +3.0, +2.9]])
    bias = torch.tensor([+8.0, 0.0, +4.0, +4.0])
    common = {
        "top_k": 2,
        "renormalize": False,
        "num_expert_group": 2,
        "topk_group": 1,
        "scoring_func": "sigmoid",
    }
    _, biased_ids = grouped_topk(logits, e_score_correction_bias=bias, **common)
    _, plain_ids = grouped_topk(logits, e_score_correction_bias=None, **common)
    assert set(biased_ids[0].tolist()) == {2, 3}, "the sum-ruled group must win"
    assert set(plain_ids[0].tolist()) == {0, 1}, "the max-ruled group must win"


def test_strongest_expert_outside_kept_groups_never_runs():
    """A masked-out expert cannot be drawn, however large its biased score.

    Group 0's outlier alone is the biggest biased score on the token, but the
    group sum keeps group 1 — the outlier's whole group is masked to ``-inf``
    before the expert top-k.
    """
    logits = torch.zeros(1, 4)
    bias = torch.tensor([10.0, 0.01, 6.0, 5.5])
    _, ids = grouped_topk(
        logits,
        top_k=2,
        renormalize=False,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    assert 0 not in ids[0].tolist(), "the losing group is masked, outlier included"
    assert set(ids[0].tolist()) == {2, 3}


def test_renormalise_then_scale_the_selected_weights():
    """``renormalize`` makes the row sum to the routed scale, not to one.

    DeepSeek ships ``norm_topk_prob`` true with ``routed_scaling_factor`` 2.5:
    renormalised to one, then scaled — the combination V3 runs with.
    """
    logits = torch.tensor([[+2.0, +1.0, -1.0, -2.0]])
    common = {
        "top_k": 2,
        "num_expert_group": 1,
        "topk_group": 1,
        "scoring_func": "sigmoid",
    }
    renormed, _ = grouped_topk(logits, renormalize=True, routed_scaling_factor=2.5, **common)
    raw, _ = grouped_topk(logits, renormalize=False, routed_scaling_factor=2.5, **common)
    assert renormed.sum(dim=-1) == pytest.approx(2.5, abs=1e-6)
    top2 = torch.sigmoid(logits).topk(2).values
    assert raw[0].tolist() == pytest.approx((top2 * 2.5).flatten().tolist())


def test_softmax_scores_share_one_sigmoid_scores_do_not():
    """The two families differ before any routing: softmax competes, sigmoid is free.

    With every expert selected, a softmax row is a partition of one (the
    shares sum to one) while a sigmoid row is a set of independent
    probabilities that overshoot it — the property V3's renormalise exists
    to correct.
    """
    torch.manual_seed(0)
    logits = torch.randn(5, 8)
    common = {
        "top_k": 8,
        "renormalize": False,
        "num_expert_group": 2,
        "topk_group": 2,
    }
    softmaxed, _ = grouped_topk(logits, scoring_func="softmax", **common)
    sigmoided, _ = grouped_topk(logits, scoring_func="sigmoid", **common)
    assert softmaxed.sum(dim=-1) == pytest.approx(1.0, abs=1e-5)
    assert (sigmoided.sum(dim=-1) > 1.0).all(), (
        "sigmoid scores are independent probabilities, not softmax shares"
    )
    assert sigmoided.dtype == torch.float32


# --------------------------------------------------------------------------- #
# The whole function against the naive reference, across the parameter grid
# --------------------------------------------------------------------------- #
# ``topk_group`` cannot exceed the group count (and ``top_k`` never exceeds
# the surviving experts at the smallest geometry, 4 = 1 group x 4 experts).
_GRID = [
    (scoring, bias, renormalize, scale, groups, topk_group, top_k)
    for scoring, bias, renormalize, scale, groups, topk_group, top_k in itertools.product(
        ("softmax", "sigmoid"),
        (False, True),
        (False, True),
        (1.0, 2.5),
        (2, 4),
        (1, 3),
        (2, 4),
    )
    if topk_group <= groups
]


@pytest.mark.parametrize(
    "scoring,bias,renormalize,scale,groups,topk_group,top_k",
    _GRID,
)
def test_matches_naive_reference(scoring, bias, renormalize, scale, groups, topk_group, top_k):
    """Every combination must reproduce vLLM's routing exactly, weights included."""
    torch.manual_seed(0)
    n_experts = 16
    logits = torch.randn(32, n_experts)
    bias = torch.randn(n_experts) if bias else None
    kwargs = {
        "top_k": top_k,
        "renormalize": renormalize,
        "num_expert_group": groups,
        "topk_group": topk_group,
        "scoring_func": scoring,
        "routed_scaling_factor": scale,
        "e_score_correction_bias": bias,
    }
    weights, ids = grouped_topk(logits, **kwargs)
    ref_weights, ref_ids = _naive_grouped_topk(logits, **kwargs)
    _assert_same_routing(
        weights,
        ids,
        ref_weights,
        ref_ids,
        f"scoring={scoring} bias={bias is not None} renorm={renormalize} "
        f"scale={scale} groups={groups} topk_group={topk_group} top_k={top_k}",
    )


# --------------------------------------------------------------------------- #
# The device production routes on
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("scoring,bias", [("softmax", False), ("sigmoid", True)])
def test_matches_naive_reference_on_the_device(scoring, bias):
    """The rules must hold where production routes: CUDA, fp32 logits.

    The CPU grid pins the semantics; what only the device can pin is the
    numeric environment the router actually runs in — ``torch.topk`` and
    the fp32 reductions take CUDA code paths there. Seeded ``randn`` logits
    make exact score ties measure-zero, so selection must agree with the
    per-row reference *and* with the CPU run of the same seed; only
    reduction rounding may differ, which the tolerance absorbs.
    """
    torch.manual_seed(0)
    logits = torch.randn(64, 16)
    bias_cpu = torch.randn(16) if bias else None
    common = {
        "top_k": 4,
        "renormalize": True,
        "num_expert_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
        "scoring_func": scoring,
    }
    device_kwargs = {
        **common,
        "e_score_correction_bias": bias_cpu.cuda() if bias_cpu is not None else None,
    }
    weights, ids = grouped_topk(logits.cuda(), **device_kwargs)
    assert weights.is_cuda and ids.is_cuda
    ref_weights, ref_ids = _naive_grouped_topk(logits.cuda(), **device_kwargs)
    _assert_same_routing(weights, ids, ref_weights, ref_ids, f"on-device {scoring}")

    cpu_kwargs = {**common, "e_score_correction_bias": bias_cpu}
    cpu_weights, cpu_ids = grouped_topk(logits, **cpu_kwargs)
    _assert_same_routing(weights, ids, cpu_weights, cpu_ids, f"device-vs-cpu {scoring}")


@pytest.mark.gpu
def test_saturated_scores_keep_the_rules_on_the_device():
    """Saturated sigmoid scores tie at exactly 1.0 in fp32 — and the rules survive.

    fp32 sigmoid saturates once ``|logit|`` passes ~17: every high expert's
    original score is exactly 1.0, every low expert's ~9e-14. Real router
    logits reach this, and it is the one input class where CUDA and CPU
    ``topk`` may pick different winners (why vLLM pins ``sorted=True`` for
    batch invariance), so the winners themselves cannot be asserted across
    devices — but every rule that holds regardless of the tie-break must:
    the weights are whoever won's *original* scores, renormalised and scaled.
    A bias leaking into them would read 1.0 + bias, not 1.0.
    """
    torch.manual_seed(0)
    logits = torch.where(torch.arange(16) % 2 == 0, 30.0, -30.0).repeat(8, 1).cuda()
    bias = torch.randn(16).cuda()
    weights, ids = grouped_topk(
        logits,
        top_k=4,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
    )
    originals = torch.sigmoid(logits)
    # The premise this test exists for: the high half sits at exactly 1.0.
    assert originals[:, 0::2].eq(1.0).all()
    unrenormed = originals.gather(1, ids)
    expected = unrenormed / unrenormed.sum(dim=-1, keepdim=True) * 2.5
    assert torch.allclose(weights, expected, rtol=1e-6), (
        "weights must be the winners' tied original scores, renormalised"
    )


# --------------------------------------------------------------------------- #
# The contract's error paths
# --------------------------------------------------------------------------- #
def test_rejects_scoring_func_outside_the_two_families():
    with pytest.raises(ValueError, match="scoring_func"):
        grouped_topk(
            torch.randn(1, 8),
            top_k=2,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            scoring_func="relu",
        )


def test_rejects_expert_count_that_does_not_split_into_groups():
    with pytest.raises(ValueError, match="divide"):
        grouped_topk(
            torch.randn(1, 10),
            top_k=2,
            renormalize=False,
            num_expert_group=3,
            topk_group=1,
            scoring_func="sigmoid",
        )
