"""Weight-mapping unit tests: key translation, per-layer loaders, coverage.

Checkpoint key -> parameter translation, fused gate/up and QKV block
boundaries (with GQA and scale grids), and the coverage accounting
that ``load_weights`` enforces.

Usage:
    pytest tests/models/test_weight_mapping.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rapid_llm.executor.weight_utils import hf_weight_files
from rapid_llm.models import weights
from rapid_llm.models.base import CausalLM
from rapid_llm.modules import ColumnParallelLinear, QKVParallelLinear, SparseMoeBlock
from tests.registry import import_needs_cuda

import_needs_cuda()

#: The packed-mapping the text models actually use; the translator is a pure
#: function of the key and this table, so the tests exercise the production rules
#: by passing the production table.
_PACKED = CausalLM.packed_modules_mapping

# --------------------------------------------------------------------------- #
# Tier 1: key translation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key,expected",
    [
        # Untouched: these already carry rapid_llm parameter names.
        ("embed_tokens.weight", ("embed_tokens.weight", None)),
        ("lm_head.weight", ("lm_head.weight", None)),
        ("layers.3.mlp.gate_proj.weight", ("layers.3.mlp.gate_up_proj.weight", 0)),
        ("layers.3.mlp.up_proj.weight", ("layers.3.mlp.gate_up_proj.weight", 1)),
        ("layers.3.mlp.down_proj.weight", ("layers.3.mlp.down_proj.weight", None)),
        # Flattened: the module level is folded into the parameter name.
        ("norm.weight", ("norm_weight", None)),
        ("layers.3.input_layernorm.weight", ("layers.3.input_layernorm_weight", None)),
        (
            "layers.3.post_attention_layernorm.weight",
            ("layers.3.post_attention_layernorm_weight", None),
        ),
        # Projections are submodules (LinearBase), so their parameters use dots.
        ("layers.3.self_attn.o_proj.weight", ("layers.3.self_attn.o_proj.weight", None)),
        ("layers.3.self_attn.q_norm.weight", ("layers.3.self_attn.q_norm_weight", None)),
        ("layers.3.self_attn.k_norm.weight", ("layers.3.self_attn.k_norm_weight", None)),
        # MoE router. Must not be confused with the dense ``mlp.gate_proj``.
        ("layers.3.mlp.gate.weight", ("layers.3.mlp.gate_weight", None)),
        # Fused triple: all three attention projections land in one parameter,
        # and the shard id says which block of [q | k | v] each one fills.
        ("layers.3.self_attn.q_proj.weight", ("layers.3.self_attn.qkv_proj.weight", 0)),
        ("layers.3.self_attn.k_proj.weight", ("layers.3.self_attn.qkv_proj.weight", 1)),
        ("layers.3.self_attn.v_proj.weight", ("layers.3.self_attn.qkv_proj.weight", 2)),
        ("layers.3.self_attn.q_proj.bias", ("layers.3.self_attn.qkv_proj.bias", 0)),
        ("layers.3.self_attn.k_proj.bias", ("layers.3.self_attn.qkv_proj.bias", 1)),
        ("layers.3.self_attn.v_proj.bias", ("layers.3.self_attn.qkv_proj.bias", 2)),
        # Stacked experts: (expert index, projection) with gate=0, up=1, down=2.
        ("layers.3.mlp.experts.7.gate_proj.weight", ("layers.3.mlp.experts.gate_up_proj", (7, 0))),
        ("layers.3.mlp.experts.7.up_proj.weight", ("layers.3.mlp.experts.gate_up_proj", (7, 1))),
        ("layers.3.mlp.experts.7.down_proj.weight", ("layers.3.mlp.experts.down_proj", (7, 2))),
        # The fp8 scale grid of an expert stacks under its own parameter name.
        (
            "layers.3.mlp.experts.7.gate_proj.weight_scale_inv",
            ("layers.3.mlp.experts.gate_up_proj_scale_inv", (7, 0)),
        ),
    ],
)
def test_translate_text_key_names_the_right_parameter(key: str, expected: weights.Target):
    assert weights.translate_text_key(key, _PACKED) == expected


def test_dense_gate_up_fusion_misses_the_moe_router_and_experts():
    """Only the dense ``mlp.gate_proj`` may fuse; the MoE siblings keep their names."""
    assert weights.translate_text_key("layers.0.mlp.gate.weight", _PACKED) == (
        "layers.0.mlp.gate_weight",
        None,
    )
    assert weights.translate_text_key("layers.0.mlp.experts.7.gate_proj.weight", _PACKED) == (
        "layers.0.mlp.experts.gate_up_proj",
        (7, 0),
    )


def test_strip_prefix_reports_a_non_match():
    assert weights.strip_prefix("model.layers.0", "model.") == "layers.0"
    assert weights.strip_prefix("visual.blocks.0", "model.") is None
    # The empty prefix matches everything, which is what the multimodal routers
    # use as their fall-through rule.
    assert weights.strip_prefix("lm_head.weight", "") == "lm_head.weight"


# --------------------------------------------------------------------------- #
# Tier 2: the loaders the layers bind onto their parameters
# --------------------------------------------------------------------------- #


def test_q_k_and_v_fill_three_blocks_of_unequal_width():
    """All three land in one parameter, and grouped-query attention makes q wider.

    The block boundaries are the layer's own head geometry: four one-dimensional
    query heads and two key/value heads give an 8-row parameter whose q block is
    twice the k and v blocks.
    """
    proj = QKVParallelLinear(3, num_heads=4, num_kv_heads=2, head_dim=1)
    param = proj.weight
    param.data.zero_()

    loader = param.weight_loader
    loader(param, torch.full((4, 3), 1.0), 0)
    loader(param, torch.full((2, 3), 2.0), 1)
    loader(param, torch.full((2, 3), 3.0), 2)
    assert torch.equal(param.data[:4], torch.ones(4, 3))
    assert torch.equal(param.data[4:6], torch.full((2, 3), 2.0))
    assert torch.equal(param.data[6:], torch.full((2, 3), 3.0))


def test_qkv_blocks_are_equal_width_without_gqa():
    """Multi-head attention is the same rule with all three blocks equal."""
    proj = QKVParallelLinear(3, num_heads=2, num_kv_heads=2, head_dim=1)
    param = proj.weight
    param.data.zero_()

    for shard_id in range(3):
        param.weight_loader(param, torch.full((2, 3), float(shard_id + 1)), shard_id)
    assert torch.equal(param.data[:, 0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))


def test_qkv_loader_scales_block_boundaries_to_a_scale_grid():
    """An fp8 scale grid has one row per ``group_n`` channels; the same shard ids
    must place its blocks, so the loader rescales the boundaries by the ratio of
    the parameter's rows to the weight's rows."""
    proj = QKVParallelLinear(3, num_heads=4, num_kv_heads=2, head_dim=1)
    grid = torch.zeros(4, 3)  # one row per 2 weight rows: [q | k | v] -> 2+1+1
    # The loader reads only the parameter's shape, so a plain tensor standing in
    # for a scale-grid parameter exercises the rescaling.
    proj._weight_loader(grid, torch.full((2, 3), 1.0), 0)
    proj._weight_loader(grid, torch.full((1, 3), 2.0), 1)
    proj._weight_loader(grid, torch.full((1, 3), 3.0), 2)
    assert torch.equal(grid[:, 0], torch.tensor([1.0, 1.0, 2.0, 3.0]))


def test_qkv_loader_rejects_a_tensor_without_a_shard_id():
    proj = QKVParallelLinear(3, num_heads=4, num_kv_heads=2, head_dim=1)
    with pytest.raises(ValueError, match="must name its block"):
        proj.weight.weight_loader(proj.weight, torch.zeros(8, 3), None)


def test_gate_and_up_fill_opposite_halves():
    """The dense MLP's gate/up pair splits the parameter down the middle instead."""
    proj = ColumnParallelLinear(3, 8)
    param = proj.weight
    param.data.zero_()

    param.weight_loader(param, torch.full((4, 3), 1.0), 0)
    param.weight_loader(param, torch.full((4, 3), 2.0), 1)
    assert torch.equal(param.data[:4], torch.ones(4, 3))
    assert torch.equal(param.data[4:], torch.full((4, 3), 2.0))


def test_expert_gate_and_up_fill_opposite_halves_of_their_own_slice():
    """gate/up are fused *within* each expert's slice, not across experts."""
    block = SparseMoeBlock(
        SimpleNamespace(
            num_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=2,
            norm_topk_prob=True,
            hidden_size=5,
            dtype=torch.float16,
            routed_scaling_factor=1.0,
            n_shared_experts=0,
        )
    )
    param = block.experts["gate_up_proj"]  # [experts, 2 * moe_inter, hidden]
    param.data.zero_()

    param.weight_loader(param, torch.full((2, 5), 1.0), (1, 0))
    param.weight_loader(param, torch.full((2, 5), 2.0), (1, 1))
    assert torch.equal(param.data[1, :2], torch.ones(2, 5))
    assert torch.equal(param.data[1, 2:], torch.full((2, 5), 2.0))
    # Other experts untouched.
    assert param.data[0].abs().sum() == 0
    assert param.data[2].abs().sum() == 0


def test_expert_down_proj_fills_a_whole_slice():
    block = SparseMoeBlock(
        SimpleNamespace(
            num_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=2,
            norm_topk_prob=True,
            hidden_size=5,
            dtype=torch.float16,
            routed_scaling_factor=1.0,
            n_shared_experts=0,
        )
    )
    param = block.experts["down_proj"]  # [experts, hidden, moe_inter]
    param.data.zero_()

    param.weight_loader(param, torch.full((5, 2), 7.0), (2, 2))
    assert torch.equal(param.data[2], torch.full((5, 2), 7.0))
    assert param.data[:2].abs().sum() == 0


# --------------------------------------------------------------------------- #
# Tier 3: coverage accounting
# --------------------------------------------------------------------------- #


def _half_loader(param: torch.Tensor, loaded: torch.Tensor, shard_id=None) -> torch.Tensor:
    """What a packed linear's loader does, minus the tensor-parallel narrow:
    ``shard_id`` picks a dim-0 half of the parameter."""
    view = param.data
    if shard_id is not None:
        half = view.shape[0] // 2
        view = view.narrow(0, shard_id * half, half)
    if view.shape != loaded.shape:
        raise ValueError(
            f"shape {tuple(loaded.shape)} does not fit view of shape {tuple(view.shape)}"
        )
    view.copy_(loaded)
    return view


class _TwoParams(nn.Module):
    """Minimal stand-in: a plain parameter, a fused pair, and a tie target."""

    def __init__(self) -> None:
        super().__init__()
        self.plain = nn.Parameter(torch.zeros(2, 3))
        self.fused = nn.Parameter(torch.zeros(4, 3))
        self.mirror = nn.Parameter(torch.zeros(2, 3))
        # The plain and mirror parameters keep the default whole-copy loader;
        # the fused one gets the half-selecting rule a packed linear would bind.
        self.fused.weight_loader = _half_loader


def _translate(key: str) -> weights.Target:
    table: dict[str, weights.Target] = {
        "plain": ("plain", None),
        "fused_low": ("fused", 0),
        "fused_high": ("fused", 1),
        "mirror": ("mirror", None),
        "ignore_me": None,
    }
    return table.get(key, (key, None))


def _stream(*, drop: tuple[str, ...] = ()) -> list[tuple[str, torch.Tensor]]:
    """A complete checkpoint stream for :class:`_TwoParams`, minus ``drop``."""
    full = {
        "plain": torch.ones(2, 3),
        "fused_low": torch.full((2, 3), 4.0),
        "fused_high": torch.full((2, 3), 5.0),
        "mirror": torch.full((2, 3), 7.0),
        "ignore_me": torch.zeros(1),
    }
    return [(k, v) for k, v in full.items() if k not in drop]


def test_full_coverage_succeeds():
    model = _TwoParams()
    weights.load_weights(model, _stream(), _translate)

    assert torch.equal(model.plain, torch.ones(2, 3))
    assert torch.equal(model.fused[:2], torch.full((2, 3), 4.0))
    assert torch.equal(model.fused[2:], torch.full((2, 3), 5.0))
    assert torch.equal(model.mirror, torch.full((2, 3), 7.0))


def test_missing_parameter_is_reported_by_name():
    with pytest.raises(ValueError, match=r"never written.*plain"):
        weights.load_weights(_TwoParams(), _stream(drop=("plain",)), _translate)


def test_half_written_fused_parameter_is_rejected():
    """Only part of a fused parameter arriving is the failure a key-set check cannot see."""
    with pytest.raises(ValueError, match=r"partially written.*fused"):
        weights.load_weights(_TwoParams(), _stream(drop=("fused_high",)), _translate)


def test_a_parameter_written_twice_is_rejected():
    """Two keys competing for one destination silently loses whichever lost the race."""
    duplicated = [*_stream(), ("plain", torch.zeros(2, 3))]
    with pytest.raises(ValueError, match=r"plain \(12 of 6 elements\)"):
        weights.load_weights(_TwoParams(), duplicated, _translate)


def test_shape_mismatch_names_both_shapes():
    with pytest.raises(ValueError, match=r"shape \(3, 3\).*\(2, 3\)"):
        weights.load_weights(_TwoParams(), [("plain", torch.zeros(3, 3))], _translate)


def test_shape_mismatch_names_the_key_and_parameter():
    """The loader reports the shapes; the loop adds which key and parameter met."""
    with pytest.raises(ValueError, match=r"checkpoint key 'plain' -> 'plain'"):
        weights.load_weights(_TwoParams(), [("plain", torch.zeros(3, 3))], _translate)


def test_key_mapping_to_a_nonexistent_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter 'typo'"):
        weights.load_weights(_TwoParams(), [("typo", torch.zeros(2, 3))], _translate)


def test_tie_fills_a_parameter_the_checkpoint_omitted():
    """Tied checkpoints ship no ``lm_head``, and the coverage check would reject the gap."""
    model = _TwoParams()
    weights.load_weights(model, _stream(drop=("mirror",)), _translate, tied={"mirror": "plain"})
    assert torch.equal(model.mirror, model.plain)


def test_tie_does_not_override_a_shipped_tensor():
    """A tie is a fallback: an untied checkpoint's own lm_head must survive it."""
    model = _TwoParams()
    weights.load_weights(model, _stream(), _translate, tied={"mirror": "plain"})
    assert torch.equal(model.mirror, torch.full((2, 3), 7.0))


# --------------------------------------------------------------------------- #
# Checkpoint file discovery
# --------------------------------------------------------------------------- #


def test_safetensors_wins_over_legacy_bin(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "pytorch_model.bin").touch()
    assert [p.name for p in hf_weight_files(tmp_path)] == ["model.safetensors"]


def test_all_shards_are_returned_in_order(tmp_path):
    for name in ("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"):
        (tmp_path / name).touch()
    assert [p.name for p in hf_weight_files(tmp_path)] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_a_directory_without_weights_says_what_it_wanted(tmp_path):
    with pytest.raises(FileNotFoundError, match="HuggingFace checkpoint directory"):
        hf_weight_files(tmp_path)
