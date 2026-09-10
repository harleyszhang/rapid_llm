"""CPU tests for multimodal helpers (no vision tower, no GPU).

``merge_multimodal_embeddings`` placeholder splicing: replacement
positions, multiple placeholder ids, count-mismatch errors, and
higher-rank vision inputs.

Usage:
    pytest tests/multimodal/test_multimodal.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.models.interfaces import merge_multimodal_embeddings
from tests.registry import import_needs_cuda

import_needs_cuda()


def test_merge_replaces_placeholder_positions():
    input_ids = torch.tensor([[1, 42, 42, 3]])  # placeholder id = 42
    inputs_embeds = torch.zeros(1, 4, 8)
    vision = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)

    out = merge_multimodal_embeddings(input_ids, inputs_embeds, vision, placeholder_token_ids=42)

    assert torch.equal(out[0, 1], vision[0])
    assert torch.equal(out[0, 2], vision[1])
    # Non-placeholder positions must stay untouched.
    assert torch.equal(out[0, 0], torch.zeros(8))
    assert torch.equal(out[0, 3], torch.zeros(8))


def test_merge_accepts_multiple_placeholder_ids():
    input_ids = torch.tensor([[1, 42, 43, 3]])
    inputs_embeds = torch.zeros(1, 4, 4)
    vision = torch.ones(2, 4)

    out = merge_multimodal_embeddings(
        input_ids, inputs_embeds, vision, placeholder_token_ids=(42, 43)
    )
    assert torch.equal(out[0, 1], torch.ones(4))
    assert torch.equal(out[0, 2], torch.ones(4))


def test_merge_raises_on_placeholder_count_mismatch():
    """A silent pad-or-truncate on mismatch would hide processor/config drift."""
    input_ids = torch.tensor([[1, 42, 42, 42]])
    inputs_embeds = torch.zeros(1, 4, 4)
    vision = torch.zeros(2, 4)  # only 2 embeddings for 3 placeholders

    with pytest.raises(ValueError, match="does not match"):
        merge_multimodal_embeddings(input_ids, inputs_embeds, vision, placeholder_token_ids=42)


def test_merge_accepts_higher_rank_vision_input():
    """The helper must reshape any multi-image / batched vision tensor into ``[N, hidden]``."""
    input_ids = torch.tensor([[1, 7, 7, 7, 7]])
    inputs_embeds = torch.zeros(1, 5, 6)
    # 2 images, 2 patches each — total 4 placeholders, hidden=6.
    vision = torch.arange(2 * 2 * 6, dtype=torch.float32).reshape(2, 2, 6)

    out = merge_multimodal_embeddings(input_ids, inputs_embeds, vision, placeholder_token_ids=7)

    filled = out[0, 1:5].reshape(4, 6)
    assert torch.equal(filled, vision.reshape(4, 6))
