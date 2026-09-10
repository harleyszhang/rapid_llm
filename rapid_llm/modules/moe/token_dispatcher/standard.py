"""Standard (non-EP / TP) token dispatcher: the passthrough path.

Mirrors sglang's ``StandardDispatcher`` — the non-a2a case where every rank
already holds all tokens, so ``dispatch`` hands the rows straight to the runner
and ``combine`` returns the expert output unchanged. The TP all-reduce over the
expert split stays in the layer (it owns the shared-expert sum and the
deferred-AR fence), exactly as before the refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from ..utils import MoeA2ABackend
from .base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
    register_dispatcher,
)

if TYPE_CHECKING:
    from ..router import TopKOutput


class StandardDispatchOutput(NamedTuple):
    """Passthrough dispatch result: the rows and routing, untouched."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


class StandardCombineInput(NamedTuple):
    """Passthrough combine input: the expert output, untouched."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


@register_dispatcher(MoeA2ABackend.NONE)
class StandardDispatcher(BaseDispatcher):
    """Non-EP dispatch/combine: identity, since every rank holds all tokens."""

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        return StandardDispatchOutput(hidden_states, topk_output.topk_weights, topk_output.topk_ids)

    def combine(self, combine_input) -> torch.Tensor:
        # Accept either the tensor directly (the layer's fast path) or the
        # tagged StandardCombineInput; both mean "hand the rows back as they are".
        if isinstance(combine_input, StandardCombineInput):
            return combine_input.hidden_states
        return combine_input
