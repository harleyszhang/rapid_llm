"""Token-dispatch seam (stages 2 & 5): the ``BaseDispatcher`` contract + registry.

Mirrors sglang's ``token_dispatcher/base.py``: a ``BaseDispatcher`` ABC, the
``DispatchOutput``/``CombineInput`` protocols (each tagged with a ``*Format`` so
the runner can branch on layout), and a decorator registry keyed by
:class:`~rapid_llm.modules.moe.utils.MoeA2ABackend`. rapid_llm registers two
backends: the non-EP passthrough (:class:`~.standard.StandardDispatcher`) and the
all-to-all EP path (:class:`~.all_to_all.AllToAllDispatcher`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..utils import MoeA2ABackend

if TYPE_CHECKING:
    import torch

    from ..router import TopKOutput


class DispatchOutputFormat(Enum):
    """Layout tag on a dispatch result, so the runner can branch without isinstance."""

    STANDARD = "standard"
    ALL_TO_ALL = "all_to_all"


class CombineInputFormat(Enum):
    """Layout tag on a combine input (symmetric with :class:`DispatchOutputFormat`)."""

    STANDARD = "standard"
    ALL_TO_ALL = "all_to_all"


@runtime_checkable
class DispatchOutput(Protocol):
    """What a dispatcher hands the runner: the received rows plus a format tag."""

    @property
    def format(self) -> DispatchOutputFormat: ...


@runtime_checkable
class CombineInput(Protocol):
    """What the runner hands back to combine: the expert output plus a format tag."""

    @property
    def format(self) -> CombineInputFormat: ...


class BaseDispatcher(ABC):
    """The dispatch/combine seam a routed MoE layer talks to (stages 2 & 5).

    Concrete dispatchers own the comm path; the layer only sees
    :meth:`dispatch` (tokens out → this rank's expert batch) and :meth:`combine`
    (expert results → the per-token weighted sum). The all-to-all backend adds a
    two-phase ``*_a``/``*_b`` split beneath this for two-batch overlap.
    """

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput) -> DispatchOutput:
        """Route ``hidden_states`` to the ranks/experts ``topk_output`` selects."""

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        """Reduce expert results back to ``[tokens, hidden]`` on the origin rank."""


#: backend -> dispatcher class, populated by :func:`register_dispatcher`.
_DISPATCHER_REGISTRY: dict[MoeA2ABackend, type] = {}


def register_dispatcher(backend: MoeA2ABackend):
    """Register a dispatcher class under ``backend`` (sglang decorator style)."""

    def decorator(cls: type) -> type:
        _DISPATCHER_REGISTRY[backend] = cls
        return cls

    return decorator


def get_dispatcher_class(backend: MoeA2ABackend) -> type:
    """The dispatcher class registered for ``backend`` (raises if unregistered)."""
    try:
        return _DISPATCHER_REGISTRY[backend]
    except KeyError as exc:
        raise KeyError(f"no MoE dispatcher registered for backend {backend}") from exc


def get_dispatcher(backend: MoeA2ABackend, *args, **kwargs):
    """Construct the dispatcher for ``backend`` with the backend's own ctor args."""
    return get_dispatcher_class(backend)(*args, **kwargs)
