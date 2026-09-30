"""Public API surface: lazy imports keep ``import rapid_llm`` CUDA-free.

Only the names in ``__all__`` — :class:`~rapid_llm.engine.llm.LLM`, the
engines, :class:`~rapid_llm.engine.sampler.SamplingParams` — are exposed,
and each resolves on first attribute access so the import stays cheap.

Usage:
    from rapid_llm import LLM, SamplingParams
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine.async_data_parallel import AsyncDataParallelEngine
    from .engine.async_engine import AsyncLLMEngine, StreamedOutput
    from .engine.batch_planner import Request, SchedulerConfig
    from .engine.data_parallel import DataParallelEngine
    from .engine.generator import TextGenerator, VisionGenerator
    from .engine.llm import LLM
    from .engine.llm_engine import LLMEngine
    from .engine.outputs import CompletionOutput, RequestOutput
    from .engine.sampler import Sampler, SamplingParams, sample_top_p
    from .engine.scheduler import Scheduler


# Importing ``rapid_llm`` is common in CLI discovery, test collection and worker
# process startup.  Keep that operation CPU-only: engine implementations import
# the executor and Triton kernels, while lightweight values such as
# ``SamplingParams`` do not need either.  PEP 562 gives the public facade the
# same API without eagerly importing every implementation behind it.
_EXPORTS: dict[str, tuple[str, str]] = {
    "LLM": (".engine.llm", "LLM"),
    "AsyncDataParallelEngine": (".engine.async_data_parallel", "AsyncDataParallelEngine"),
    "AsyncLLMEngine": (".engine.async_engine", "AsyncLLMEngine"),
    "CompletionOutput": (".engine.outputs", "CompletionOutput"),
    "Scheduler": (".engine.scheduler", "Scheduler"),
    "DataParallelEngine": (".engine.data_parallel", "DataParallelEngine"),
    "LLMEngine": (".engine.llm_engine", "LLMEngine"),
    "Request": (".engine.batch_planner", "Request"),
    "RequestOutput": (".engine.outputs", "RequestOutput"),
    "Sampler": (".engine.sampler", "Sampler"),
    "SamplingParams": (".engine.sampler", "SamplingParams"),
    "SchedulerConfig": (".engine.batch_planner", "SchedulerConfig"),
    "StreamedOutput": (".engine.async_engine", "StreamedOutput"),
    "TextGenerator": (".engine.generator", "TextGenerator"),
    "VisionGenerator": (".engine.generator", "VisionGenerator"),
    "sample_top_p": (".engine.sampler", "sample_top_p"),
}


def __getattr__(name: str) -> Any:
    """Load a public symbol on first use and cache it in this module."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())


__version__ = "0.11.0"

__all__ = [
    "LLM",
    "AsyncDataParallelEngine",
    "AsyncLLMEngine",
    "CompletionOutput",
    "DataParallelEngine",
    "LLMEngine",
    "Request",
    "RequestOutput",
    "Sampler",
    "SamplingParams",
    "Scheduler",
    "SchedulerConfig",
    "StreamedOutput",
    "TextGenerator",
    "VisionGenerator",
    "__version__",
    "sample_top_p",
]
