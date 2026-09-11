"""Generation engines: two batching strategies over one shared executor.

:class:`ContinuousBatchingEngine` interleaves prefills and decodes step by
step while :class:`LLMEngine` runs one-shot batches; both drive the same
:class:`~rapid_llm.executor.executor.Executor`. Imports are lazy so the
package import stays CUDA-free.

Usage:
    from rapid_llm.engine import LLMEngine, SamplingParams
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_data_parallel import AsyncDataParallelEngine
    from .async_engine import AsyncLLMEngine, StreamedOutput
    from .continuous_engine import ContinuousBatchingEngine
    from .data_parallel import DataParallelEngine
    from .generator import TextGenerator, VisionGenerator
    from .llm import LLM
    from .llm_engine import LLMEngine
    from .outputs import CompletionOutput, RequestOutput
    from .reasoning import ReasoningSplitter
    from .sampler import BatchedSamplingParams, Sampler, SamplingParams, sample_top_p
    from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig
    from .tool_parser import (
        DeepSeekToolParser,
        QwenToolParser,
        ToolCall,
        ToolCallDelta,
        ToolParser,
        ToolStream,
    )


# The submodules pull the executor and the Triton kernels; values such as
# ``SamplingParams`` or ``SchedulerConfig`` do not need either. Resolve the
# facade lazily so importing one lightweight symbol stays CPU-only.
_EXPORTS: dict[str, tuple[str, str]] = {
    "LLM": (".llm", "LLM"),
    "AsyncDataParallelEngine": (".async_data_parallel", "AsyncDataParallelEngine"),
    "AsyncLLMEngine": (".async_engine", "AsyncLLMEngine"),
    "BatchedSamplingParams": (".sampler", "BatchedSamplingParams"),
    "CompletionOutput": (".outputs", "CompletionOutput"),
    "ContinuousBatchingEngine": (".continuous_engine", "ContinuousBatchingEngine"),
    "DataParallelEngine": (".data_parallel", "DataParallelEngine"),
    "DeepSeekToolParser": (".tool_parser", "DeepSeekToolParser"),
    "LLMEngine": (".llm_engine", "LLMEngine"),
    "QwenToolParser": (".tool_parser", "QwenToolParser"),
    "ReasoningSplitter": (".reasoning", "ReasoningSplitter"),
    "Request": (".scheduler", "Request"),
    "RequestOutput": (".outputs", "RequestOutput"),
    "RequestStatus": (".scheduler", "RequestStatus"),
    "Sampler": (".sampler", "Sampler"),
    "SamplingParams": (".sampler", "SamplingParams"),
    "Scheduler": (".scheduler", "Scheduler"),
    "SchedulerConfig": (".scheduler", "SchedulerConfig"),
    "StreamedOutput": (".async_engine", "StreamedOutput"),
    "TextGenerator": (".generator", "TextGenerator"),
    "ToolCall": (".tool_parser", "ToolCall"),
    "ToolCallDelta": (".tool_parser", "ToolCallDelta"),
    "ToolParser": (".tool_parser", "ToolParser"),
    "ToolStream": (".tool_parser", "ToolStream"),
    "VisionGenerator": (".generator", "VisionGenerator"),
    "sample_top_p": (".sampler", "sample_top_p"),
}


def __getattr__(name: str) -> Any:
    """Resolve the engine facade without importing unrelated GPU modules."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())


__all__ = [
    "LLM",
    "AsyncDataParallelEngine",
    "AsyncLLMEngine",
    "BatchedSamplingParams",
    "CompletionOutput",
    "ContinuousBatchingEngine",
    "DataParallelEngine",
    "DeepSeekToolParser",
    "LLMEngine",
    "QwenToolParser",
    "ReasoningSplitter",
    "Request",
    "RequestOutput",
    "RequestStatus",
    "Sampler",
    "SamplingParams",
    "Scheduler",
    "SchedulerConfig",
    "StreamedOutput",
    "TextGenerator",
    "ToolCall",
    "ToolCallDelta",
    "ToolParser",
    "ToolStream",
    "VisionGenerator",
    "sample_top_p",
]
