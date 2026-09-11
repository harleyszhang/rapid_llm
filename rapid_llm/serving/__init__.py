"""The serving front end: parsing, templating, tools and SSE.

This layer only translates between the wire protocol and the engine's
token-level API: request models here, chat templating and the streaming
output parsers (reasoning, tool calls) beside them. Scheduling and
inference stay in :mod:`rapid_llm.engine`.

Re-exports the request models :class:`CompletionRequest` and
:class:`ChatCompletionRequest`; the server itself lives in
:mod:`rapid_llm.serving.api_server`. The models are pydantic-backed (the
``serve`` extra), so they resolve lazily: the parsers next door import
cleanly without it.

Usage:
    from rapid_llm.serving import CompletionRequest
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import ChatCompletionRequest, CompletionRequest

# The only pydantic consumers in this package are the wire models; everything
# else (tool parsing, reasoning splitting) must import without the extra.
_EXPORTS: dict[str, tuple[str, str]] = {
    "ChatCompletionRequest": (".protocol", "ChatCompletionRequest"),
    "CompletionRequest": (".protocol", "CompletionRequest"),
}


def __getattr__(name: str) -> Any:
    """Resolve the wire models without importing pydantic up front."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())


__all__ = ["ChatCompletionRequest", "CompletionRequest"]
