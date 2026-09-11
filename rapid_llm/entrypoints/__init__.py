"""Server entrypoints: the OpenAI-compatible HTTP API.

Re-exports the request models :class:`CompletionRequest` and
:class:`ChatCompletionRequest`; the server itself lives in
:mod:`rapid_llm.entrypoints.api_server`.

Usage:
    from rapid_llm.entrypoints import CompletionRequest
"""

from .protocol import ChatCompletionRequest, CompletionRequest

__all__ = ["ChatCompletionRequest", "CompletionRequest"]
