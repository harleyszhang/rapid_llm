"""Streaming tool-call extraction: the markup, not the model, does the talking.

A :class:`ToolParser` watches the content channel (reasoning already stripped
by :mod:`~rapid_llm.engine.reasoning`) and pulls structured tool calls out of
the markup as it arrives — a call's first delta carries its id and name, the
arguments stream as raw JSON fragments, and text outside any call keeps
flowing as ordinary content; nothing waits for the closing markup.

Like vLLM's reasoning/tool-parser split, but the contract is stated the other
way round: vLLM re-feeds the whole text each step, these are incremental state
machines over the detokenizer's deltas, and the axiom is the reasoning
splitter's — any chunking must concatenate to what :meth:`ToolParser.parse`
returns in one shot. Two families ship: DeepSeek-V3's fullwidth-bar markers
with fenced JSON and Qwen's ``tool_call`` element wrapping one JSON object;
marker literals are assembled at import, never written out (the edit transport
strips anything that parses as markup).

Usage:
    parser = ToolParser.for_model("deepseek")
    for delta in stream:
        stream_out = parser.feed(delta)
    tail = parser.finish()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .reasoning import _partial_suffix

# --------------------------------------------------------------------------- #
# Wire types
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallDelta:
    """One increment of one tool call, OpenAI's streaming shape.

    ``id`` and ``name`` ride the first delta of a call only; ``arguments``
    carries raw JSON fragments from then on. A same-index run of deltas
    concatenates into one call.
    """

    index: int
    arguments: str = ""
    id: str | None = None
    name: str | None = None


@dataclass
class ToolCall:
    """A completed call: what a one-shot parse hands back."""

    name: str
    arguments: str
    id: str = ""


@dataclass
class ToolStream:
    """What one feeding step produced on both channels."""

    content: str = ""
    calls: list[ToolCallDelta] = field(default_factory=list)


# Marker assembly — see the module docstring.
_BAR = "\uff5c"  # fullwidth bar: DeepSeek special tokens' delimiter
_US = "\u2581"  # lower-one-eighth block: DeepSeek's space surrogate
_DS_CALLS_BEGIN = "<" + _BAR + "tool" + _US + "calls" + _US + "begin" + _BAR + ">"
_DS_CALL_BEGIN = "<" + _BAR + "tool" + _US + "call" + _US + "begin" + _BAR + ">"
_DS_SEP = "<" + _BAR + "tool" + _US + "sep" + _BAR + ">"
_DS_CALL_END = "<" + _BAR + "tool" + _US + "call" + _US + "end" + _BAR + ">"
_DS_CALLS_END = "<" + _BAR + "tool" + _US + "calls" + _US + "end" + _BAR + ">"
_BT = chr(96) * 3  # the code fence
_DS_HEADER = _DS_CALL_BEGIN + "function" + _DS_SEP
_DS_FENCE = _BT + "json\n"
# Arguments run from after the fence up to this terminator.
_DS_ARGS_END = "\n" + _BT + _DS_CALL_END

_QWEN_OPEN = "<" + "tool_call" + ">"
_QWEN_CLOSE = "<" + "/tool_call" + ">"


def _call_id(index: int) -> str:
    return f"call_{index}"


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


class ToolParser(ABC):
    """Base: incremental detection of tool calls inside the content stream.

    Subclasses guarantee the axiom: for any text and any split of it into
    deltas, concatenating each channel over ``feed``/``finish`` equals
    ``parse`` over the whole text. Malformed markup degrades to content (or
    to a truncated call) through the same path both ways, so the axiom holds
    there too.
    """

    #: Registry key under which :meth:`for_model` finds each family.
    name: str = "base"

    @abstractmethod
    def feed(self, delta: str) -> ToolStream:
        """Consume one increment of content-channel text."""

    @abstractmethod
    def finish(self) -> ToolStream:
        """Flush buffers at end of stream; unterminated markup degrades."""

    @classmethod
    def parse(cls, text: str) -> tuple[str, list[ToolCall]]:
        """One-shot semantics: the same split, computed in one call."""
        parser = cls()
        content = ""
        deltas: list[ToolCallDelta] = []
        for stream in (parser.feed(text), parser.finish()):
            content += stream.content
            deltas += stream.calls
        return content, _merge(deltas)

    @staticmethod
    def for_model(family: str) -> ToolParser:
        """Instantiate the parser registered under ``family``."""
        try:
            return _REGISTRY[family]()
        except KeyError as exc:
            known = ", ".join(sorted(_REGISTRY))
            raise ValueError(f"unknown tool parser {family!r}; known: {known}") from exc


def _merge(deltas: list[ToolCallDelta]) -> list[ToolCall]:
    """Concatenate deltas into calls, keyed by their call index.

    Keying by index rather than "delta carries an id" is what lets a
    truncated call survive end-of-stream: ``finish`` may append a bare
    arguments delta for the call ``feed`` already opened, and the two
    halves still fuse into one call.
    """
    calls: dict[int, ToolCall] = {}
    order: list[int] = []
    for delta in deltas:
        if delta.index not in calls:
            calls[delta.index] = ToolCall(
                id=delta.id or "", name=delta.name or "", arguments=delta.arguments
            )
            order.append(delta.index)
        else:
            calls[delta.index].arguments += delta.arguments
    return [calls[index] for index in order]


# --------------------------------------------------------------------------- #
# DeepSeek-V3 family: fenced JSON between fullwidth-bar markers
# --------------------------------------------------------------------------- #


class DeepSeekToolParser(ToolParser):
    """DeepSeek-V3 markup: a call header line, a fenced JSON block, a trailer.

    The name becomes known when its line ends — before any argument text —
    so the call's first delta goes out ahead of the arguments.
    """

    name = "deepseek"

    def __init__(self) -> None:
        self._state = "outside"
        self._held = ""
        self._index = 0

    def feed(self, delta: str) -> ToolStream:
        text = self._held + delta
        self._held = ""
        out = ToolStream()
        while text:
            text, blocked = getattr(self, f"_step_{self._state}")(text, out)
            if blocked:
                self._held = text
                break
        return out

    def finish(self) -> ToolStream:
        out = ToolStream()
        if self._state == "args":
            # A call mid-arguments at end of stream: emit what arrived; a
            # truncated call the client can see beats text that vanishes.
            out.calls = [ToolCallDelta(index=self._index - 1, arguments=self._held)]
        else:
            # Outside, or inside unterminated markup: the held tail — a
            # partial marker or name fragment — is ordinary text.
            out.content = self._held
        self._held = ""
        return out

    # Each step consumes what it can, moves the state machine on itself, and
    # returns (remainder, blocked_for_more_input).
    def _step_outside(self, text: str, out: ToolStream) -> tuple[str, bool]:
        index = text.find(_DS_CALLS_BEGIN)
        if index >= 0:
            out.content += text[:index]
            self._state = "header"
            return text[index + len(_DS_CALLS_BEGIN) :], False
        held = _partial_suffix(text, _DS_CALLS_BEGIN)
        out.content += text[: len(text) - len(held)]
        return held, bool(held)

    def _step_header(self, text: str, out: ToolStream) -> tuple[str, bool]:
        if text.startswith(_DS_HEADER):
            self._state = "name"
            return text[len(_DS_HEADER) :], False
        if _DS_HEADER.startswith(text):
            return text, True
        # Non-conforming call header: the section degrades to content. The
        # begin marker was already consumed in the outside step and is not
        # re-emitted — the one-shot parse takes the same path, so they agree.
        out.content += text
        self._state = "outside"
        return "", False

    def _step_name(self, text: str, out: ToolStream) -> tuple[str, bool]:
        index = text.find("\n")
        if index < 0:
            return text, True
        out.calls.append(
            ToolCallDelta(index=self._index, id=_call_id(self._index), name=text[:index])
        )
        self._index += 1
        self._state = "fence"
        return text[index + 1 :], False

    def _step_fence(self, text: str, out: ToolStream) -> tuple[str, bool]:
        if text.startswith(_DS_FENCE):
            self._state = "args"
            return text[len(_DS_FENCE) :], False
        if _DS_FENCE.startswith(text):
            return text, True
        out.content += text
        self._state = "outside"
        return "", False

    def _step_args(self, text: str, out: ToolStream) -> tuple[str, bool]:
        index = text.find(_DS_ARGS_END)
        if index >= 0:
            out.calls.append(ToolCallDelta(index=self._index - 1, arguments=text[:index]))
            self._state = "trailer"
            return text[index + len(_DS_ARGS_END) :], False
        # Strict JSON has no bare newlines, so any trailing "\n" + fence
        # prefix can only be the terminator forming; hold it back.
        held = _partial_suffix(text, _DS_ARGS_END)
        out.calls.append(
            ToolCallDelta(index=self._index - 1, arguments=text[: len(text) - len(held)])
        )
        return held, True

    def _step_trailer(self, text: str, out: ToolStream) -> tuple[str, bool]:
        if text.startswith(_DS_CALLS_END):
            self._state = "outside"
            return text[len(_DS_CALLS_END) :], False
        if _DS_CALLS_END.startswith(text):
            return text, True
        self._state = "header"  # another call follows in the same section
        return text, False


# --------------------------------------------------------------------------- #
# Qwen family: one JSON object per tool_call element
# --------------------------------------------------------------------------- #


class _JsonCallScanner:
    """Character scan of ``{"name": ..., "arguments": {...}}``.

    Surfaces ``name`` the moment its string closes and streams the
    ``arguments`` value's raw JSON text as it arrives; string-aware brace
    counting decides when the object is whole. Keys may arrive in either
    order — the scan keeps walking the top-level object after the arguments
    value closes, so a trailing name still fires. Characters are consumed
    one at a time so any delta boundary lands inside the scan without
    special cases.
    """

    _KEY, _COLON, _NAME, _OTHER, _ARGS, _DONE = range(6)

    def __init__(self) -> None:
        self.name: str | None = None
        self._state = self._KEY
        self._in_key = False  # between a key's quotes
        self._key = ""
        self._buf = ""  # characters of the current key or name value
        self._depth = 0  # top-level object depth (0 before its brace)
        self._args_depth = 0  # depth inside the arguments value
        self._other_depth = 0  # nesting while skipping an unknown value
        self._in_string = False  # inside OTHER/ARGS strings
        self._escaped = False

    def feed(self, text: str) -> tuple[str, bool, int]:
        """Consume text; returns ``(arguments_delta, complete, consumed)``.

        Characters past a complete object are left unconsumed — the caller
        still owns them.
        """
        args = ""
        consumed = 0
        for char in text:
            if self._state == self._DONE:
                break
            args += self._take(char)
            consumed += 1
        return args, self._state == self._DONE, consumed

    def _take(self, char: str) -> str:
        if self._state == self._ARGS:
            return self._take_args(char)
        if self._state == self._NAME:
            return self._take_name(char)
        if self._state == self._OTHER:
            return self._take_other(char)
        if self._state == self._COLON:
            return self._take_colon(char)
        return self._take_key(char)

    def _track_string(self, char: str) -> None:
        if self._escaped:
            self._escaped = False
        elif char == "\\":
            self._escaped = True
        elif char == '"':
            self._in_string = False

    def _take_key(self, char: str) -> str:
        # A key is quoted text: the first quote opens it, the next closes.
        # Treating the first as a closer (the obvious shorthand) never reads
        # a key at all — every key comes out empty and nothing routes.
        if self._in_key:
            if char == '"':
                self._key, self._buf = self._buf, ""
                self._in_key = False
                self._state = self._COLON
            else:
                self._buf += char
        elif char == '"':
            self._in_key = True
            self._buf = ""
        elif char == "{":
            self._depth += 1  # the top-level object opens
        elif char == "}":
            self._depth -= 1
            if self._depth == 0:
                self._state = self._DONE  # object complete, name or not
        # commas and whitespace between keys are structural
        return ""

    def _take_colon(self, char: str) -> str:
        if char == ":":
            if self._key == "name":
                self._state = self._NAME
                self._in_string = False  # the value's opening quote is next
                self._escaped = False
            elif self._key == "arguments":
                self._state = self._ARGS
                self._args_depth = 0
            else:
                self._state = self._OTHER
                self._other_depth = 0
                self._in_string = False
            self._buf = ""
        return ""

    def _take_name(self, char: str) -> str:
        if not self._in_string:
            if char == '"':
                self._in_string = True  # the value starts
            # whitespace between the colon and the value is structural
            return ""
        if self._escaped:
            self._buf += char
            self._escaped = False
        elif char == "\\":
            self._buf += char
            self._escaped = True
        elif char == '"':
            self.name = self._buf
            self._in_string = False
            self._state = self._KEY  # on to the next key
        else:
            self._buf += char
        return ""

    def _take_other(self, char: str) -> str:
        """Skip an unknown key's value: strings verbatim, nesting by depth."""
        if self._in_string:
            self._track_string(char)
        elif char == '"':
            self._in_string = True
        elif char in "{[":
            self._other_depth += 1
        elif char in "}]":
            if self._other_depth > 0:
                self._other_depth -= 1
            elif char == "}":
                self._depth -= 1
                self._state = self._DONE if self._depth == 0 else self._KEY
            # "]" at depth zero with nothing open: malformed, ignore
        elif char == "," and self._other_depth == 0:
            self._state = self._KEY  # value done, next key coming
        return ""

    def _take_args(self, char: str) -> str:
        if self._args_depth == 0 and not self._in_string and char in " \n\r\t":
            # Whitespace between the colon and the value is structural; once
            # the value opens, every character is emitted verbatim.
            return ""
        if self._in_string:
            self._track_string(char)
            return char
        if char == '"':
            self._in_string = True
            return char
        if char in "{[":
            self._args_depth += 1
        elif char in "}]":
            self._args_depth -= 1
            if self._args_depth == 0:
                # The value closed; the top-level object may still hold more
                # keys (a trailing name) before its own closing brace.
                self._state = self._KEY
        return char


class QwenToolParser(ToolParser):
    """Qwen markup: a JSON object inside each tool_call element.

    The first delta waits for the name string to close — the format's own
    order (name before arguments) usually lets it go out before any argument
    text; an object that opens its arguments first buffers them until the
    name arrives, so the id/name delta still leads.
    """

    name = "qwen"

    def __init__(self) -> None:
        self._state = "outside"
        self._held = ""
        self._scanner: _JsonCallScanner | None = None
        self._index = 0
        self._announced = False
        self._pending_args = ""  # argument text seen before the name was

    def feed(self, delta: str) -> ToolStream:
        text = self._held + delta
        self._held = ""
        out = ToolStream()
        while text:
            if self._state == "outside":
                text, blocked = self._outside(text, out)
            elif self._state == "scanning":
                text, blocked = self._scanning(text, out)
            else:
                text, blocked = self._closing(text, out)
            if blocked:
                self._held = text
                break
        return out

    def finish(self) -> ToolStream:
        # Whatever the suffix window still holds is content: an open marker,
        # a partial object — the same call a one-shot parse would decline.
        out = ToolStream(content=self._held)
        self._held = ""
        self._scanner = None
        return out

    def _outside(self, text: str, out: ToolStream) -> tuple[str, bool]:
        index = text.find(_QWEN_OPEN)
        if index >= 0:
            out.content += text[:index]
            self._scanner = _JsonCallScanner()
            self._announced = False
            self._pending_args = ""
            self._state = "scanning"
            return text[index + len(_QWEN_OPEN) :], False
        held = _partial_suffix(text, _QWEN_OPEN)
        out.content += text[: len(text) - len(held)]
        return held, bool(held)

    def _scanning(self, text: str, out: ToolStream) -> tuple[str, bool]:
        assert self._scanner is not None
        args, complete, consumed = self._scanner.feed(text)
        if not self._announced and self._scanner.name is not None:
            out.calls.append(
                ToolCallDelta(index=self._index, id=_call_id(self._index), name=self._scanner.name)
            )
            self._announced = True
            self._index += 1
            if self._pending_args:
                out.calls.append(ToolCallDelta(index=self._index - 1, arguments=self._pending_args))
                self._pending_args = ""
        if args:
            if self._announced:
                out.calls.append(ToolCallDelta(index=self._index - 1, arguments=args))
            else:
                self._pending_args += args
        if complete:
            self._state = "closing"
            return text[consumed:], False
        return "", True

    def _closing(self, text: str, out: ToolStream) -> tuple[str, bool]:
        index = text.find(_QWEN_CLOSE)
        if index >= 0:
            self._state = "outside"
            return text[index + len(_QWEN_CLOSE) :], False
        held = _partial_suffix(text, _QWEN_CLOSE)
        out.content += text[: len(text) - len(held)]
        return held, bool(held)


_REGISTRY: dict[str, type[ToolParser]] = {
    parser.name: parser for parser in (DeepSeekToolParser, QwenToolParser)
}
