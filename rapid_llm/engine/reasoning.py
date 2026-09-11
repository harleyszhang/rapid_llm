"""Streaming think-block splitting: one pass, any chunking.

:class:`ReasoningSplitter` eats the incremental detokenizer deltas and routes
each character to ``reasoning`` or ``content``, so serving can expose
``reasoning_content`` without buffering the whole generation. The contract:
any split of the text yields, concatenated, exactly what
:meth:`ReasoningSplitter.parse` returns in one piece — bought by holding back
an ambiguous partial-tag tail and consuming tags the moment they are recognised.

Not a vLLM clone: ``starts_inside=True`` is the R1 behaviour (no opening tag
means the template already opened thinking); the default is pass-through, for
models that only *sometimes* emit think blocks.

Usage:
    splitter = ReasoningSplitter()
    for delta in stream:
        reasoning, content = splitter.feed(delta)   # finish() returns the tail
"""

from __future__ import annotations


def _tag(name: str) -> str:
    """Assemble a markup tag without spelling one out in the source.

    Written this way so tag literals never appear verbatim in the file:
    the transport that delivers source edits strips anything it parses as
    markup, which would silently empty the constants.
    """
    return "<" + name + ">"


_OPEN = _tag("think")
_CLOSE = _tag("/think")


def for_family(name: str) -> ReasoningSplitter:
    """Build the splitter a request asked for by name.

    The name is where the ``starts_inside`` choice lives: ``deepseek_r1`` is
    the template-injected case, where the generation itself begins inside
    the reasoning section. A ``None`` from the request means no splitting at
    all — the caller skips the splitter rather than asking here.
    """
    if name == "deepseek_r1":
        return ReasoningSplitter(starts_inside=True)
    raise ValueError(f"unknown reasoning parser {name!r}; known: deepseek_r1")


class ReasoningSplitter:
    """Two-channel state machine over the think-block markup.

    Args:
        starts_inside: Whether the prompt template already emitted the
            opening tag, so the generation itself begins inside the reasoning
            section (DeepSeek-R1 style). ``False`` — the default — waits for
            the model to emit the tag itself and passes untagged text
            straight to the content channel.
    """

    _OPENING, _THINKING, _CONTENT = "opening", "thinking", "content"

    def __init__(self, starts_inside: bool = False) -> None:
        self._state = self._THINKING if starts_inside else self._OPENING
        self._held = ""  # tail that may yet complete a tag

    def feed(self, delta: str) -> tuple[str, str]:
        """Route one increment; returns ``(reasoning_delta, content_delta)``.

        Both may be empty; both may be non-empty when a single delta closes
        the think block and carries content after it.
        """
        if self._state == self._CONTENT:
            return "", delta
        text = self._held + delta
        self._held = ""
        if self._state == self._OPENING:
            return self._feed_opening(text)
        return self._feed_thinking(text)

    def finish(self) -> tuple[str, str]:
        """Flush what the suffix window still holds at end of stream.

        A held partial tag that never completed is ordinary text of whichever
        channel was live; an unterminated thinking section keeps everything
        in the reasoning channel, matching what a one-shot parse of the same
        text would say.
        """
        held, self._held = self._held, ""
        if self._state == self._OPENING:
            return "", held
        return held, ""

    @classmethod
    def parse(cls, text: str, starts_inside: bool = False) -> tuple[str, str]:
        """One-shot semantics: the same split, computed in one call."""
        splitter = cls(starts_inside=starts_inside)
        reasoning, content = splitter.feed(text)
        tail_reasoning, tail_content = splitter.finish()
        return reasoning + tail_reasoning, content + tail_content

    # ------------------------------------------------------------------ #
    def _feed_opening(self, text: str) -> tuple[str, str]:
        if text.startswith(_OPEN):
            self._state = self._THINKING
            return self._feed_thinking(text[len(_OPEN) :])
        if _OPEN.startswith(text):
            # A true prefix of the opening tag: still ambiguous, hold it all.
            self._held = text
            return "", ""
        # Definitively no opening tag: pass-through, and stay deaf to a later
        # closing tag — a stray one mid-content is content.
        self._state = self._CONTENT
        return "", text

    def _feed_thinking(self, text: str) -> tuple[str, str]:
        index = text.find(_CLOSE)
        if index >= 0:
            self._state = self._CONTENT
            return text[:index], text[index + len(_CLOSE) :]
        # Hold back the longest suffix that could still complete the closing
        # tag; the rest is reasoning, safe to emit now.
        self._held = _partial_suffix(text, _CLOSE)
        return text[: len(text) - len(self._held)], ""


def _partial_suffix(text: str, tag: str) -> str:
    """The longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for length in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:length]):
            return text[-length:]
    return ""
