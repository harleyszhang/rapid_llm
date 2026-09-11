"""The streaming think-block splitter must agree with its one-shot self.

The correctness axiom of the parser: for any text and any way of cutting it
into deltas, the concatenated channel outputs of ``feed`` plus ``finish``
equal ``parse`` over the whole text. Everything else here pins the semantics
that axiom rides on — tag consumption, the partial-tag suffix window, and the
two opening modes.

Usage:
    pytest tests/engine/test_reasoning.py
"""

from __future__ import annotations

import itertools

import pytest

from rapid_llm.engine.reasoning import ReasoningSplitter, for_family

# Tag spellings are assembled, never written out: the transport delivering
# source edits strips anything it parses as markup.
OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"


def _stream(parts: list[str], starts_inside: bool = False) -> tuple[str, str]:
    """Feed the parts through one splitter and concatenate the channels."""
    splitter = ReasoningSplitter(starts_inside=starts_inside)
    reasoning = content = ""
    for part in parts:
        delta_r, delta_c = splitter.feed(part)
        reasoning += delta_r
        content += delta_c
    tail_r, tail_c = splitter.finish()
    return reasoning + tail_r, content + tail_c


# --------------------------------------------------------------------------- #
# One-shot semantics
# --------------------------------------------------------------------------- #
def test_parse_splits_a_closed_block():
    text = OPEN + "weigh options" + CLOSE + "final answer"
    assert ReasoningSplitter.parse(text) == ("weigh options", "final answer")


def test_parse_passes_untagged_text_through_as_content():
    """No opening tag, no reasoning section — the whole text is content.

    This is the deliberate divergence from vLLM's R1 parser, which would
    treat the same text as reasoning: models that only *sometimes* emit think
    blocks must not have their plain answers filed under reasoning_content.
    """
    assert ReasoningSplitter.parse("just an answer") == ("", "just an answer")


def test_parse_keeps_an_unterminated_block_in_reasoning():
    assert ReasoningSplitter.parse(OPEN + "still thinking") == ("still thinking", "")


def test_parse_starts_inside_treats_leading_text_as_reasoning():
    """R1 mode: the template opened the block, so text up to the close reasons."""
    text = "weigh options" + CLOSE + "final answer"
    assert ReasoningSplitter.parse(text, starts_inside=True) == (
        "weigh options",
        "final answer",
    )


def test_a_stray_closing_tag_mid_content_is_content():
    """After pass-through engages, markup has no special meaning."""
    text = "answer with " + CLOSE + " in it"
    assert ReasoningSplitter.parse(text) == ("", text)


def test_empty_input_yields_empty_channels():
    assert ReasoningSplitter.parse("") == ("", "")


# --------------------------------------------------------------------------- #
# Streaming mechanics
# --------------------------------------------------------------------------- #
def test_tags_split_across_deltas_are_consumed_not_emitted():
    """Every character of both tags vanishes; the payload survives intact."""
    parts = ["<th", "ink>", "ab", "c</th", "ink>", "de", "f"]
    assert _stream(parts) == ("abc", "def")


def test_single_character_deltas_match_the_one_shot_parse():
    """The hardest chunking there is: one character per delta."""
    text = OPEN + "weigh options" + CLOSE + "final answer"
    assert _stream(list(text)) == ReasoningSplitter.parse(text)


@pytest.mark.parametrize(
    "text",
    [
        OPEN + "reason" + CLOSE + "content",
        OPEN + "reason",
        OPEN + "reason" + CLOSE,
        "no markup at all",
        OPEN + "a</" + CLOSE + "b" + CLOSE + "c",  # inner partial tag, then real
        OPEN + "r" + CLOSE + "c1" + CLOSE + "c2",  # stray close inside content
        "<",
        OPEN[:3],
    ],
)
def test_every_two_cut_split_agrees_with_parse(text):
    """The axiom, checked exhaustively over all two-point cut positions."""
    expected = ReasoningSplitter.parse(text)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            parts = [text[:i], text[i:j], text[j:]]
            assert _stream([p for p in parts if p]) == expected, (i, j)


@pytest.mark.parametrize(
    "text",
    [
        "reason" + CLOSE + "content",
        "reason",
        CLOSE + "content",
    ],
)
def test_every_two_cut_split_agrees_with_parse_inside_mode(text):
    """The same axiom for R1-style streams that begin mid-think."""
    expected = ReasoningSplitter.parse(text, starts_inside=True)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            parts = [text[:i], text[i:j], text[j:]]
            assert _stream([p for p in parts if p], starts_inside=True) == expected, (
                i,
                j,
            )


def test_a_delta_can_close_the_block_and_carry_content():
    """One feed may emit on both channels; the tuple carries both."""
    splitter = ReasoningSplitter()
    assert splitter.feed(OPEN) == ("", "")
    assert splitter.feed("abc") == ("abc", "")
    reasoning, content = splitter.feed(CLOSE + "tail")
    assert reasoning == ""
    assert content == "tail"


def test_partial_opening_tag_at_end_of_stream_flushes_as_content():
    """A held "<thi" that never completes is ordinary text, not reasoning."""
    assert _stream(["an<thi"]) == ("", "an<thi")


def test_empty_deltas_change_nothing():
    assert _stream(["", OPEN, "", "r", "", CLOSE, "c", ""]) == ("r", "c")


def test_every_ordering_of_a_short_text_agrees():
    """A wider sweep: all deltas of size <= 2, all compositions, one text."""
    text = OPEN + "ab" + CLOSE + "cd"
    expected = ReasoningSplitter.parse(text)
    # Cut positions after every 1-2 characters: cover tag-aligned and
    # tag-straddling boundaries alike.
    for cuts in itertools.combinations(range(1, len(text)), 2):
        parts = [text[: cuts[0]], text[cuts[0] : cuts[1]], text[cuts[1] :]]
        assert _stream(parts) == expected, cuts


# --------------------------------------------------------------------------- #
# The named factory
# --------------------------------------------------------------------------- #
def test_the_registered_family_is_the_template_injected_case():
    """``deepseek_r1`` means the prompt template already opened the block."""
    splitter = for_family("deepseek_r1")
    assert splitter.feed("ponder") == ("ponder", "")


def test_unknown_family_names_are_rejected():
    with pytest.raises(ValueError, match="unknown reasoning parser"):
        for_family("bogus")
