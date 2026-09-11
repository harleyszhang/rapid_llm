"""Streaming tool parsers must agree with their one-shot selves.

The correctness axiom, same as the reasoning splitter's: for any text and any
way of cutting it into deltas, the concatenated channels of ``feed`` +
``finish`` equal ``parse`` over the whole text — checked exhaustively over
all two-cut splits for every fixture. The fixtures pin the semantics the
axiom rides on: marker consumption, the partial-marker suffix window,
first-delta timing (id/name before arguments), string-aware brace counting,
and the degrade-to-content path for malformed markup.

Usage:
    pytest tests/engine/test_tool_parser.py
"""

from __future__ import annotations

import json

import pytest

# Marker spellings are assembled, never written out: the transport delivering
# source edits strips anything it parses as markup.
from rapid_llm.engine.tool_parser import (
    _DS_ARGS_END,
    _DS_CALLS_BEGIN,
    _DS_CALLS_END,
    _DS_FENCE,
    _DS_HEADER,
    _QWEN_CLOSE,
    _QWEN_OPEN,
    DeepSeekToolParser,
    QwenToolParser,
    ToolCallDelta,
    ToolParser,
)


def _call(name: str, args: str, id_: str = ""):
    return (id_, name, args)


def _merge(deltas: list[ToolCallDelta]) -> list[list]:
    """Fold stream deltas into ``[id, name, arguments]`` rows.

    Keyed by the delta's call ``index`` — the same key the wire protocol
    uses — so an orphan arguments delta from ``finish`` fuses with the call
    ``feed`` already opened instead of crashing into it.
    """
    rows: dict[int, list] = {}
    order: list[int] = []
    for delta in deltas:
        if delta.index not in rows:
            rows[delta.index] = [delta.id or "", delta.name or "", delta.arguments]
            order.append(delta.index)
        else:
            rows[delta.index][2] += delta.arguments
    return [rows[index] for index in order]


def _stream(parser: ToolParser, parts: list[str]) -> tuple[str, list[list]]:
    content = ""
    deltas: list[ToolCallDelta] = []
    for part in parts:
        step = parser.feed(part)
        content += step.content
        deltas += step.calls
    step = parser.finish()
    content += step.content
    deltas += step.calls
    return content, _merge(deltas)


def _assert_axiom(parser_cls: type[ToolParser], text: str) -> None:
    """Every two-cut split of ``text`` must match its one-shot parse."""
    expected_content, expected_calls = parser_cls.parse(text)
    expected = [[c.id, c.name, c.arguments] for c in expected_calls]
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            parts = [p for p in (text[:i], text[i:j], text[j:]) if p]
            content, calls = _stream(parser_cls(), parts)
            assert content == expected_content, (parser_cls.name, i, j, repr(content))
            assert calls == expected, (parser_cls.name, i, j, calls)


# --------------------------------------------------------------------------- #
# Fixtures: well-formed, malformed, adversarial
# --------------------------------------------------------------------------- #
DS_TWO_CALLS = (
    "Let me check. "
    + _DS_CALLS_BEGIN
    + _DS_HEADER
    + "get_weather\n"
    + _DS_FENCE
    + '{"city": "Tokyo"}'
    + _DS_ARGS_END
    + _DS_HEADER
    + "get_time\n"
    + _DS_FENCE
    + '{"tz": "JST"}'
    + _DS_ARGS_END
    + _DS_CALLS_END
    + " Done."
)
QWEN_ONE_CALL = (
    "Intro "
    + _QWEN_OPEN
    + '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    + _QWEN_CLOSE
    + " outro"
)
#: arguments-first key order: the scanner must keep walking the object so the
#: trailing name still fires before the call's deltas go out.
QWEN_ARGS_FIRST = (
    "a" + _QWEN_OPEN + '{"arguments": {"city": "Paris"}, "name": "get_weather"}' + _QWEN_CLOSE + "b"
)
#: escaped quotes and braces inside argument strings: string-aware counting.
_ARGS_RAW = '{"code": "say \\"hi\\" {}", "n": 2}'
QWEN_ESCAPES = _QWEN_OPEN + '{"name": "run", "arguments": ' + _ARGS_RAW + "}" + _QWEN_CLOSE


# --------------------------------------------------------------------------- #
# One-shot semantics
# --------------------------------------------------------------------------- #
def test_deepseek_parse_extracts_two_calls_and_surrounding_content():
    content, calls = DeepSeekToolParser.parse(DS_TWO_CALLS)
    assert content == "Let me check.  Done."
    assert [(c.name, c.arguments, c.id) for c in calls] == [
        ("get_weather", '{"city": "Tokyo"}', "call_0"),
        ("get_time", '{"tz": "JST"}', "call_1"),
    ]


def test_qwen_parse_extracts_the_call_and_content():
    content, calls = QwenToolParser.parse(QWEN_ONE_CALL)
    assert content == "Intro  outro"
    assert [(c.name, c.arguments) for c in calls] == [("get_weather", '{"city": "Tokyo"}')]


def test_qwen_arguments_first_order_still_names_the_call():
    content, calls = QwenToolParser.parse(QWEN_ARGS_FIRST)
    assert content == "ab"
    assert [(c.name, c.arguments) for c in calls] == [("get_weather", '{"city": "Paris"}')]


def test_qwen_arguments_text_is_valid_json_verbatim():
    """The arguments channel is the value's raw JSON text, byte for byte."""
    _, calls = QwenToolParser.parse(QWEN_ESCAPES)
    assert calls[0].arguments == _ARGS_RAW
    assert json.loads(calls[0].arguments)["code"] == 'say "hi" {}'
    assert json.loads(calls[0].arguments)["n"] == 2


def test_untagged_text_is_pure_content_for_both_families():
    for parser in (DeepSeekToolParser(), QwenToolParser()):
        content, calls = parser.parse("plain content only")
        assert content == "plain content only"
        assert calls == []


# --------------------------------------------------------------------------- #
# Streaming mechanics
# --------------------------------------------------------------------------- #
def test_the_first_delta_carries_id_and_name_before_arguments():
    """OpenAI's contract: the call's identity leads, arguments stream after."""
    parser = QwenToolParser()
    step = parser.feed(_QWEN_OPEN + '{"name": "get_weat')
    assert step.calls == []  # the name string has not closed yet
    step = parser.feed('her", "arguments": {"ci')
    first = step.calls[0]
    assert first.id == "call_0"
    assert first.name == "get_weather"
    assert first.arguments == ""  # nothing of the value had arrived at close
    # the same step may already stream argument text once the name closes;
    # those deltas carry the index and arguments only, never identity
    for later in step.calls[1:]:
        assert later.id is None and later.name is None
        assert later.index == first.index
    step = parser.feed('ty": "Tokyo"}}' + _QWEN_CLOSE)
    streamed = "".join(delta.arguments for delta in step.calls)
    assert '{"ci' + streamed == '{"city": "Tokyo"}'


def test_a_marker_split_across_deltas_is_consumed_not_emitted():
    parser = DeepSeekToolParser()
    step = parser.feed("answer " + _DS_CALLS_BEGIN[:6])
    assert step.content == "answer "
    step = parser.feed(_DS_CALLS_BEGIN[6:] + _DS_HEADER + "f\n" + _DS_FENCE + "{}")
    assert step.content == ""
    step = parser.feed(_DS_ARGS_END + "after")
    assert step.content == "after"
    step = parser.finish()
    assert step.content == ""  # the section closed cleanly; nothing held
    assert step.calls == []


def test_a_truncated_call_at_end_of_stream_emits_what_arrived():
    """A stream that dies mid-arguments still hands the client the pieces."""
    text = _DS_CALLS_BEGIN + _DS_HEADER + "half\n" + _DS_FENCE + '{"a": '
    content, calls = DeepSeekToolParser.parse(text)
    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "half"
    assert calls[0].arguments == '{"a": '


def test_a_partial_marker_at_end_of_stream_flushes_as_content():
    for parser_cls, marker in ((QwenToolParser, _QWEN_OPEN), (DeepSeekToolParser, _DS_CALLS_BEGIN)):
        content, calls = parser_cls.parse("tail " + marker[:4])
        assert content == "tail " + marker[:4]
        assert calls == []


def test_for_model_rejects_unknown_families():
    with pytest.raises(ValueError, match="unknown tool parser"):
        ToolParser.for_model("nope")


def test_for_model_builds_each_registered_family():
    for family, cls in (("deepseek", DeepSeekToolParser), ("qwen", QwenToolParser)):
        assert isinstance(ToolParser.for_model(family), cls)


# --------------------------------------------------------------------------- #
# The axiom
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "parser_cls,text",
    [
        (DeepSeekToolParser, DS_TWO_CALLS),
        (DeepSeekToolParser, "pre " + _DS_CALLS_BEGIN + _DS_HEADER + "half\n"),
        (DeepSeekToolParser, "plain content only"),
        (DeepSeekToolParser, _DS_CALLS_BEGIN[:5] + "tail"),
        (QwenToolParser, QWEN_ONE_CALL),
        (QwenToolParser, QWEN_ARGS_FIRST),
        (QwenToolParser, QWEN_ESCAPES),
        (QwenToolParser, "plain content only"),
        (QwenToolParser, "a" + _QWEN_OPEN[:4]),
        (QwenToolParser, _QWEN_OPEN + '{"name": "x"'[:5]),
    ],
    ids=lambda value: value if isinstance(value, str) else f"{value.name}-axiom",
)
def test_every_two_cut_split_agrees_with_parse(parser_cls, text):
    _assert_axiom(parser_cls, text)


def test_single_character_deltas_agree_for_both_families():
    for parser_cls, text in (
        (DeepSeekToolParser, DS_TWO_CALLS),
        (QwenToolParser, QWEN_ONE_CALL),
        (QwenToolParser, QWEN_ESCAPES),
    ):
        expected = parser_cls.parse(text)
        content, calls = _stream(parser_cls(), list(text))
        assert content == expected[0]
        assert calls == [[c.id, c.name, c.arguments] for c in expected[1]]
