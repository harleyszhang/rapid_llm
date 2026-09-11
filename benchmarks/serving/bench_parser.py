"""F8 parser cost: what streaming reasoning/tool parsing adds per token.

The parsers live on the serving path between the detokenizer and the wire, so
the honest number is per-token CPU latency, measured on the same incremental
chunks a detokenizer would emit. Three configurations — no parsing, reasoning
splitting only, reasoning plus tool parsing — over one mixed corpus; the
baseline row is the cost of the no-op loop itself, so the parser rows are
their difference from it, not inflated by it.

Usage:
    python benchmarks/serving/bench_parser.py --json docs/benchmark_logs/kernels/parser_v0.11.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.benchmark import write_json_log
from rapid_llm.serving.reasoning import _CLOSE as _UNTHINK
from rapid_llm.serving.reasoning import _OPEN as _THINK
from rapid_llm.serving.tool_parser import (
    _DS_ARGS_END,
    _DS_CALLS_BEGIN,
    _DS_CALLS_END,
    _DS_FENCE,
    _DS_HEADER,
)


def build_corpus(chunks: int) -> list[str]:
    """One mixed stream: think block, plain content, a tool-call section.

    Sized in detokenizer-like increments (~4 characters) so the parsers see
    tag boundaries straddling chunks, which is the case their suffix windows
    exist for — a best-case corpus of tag-aligned chunks would under-report.
    """
    turn = (
        _THINK
        + "weigh the request, gather the constraints, decide. "
        + "The answer needs one tool call. "
        + _UNTHINK
        + "Let me look that up. "
        + _DS_CALLS_BEGIN
        + _DS_HEADER
        + "get_weather\n"
        + _DS_FENCE
        + '{"city": "Tokyo", "unit": "celsius"}'
        + _DS_ARGS_END
        + _DS_CALLS_END
        + " It is 21 degrees. "
    )
    text = turn * (chunks * 4 // len(turn) + 1)
    return [text[i : i + 4] for i in range(0, len(text), 4)]


def run_once(chunks: list[str], *, reasoning: bool, tools: bool) -> float:
    from rapid_llm.serving.reasoning import ReasoningSplitter
    from rapid_llm.serving.tool_parser import DeepSeekToolParser

    splitter = ReasoningSplitter(starts_inside=False) if reasoning else None
    tool_parser = DeepSeekToolParser() if tools else None
    sink = 0
    start = time.perf_counter()
    for chunk in chunks:
        reasoning_text, content_text = ("", chunk)
        if splitter is not None:
            reasoning_text, content_text = splitter.feed(chunk)
            sink += len(reasoning_text)
        if tool_parser is not None:
            step = tool_parser.feed(content_text)
            sink += len(step.content) + len(step.calls)
        else:
            sink += len(content_text)
    elapsed = time.perf_counter() - start
    assert sink > 0, "the loop must do real work or the timing is fiction"
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=int, default=8000, help="detokenizer-sized deltas (~4 chars)")
    ap.add_argument("--iters", type=int, default=9, help="rounds per config, median reported")
    ap.add_argument(
        "--tpot-ms",
        type=float,
        default=None,
        help="decode TPOT to quote the parser cost against, e.g. from bench_mla",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    chunks = build_corpus(args.chunks)
    tokens = len(chunks)  # one chunk stands for one detokenized token
    configs = {
        "off": {"reasoning": False, "tools": False},
        "reasoning": {"reasoning": True, "tools": False},
        "reasoning+tools": {"reasoning": True, "tools": True},
    }

    results = {}
    for label, kwargs in configs.items():
        # Warm-up round outside the measurement, then medians over iters.
        run_once(chunks, **kwargs)
        elapsed = statistics.median(run_once(chunks, **kwargs) for _ in range(args.iters))
        per_token_us = elapsed / tokens * 1e6
        results[label] = {
            "total_ms": round(elapsed * 1000, 2),
            "tokens": tokens,
            "per_token_us": round(per_token_us, 3),
        }
        extra = ""
        if args.tpot_ms is not None and label != "off":
            delta_us = per_token_us - results["off"]["per_token_us"]
            extra = f"  (+{delta_us:.2f} us/token over off, "
            f"{delta_us / (args.tpot_ms * 1000) * 100:.3f}% of {args.tpot_ms:.2f} ms TPOT)"
        print(
            f"{label:16s} {elapsed * 1000:8.2f} ms for {tokens} tokens "
            f"({per_token_us:.2f} us/token){extra}"
        )

    if args.json:
        write_json_log(args.json, vars(args), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
