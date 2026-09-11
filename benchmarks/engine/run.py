"""Unified CLI for engine-level benchmarks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.engine import (
    bench_kv_transfer,
    bench_pipeline,
    continuous,
    cpu,
    optimizations,
    quant,
    scheduler,
)


def _add_command(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
    configure: Callable[[argparse.ArgumentParser], None],
    run: Callable[[argparse.Namespace], int],
) -> None:
    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    configure(parser)
    parser.set_defaults(_run=run)


def _configure_scheduler(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="scheduler_command", required=True)
    for command in scheduler.COMMANDS:
        _add_command(subparsers, command.name, command.help, command.configure, command.run)
    _add_command(
        subparsers,
        "continuous",
        "continuous batching versus static batching, offline and online",
        continuous.configure,
        continuous.run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scheduler_parser = subparsers.add_parser(
        "scheduler",
        help="scheduler, serving, and continuous-batching benchmarks",
    )
    _configure_scheduler(scheduler_parser)

    _add_command(
        subparsers,
        "optimizations",
        "optimization-feature A/B matrix",
        optimizations.configure,
        optimizations.run,
    )
    _add_command(
        subparsers,
        "quant",
        "offline quantization matrix",
        quant.configure,
        quant.run,
    )
    _add_command(
        subparsers,
        "cpu",
        "CPU model-forward latency",
        cpu.configure,
        cpu.run,
    )
    _add_command(
        subparsers,
        "pipeline",
        "launch/harvest pipeline depth A/B (host sync wait share)",
        bench_pipeline.configure,
        bench_pipeline.run,
    )
    _add_command(
        subparsers,
        "kv-transfer",
        "CPU KV tier acceptance: parity, TPOT tax, overflow reuse",
        bench_kv_transfer.configure,
        bench_kv_transfer.run,
    )

    args = parser.parse_args(argv)
    run = args._run
    for internal in ("_run", "command", "scheduler_command"):
        if hasattr(args, internal):
            delattr(args, internal)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
