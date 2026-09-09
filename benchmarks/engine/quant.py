"""Offline quantisation matrix: speed, memory, and two independent accuracy references.

One row per configuration, where a configuration is the cross product of
``quantisation x kv-cache dtype x TP x DP x CUDA graph``. Every row carries both
halves of the trade — tokens per second *and* how far the output moved — because a
quantisation table that reports only throughput cannot distinguish a working kernel
from a broken one.

Two accuracy columns, and they answer different questions:

* ``golden`` compares against the bf16 / eager / TP=1 baseline recorded by
  ``scripts/golden_tokens.py`` into ``tests/golden/data/``, produced by *this* engine.
  Same engine on both sides, so a deviation is attributable to the axis under test.
  Reported two ways: ``prefix`` is the fraction of each completion reproduced before
  the first differing token, and ``pos`` is plain position-wise agreement. Read the
  first one — greedy decoding is chaotic, so once a token differs the rest of the
  sequence is unrelated and ``pos`` decays toward chance no matter how small the
  numerical error was.
* ``vs HF`` compares against HuggingFace, loaded at the dtype the checkpoint's
  own config declares. That difference contains the whole engine — kernels,
  attention, sampling — so it is a sanity bound, not a measure of quantisation
  error. Skip it with ``--skip-hf`` when only the axes matter.

The golden run reuses the recorded prompt set but not its pinned KV pool, so the
bf16 / eager / TP=1 row acts as a control: it must score ``1.000``. If it does not,
the pool geometry is itself moving tokens and no other row's deviation is
attributable — the script says so instead of publishing the column.

Each configuration is measured in its own spawned process. Not tidiness: a process
joins one TP group for its lifetime, so ``tp1`` and ``tp2`` rows cannot share one,
DP replicas are processes of their own, and a peak-memory reading is only meaningful
in a process that allocated nothing else first. Isolation also means one row that
OOMs or hangs is reported as a failed cell rather than taking the matrix with it.

A ``tp>1`` row is measured through the continuous-batching engine, marked ``+cb`` in
the label, because that is the only path whose executor broadcasts each step's plan
to the follower ranks. Comparing such a row against a lockstep ``tp1`` row prices
the scheduler along with the sharding, so pass ``--engine continuous`` to put the
baseline on the same path. Under TP the memory columns are rank 0's shard, not the
whole model.

Usage:
    # single axis: which schemes cost what
    python benchmarks/engine/run.py quant --model-dir $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
        --schemes fp16 fp8 int4 nvfp4

    # quantisation x TP x graph, both sides on the continuous-batching engine
    python benchmarks/engine/run.py quant --model-dir ... --schemes fp8 int4 \
        --tp 1 2 --engine continuous --cuda-graph --no-cuda-graph --skip-hf

    # KV cache fp8, data parallel
    python benchmarks/engine/run.py quant --model-dir ... --kv-cache-dtype auto fp8_e4m3 --dp 1 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue as queue_module
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from rapid_llm.benchmark import PROMPTS, checkpoint_dtype, dtype_tag, expand_prompts, gpu_tag

_MAX_GEN = 64
_BATCH = 4

#: Where ``scripts/golden_tokens.py`` writes the bf16 / eager / TP=1 baselines.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO_ROOT / "tests" / "golden" / "data"

#: A checkpoint load, a KV profile, a capture, warmups and three generation passes.
#: Its job is to turn a wedged rank into a reported failure rather than a hung matrix.
_SPEC_TIMEOUT_S = 1800.0

#: Model codenames used by ``--all``, relative to ``$RAPID_LLM_MODELZOO``.
_MODELZOO_ENV = "RAPID_LLM_MODELZOO"
_ALL_MODELS = (
    "Qwen/Qwen2___5-0___5B-Instruct",
    "Qwen3/Qwen3-4B-Thinking-2507",
)


@dataclass(frozen=True)
class RunSpec:
    """One cell of the matrix. Every field appears in :attr:`label`."""

    scheme: str | None = None
    kv_cache_dtype: str = "auto"
    tp: int = 1
    dp: int = 1
    graph: bool = True
    #: ``"auto"`` picks the lockstep batch loop, or the continuous-batching engine
    #: when the cell needs one. ``"continuous"`` forces it, which is how a TP=1
    #: baseline is made comparable to a TP=2 row.
    engine: str = "auto"

    @property
    def engine_kind(self) -> str:
        """Which measurement path this cell runs on.

        Above ``tp=1`` the choice is not free: only the continuous-batching
        engine's executor broadcasts each step's plan to the follower ranks, and
        the lockstep loop has no channel to hand a follower its work. So a TP row
        is a continuous-batching row whether or not anyone asked, and comparing it
        against a lockstep TP=1 row would price the scheduler as if it were the
        sharding. Force ``engine="continuous"`` on the baseline to avoid that.
        """
        if self.dp > 1:
            return "dp"
        if self.engine == "continuous" or self.tp > 1:
            return "continuous"
        return "batch"

    @property
    def label(self) -> str:
        """Self-describing row name, e.g. ``fp8+kvfp8+tp2+graph+cb``.

        Every axis is spelled out even at its default, because a json row is read
        long after the command line that produced it is gone.
        """
        parts = [self.scheme or "bf16"]
        if self.kv_cache_dtype != "auto":
            parts.append("kv" + self.kv_cache_dtype.replace("_", "").replace("e4m3", "fp8"))
        parts.append(f"tp{self.tp}")
        if self.dp > 1:
            parts.append(f"dp{self.dp}")
        parts.append("graph" if self.graph else "eager")
        if self.engine_kind == "continuous":
            parts.append("cb")
        return "+".join(parts)

    @property
    def gpus_needed(self) -> int:
        return self.tp * self.dp


@dataclass
class Row:
    """One measured (or failed) cell, ready for the table and for json."""

    config: str
    model: str
    # None on a DP row: the coordinator has no per-step callback, so those rows
    # measure aggregate throughput only. Reported as "—" rather than as zero.
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    tps: float = 0.0
    total_s: float = 0.0
    gen_tokens: int = 0
    batch: int = 0
    peak_mem_gb: float | None = None
    model_mem_gb: float | None = None
    kv_cache_tokens: int | None = None
    # Whether decode steps *went through* a graph. Capturing and replaying are
    # separate facts, and a config where the grid never matches decodes eager with
    # every graph still resident.
    graph_installed: bool | None = None
    graph_replays: int | None = None
    golden_match_rate: float | None = None
    # Fraction of each completion reproduced *before* the first differing token,
    # averaged over sequences. The interpretable accuracy number: position-wise
    # agreement past a divergence is chance, this is not.
    golden_prefix_rate: float | None = None
    golden_exact: str | None = None
    token_match_rate: float | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.total_s > 0.0


# --------------------------------------------------------------------------- #
# Child process: build one engine, measure it, generate the comparison texts
# --------------------------------------------------------------------------- #
def _lite_kwargs(
    spec: RunSpec, max_seq_len: int, kv_tokens: int | None, *, with_tp: bool = True
) -> dict[str, Any]:
    """Engine kwargs for one spec.

    ``with_tp=False`` leaves ``tensor_parallel_size`` out for callers that pass it
    themselves — the continuous and DP paths both take it as a named argument, and
    duplicating it is a TypeError rather than a silently ignored value.
    """
    kwargs: dict[str, Any] = {"max_seq_len": max_seq_len, "use_cuda_graph": spec.graph}
    if kv_tokens:
        kwargs["max_gpu_num_blocks"] = kv_tokens
    if spec.scheme:
        kwargs["quantization"] = spec.scheme
    if spec.kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = spec.kv_cache_dtype
    if with_tp and spec.tp > 1:
        kwargs["tensor_parallel_size"] = spec.tp
    return kwargs


def _golden_texts(generate, cases, penalties) -> dict[str, list[str]]:
    """Replay the recorded golden cases through this engine, keyed as the file is."""
    from rapid_llm import SamplingParams
    from tests.golden.cases import case_key

    out: dict[str, list[str]] = {}
    for name, prompts, max_gen_len in cases:
        for penalty in penalties:
            params = SamplingParams(
                temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=penalty
            )
            out[case_key(name, penalty)] = list(generate(prompts, params))
    return out


def _measure_lite(payload: dict[str, Any]) -> dict[str, Any]:
    """Stream-timed measurement of one in-process engine (TP included)."""
    from rapid_llm import SamplingParams
    from rapid_llm.benchmark import LiteBackend, footprint_stats

    spec: RunSpec = payload["spec"]
    torch.cuda.reset_peak_memory_stats()
    backend = LiteBackend(
        payload["model"],
        **_lite_kwargs(spec, payload["max_seq_len"], payload["kv_tokens"]),
    )
    prompts = expand_prompts(PROMPTS, payload["batch"])
    speed = backend.measure(prompts, payload["max_gen"], greedy=True)
    torch.cuda.synchronize()
    generator = backend.generator
    report: dict[str, Any] = {
        "ttft_ms": speed.ttft_ms,
        "tpot_ms": speed.tpot_ms,
        "tps": speed.tps,
        "total_s": speed.total_s,
        "gen_tokens": speed.gen_tokens,
        "batch": speed.batch,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        # Read after the timed run, so the replay count is what those decodes produced.
        **footprint_stats(generator.engine.model_runner),
    }
    report["texts"] = generator.generate(
        prompts, SamplingParams(temperature=0.0, max_gen_len=payload["max_gen"])
    )
    if payload["golden"]:
        report["golden"] = _golden_texts(
            generator.generate, payload["golden_cases"], payload["penalties"]
        )
    backend.close()
    return report


def _measure_dp(payload: dict[str, Any]) -> dict[str, Any]:
    """Latency-only measurement through the DP coordinator.

    Nothing is loaded in this process — the replicas own the weights — so the memory
    and graph columns are unavailable here rather than zero, and the throughput is
    the aggregate over replicas. TTFT/TPOT are absent because ``generate`` returns
    once, with no per-step callback to time.
    """
    from rapid_llm import DataParallelEngine, SamplingParams

    spec: RunSpec = payload["spec"]
    prompts = expand_prompts(PROMPTS, payload["batch"] * spec.dp)
    params = SamplingParams(temperature=0.0, max_gen_len=payload["max_gen"])
    kwargs = _lite_kwargs(spec, payload["max_seq_len"], payload["kv_tokens"], with_tp=False)

    with DataParallelEngine(
        model=payload["model"],
        data_parallel_size=spec.dp,
        tensor_parallel_size=spec.tp,
        max_num_seqs=payload["batch"],
        **kwargs,
    ) as engine:

        def generate(prompt_list, sampling):
            return [out.text for out in engine.generate(prompt_list, sampling)]

        generate(prompts, SamplingParams(temperature=0.0, max_gen_len=8))  # warm every replica
        torch.cuda.synchronize()
        start = time.perf_counter()
        texts = generate(prompts, params)
        total = time.perf_counter() - start

        tokenizer = engine.tokenizer
        gen_tokens = sum(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)
        report: dict[str, Any] = {
            "tps": gen_tokens / total if total else 0.0,
            "total_s": total,
            "gen_tokens": gen_tokens,
            "batch": len(prompts),
            "texts": texts,
        }
        if payload["golden"]:
            report["golden"] = _golden_texts(
                generate, payload["golden_cases"], payload["penalties"]
            )
    return report


def _measure_cb(payload: dict[str, Any]) -> dict[str, Any]:
    """Step-timed measurement through the continuous-batching engine.

    The only path that can drive a tensor-parallel group: ``from_pretrained``
    spawns the follower ranks and its executor broadcasts each step's plan to
    them, which is what a sharded forward needs. Timing is taken around
    :meth:`step` rather than from a stream callback, so TTFT and TPOT keep the
    same definitions as the lockstep rows.

    Under TP the memory columns describe **rank 0 only** — its shard of the
    weights and its own allocator peak. The other rank holds a shard of the same
    size, so the figure is per-GPU rather than per-model.
    """
    from rapid_llm import SamplingParams
    from rapid_llm.benchmark import footprint_stats, run_requests
    from rapid_llm.engine import ContinuousBatchingEngine

    spec: RunSpec = payload["spec"]
    prompts = expand_prompts(PROMPTS, payload["batch"])
    torch.cuda.reset_peak_memory_stats()
    engine = ContinuousBatchingEngine.from_pretrained(
        payload["model"],
        max_num_seqs=max(payload["batch"], 8),
        tensor_parallel_size=spec.tp,
        **_lite_kwargs(spec, payload["max_seq_len"], payload["kv_tokens"], with_tp=False),
    )
    try:

        def generate(prompt_list, sampling):
            return [out.outputs[0].text for out in engine.generate(prompt_list, sampling)]

        for _ in range(2):  # autotune + allocator, so the measured run is steady state
            generate(prompts, SamplingParams(temperature=0.0, max_gen_len=8))

        params = SamplingParams(temperature=0.0, max_gen_len=payload["max_gen"])
        run = run_requests(engine, prompts, params)
        speed = run.result(len(prompts))
        texts = run.texts
        tokenizer = engine.tokenizer
        gen_tokens = sum(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)

        report: dict[str, Any] = {
            "ttft_ms": speed.ttft_ms,
            "tpot_ms": speed.tpot_ms,
            "tps": gen_tokens / run.total_s if run.total_s else 0.0,
            "total_s": run.total_s,
            "gen_tokens": gen_tokens,
            "batch": len(prompts),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
            **footprint_stats(engine.engine.model_runner),
            "texts": texts,
            "note": "rank 0 shard" if spec.tp > 1 else "",
        }
        if payload["golden"]:
            report["golden"] = _golden_texts(
                generate, payload["golden_cases"], payload["penalties"]
            )
    finally:
        engine.shutdown()
    return report


def _measure_hf(payload: dict[str, Any]) -> dict[str, Any]:
    """HuggingFace reference row, at the checkpoint's declared dtype."""
    from rapid_llm.benchmark import HFBackend

    torch.cuda.reset_peak_memory_stats()
    backend = HFBackend(payload["model"])
    prompts = expand_prompts(PROMPTS, payload["batch"])
    speed = backend.measure(prompts, payload["max_gen"], greedy=True)
    torch.cuda.synchronize()
    texts = [backend.tokenizer.decode(row, skip_special_tokens=True) for row in backend._last_gen]
    return {
        "ttft_ms": speed.ttft_ms,
        "tpot_ms": speed.tpot_ms,
        "tps": speed.tps,
        "total_s": speed.total_s,
        "gen_tokens": speed.gen_tokens,
        "batch": speed.batch,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "texts": texts,
    }


def _child(payload: dict[str, Any], out: mp.Queue) -> None:
    """Measure one cell in a fresh process; a failure travels back as a traceback."""
    measure = {
        "hf": _measure_hf,
        "dp": _measure_dp,
        "continuous": _measure_cb,
        "batch": _measure_lite,
    }
    try:
        kind = "hf" if payload["kind"] == "hf" else payload["spec"].engine_kind
        report = measure[kind](payload)
    except Exception:
        out.put(("error", traceback.format_exc()))
    else:
        out.put(("ok", report))


def _run_child(payload: dict[str, Any], timeout_s: float) -> tuple[str, Any]:
    """Run one cell to completion. Returns ``("ok", report)`` or ``("error", text)``."""
    context = mp.get_context("spawn")
    out: mp.Queue = context.Queue()
    # Not a daemon: a TP row spawns followers and a DP row spawns replicas, and a
    # daemonic process is not allowed children.
    process = context.Process(target=_child, args=(payload, out), daemon=False)
    process.start()
    try:
        try:
            return out.get(timeout=timeout_s)
        except queue_module.Empty:
            return "error", f"produced nothing in {timeout_s:.0f}s (treated as a hang)"
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()
            process.join(timeout=30.0)


# --------------------------------------------------------------------------- #
# Parent process: scoring and reporting
# --------------------------------------------------------------------------- #
def _token_agreement(tokenizer, want: list[str], got: list[str]) -> tuple[int, int, int, int]:
    """``(matching positions, compared positions, exact sequences, leading matches)``.

    Position-wise over the tokenised completions, which is stricter than comparing
    words: a scheme that produces the same prose with different tokenisation is not
    producing the same tokens.

    The fourth number exists because greedy decoding is chaotic. One differing token
    changes the context every later step conditions on, so the suffixes are unrelated
    and the position-wise rate collapses toward chance — it says "diverged" and
    nothing more. Counting the *leading* agreement instead says how far a config
    tracked the baseline, which is the quantity that orders schemes.
    """
    matched = compared = exact = leading = 0
    for want_text, got_text in zip(want, got, strict=False):
        want_ids = tokenizer(want_text, add_special_tokens=False).input_ids
        got_ids = tokenizer(got_text, add_special_tokens=False).input_ids
        length = min(len(want_ids), len(got_ids))
        pairs = list(zip(want_ids[:length], got_ids[:length], strict=True))
        matched += sum(1 for a, b in pairs if a == b)
        leading += next((i for i, (a, b) in enumerate(pairs) if a != b), length)
        compared += length
        exact += want_text == got_text
    return matched, compared, exact, leading


def _score_golden(tokenizer, baseline: dict[str, list[str]], got: dict[str, list[str]]):
    """Agreement against the recorded baseline, over the cases both have."""
    matched = compared = exact = sequences = leading = 0
    for key, want in baseline.items():
        if key not in got:
            continue
        m, c, e, lead = _token_agreement(tokenizer, want, got[key])
        matched, compared, exact, leading = matched + m, compared + c, exact + e, leading + lead
        sequences += len(want)
    if not compared:
        return None, None, None
    return matched / compared, leading / compared, f"{exact}/{sequences}"


def _render(rows: list[Row]) -> str:
    header = (
        "| Config | Model mem | Peak | KV tok | graph | TTFT (ms) | TPOT (ms) | TPS "
        "| golden prefix | golden pos | exact | vs HF |"
    )
    lines = [header, "|" + "---|" * 12]
    for r in rows:
        if not r.ok:
            lines.append(f"| {r.config} | FAILED: {r.note} |" + " |" * 10)
            continue

        def num(value, spec="{:.2f}"):
            return "—" if value is None else spec.format(value)

        graph = "—"
        if r.graph_installed is not None:
            graph = "off" if not r.graph_installed else f"{r.graph_replays or 0} replays"
        lines.append(
            f"| {r.config} | {num(r.model_mem_gb)} GB | {num(r.peak_mem_gb)} GB "
            f"| {'—' if r.kv_cache_tokens is None else format(r.kv_cache_tokens, ',')} "
            f"| {graph} | {num(r.ttft_ms, '{:.1f}')} | {num(r.tpot_ms)} | {r.tps:.1f} "
            f"| {num(r.golden_prefix_rate, '{:.3f}')} "
            f"| {num(r.golden_match_rate, '{:.3f}')} | {r.golden_exact or '—'} "
            f"| {num(r.token_match_rate, '{:.3f}')} |"
        )
    return "\n".join(lines)


def _meta(model_dir: str, batch: int, max_gen: int) -> dict[str, Any]:
    import subprocess

    import triton

    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "model_dir": model_dir,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "commit": commit,
        "command": " ".join(sys.argv),
        "batch": batch,
        "max_gen_len": max_gen,
        "date": date.today().isoformat(),
    }


def benchmark_model(
    model_dir: str,
    specs: list[RunSpec],
    *,
    batch: int = _BATCH,
    max_gen: int = _MAX_GEN,
    max_seq_len: int = 1024,
    kv_tokens: int | None = None,
    skip_hf: bool = False,
    use_golden: bool = True,
    timeout_s: float = _SPEC_TIMEOUT_S,
) -> list[Row]:
    """Measure every spec on one checkpoint, one child process each."""
    from transformers import AutoTokenizer

    from tests.golden.cases import CASES, PENALTIES

    model_name = Path(model_dir).name
    visible = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    golden_path = _GOLDEN_DIR / f"{model_name}.json"
    baseline: dict[str, list[str]] | None = None
    if use_golden and golden_path.exists():
        recorded = json.loads(golden_path.read_text())
        # Only the plain text cases: the file may also hold per-scheme and
        # continuous-batching keys, which were recorded through other paths.
        wanted = {key for name, _, _ in CASES for key in (name, f"{name}_rp1.1")}
        baseline = {k: v for k, v in recorded.items() if k in wanted}
        print(f"golden baseline: {golden_path.name} ({len(baseline)} cases)")
    elif use_golden:
        print(f"no golden baseline at {golden_path} — golden column will be empty")

    common = {
        "model": model_dir,
        "batch": batch,
        "max_gen": max_gen,
        "max_seq_len": max_seq_len,
        "kv_tokens": kv_tokens,
        "golden": baseline is not None,
        "golden_cases": CASES,
        "penalties": PENALTIES,
    }

    rows: list[Row] = []
    hf_texts: list[str] | None = None
    # The HF baseline follows the checkpoint's own dtype like every lite row does; the
    # tag carries it, so two runs cannot silently switch precision.
    hf_tag = dtype_tag(checkpoint_dtype(model_dir))
    if not skip_hf:
        print(f"\n=== {model_name} — HF {hf_tag} baseline ===")
        status, payload = _run_child({**common, "kind": "hf", "golden": False}, timeout_s)
        if status == "ok":
            hf_texts = payload.pop("texts")
            rows.append(Row(config=f"HF {hf_tag}", model=model_name, **payload))
            print(f"  TPS {rows[-1].tps:.1f} | peak {rows[-1].peak_mem_gb:.2f} GB")
        else:
            rows.append(Row(config=f"HF {hf_tag}", model=model_name, note=_first_line(payload)))
            print(f"  FAILED: {_first_line(payload)}")

    for spec in specs:
        print(f"\n=== {model_name} — {spec.label} ===")
        if spec.gpus_needed > visible:
            note = f"needs {spec.gpus_needed} GPUs, {visible} visible"
            rows.append(Row(config=spec.label, model=model_name, note=note))
            print(f"  SKIPPED: {note}")
            continue

        status, payload = _run_child({**common, "kind": "lite", "spec": spec}, timeout_s)
        if status == "error":
            rows.append(Row(config=spec.label, model=model_name, note=_first_line(payload)))
            print(f"  FAILED: {_first_line(payload)}")
            continue

        texts = payload.pop("texts")
        golden_texts = payload.pop("golden", None)
        row = Row(config=spec.label, model=model_name, **payload)
        if baseline is not None and golden_texts is not None:
            row.golden_match_rate, row.golden_prefix_rate, row.golden_exact = _score_golden(
                tokenizer, baseline, golden_texts
            )
        if hf_texts is not None:
            matched, compared, _, _ = _token_agreement(tokenizer, hf_texts, texts)
            row.token_match_rate = matched / compared if compared else None
        rows.append(row)
        print(
            f"  TPS {row.tps:.1f} | model {row.model_mem_gb or 0:.2f} GB "
            f"| KV {row.kv_cache_tokens or 0:,} tok "
            f"| graph {'on' if row.graph_installed else 'off'} "
            f"({row.graph_replays or 0} replays) "
            f"| golden "
            f"{'—' if row.golden_prefix_rate is None else f'{row.golden_prefix_rate:.3f} prefix'}"
        )

    _check_control(rows)
    return rows


def _first_line(text: str) -> str:
    """Last line of a traceback, which is the exception; enough to name the failure."""
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1][:160] if lines else "unknown failure"


def _check_control(rows: list[Row]) -> None:
    """The bf16 / eager / TP=1 row is the control for the golden column.

    It ran the same weights, the same kernels and the same width as the recording,
    so anything other than 1.000 means the *harness* moved tokens — a different KV
    pool size, a changed default — and every other row's deviation is then a mix of
    that and the axis under test. Saying so is the difference between a column that
    can be quoted and one that cannot.
    """
    control = next(
        (r for r in rows if r.config.startswith("bf16+tp1+eager") and r.ok),
        None,
    )
    if control is None or control.golden_match_rate is None:
        return
    if control.golden_match_rate < 1.0:
        print(
            f"\nWARNING: control row {control.config} scores "
            f"{control.golden_match_rate:.3f} against its own recorded baseline "
            f"({control.golden_exact} exact). The golden column measures the harness "
            f"as well as the quantisation in this run; re-record with "
            f"scripts/golden_tokens.py before quoting it."
        )
    else:
        print(f"\ncontrol row {control.config} reproduces the golden baseline exactly")


def _specs_from_args(args: argparse.Namespace) -> list[RunSpec]:
    """Cross product of the requested axes, in a deterministic order."""
    schemes = [None if s in ("None", "none", "fp16", "bf16") else s for s in args.schemes]
    graphs: list[bool] = []
    if args.cuda_graph or not args.no_cuda_graph:
        graphs.append(True)
    if args.no_cuda_graph:
        graphs.append(False)
    return [
        RunSpec(scheme=scheme, kv_cache_dtype=kv, tp=tp, dp=dp, graph=graph, engine=args.engine)
        for scheme in schemes
        for kv in args.kv_cache_dtype
        for tp in args.tp
        for dp in args.dp
        for graph in graphs
    ]


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", help="Checkpoint directory to benchmark")
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Run the representative models under ${_MODELZOO_ENV}",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["fp16", "fp8", "int4"],
        help="Quantisation schemes; fp16/bf16/none mean the checkpoint's own dtype",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        nargs="+",
        default=["auto"],
        choices=["auto", "fp8", "fp8_e4m3"],
    )
    parser.add_argument("--tp", nargs="+", type=int, default=[1], help="Tensor parallel sizes")
    parser.add_argument("--dp", nargs="+", type=int, default=[1], help="Data parallel sizes")
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "continuous"],
        help=(
            "auto: the lockstep batch loop, or continuous batching where a cell needs "
            "it (tp>1). continuous: force it everywhere, so a tp1 row is a like-for-like "
            "baseline for a tp2 one instead of differing in scheduler as well"
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture decode graphs (the default); pass both flags to measure each",
    )
    parser.add_argument("--no-cuda-graph", action="store_true", help="Decode eager")
    parser.add_argument("--batch", type=int, default=_BATCH, help="Prompts per replica")
    parser.add_argument("--max-gen", type=int, default=_MAX_GEN)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--max-gpu-num-blocks",
        type=int,
        default=None,
        help="KV cache tokens; profiled when omitted, which is what the memory columns measure",
    )
    parser.add_argument("--skip-hf", action="store_true", help="Skip the HuggingFace baseline row")
    parser.add_argument("--no-golden", action="store_true", help="Skip the golden comparison")
    parser.add_argument("--spec-timeout", type=float, default=_SPEC_TIMEOUT_S)
    parser.add_argument("--json", help="Output path; a default under docs/benchmark_logs is used")


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    # The HF row and the lite children import transformers / rapid_llm inside
    # their processes. A bare interpreter (a container's system python, say)
    # lacks them; failing here names the fix instead of a mid-matrix traceback.
    missing = [
        name for name in ("transformers", "rapid_llm") if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(
            f"this interpreter lacks {', '.join(missing)} — benchmarks run on the "
            "project environment, e.g. "
            f".venv/bin/python {Path(__file__).resolve().relative_to(Path.cwd())} ...",
            file=sys.stderr,
        )
        return 1

    if args.all:
        root = os.environ.get(_MODELZOO_ENV)
        if not root:
            print(f"--all needs ${_MODELZOO_ENV} set", file=sys.stderr)
            return 1
        models = [str(Path(root) / name) for name in _ALL_MODELS]
        models = [m for m in models if Path(m).exists()]
    elif args.model_dir:
        models = [args.model_dir]
    else:
        raise SystemExit("pass --model-dir or --all")

    specs = _specs_from_args(args)
    print(f"{len(specs)} configuration(s) x {len(models)} model(s):")
    for spec in specs:
        print(f"  {spec.label}")

    rows: list[Row] = []
    for model_dir in models:
        rows.extend(
            benchmark_model(
                model_dir,
                specs,
                batch=args.batch,
                max_gen=args.max_gen,
                max_seq_len=args.max_seq_len,
                kv_tokens=args.max_gpu_num_blocks,
                skip_hf=args.skip_hf,
                use_golden=not args.no_golden,
                timeout_s=args.spec_timeout,
            )
        )

    print("\n" + "=" * 70)
    print(_render(rows))

    out_path = Path(args.json) if args.json else _default_json_path(models[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "meta": _meta(models[0], args.batch, args.max_gen),
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nJSON saved to {out_path}")
    return 0 if all(r.ok for r in rows) else 1


def _default_json_path(model_dir: str) -> Path:
    gpu = gpu_tag()
    stamp = date.today().strftime("%Y%m%d")
    return (
        _REPO_ROOT
        / "docs"
        / "benchmark_logs"
        / f"bench_quant_{Path(model_dir).name}_{gpu}_{stamp}.json"
    )

