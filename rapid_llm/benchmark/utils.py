"""Runtime helpers shared by every bench script: GPU hygiene, memory footprints,
JSON logs, table rendering, tokenizer and endpoint readiness.

Nothing here knows about engines or metrics — that is :mod:`backends` and
:mod:`metrics`.

Usage:
    from rapid_llm.benchmark import write_json_log, free_gpu, require_gpus, get_tokenizer
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_module
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def kv_args(pairs: list[str]) -> dict:
    """``key=value`` pairs into kwargs, values json-decoded (``false``/``1.0``/...).

    The one parser behind every ``--engine-arg`` / ``--vllm-arg`` flag, so a
    feature A/B spells its value the same way whichever runner takes it.
    """
    out: dict = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=value, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def median_round(rounds: list, key):
    """The middle round by ``key`` — the one reported when ``--iters > 1``.

    An actual round rather than a per-field median: mixing the fastest round's
    TTFT with another's TPOT describes a run that never happened. Upper median
    for even counts (``len // 2`` of the sorted list), as sglang reports.
    """
    return sorted(rounds, key=key)[len(rounds) // 2]


def gpu_tag() -> str:
    """Filename-safe GPU tag: ``NVIDIA H100 80GB HBM3`` -> ``h100``.

    Vendor and memory words (``80gb``, ``hbm3``) are dropped, the rest is
    lowercased and joined. ``cpu`` without CUDA, ``gpu`` if nothing survives.
    """
    if not torch.cuda.is_available():
        return "cpu"

    vendor_words = {"nvidia", "geforce", "tesla", "quadro"}

    def is_model_word(word: str) -> bool:
        return (
            word not in vendor_words  # vendor / product line
            and not word.endswith("gb")  # VRAM capacity, e.g. 80gb
            and "hbm" not in word  # VRAM type, e.g. hbm3
        )

    # "-" is a separator too: "A100-SXM4-80GB" drops only its memory segment.
    words = torch.cuda.get_device_name(0).lower().replace("-", " ").split()
    return "".join(word for word in words if is_model_word(word)) or "gpu"


def free_gpu() -> None:
    """Release the CUDA caching allocator's view of a torn-down engine.

    Engine/generator/executor/KV manager hold mutual references, so without
    an explicit gc pass the memory is not returned: a second backend built in
    the same process then profiles a KV budget of zero tokens.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def reset_peak_mem() -> None:
    """Start a new peak-memory window (call before building the thing under test)."""
    torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    """Peak allocated bytes since :func:`reset_peak_mem`, in GiB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


def describe_footprint(runner, replicas: int = 1) -> tuple[float, int]:
    """``(weight GiB, KV pool capacity in tokens)``, read from ``ModelRunner``'s tensors.

    ``replicas`` is the TP rank count: the runner holds only this rank's shard, so
    a whole replica's weights are ``replicas`` times that.
    """
    weight_bytes = sum(p.numel() * p.element_size() for p in runner.model.parameters())
    kv_tokens = runner.kv_cache_manager.gpu_kv_buffer[0].shape[0]
    return weight_bytes * replicas / (1024**3), kv_tokens


def footprint_stats(runner) -> dict:
    """The memory and graph columns every offline benchmark reports per row."""
    weights_gib, kv_tokens = describe_footprint(runner)
    manager = runner._graph_manager
    return {
        "model_mem_gb": weights_gib,
        "kv_cache_tokens": kv_tokens,
        "graph_installed": manager is not None,
        "graph_replays": None if manager is None else manager.replays,
    }


def timed_rounds(run, iters: int):
    """Run ``iters`` sync-bounded rounds; return the median one as ``(seconds, value)``.

    Latency and payload come from the *same* round (see :func:`median_round`), so
    a reported token count belongs to the reported wall clock.
    """
    rounds: list[tuple[float, object]] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = run()
        torch.cuda.synchronize()
        rounds.append((time.perf_counter() - start, value))
    return median_round(rounds, key=lambda r: r[0])


def measure_generate(
    generate,
    prompts: list[str],
    *,
    gen_len: int,
    iters: int,
    tokenizer,
    warmup_prompts: list[str] | None = None,
) -> tuple[float, int, list[str]]:
    """Measure a one-shot ``generate`` path: warm up, time ``iters`` rounds, count tokens.

    Args:
        generate: ``(prompts, params) -> [output]``, each output having ``.text``.
        warmup_prompts: Prompts for the warm-up round; the measured ones by default.
            A script measuring cache hits must pass prompts from *outside* the
            workload, or the warm-up has already written the prefixes under test
            into the cache and every row reports a hit rate it did not earn.

    Returns:
        ``(wall clock in seconds, output tokens, texts)`` — all three of the
        median round, so they describe one run rather than three.
    """
    from .workloads import sampling_params

    generate(warmup_prompts or prompts, sampling_params(8))
    latency, outputs = timed_rounds(lambda: generate(prompts, sampling_params(gen_len)), iters)
    texts = [out.text for out in outputs]
    return latency, count_gen_tokens(texts, tokenizer), texts


def count_gen_tokens(texts: list[str], tokenizer) -> int:
    """Re-tokenise generated text to count output tokens (vLLM's own method)."""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


def report_agreement(reference: list[str], rows: list[tuple[str, list[str]]]) -> None:
    """Every configuration must return the same completions; a low rate is a bug flag.

    Greedy sampling must be routing-independent: a shared prefix that hits the
    cache is *copied* K/V, not recomputed, so it can differ from a fresh prefill
    in the last bits — and an fp16 greedy tie can flip on that. The agreement
    rate is the flag that says the reuse is not merely inexact but wrong.
    """
    for label, texts in rows:
        if len(texts) != len(reference):
            continue
        same = sum(a == b for a, b in zip(reference, texts, strict=True))
        empty = sum(not text for text in texts)
        print(
            f"{label}: {same}/{len(reference)} completions identical to the baseline, {empty} empty"
        )


def require_gpus(min_count: int = 1) -> int:
    """Exit unless CUDA exposes ``min_count`` devices; returns the visible count."""
    visible = torch.cuda.device_count()
    if visible < min_count:
        print(
            f"requires {min_count} CUDA device(s), found {visible}",
            file=sys.stderr,
        )
        sys.exit(1)
    return visible


def timestamped_log_path(log_dir: str | Path, prefix: str) -> Path:
    """``<log_dir>/<prefix>_<stamp>.json`` — the --log-dir naming convention."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{prefix}_{stamp}.json"


def environment() -> dict:
    """The hardware/software facts every benchmark report must carry.

    Collected rather than hand-written so the numbers cannot drift from the
    machine that actually produced them: GPU model and count, SM count and
    compute capability, interconnect topology (``PHB`` in the nvidia-smi topo
    matrix means PCIe host bridge, i.e. no NVLink), driver and library
    versions, host CPU/memory, and the inference mode.
    """
    import os
    import platform
    import subprocess

    import transformers
    import triton

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    topo = ""
    driver = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        ).stdout
        topo = "; ".join(
            line.strip()
            for line in out.splitlines()
            if line.startswith("GPU0") or line.startswith("GPU1")
        )
    except Exception:
        topo = "unavailable"
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except Exception:
        driver = "unavailable"
    return {
        "gpu_model": gpu.name if gpu else "no CUDA device",
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gib": round(gpu.total_memory / 1024**3, 1) if gpu else None,
        "sm_count": gpu.multi_processor_count if gpu else None,
        "compute_capability": f"sm_{gpu.major}{gpu.minor}" if gpu else None,
        "driver_version": driver,
        "interconnect_topology": topo or "unknown",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "inference_mode": "offline (all prompts submitted at once, no serving queue)",
    }


def write_json_log(path: str | Path, config: dict, results) -> None:
    """One JSON shape for every benchmark: {"config": ..., "results": ...}.

    A ``timestamp`` is stamped into the config unless the caller supplied one,
    and the machine/library facts from :func:`environment` are stamped in too —
    every benchmark report owes its reader the environment the numbers came
    from, and collecting it here means no bench script can forget it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        **config,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": config.get("environment") or environment(),
    }
    path.write_text(json.dumps({"config": config, "results": results}, indent=2, default=str))
    print(f"-> {path}")


def print_row_table(headers: list[str], widths: list[int], rows: list[list[str]]) -> None:
    """Aligned rows between two rules: first column left-aligned, the rest right.

    The caller formats each cell, so a column can hold a number, a ratio or ``—``
    without this function knowing which.
    """
    fmt = "".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    rule = "─" * sum(widths)
    print(f"\n{rule}")
    print(fmt.format(*headers))
    print(rule)
    for row in rows:
        print(fmt.format(*row))
    print(rule)


def print_run_header(title: str, fields: dict[str, object], *, width: int = 91) -> None:
    """The banner every run opens with: model, workload knobs, then the device.

    The device line is not decoration: a table without it is an anecdote, so the
    scripts that print comparison tables all open with this.
    """
    print(f"\n{'=' * width}")
    print(f"{title}  |  " + "  ".join(f"{k}={v}" for k, v in fields.items()))
    print(f"gpu={torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"{'=' * width}")


def get_tokenizer(model_path: str):
    """The benchmark's tokenizer (sglang ``utils.get_tokenizer`` mirror).

    Local checkpoints load straight off disk; repo ids go through
    ``AutoTokenizer`` with ``trust_remote_code`` so custom tokenizers resolve.
    """
    from transformers import AutoTokenizer

    assert model_path, "model path is required"
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def wait_for_endpoint(url: str, timeout_s: float = 3600.0) -> float:
    """Block until the endpoint answers 200; returns the observed latency (s).

    sglang ``bench_utils`` mirror: probes the URL in a loop so a slow engine
    bring-up (graph capture, KV profiling) is waited out, not raced. The
    returned value is the last probe's round-trip — the heartbeat latency the
    serving benchmark reports alongside its percentiles.
    """
    import requests

    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"endpoint {url} not ready after {timeout_s:.0f}s")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return time.monotonic() - start
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)


def run_in_spawned_process(
    target: Callable[[Any, mp.Queue], None],
    arg: Any,
    timeout_s: float,
    join_s: float = 60.0,
) -> tuple[str, Any]:
    """Run ``target(arg, queue)`` in a fresh spawned process; return what it queued.

    The one process-lifecycle loop the parallel/serving scenarios share: a probe
    (an engine build plus its measurement) must run in its own process so a
    wedged rank cannot take the parent down and each build gets a clean GPU. The
    child reports through the queue with ``queue.put((status, payload))`` — by
    convention ``"ok"`` with a result or ``"error"`` with a traceback string;
    this returns that tuple verbatim, or ``("timeout", msg)`` when nothing arrived
    in ``timeout_s``.

    Not a daemon: with ``tp_size > 1`` the child spawns the follower ranks, and a
    daemonic process may not have children. The process is always joined and, if
    still alive, terminated — TP followers and DP replicas hold GPU memory the
    next configuration would otherwise discover as an OOM.
    """
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(target=target, args=(arg, result_queue), daemon=False)
    process.start()
    try:
        try:
            return result_queue.get(timeout=timeout_s)
        except queue_module.Empty:
            return "timeout", f"no result within {timeout_s:.0f}s"
    finally:
        process.join(timeout=join_s)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()
            process.join(timeout=30.0)
