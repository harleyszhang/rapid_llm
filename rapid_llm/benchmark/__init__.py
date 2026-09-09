"""Benchmark library — the sglang-style layering, inside the package it measures.

The implementation lives here, next to the engine, the way sglang keeps
``python/sglang/benchmark/`` inside the package; the top-level ``benchmarks/``
holds only scenario scripts that import from here — nothing is implemented
twice, and there is no ``dp`` module: DP is one backend (:class:`DPBackend`).

Layering, bottom-up: ``datasets/`` (workload rows via ``get_dataset``),
``workloads`` (prompt/sampling presets), ``metrics`` (TTFT/TPOT/TPS),
``stream_metrics`` (steady-state windows), ``backends`` (one ABC, one
implementation per engine, ``measure_rows`` the only drive loop), ``utils``
(GPU hygiene, footprints, JSON logs, tables, tokenizer, endpoint).

Runnable on top, each a thin orchestration of the above:

    python -m rapid_llm.benchmark.offline_throughput   # 3-arm offline throughput
    python -m rapid_llm.benchmark.one_batch            # batch-size latency scan
    python -m rapid_llm.benchmark.serving              # online SSE serving
    python -m rapid_llm.benchmark.models               # the model-zoo suite

Usage:
    from rapid_llm.benchmark import BenchResult, make_backend, write_json_log
"""

from .backends import (
    ARMS,
    Backend,
    DPBackend,
    EngineBackend,
    HFBackend,
    LiteBackend,
    VisionBackend,
    VLLMBackend,
    build_arm,
    checkpoint_dtype,
    dtype_tag,
    make_backend,
    uniform_rows,
)
from .datasets import DatasetRow, add_dataset_args, get_dataset
from .metrics import (
    BenchResult,
    RequestRun,
    pctl,
    print_table,
    result_from_requests,
    run_requests,
    steps_to_result,
)
from .stream_metrics import BatchStreamRecorder, SteadyStateWindow
from .utils import (
    count_gen_tokens,
    describe_footprint,
    environment,
    footprint_stats,
    free_gpu,
    get_tokenizer,
    gpu_tag,
    kv_args,
    measure_generate,
    median_round,
    peak_mem_gb,
    print_row_table,
    print_run_header,
    report_agreement,
    require_gpus,
    reset_peak_mem,
    run_in_spawned_process,
    timed_rounds,
    timestamped_log_path,
    wait_for_endpoint,
    write_json_log,
)
from .workloads import GREEDY_PARAMS, PROMPTS, SAMPLE_KW, expand_prompts, sampling_params

__all__ = [
    "ARMS",
    "GREEDY_PARAMS",
    "PROMPTS",
    "SAMPLE_KW",
    "Backend",
    "BatchStreamRecorder",
    "BenchResult",
    "DPBackend",
    "DatasetRow",
    "EngineBackend",
    "HFBackend",
    "LiteBackend",
    "RequestRun",
    "SteadyStateWindow",
    "VLLMBackend",
    "VisionBackend",
    "add_dataset_args",
    "build_arm",
    "checkpoint_dtype",
    "count_gen_tokens",
    "describe_footprint",
    "dtype_tag",
    "environment",
    "expand_prompts",
    "footprint_stats",
    "free_gpu",
    "get_dataset",
    "get_tokenizer",
    "gpu_tag",
    "kv_args",
    "make_backend",
    "measure_generate",
    "median_round",
    "pctl",
    "peak_mem_gb",
    "print_row_table",
    "print_run_header",
    "print_table",
    "report_agreement",
    "require_gpus",
    "reset_peak_mem",
    "result_from_requests",
    "run_in_spawned_process",
    "run_requests",
    "sampling_params",
    "steps_to_result",
    "timed_rounds",
    "timestamped_log_path",
    "uniform_rows",
    "wait_for_endpoint",
    "write_json_log",
]
