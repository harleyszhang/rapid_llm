"""Unit tests for DP-attention benchmark comparison semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).parents[2] / "benchmarks/parallelism/bench_dp_attention.py"
_SPEC = importlib.util.spec_from_file_location("bench_dp_attention", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)
_attach_comparisons = _BENCHMARK._attach_comparisons
_shape_matched_reference = _BENCHMARK._shape_matched_reference


class _BatchSensitiveEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, _params):
        self.calls.append(prompts)
        return [SimpleNamespace(text=f"{prompt}|batch={len(prompts)}") for prompt in prompts]


def test_shape_matched_reference_replays_round_robin_buckets_in_request_order():
    engine = _BatchSensitiveEngine()
    prompts = ["p0", "p1", "p2", "p3", "p4"]

    texts = _shape_matched_reference(engine, prompts, object(), replicas=2)

    assert engine.calls == [["p0", "p2", "p4"], ["p1", "p3"]]
    assert texts == [
        "p0|batch=3",
        "p1|batch=2",
        "p2|batch=3",
        "p3|batch=2",
        "p4|batch=3",
    ]


def test_exact_completions_use_shape_matched_reference():
    scenarios = {"balanced": ["p0", "p1", "p2", "p3"]}
    baseline = {
        "arm": "local1",
        "dp": 1,
        "parallel_degree": 1,
        "parity_references": {
            "1": {"balanced": ["full-0", "full-1", "full-2", "full-3"]},
            "2": {"balanced": ["bucket-0", "bucket-1", "bucket-2", "bucket-3"]},
        },
        "scenarios": [
            {
                "scenario": "balanced",
                "tps": 10.0,
                "texts": ["full-0", "full-1", "full-2", "full-3"],
            }
        ],
    }
    dpa = {
        "arm": "dpa2",
        "dp": 2,
        "parallel_degree": 2,
        "scenarios": [
            {
                "scenario": "balanced",
                "tps": 15.0,
                "texts": ["bucket-0", "wrong", "bucket-2", "bucket-3"],
            }
        ],
    }

    _attach_comparisons([baseline, dpa], scenarios)

    assert baseline["scenarios"][0]["exact_completions"] == 4
    assert dpa["scenarios"][0]["exact_completions"] == 3
