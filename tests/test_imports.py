"""Public imports must not eagerly initialise the GPU execution stack.

Importing ``rapid_llm.sampling`` (or engine / model components) must
leave torch CUDA internals and Triton untouched, so lightweight
consumers pay no GPU tax — asserted via the loaded-module diff.

Usage:
    pytest tests/test_imports.py
"""

from __future__ import annotations

import subprocess
import sys


def _imported_modules(statement: str) -> set[str]:
    script = (
        f"{statement}\n"
        "import sys\n"
        "print('\\n'.join(sorted(name for name in sys.modules "
        "if name.startswith('rapid_llm'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def test_sampling_import_does_not_load_executor_or_kernels() -> None:
    modules = _imported_modules("from rapid_llm import SamplingParams")

    assert "rapid_llm.engine.sampler" in modules
    assert not any(name.startswith("rapid_llm.executor") for name in modules)
    assert not any(name.startswith("rapid_llm.kernels") for name in modules)


def test_engine_submodule_import_does_not_load_implementations() -> None:
    modules = _imported_modules("from rapid_llm.engine.scheduler import Scheduler")

    assert "rapid_llm.engine.scheduler" in modules
    assert "rapid_llm.engine.llm_engine" not in modules
    assert "rapid_llm.engine.batch_planner" in modules


def test_model_components_do_not_eagerly_import_triton_kernels() -> None:
    modules = _imported_modules("from rapid_llm.modules import RotaryEmbedding, ReplicatedLinear")

    assert "rapid_llm.modules.rotary_embedding" in modules
    assert "rapid_llm.modules.linear" in modules
    assert not any(name.startswith("rapid_llm.kernels") for name in modules)
    assert "triton" not in modules
