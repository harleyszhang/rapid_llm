"""Shared fixtures and collection policy for the rapid_llm test suite.

Three policies live here:

* Which modules a machine without CUDA may import at all: answered from the
  per-module declarations in :mod:`tests.registry` (parsed with :mod:`ast`,
  so finding out never imports anything).
* What a marker means on this machine: ``gpu``/``weights`` become skips, and
  the golden gate reports UNVERIFIED -- never a silent skip -- or fails
  under ``RAPID_LLM_GOLDEN_STRICT=1``.
* What must hold around every test: seeded torch state, a pinned dispatch
  switch, and a rank grid restored to a world of one.

Usage:
    pytest tests/   # collection policy from this file applies automatically
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.registry import IMPORT_NEEDS_CUDA, declares

# tests/conftest.py -> tests/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Checkpoint the integration tier uses by default. Override with
#: ``RAPID_LLM_TEST_MODEL_DIR`` to point at any other checkpoint.
DEFAULT_MODEL_NAME = "Qwen2.5-0.5B"

#: Shared checkpoint root, the same variable the benchmark scripts read. Honoured
#: here too so a machine that keeps weights outside the repository runs the
#: integration tier without symlinking them into ``my_weight/``.
MODELZOO_ENV = "RAPID_LLM_MODELZOO"

#: Directories whose modules reach the Triton runtime at module scope: the
#: whole directory leaves collection on a machine without CUDA. Everything
#: else answers for itself -- a module that cannot be imported there says so
#: in its own source, with an :func:`~tests.registry.import_needs_cuda`
#: declaration (see ``tests/registry.py`` for why a marker cannot do this).
_GPU_ONLY_DIRS = ("kernels",)

#: Directories that constitute the golden gate — these must never silently skip.
_GOLDEN_DIRS = ("golden",)

#: When set, golden tests that cannot run become hard FAILs instead of xfail.
_GOLDEN_STRICT = os.environ.get("RAPID_LLM_GOLDEN_STRICT", "") == "1"

#: Marker `_golden_outcome` adds under the strict gate; `pytest_runtest_setup`
#: turns it into a FAILED report. Registered in `pytest_configure`.
_GOLDEN_GATE_FAIL = "golden_gate_fail"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Do not import CUDA-bound test modules on a machine without CUDA.

    Marker selection happens after Python imports a test module, so a module
    whose *import* needs the Triton/CUDA runtime must leave collection before
    pytest reaches it: files under ``_GPU_ONLY_DIRS`` by location, anything
    else by the declaration in its own source (:func:`tests.registry.declares`
    -- parsed, never imported).
    """
    if torch.cuda.is_available():
        return None
    try:
        relative = collection_path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    if any(relative.parts[:2] == ("tests", d) for d in _GPU_ONLY_DIRS):
        return True
    if collection_path.suffix != ".py":
        return None
    return declares(collection_path, IMPORT_NEEDS_CUDA) or None


def checkpoint_candidates(reference: str) -> list[Path]:
    """Every place a checkpoint reference may resolve to, most specific first.

    ``reference`` is a path (absolute, or relative to the repository root) or a
    bare checkpoint name. The reference itself comes first, then its final
    component under the shared checkpoint root -- so a machine that keeps weights
    outside the repository finds them without symlinking them in. Finally
    ``my_weight/`` is searched for whichever single usable checkpoint it holds: a
    machine that downloaded ``-Instruct`` where a config names the base model
    still runs the tier instead of skipping all of it, which is how the failures
    hiding behind those skips stay visible.

    Public because ``tests/evals`` resolves the checkpoint each of its configs
    names, and the chain must not be answered in two places.
    """
    path = Path(reference).expanduser()
    candidates = [path if path.is_absolute() else REPO_ROOT / path]

    name = path.name
    zoo = os.environ.get(MODELZOO_ENV)
    if zoo:
        root = Path(zoo).expanduser()
        candidates += [root / name, *sorted(root.glob(f"*/{name}"))]

    usable = [candidate for candidate in candidates if checkpoint_problem(candidate) is None]
    if usable:
        return usable
    local = sorted(p for p in (REPO_ROOT / "my_weight").glob("*") if checkpoint_problem(p) is None)
    return local or candidates


def _resolve_model_dir() -> Path:
    """Absolute path of the checkpoint under test, without validating it.

    An explicit ``RAPID_LLM_TEST_MODEL_DIR`` is the whole answer: a caller that
    named a checkpoint does not want a fallback silently substituted for it.
    Otherwise the first usable candidate, or the first candidate when none is
    usable -- the caller reports that one's problem rather than testing nothing.
    """
    explicit = os.environ.get("RAPID_LLM_TEST_MODEL_DIR")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else REPO_ROOT / path
    return checkpoint_candidates(DEFAULT_MODEL_NAME)[0]


def checkpoint_problem(path: Path) -> str | None:
    """Describe why ``path`` is unusable as a checkpoint, or ``None`` if it is.

    Public because ``tests/evals`` gates on checkpoints named in its own configs
    rather than on the one this file resolves, and "what counts as a usable
    checkpoint" must not be answered in two places.
    """
    if not path.is_dir():
        return f"no such directory: {path}"
    if not (path / "config.json").is_file():
        return f"no config.json in {path}"
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        return f"no *.safetensors or *.bin weights in {path}"
    return None


def _is_golden(nodeid: str) -> bool:
    """Whether this test item belongs to the golden gate suite."""
    return any(f"tests/{d}/" in nodeid or f"tests\\{d}\\" in nodeid for d in _GOLDEN_DIRS)


# --------------------------------------------------------------------------- #
# Collection policy
# --------------------------------------------------------------------------- #
def pytest_configure(config: pytest.Config) -> None:
    """Register the marker the strict golden gate adds to unrunnable tests."""
    config.addinivalue_line(
        "markers",
        "golden_gate_fail(reason): a golden test this machine cannot run while "
        "RAPID_LLM_GOLDEN_STRICT=1; reported as FAILED, never as a skip",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Report a strict-gated golden test as FAILED instead of skipped.

    No marker combination expresses "cannot run *and* must fail": ``skip``
    outranks ``xfail``, and ``xfail(run=False)`` reports xfailed even under
    ``strict=True``. Failing from setup does say FAILED, which is what the
    strict gate promises.
    """
    gate = item.get_closest_marker(_GOLDEN_GATE_FAIL)
    if gate is not None:
        pytest.fail(gate.kwargs.get("reason", "golden gate: cannot run"))


def _golden_outcome(reason: str):
    """How a golden test that cannot run is reported on this machine.

    UNVERIFIED -- yellow/orange in CI, never green -- unless
    ``RAPID_LLM_GOLDEN_STRICT=1`` upgrades it to a hard failure.
    """
    if _GOLDEN_STRICT:
        return getattr(pytest.mark, _GOLDEN_GATE_FAIL)(reason=f"GOLDEN GATE FAIL: {reason}")
    return pytest.mark.xfail(reason=f"UNVERIFIED: {reason}", run=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply directory-based marks, then skip what the machine cannot run.

    For golden tests the outcome is never a silent skip; see
    :func:`_golden_outcome`.
    """
    model_dir = _resolve_model_dir()
    checkpoint_problem_reason = checkpoint_problem(model_dir)
    cuda_missing = not torch.cuda.is_available()

    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_weights = pytest.mark.skip(reason=f"needs a checkpoint: {checkpoint_problem_reason}")

    for item in items:
        # Triton kernels cannot run on CPU; mark by location so new kernel
        # tests inherit the requirement automatically.
        if any(
            f"tests/{d}/" in item.nodeid or f"tests\\{d}\\" in item.nodeid for d in _GPU_ONLY_DIRS
        ):
            item.add_marker(pytest.mark.gpu)

        is_golden_test = _is_golden(item.nodeid)

        if cuda_missing and "gpu" in item.keywords:
            item.add_marker(_golden_outcome("no CUDA device") if is_golden_test else skip_gpu)

        if checkpoint_problem_reason and "weights" in item.keywords:
            item.add_marker(
                _golden_outcome(checkpoint_problem_reason) if is_golden_test else skip_weights
            )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
#: ``parallel_state``'s coordinates after ``_reset_grid()``, read through the
#: public accessors. ``_restore_parallel_grid`` compares against this.
_WORLD_OF_ONE = (0, 1, 0, 1, 0, 1, False, False, False, False, False)


@pytest.fixture(autouse=True)
def _restore_parallel_grid():
    """Restore the rank grid after every test; accuse a test that left it dirty.

    ``rapid_llm.distributed.parallel_state`` keeps the grid in module globals,
    and a teardown that cannot finish -- ``destroy_parallel`` raising out of
    ``dist.destroy_process_group`` when its peer is already gone -- leaves
    ``tp_size=2`` behind for the rest of the process: the next engine build
    reads a world nobody asked for, and tests fail far from the leak.

    Fixture teardown is innermost-first, so this conftest-level finalizer runs
    *after* module-local teardowns (the ``_reset_grid`` fixtures in
    tests/distributed and friends) -- anything still standing belongs to the
    test that just ended. The state is cleared through ``abandon_parallel``:
    ``destroy_parallel`` may be the very call that hung or raised, and this
    path must not depend on it.
    """
    yield
    from rapid_llm.distributed import parallel_state as ps

    # Every public coordinate the module exposes, against _reset_grid's
    # defaults; group objects are observed by presence.
    observed = (
        ps.get_tensor_model_parallel_rank(),
        ps.get_tensor_model_parallel_world_size(),
        ps.get_data_parallel_rank(),
        ps.get_data_parallel_world_size(),
        ps.get_ep_rank(),
        ps.get_ep_world_size(),
        ps.expert_parallel_enabled(),
        ps.dp_attention_enabled(),
        ps.get_tensor_model_parallel_group() is not None,
        ps.get_data_parallel_group() is not None,
        ps.get_ep_group() is not None,
    )
    if observed == _WORLD_OF_ONE:
        return
    ps.abandon_parallel()
    pytest.fail(
        f"left the parallel_state grid dirty: {observed} != {_WORLD_OF_ONE}; "
        "the test's teardown did not return the process to a world of one "
        "(the grid was reset, so later tests are not poisoned)"
    )


@pytest.fixture(autouse=True)
def _reset_torch_state():
    """Seed every test identically and drop cached blocks afterwards.

    Kernel tests compare against a reference on random inputs, so an unseeded
    run would be non-reproducible: a failure could not be re-triggered.
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: Hermetic dispatch, process-wide: a developer's frozen records must not flip
#: the suite. This has to be set at conftest *import*, not only in the
#: function-scoped fixture below — module/session-scoped fixtures (engine
#: builders in tests/engine, tests/golden, ...) are instantiated before
#: function-scoped autouse fixtures, and a dispatch made there caches its
#: decision on the global registry for the rest of the session.
os.environ["RAPID_LLM_FROZEN_RANK"] = "0"


@pytest.fixture(autouse=True)
def _frozen_rank_off():
    """Re-pin the switch per test, so one opting in cannot leak the opt-in.

    A test opting into frozen ranking sets ``RAPID_LLM_FROZEN_RANK=1`` (via
    monkeypatch) for its own duration; that undo restores the process-wide
    ``"0"`` from this module's import, and this re-pin also covers a test
    that unsets the variable outright.

    A plain assignment, not ``monkeypatch.setenv``: a fixture that *depends*
    on monkeypatch is set up before every other fixture and torn down after
    them all, so the dependency would defer every test's monkeypatch undo
    past the grid guard above -- and a grid simulated with
    ``monkeypatch.setattr`` would still be standing when the guard looks.
    """
    os.environ["RAPID_LLM_FROZEN_RANK"] = "0"


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Validated checkpoint directory; skips the test when it is unusable."""
    path = _resolve_model_dir()
    problem = checkpoint_problem(path)
    if problem:
        pytest.skip(f"needs a checkpoint: {problem}")
    return path


@pytest.fixture
def cuda_available() -> bool:
    """Skip unless CUDA is present. Prefer the ``gpu`` mark in new tests."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return True


def needs_capability(minimum: tuple[int, int], feature: str):
    """Mark a test as needing a CUDA device of at least ``minimum`` capability.

    The tier half is the ``gpu`` mark, the same reasoning as ``needs_gpus`` in
    tests/distributed: a CPU-tier run must not reach for kernels it has no
    device to test. The skip half tells "this box cannot verify the claim"
    apart from "the claim is false": fp8 e4m3 starts at sm_89 and nvfp4 at
    sm_100, so a box with an older device steps aside naming what it has
    instead of blaming the kernel for the hardware's age.

    Args:
        minimum: ``(major, minor)`` the device must reach, e.g. ``(8, 9)``.
        feature: What the floor buys, named in the skip reason.
    """
    found = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    skip = pytest.mark.skipif(
        found < minimum,
        reason=(
            f"needs a CUDA device of capability >= {minimum[0]}.{minimum[1]} "
            f"for {feature}; this box has {found[0]}.{found[1]}"
        ),
    )

    def decorate(func):
        return pytest.mark.gpu(skip(func))

    return decorate
