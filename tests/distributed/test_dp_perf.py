"""Performance guard for data parallelism: two replicas must actually serve twice.

Each width serves the same prompt set *per replica*, so the total work grows with
the replica count and throughput is what scales. Splitting one fixed set instead
would halve each replica's batch rather than its step count, and a small-model
decode is bandwidth-bound enough that a batch of 6 costs the same step time as a
batch of 12 -- that measures flat at 1.0x and is correct behaviour, not a
serialised coordinator. ``benchmarks/bench_data_parallel.py`` names the two modes
``weak`` (this one) and ``strong`` (the flat one).

Usage:
    pytest tests/distributed/test_dp_perf.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rapid_llm import DataParallelController, SamplingParams
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

_MAX_SEQ_LEN = 512
_KV_TOKENS = 4096
_MAX_GEN = 64

#: The per-replica batch. Enough requests that per-dispatch overhead is amortised
#: and every replica stays busy for the whole measurement; short enough to keep the
#: test under a minute. Each width serves this many prompts *per replica*.
_PROMPTS = [
    "Explain what a GPU does.",
    "Write a short poem about rain.",
    "List four prime numbers.",
    "What is the capital of Japan?",
    "Describe the colour blue.",
    "Name three programming languages.",
    "Summarise the water cycle.",
    "Why is the sky blue?",
    "Give one use of a hash table.",
    "What does a compiler do?",
    "Name a famous bridge.",
    "Explain recursion briefly.",
]

#: Two cards on an idle machine measure ~1.9x. Set low enough that a busy GPU or a
#: noisy neighbour cannot fail the build, high enough that a serialised
#: coordinator (~1.0x) cannot pass it.
_MIN_SCALING = 1.4

_GREEDY = SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN, repetition_penalty=1.0)


def _serve(model_dir: Path, replicas: int) -> tuple[float, int]:
    """Serve one copy of the prompt set per replica; return seconds and answers.

    ``max_num_seqs`` stays at the per-replica width, so every replica runs the
    same batch it would alone and the comparison isolates the second card rather
    than a change in batch shape. A short warm-up dispatch first: the first
    generation on a fresh replica pays for lazily compiled Triton kernels, and
    charging that to the two-replica run (which pays it twice, in parallel) would
    measure the compiler.
    """
    prompts = _PROMPTS * replicas
    engine = DataParallelController(
        model=str(model_dir),
        data_parallel_size=replicas,
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_TOKENS,
        max_num_seqs=len(_PROMPTS),
        use_cuda_graph=False,
    )
    try:
        engine.generate(_PROMPTS[:replicas], SamplingParams(temperature=0.0, max_gen_len=4))

        started = time.perf_counter()
        outputs = engine.generate(prompts, _GREEDY)
        elapsed = time.perf_counter() - started
        return elapsed, sum(1 for output in outputs if output.outputs[0].text.strip())
    finally:
        engine.shutdown()


@pytest.fixture(scope="module")
def scaling(model_dir: Path) -> tuple[tuple[float, int], tuple[float, int]]:
    """One measurement per width, run one after the other so neither is starved."""
    one = _serve(model_dir, replicas=1)
    two = _serve(model_dir, replicas=2)
    return one, two


@needs_gpus(2)
def test_a_second_replica_adds_its_own_throughput(scaling):
    """Two replicas at the same per-replica width must serve close to twice the requests.

    Judged on requests per second rather than wall clock: the two widths do not do
    the same amount of work, so only a rate is comparable -- and a rate already
    rules out "half the work done twice as fast".
    """
    (one_s, one_n), (two_s, two_n) = scaling
    one_rate, two_rate = one_n / one_s, two_n / two_s
    ratio = two_rate / one_rate
    print(f"\ndp weak scaling: {one_rate:.1f} -> {two_rate:.1f} req/s ({ratio:.2f}x)")
    assert ratio > _MIN_SCALING, (
        f"two replicas served {two_rate:.1f} req/s against {one_rate:.1f} on one "
        f"({ratio:.2f}x, wanted >{_MIN_SCALING}x): the replicas are not running "
        f"concurrently"
    )


@needs_gpus(2)
def test_both_widths_answered_every_prompt(scaling):
    """Guards the ratio: a width that dropped prompts would look fast by doing less."""
    for width, (_elapsed, answered) in zip((1, 2), scaling, strict=True):
        expected = len(_PROMPTS) * width
        assert answered == expected, f"dp={width} answered {answered}/{expected}"
