"""Chunked prefill: what a step does when a prompt does not fit one chunk.

A mixed batch of short and long prompts runs with a small chunk budget;
only completed chunks may sample, and both requests must finish with
their own text.

Usage:
    pytest tests/engine/test_chunked_prefill.py
"""

from __future__ import annotations

import gc

import pytest
import torch

from rapid_llm.engine.batch_planner import SchedulerConfig
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Scheduler

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

_MAX_SEQ_LEN = 1024
_KV_BLOCKS = 8192
#: Small enough that a few hundred words of prompt take several chunks, which is
#: what puts a partial and a completed chunk in the same grid.
_CHUNK = 64

GREEDY = SamplingParams(temperature=0.0, max_gen_len=8, repetition_penalty=1.0)

SHORT = "The capital of France is"
#: Comfortably more than one chunk, and deliberately repetitive: the point is its
#: length, and repeated tokens keep it from wandering into the short prompt's
#: vocabulary, which would blunt the leakage assertion below.
LONG = "banana " * 200


@pytest.fixture
def engine(model_dir):
    """Continuous-batching engine with chunking turned down to :data:`_CHUNK`."""
    built = Scheduler.from_pretrained(
        str(model_dir),
        max_seq_len=_MAX_SEQ_LEN,
        max_num_seqs=4,
        max_gpu_num_blocks=_KV_BLOCKS,
        use_cuda_graph=False,
    )
    config = SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_chunk_size=_CHUNK)
    built.config = config
    built.planner.config = config
    yield built
    del built
    gc.collect()
    torch.cuda.empty_cache()


def test_mixed_grid_samples_only_the_completed_chunk(engine) -> None:
    """One step, two admissions: only the prompt that finished emits a token.

    Both requests are admitted into the same prefill grid, so their logits come
    back as one ``[2, vocab]`` tensor — but only row 0 belongs to a finished
    prompt. Pairing "the requests that completed" with "the leading rows of the
    batch" is the same list only when *every* chunk completes; here it is not,
    and the mismatch used to surface as a broadcast error while scattering the
    sampled tokens into the generated grid.
    """
    short = engine.add_request(SHORT, GREEDY)
    long = engine.add_request(LONG, GREEDY)

    advanced = engine.step()

    assert [r.request_id for r in advanced] == [short.request_id]
    assert short.delta != ""
    assert long.output_token_ids == []
    # The long prompt holds a slot and exactly one chunk of KV.
    assert long.num_computed_tokens == _CHUNK
    assert not long.prefill_done


def test_both_requests_finish_and_keep_their_own_text(engine) -> None:
    """A chunked prompt resumes to completion beside a request already decoding.

    The short request is decoding while the long one is still being prefilled,
    so every remaining chunk runs in a step that also runs a decode pass. Text
    crossing between the two would mean the resumed chunks wrote their K/V into
    somebody else's slot.
    """
    short = engine.add_request(SHORT, GREEDY)
    long = engine.add_request(LONG, GREEDY)

    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        assert steps < 200, "chunked prefill failed to make progress"

    assert short.text and long.text
    assert short.finish_reason and long.finish_reason
    # A prompt of repeated "banana" continues itself; the short prompt's answer
    # has no reason to contain it unless the two swapped cache rows.
    assert "banana" not in short.text
