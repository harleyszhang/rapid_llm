"""GPU accuracy: tier off/on output parity and store/load activity.

ROADMAP v0.12.0 acceptance: the CPU tier must produce the same greedy tokens
as the GPU-only path, and actual store/load transfers must happen.  The test
drives a full engine (``ContinuousBatchingEngine`` via ``build_engine``), uses
``build_cpu_tier`` from the executor package, and compares token sequences.

The single-engine approach (not a ref/eq build pair) means both arms share the
same weight loading and graph capture — the measured delta is only the tier.
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.benchmark import sampling_params as _sampling_params
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.llm_engine import LLMEngine
from rapid_llm.engine.scheduler import SchedulerConfig
from rapid_llm.executor.executor import UniProcExecutor
from rapid_llm.executor.kv_offload import build_cpu_tier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CKPT = "my_weight/Qwen3-0.6B"
BLOCKS = 4096
CPU_BLOCKS = 2048
MAX_SEQ_LEN = 1024
MAX_NUM_SEQS = 8
GEN_LEN = 32
PROMPTS = [
    "What is the capital of France?",
    "Name three primary colors.",
    "Explain gravity in one sentence.",
    "What is 2 plus 2?",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?",
    "What year did World War II end?",
]


def _build_engine(tier: bool):
    """Construct an engine with or without CPU tier offloading."""
    config = SchedulerConfig(
        max_seq_len=MAX_SEQ_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_cache=True,
    )
    llm = LLMEngine(
        CKPT,
        max_seq_len=MAX_SEQ_LEN,
        max_gpu_num_blocks=BLOCKS,
        use_cuda_graph=True,
    )
    executor = UniProcExecutor(llm, config.max_num_seqs, config.max_seq_len)
    manager = build_cpu_tier(llm.model_runner.kv_cache_manager, CPU_BLOCKS) if tier else None
    return ContinuousBatchingEngine(llm, config, executor, offloading=manager), manager


class TestAccuracyParity:
    """Tier on vs off: greedy tokens must match, tier must show activity."""

    def test_tier_on_tokens_match_tier_off(self):
        off_engine, _ = _build_engine(tier=False)
        on_engine, on_manager = _build_engine(tier=True)

        try:
            off_texts = off_engine.generate(PROMPTS, _sampling_params(GEN_LEN))
            on_texts = on_engine.generate(PROMPTS, _sampling_params(GEN_LEN))

            assert len(off_texts) == len(on_texts) == len(PROMPTS)
            for i, (off_t, on_t) in enumerate(zip(off_texts, on_texts, strict=True)):
                assert off_t == on_t, (
                    f"request {i} differs:\n  tier off: {off_t!r}\n  tier on:  {on_t!r}"
                )

            stats = on_manager.get_stats()
            assert stats.stores > 0, f"expected stores > 0, got {stats.stores}"
        finally:
            del off_engine, on_engine
            torch.cuda.empty_cache()

    def test_tier_on_shows_store_and_load_activity(self):
        """Constrained GPU pool + shared prefixes -> eviction -> loads."""
        pool = 256  # tokens = 16 scheduler blocks; small enough to force eviction
        cpu_blocks = 1024
        # Each prompt: "You are assistant {g}. " + filler * 16 + unique suffix
        _FILLER = "Follow every instruction carefully and answer as precisely as you can. "
        prefixes = [f"You are assistant number {g}. " + _FILLER * 16 for g in range(4)]
        prompts = [
            f"{prefixes[g]}Question {i}: what is {i} plus {i + 1}?"
            for i, g in enumerate([g for g in range(4) for _ in range(2)])
        ]
        config = SchedulerConfig(
            max_seq_len=MAX_SEQ_LEN,
            max_num_seqs=4,
            enable_prefix_cache=True,
        )
        llm = LLMEngine(
            CKPT,
            max_seq_len=MAX_SEQ_LEN,
            max_gpu_num_blocks=pool,
            use_cuda_graph=True,
        )
        executor = UniProcExecutor(llm, config.max_num_seqs, config.max_seq_len)
        manager = build_cpu_tier(llm.model_runner.kv_cache_manager, cpu_blocks)
        engine = ContinuousBatchingEngine(llm, config, executor, offloading=manager)
        try:
            engine.generate(prompts, _sampling_params(8))  # wave 1: fill + evict
            engine.generate(prompts, _sampling_params(8))  # wave 2: HIT -> loads
            stats = manager.get_stats()
            assert stats.stores > 0, f"stores: {stats.stores}"
            assert stats.loads > 0, f"loads: {stats.loads}"
        finally:
            del engine
            torch.cuda.empty_cache()

    def test_tier_on_stats_have_expected_fields(self):
        """Smoke check: the stats dict has the right shape."""
        engine, manager = _build_engine(tier=True)
        try:
            engine.generate(PROMPTS[:4], _sampling_params(8))
            stats = manager.get_stats()
            for field in ("stores", "loads", "store_bytes", "load_bytes", "misses", "hit_rate"):
                assert hasattr(stats, field), f"missing field: {field}"
        finally:
            del engine
            torch.cuda.empty_cache()
