"""Dataset primitives shared by every workload source (sglang ``datasets.common`` mirror).

A dataset answers one question — *which prompts, and how long does each answer
run?* — as a list of :class:`DatasetRow`. The row is the currency every runner
accepts, the way vLLM's ``SampleRequest`` is for its benchmark suite.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

ASSISTANT_SUFFIX = "Assistant:"


@dataclass
class DatasetRow:
    """One benchmark request: a prompt plus the output length to run it to.

    ``prompt_len`` is the token count the row was built for (the runner sizes
    KV budgets from it); when the prompt travels as text, ``text_prompt_len``
    is the re-tokenized length the server will actually see.
    """

    prompt: Any
    prompt_len: int
    output_len: int
    text_prompt_len: int | None = None

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len


@dataclass
class BaseDataset(ABC):
    @classmethod
    @abstractmethod
    def from_args(cls, args: Namespace) -> "BaseDataset": ...

    @abstractmethod
    def load(
        self,
        tokenizer: Any,
        model_id: str | None = None,
    ) -> list[DatasetRow]: ...


def compute_random_lens(full_len: int, range_ratio: float, num: int) -> list[int]:
    """``num`` lengths in ``[full_len * range_ratio, full_len]``.

    ``range_ratio=1.0`` pins every row to exactly ``full_len`` — what a
    length-controlled scenario wants. ``full_len=0`` is valid (no output).
    """
    if full_len <= 0:
        return [0] * num
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


@lru_cache(maxsize=1)
def get_available_tokens(tokenizer) -> list[int]:
    """Sampling pool: every vocab id except the tokenizer's special tokens.

    Special ids decode to markup like ``<|im_start|>`` that would both pollute
    the prompt and skew the re-encode length; vLLM's sampler excludes them the
    same way. Canonical order: vocab dict iteration order varies across
    tokenizer versions, which would break --seed reproducibility.
    """
    special = set(tokenizer.all_special_ids)
    return sorted(
        token_id
        for token_id in tokenizer.get_vocab().values()
        if isinstance(token_id, int) and token_id not in special
    )


def gen_prompt_decode_to_target_len(
    tokenizer,
    token_sequence: list[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
) -> tuple[str, list[int], int]:
    """Decode, re-encode, and pad/truncate until the length matches the target.

    N consecutive tokens decode to a string that re-tokenizes to != N tokens
    (``[6880, 6881] -> [' calls', 'here'] -> [1650, 939, 486]`` for GPT2), so a
    prompt built as ``decode(ids)`` drifts unless corrected. This is vLLM's
    ``gen_prompt_decode_to_target_len`` mirror: iterate decode/re-encode with
    pad-or-truncate, at most ``max_retry`` times, and report the residual
    mismatch when the budget runs out (a warning, not an exception — the drift
    is bounded and every engine sees the same text anyway).
    """
    remain = max_retry
    token_mismatch = 0
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if remain <= 0:
            token_mismatch = len(token_sequence) - target_token_len
            break
        if len(token_sequence) == target_token_len:
            break
        if len(token_sequence) < target_token_len:
            token_sequence.extend(
                np.random.randint(0, tokenizer.vocab_size,
                                  size=target_token_len - len(token_sequence)).tolist()
            )
        else:
            token_sequence = token_sequence[:target_token_len]
        remain -= 1
    return prompt, token_sequence, token_mismatch
