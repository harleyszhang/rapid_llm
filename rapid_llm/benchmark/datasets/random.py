"""Synthetic length-controlled workloads (sglang ``datasets.random`` mirror).

Two sources feed the same length machinery:

- ``random-ids``: token ids sampled from the vocabulary — deterministic under
  ``--seed``, no dataset dependency, the suite's default.
- ``random``: token ids drawn from a local ShareGPT file, repeated/truncated to
  length — real-token statistics, still length-controlled.

This is also where the classic token-length drift trap is handled: a prompt
built as ``decode(ids)`` re-tokenizes to a different length (special tokens,
byte fallbacks), so when the prompt travels as text the target length is
reduced by the tokenizer's special-token count, the same correction vLLM's
RandomDataset applies via decode→re-encode→truncate.
"""

import json
import random
from argparse import Namespace
from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase

from .common import (
    ASSISTANT_SUFFIX,
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt_decode_to_target_len,
    get_available_tokens,
)


@dataclass
class RandomDataset(BaseDataset):
    input_len: int
    output_len: int
    num_requests: int
    range_ratio: float
    dataset_path: str
    return_text: bool
    random_sample: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "RandomDataset":
        return cls(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_requests=args.num_prompts,
            range_ratio=args.random_range_ratio,
            dataset_path=args.dataset_path,
            return_text=not getattr(args, "tokenize_prompt", False),
            random_sample=(args.dataset_name == "random"),
        )

    def load(self, tokenizer: PreTrainedTokenizerBase, model_id=None) -> list[DatasetRow]:
        return sample_random_requests(
            input_len=self.input_len,
            output_len=self.output_len,
            num_prompts=self.num_requests,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            random_sample=self.random_sample,
            return_text=self.return_text,
        )


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
) -> list[DatasetRow]:
    """Build ``num_prompts`` rows with length-controlled prompts and outputs.

    The offline mirror of sglang's sampler, minus the network download: the
    ShareGPT-sampled mode requires a local ``dataset_path`` (the suite runs
    offline on purpose), and fails loudly rather than silently reaching the hub.
    """
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )

    if return_text:
        # Need to truncate input_len as server encode will add special token.
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        for i in range(num_prompts):
            input_lens[i] = max(1, input_lens[i] - num_special_tokens)

    mismatches = 0
    if random_sample:
        # Sample token ids from a local ShareGPT file and repeat/truncate them
        # to satisfy the input_lens.
        try:
            with open(dataset_path) as f:
                dataset = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(
                f"--dataset-name random samples from a ShareGPT-style json; "
                f"pass --dataset-path with a local file (got {dataset_path!r}: {exc})"
            ) from exc
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: list[DatasetRow] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            if return_text:
                # Length control applies to what the server will re-encode, so
                # the repeated/truncated ids go through the decode/re-encode
                # fixup instead of being trusted as-is.
                input_content, input_ids, mismatch = gen_prompt_decode_to_target_len(
                    tokenizer, input_ids, input_lens[i]
                )
                mismatches += mismatch != 0
            else:
                input_content = input_ids
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=len(input_ids),
                    output_len=output_lens[i],
                )
            )
    else:
        # Deterministic vocabulary sequence (vLLM's offset+index+arange walk,
        # drawn over the non-special ids; special ids re-encode badly).
        allowed = get_available_tokens(tokenizer)
        offsets = np.random.randint(0, len(allowed), size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            ids = [allowed[(int(offsets[i]) + i + j) % len(allowed)] for j in range(input_lens[i])]
            if return_text:
                prompt, ids, mismatch = gen_prompt_decode_to_target_len(
                    tokenizer, ids, input_lens[i]
                )
                mismatches += mismatch != 0
            else:
                prompt = ids
            input_requests.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=len(ids),
                    output_len=output_lens[i],
                )
            )

    if mismatches:
        print(
            f"warning: {mismatches}/{num_prompts} prompts kept a residual length "
            "drift after decode/re-encode (bounded by the retry budget; every "
            "engine tokenizes the same text, so the comparison stays fair)"
        )
    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


# ASSISTANT_SUFFIX is re-exported for the custom/sharegpt samplers.
__all__ = ["ASSISTANT_SUFFIX", "RandomDataset", "sample_random_requests"]
