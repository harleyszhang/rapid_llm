"""ShareGPT workloads from a local json (sglang ``datasets.sharegpt`` mirror).

No hub download — this repository benchmarks offline by design. ``load`` filters
conversations to at least two turns, keeps the first two (user prompt,
assistant completion), shuffles, tokenizes, and prunes rows under 2 tokens or
over ``context_len``. The output length is the completion's real token count
unless ``fixed_output_len`` pins it — the throughput scenarios' usual choice.
``prompt_suffix`` swaps in a custom assistant suffix; ``apply_chat_template``
wraps the prompt with the tokenizer's chat template (bos stripped).

Usage:
    rows = ShareGPTDataset(dataset_path=..., num_requests=64, ...).load(tokenizer)
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
)


@dataclass
class ShareGPTDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: int | None
    context_len: int | None
    prompt_suffix: str
    apply_chat_template: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "ShareGPTDataset":
        assert not getattr(args, "tokenize_prompt", False)
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> list[DatasetRow]:
        return sample_sharegpt_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
            prompt_suffix=self.prompt_suffix,
            apply_chat_template=self.apply_chat_template,
        )


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int | None = None,
    context_len: int | None = None,
    prompt_suffix: str | None = "",
    apply_chat_template=False,
) -> list[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    if not dataset_path:
        raise FileNotFoundError(
            "--dataset-name sharegpt requires --dataset-path pointing to a local "
            "ShareGPT-style json (no network download in offline benchmarks)"
        )

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

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
    filtered_dataset: list[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                _remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset


def _remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text
