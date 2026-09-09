"""Dataset registry (sglang ``datasets.__init__`` mirror).

``get_dataset`` is the factory every runner goes through: seed first (both RNGs,
so ``--seed`` reproducibility covers numpy and stdlib sampling), then build the
dataset from CLI args and load its rows. ``add_dataset_args`` attaches the
shared CLI surface so each runner re-declares none of it.
"""

# Aliased: ``from .random import ...`` below makes Python
# setattr the submodule over this module's ``random`` global (classic shadow).
import random as _random
from argparse import ArgumentParser, Namespace

import numpy as np

from .common import BaseDataset, DatasetRow
from .custom import CustomDataset
from .random import RandomDataset
from .sharegpt import ShareGPTDataset

DATASET_MAPPING: dict[str, type[BaseDataset]] = {
    "sharegpt": ShareGPTDataset,
    "custom": CustomDataset,
    # "random" (ShareGPT-sampled, local file) vs "random-ids" (vocabulary
    # integers) share one class, keyed on the same flag sglang uses.
    "random": RandomDataset,
    "random-ids": RandomDataset,
}


def add_dataset_args(parser: ArgumentParser) -> None:
    """The dataset CLI surface shared by the offline and serving runners."""
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random-ids",
        choices=sorted(DATASET_MAPPING),
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Path to a local dataset file (ShareGPT-style json for "
        "sharegpt/random, jsonl for custom). No network access.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Target input tokens per request (random datasets only).",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Target output tokens per request (random datasets only).",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Length spread: 1.0 pins every row to the target length, "
        "0.0 spreads uniformly in (0, target].",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length override for sharegpt/custom datasets.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="Prune rows longer than this (prompt + output).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for row sampling.")
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="",
        help="Suffix appended to sharegpt/custom prompts (after stripping a "
        "trailing 'Assistant:').",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Wrap sharegpt/custom prompts in the tokenizer's chat template.",
    )
    parser.add_argument(
        "--tokenize-prompt",
        action="store_true",
        help="Keep random-dataset prompts as token-id lists instead of text.",
    )


def get_dataset(
    args: Namespace,
    tokenizer,
    model_id: str | None = None,
) -> list[DatasetRow]:
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random-ids"

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    _random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_cls = DATASET_MAPPING[dataset_name]
    dataset = dataset_cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "DATASET_MAPPING",
    "BaseDataset",
    "DatasetRow",
    "add_dataset_args",
    "get_dataset",
]
