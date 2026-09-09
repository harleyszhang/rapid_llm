"""Workload definitions: what to generate, and how to sample.

The prompt sets and sampling presets every scenario script builds on, plus the
helpers (``expand_prompts``, ``sampling_params``) that keep workload shapes
reproducible across scripts.

Usage:
    from rapid_llm.benchmark import PROMPTS, GREEDY_PARAMS, sampling_params
"""

from __future__ import annotations

PROMPTS = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
]

#: Non-greedy defaults, matching rapid_llm's sampling branch.
SAMPLE_KW = {"temperature": 0.7, "top_p": 0.8}

#: Greedy, with repetition penalty and early exit off. A benchmark's token count
#: must not depend on a heuristic that fires for some rows and not others — that
#: would give the two columns different denominators.
GREEDY_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "stop_on_repeat": False,
}


def expand_prompts(prompts: list[str], batch: int) -> list[str]:
    """Cycle ``prompts`` up to ``batch`` entries."""
    return (prompts * ((batch // len(prompts)) + 1))[:batch]


def sampling_params(max_gen_len: int, greedy: bool = True):
    """The benchmark's ``SamplingParams``: :data:`GREEDY_PARAMS` or :data:`SAMPLE_KW`."""
    from rapid_llm import SamplingParams

    return SamplingParams(max_gen_len=max_gen_len, **(GREEDY_PARAMS if greedy else SAMPLE_KW))
