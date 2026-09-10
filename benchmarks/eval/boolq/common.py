"""BoolQ dataset plumbing shared by both arms — prompts, scoring, data access.

Prompt format follows sglang's ``benchmark/boolq/bench_sglang.py``: 5-shot,
``Question: {question}{passage}\nAnswer:`` with the shot's answer appended
directly. One deliberate deviation: completions are stripped before the
``True``/``False`` comparison (sglang compares raw text, so a leading space —
what tokenizers emit after ``Answer:`` — would count as wrong).

Usage:
    prompts, labels = build_prompts(train, validation, num_questions=200)
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

from benchmarks.eval._common import fetch_jsonl
from tests.evals.dataset import read_jsonl

#: SuperGLUE's official distribution (the GCS ``storage.googleapis.com/boolq``
#: source sglang's README links has been returning 403). The zip holds
#: ``BoolQ/train.jsonl`` (9427 rows) and ``BoolQ/val.jsonl`` (3270 rows) —
#: SuperGLUE names its dev split ``val``.
BOOLQ_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"

#: Completions are cut at the first newline, like sglang's ``stop=["\n"]``.
STOP = ("\n",)

#: What a completion must normalise to; anything else counts as invalid.
ANSWERS = ("True", "False")


def load_boolq() -> tuple[list[dict], list[dict]]:
    """Return ``(train, validation)`` BoolQ splits, downloading once.

    Raises:
        RuntimeError: The SuperGLUE zip is neither cached nor reachable.
    """
    cache = Path.home() / ".cache" / "rapid_llm" / "evals" / "boolq"
    train, validation = cache / "train.jsonl", cache / "validation.jsonl"
    if not (train.is_file() and validation.is_file()):
        archive = fetch_jsonl(BOOLQ_ZIP_URL, "boolq", "BoolQ.zip")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache)
        # The zip's dev split is ``BoolQ/val.jsonl``; expose it under the
        # sglang-facing name so the cache layout matches the other datasets.
        train.write_bytes((cache / "BoolQ" / "train.jsonl").read_bytes())
        validation.write_bytes((cache / "BoolQ" / "val.jsonl").read_bytes())
    return read_jsonl(train), read_jsonl(validation)


def _answer(row: dict) -> str:
    """The row's ``"True"``/``"False"`` answer under either source's name.

    SuperGLUE's zip calls the field ``label``; the HF ``google/boolq`` parquet
    that sglang's bench defaults to calls it ``answer``. Accepting both keeps
    a pre-seeded cache in either layout working.
    """
    value = row.get("answer", row.get("label"))
    if not isinstance(value, bool):
        raise ValueError(f"BoolQ row has no boolean answer: {sorted(row)}")
    return str(value)


def _format_shot(row: dict) -> str:
    return f"Question: {row['question']}{row['passage']}\nAnswer:{_answer(row)}\n\n"


def build_prompts(
    train: Sequence[dict],
    test: Sequence[dict],
    *,
    num_questions: int,
    num_shots: int,
) -> tuple[list[str], list[str]]:
    """Return 5-shot prompts and their ``"True"``/``"False"`` labels.

    The shots are the first ``num_shots`` train records and the questions the
    first ``num_questions`` validation records — fixed prefixes, mirroring
    ``tests.evals.gsm8k.build_prompts``, so runs are comparable across engines
    and repeatable on one engine.
    """
    if num_shots > len(train):
        raise ValueError(f"need {num_shots} shots but train split has {len(train)}")

    shots = "".join(_format_shot(row) for row in train[:num_shots])
    prompts, labels = [], []
    for row in test[:num_questions]:
        prompts.append(f"{shots}Question: {row['question']}{row['passage']}\nAnswer:")
        labels.append(_answer(row))
    return prompts, labels


def score(completions: Sequence[str], labels: Sequence[str]) -> tuple[float, float]:
    """Return ``(accuracy, invalid_rate)``; a stripped exact match on True/False.

    An unparsable completion (neither answer, or empty) is wrong *and* counted
    invalid — the same two-number contract GSM8K uses, separating "the model
    picked the wrong side" from "the harness never saw an answer".
    """
    if len(completions) != len(labels):
        raise ValueError(f"{len(completions)} completions for {len(labels)} labels")
    if not labels:
        return 0.0, 0.0

    correct = invalid = 0
    for completion, label in zip(completions, labels, strict=True):
        prediction = completion.strip()
        if prediction not in ANSWERS:
            invalid += 1
        correct += prediction == label
    return correct / len(labels), invalid / len(labels)
