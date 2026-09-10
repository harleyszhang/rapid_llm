"""HellaSwag dataset plumbing shared by both arms.

Prompt format follows sglang's ``benchmark/hellaswag/bench_sglang.py``: 20-shot
of ``{activity_label}: {ctx} {correct ending}``, then each question's
``{activity_label}: {ctx} `` scored against all four endings by continuation
log-likelihood. One deliberate deviation: the scored questions start *after*
the shots (sglang scores from row 0, so its first 20 questions are also the
few-shot examples — the answers sit verbatim in the prompt).

Scoring reports both rulings the literature uses:
``acc`` — total log-prob argmax (lm-eval-harness), and ``acc_len`` —
length-normalised argmax (sglang's ``sgl.select`` default).

Usage:
    rows = load_hellaswag()
    items = build_items(rows, num_questions=200, num_shots=20)
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.eval._common import fetch_jsonl
from tests.evals.dataset import read_jsonl

#: The validation split sglang's bench uses (10042 rows). HellaSwag's train
#: split is 4x larger and would dominate the download for zero benefit: only
#: 20 rows of it would ever be read.
HELLASWAG_VAL_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
)


def load_hellaswag() -> list[dict]:
    """Return the HellaSwag validation rows, downloading once.

    Raises:
        RuntimeError: The file is neither cached nor reachable.
    """
    path = fetch_jsonl(HELLASWAG_VAL_URL, "hellaswag", "hellaswag_val.jsonl")
    return read_jsonl(path)


def _shot(row: dict) -> str:
    return f"{row['activity_label']}: {row['ctx']} {row['endings'][row['label']]}\n\n"


def build_items(
    rows: Sequence[dict],
    *,
    num_questions: int,
    num_shots: int,
) -> tuple[str, list[dict]]:
    """Return ``(shots, items)`` where each item scores one question.

    ``item["prefix"]`` ends with a trailing space (sglang's format), the four
    ``item["endings"]`` are scored as continuations of it, and
    ``item["label"]`` is the index of the gold ending. Questions are taken
    from row ``num_shots`` on, so no scored question appears in the shots.
    """
    if num_shots + num_questions > len(rows):
        raise ValueError(
            f"need {num_shots} shots + {num_questions} questions "
            f"but the validation split has {len(rows)} rows"
        )
    shots = "".join(_shot(row) for row in rows[:num_shots])
    items = []
    for row in rows[num_shots : num_shots + num_questions]:
        items.append(
            {
                "prefix": f"{shots}{row['activity_label']}: {row['ctx']} ",
                "endings": list(row["endings"]),
                "label": int(row["label"]),
            }
        )
    return shots, items


def score(
    logprobs: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[int],
) -> tuple[float, float]:
    """Return ``(acc, acc_len)`` from per-ending per-token log-prob lists.

    ``logprobs[i][e]`` is question ``i`` ending ``e``'s token log-probs (empty
    endings score ``-inf`` on both rulings). ``acc`` argmaxes the sum — the
    lm-eval-harness ruling; ``acc_len`` argmaxes the mean — what sglang's
    ``token_length_normalized`` select does.
    """
    if len(logprobs) != len(labels):
        raise ValueError(f"{len(logprobs)} questions scored for {len(labels)} labels")
    if not labels:
        return 0.0, 0.0

    import math

    correct = correct_len = 0
    for endings, label in zip(logprobs, labels, strict=True):
        totals = [sum(e) if e else -math.inf for e in endings]
        means = [sum(e) / len(e) if e else -math.inf for e in endings]
        correct += totals.index(max(totals)) == label
        correct_len += means.index(max(means)) == label
    return correct / len(labels), correct_len / len(labels)
