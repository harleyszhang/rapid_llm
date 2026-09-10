"""HellaSwag accuracy on the rapid_llm engine — the optimised-mode arm.

Scores each of a question's four endings by continuation log-likelihood via
the engine's ``prompt_logprobs``: the (prefix, ending) token ids are fed
straight to ``generate_text``, so the token boundary is byte-identical to the
HF arm's. CUDA graphs stay on (the engine default).

Usage:
    python -m benchmarks.eval.hellaswag.bench_rapid_vllm \
        --model-dir Qwen2.5-0.5B --num-questions 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.eval._common import (
    append_result_log,
    build_rapid_llm,
    continuation_ids,
    rapid_continuation_logprobs,
    resolve_model_dir,
    timed,
)
from benchmarks.eval.hellaswag.common import build_items, load_hellaswag, score


def main() -> int:
    ap = argparse.ArgumentParser(description="HellaSwag accuracy via the rapid_llm engine")
    ap.add_argument("--model-dir", required=True, help="path, or a name under the model zoo")
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=20)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="sequences per forward; prompt log-probs materialise the full "
        "prefill logits grid, so this is memory-bound, not speed-bound",
    )
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument(
        "--eager",
        action="store_true",
        help="disable CUDA graphs (debugging; slower and off the default config)",
    )
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    rows = load_hellaswag()
    _shots, items = build_items(rows, num_questions=args.num_questions, num_shots=args.num_shots)
    labels = [item["label"] for item in items]

    llm = build_rapid_llm(
        model_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        use_cuda_graph=None if not args.eager else False,
    )
    # Flatten (question, ending) -> one scoring request each; the engine sees
    # an unstructured batch and the reshape back to fours happens at scoring.
    pairs = [
        continuation_ids(llm.tokenizer, item["prefix"], ending)
        for item in items
        for ending in item["endings"]
    ]
    flat, latency_s = timed(rapid_continuation_logprobs, llm, pairs, batch_size=args.batch_size)
    per_question = [flat[i : i + 4] for i in range(0, len(flat), 4)]
    accuracy, accuracy_len = score(per_question, labels)

    print(f"\nHellaSwag / rapid_llm — {model_dir}")
    print(f"acc           {accuracy:.4f}  (total log-prob, lm-eval ruling)")
    print(f"acc_len       {accuracy_len:.4f}  (length-normalised, sglang ruling)")
    print(f"questions     {len(labels)} ({args.num_shots}-shot, batch {args.batch_size})")
    print(f"latency       {latency_s:.1f} s ({len(labels) / latency_s:.2f} q/s)")

    log = append_result_log(
        "hellaswag",
        "rapid",
        {
            "engine": "rapid_llm",
            "model_dir": str(model_dir),
            "num_questions": len(labels),
            "num_shots": args.num_shots,
            "batch_size": args.batch_size,
            "cuda_graph": "off" if args.eager else "on (default)",
            "accuracy": accuracy,
            "accuracy_len": accuracy_len,
            "latency_s": latency_s,
            "questions_per_second": len(labels) / latency_s,
        },
    )
    print(f"-> {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
