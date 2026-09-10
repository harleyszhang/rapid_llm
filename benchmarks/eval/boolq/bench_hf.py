"""BoolQ accuracy on transformers — the HF baseline arm.

Same prompts and scoring as the rapid_llm arm (``common.py``); greedy
decoding, left-padded static batches. The reference numbers in
``docs/eval_models.md`` cite this arm.

Usage:
    python -m benchmarks.eval.boolq.bench_hf \
        --model-dir Qwen2.5-0.5B --num-questions 200 [--chat-template]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.eval._common import (
    append_result_log,
    hf_generate,
    load_hf,
    resolve_model_dir,
    timed,
)
from benchmarks.eval.boolq.common import STOP, build_prompts, load_boolq, score
from tests.evals.runner import truncate_at_stop


def main() -> int:
    ap = argparse.ArgumentParser(description="BoolQ accuracy via HuggingFace transformers")
    ap.add_argument("--model-dir", required=True, help="path, or a name under the model zoo")
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chat-template", action="store_true")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    train, validation = load_boolq()
    prompts, labels = build_prompts(
        train, validation, num_questions=args.num_questions, num_shots=args.num_shots
    )

    model, tokenizer = load_hf(model_dir)
    (texts, generated_tokens), latency_s = timed(
        hf_generate,
        model,
        tokenizer,
        prompts,
        max_new_tokens=6,
        batch_size=args.batch_size,
        chat_template=args.chat_template,
    )
    completions = [truncate_at_stop(t, STOP) for t in texts]
    accuracy, invalid_rate = score(completions, labels)

    print(f"\nBoolQ / HF — {model_dir}")
    print(f"accuracy      {accuracy:.4f}")
    print(f"invalid_rate  {invalid_rate:.4f}")
    print(f"questions     {len(labels)} ({args.num_shots}-shot, batch {args.batch_size})")
    print(
        f"latency       {latency_s:.1f} s "
        f"({len(labels) / latency_s:.2f} q/s, {generated_tokens / latency_s:.1f} tok/s)"
    )

    log = append_result_log(
        "boolq",
        "hf",
        {
            "engine": "hf",
            "model_dir": str(model_dir),
            "num_questions": len(labels),
            "num_shots": args.num_shots,
            "batch_size": args.batch_size,
            "chat_template": str(args.chat_template),
            "accuracy": accuracy,
            "invalid_rate": invalid_rate,
            "latency_s": latency_s,
            "generated_tokens": generated_tokens,
            "questions_per_second": len(labels) / latency_s,
            "tokens_per_second": generated_tokens / latency_s,
        },
    )
    print(f"-> {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
