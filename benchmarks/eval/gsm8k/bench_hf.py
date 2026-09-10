"""GSM8K accuracy on transformers — the HF baseline arm.

Same prompts (``build_prompts``), stop markers and scoring (``score``) as the
rapid_llm arm; only the generation stack differs, so the two accuracies are
directly comparable. The reference numbers in ``docs/eval_models.md`` cite
this arm.

Usage:
    python -m benchmarks.eval.gsm8k.bench_hf \
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
from tests.evals.dataset import load_gsm8k
from tests.evals.gsm8k import STOP, build_prompts, score
from tests.evals.runner import truncate_at_stop


def main() -> int:
    ap = argparse.ArgumentParser(description="GSM8K accuracy via HuggingFace transformers")
    ap.add_argument("--model-dir", required=True, help="path, or a name under the model zoo")
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chat-template", action="store_true")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    train, test = load_gsm8k()
    prompts, labels = build_prompts(
        train, test, num_questions=args.num_questions, num_shots=args.num_shots
    )

    model, tokenizer = load_hf(model_dir)
    (texts, generated_tokens), latency_s = timed(
        hf_generate,
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_gen_len,
        batch_size=args.batch_size,
        chat_template=args.chat_template,
    )

    completions = [truncate_at_stop(t, STOP) for t in texts]
    accuracy, invalid_rate = score(completions, labels)

    print(f"\nGSM8K / HF — {model_dir}")
    print(f"accuracy      {accuracy:.4f}")
    print(f"invalid_rate  {invalid_rate:.4f}")
    print(
        f"questions     {len(labels)} ({args.num_shots}-shot, batch {args.batch_size}, "
        f"max_gen_len {args.max_gen_len})"
    )
    print(
        f"latency       {latency_s:.1f} s "
        f"({len(labels) / latency_s:.2f} q/s, {generated_tokens / latency_s:.1f} tok/s)"
    )

    log = append_result_log(
        "gsm8k",
        "hf",
        {
            "engine": "hf",
            "model_dir": str(model_dir),
            "num_questions": len(labels),
            "num_shots": args.num_shots,
            "max_gen_len": args.max_gen_len,
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
