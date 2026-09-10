"""BoolQ accuracy on the rapid_llm engine — the optimised-mode arm.

Same prompts and scoring as the HF arm (``common.py``); greedy decoding, CUDA
graphs on (the engine default). The engine has no stop strings, so a few extra
tokens are decoded and cut afterwards — at six tokens per question the waste
is bounded and identical on both arms.

Usage:
    python -m benchmarks.eval.boolq.bench_rapid_vllm \
        --model-dir Qwen2.5-0.5B --num-questions 200 [--chat-template]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.eval._common import (
    append_result_log,
    build_rapid_llm,
    resolve_model_dir,
    timed,
)
from benchmarks.eval.boolq.common import STOP, build_prompts, load_boolq, score
from tests.evals.runner import as_user_turn, generate_completions


def main() -> int:
    ap = argparse.ArgumentParser(description="BoolQ accuracy via the rapid_llm engine")
    ap.add_argument("--model-dir", required=True, help="path, or a name under the model zoo")
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument(
        "--eager",
        action="store_true",
        help="disable CUDA graphs (debugging; slower and off the default config)",
    )
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    train, validation = load_boolq()
    prompts, labels = build_prompts(
        train, validation, num_questions=args.num_questions, num_shots=args.num_shots
    )

    llm = build_rapid_llm(
        model_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        use_cuda_graph=None if not args.eager else False,
    )
    if args.chat_template:
        prompts = [as_user_turn(llm, p) for p in prompts]
    # Six tokens cover " True\n" plus slack for a model that opens with a
    # newline; sglang's five plus the newline stop is the same budget.
    run, latency_s = timed(
        generate_completions,
        llm,
        prompts,
        max_gen_len=6,
        batch_size=args.batch_size,
        stop=STOP,
    )

    accuracy, invalid_rate = score(run.completions, labels)
    print(f"\nBoolQ / rapid_llm — {model_dir}")
    print(f"accuracy      {accuracy:.4f}")
    print(f"invalid_rate  {invalid_rate:.4f}")
    print(f"questions     {len(labels)} ({args.num_shots}-shot, batch {args.batch_size})")
    print(
        f"latency       {latency_s:.1f} s "
        f"({len(labels) / latency_s:.2f} q/s, {run.generated_tokens / latency_s:.1f} tok/s)"
    )

    log = append_result_log(
        "boolq",
        "rapid",
        {
            "engine": "rapid_llm",
            "model_dir": str(model_dir),
            "num_questions": len(labels),
            "num_shots": args.num_shots,
            "batch_size": args.batch_size,
            "chat_template": str(args.chat_template),
            "cuda_graph": "off" if args.eager else "on (default)",
            "accuracy": accuracy,
            "invalid_rate": invalid_rate,
            "latency_s": latency_s,
            "generated_tokens": run.generated_tokens,
            "questions_per_second": len(labels) / latency_s,
            "tokens_per_second": run.generated_tokens / latency_s,
        },
    )
    print(f"-> {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
