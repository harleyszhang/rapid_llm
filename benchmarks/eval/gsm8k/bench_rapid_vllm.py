"""GSM8K accuracy on the rapid_llm engine — the optimised-mode arm.

Delegates to :func:`tests.evals.gsm8k.evaluate_gsm8k` outright: prompts, stop
markers and scoring are that module's, so the benchmark and the regression
suite can never drift apart. CUDA graphs stay on (the engine default) — this
arm exists to measure the shipped configuration.

Usage:
    python -m benchmarks.eval.gsm8k.bench_rapid_vllm \
        --model-dir Qwen2.5-0.5B --num-questions 200 [--chat-template]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.eval._common import append_result_log, resolve_model_dir
from tests.evals.gsm8k import evaluate_gsm8k


def main() -> int:
    ap = argparse.ArgumentParser(description="GSM8K accuracy via the rapid_llm engine")
    ap.add_argument("--model-dir", required=True, help="path, or a name under the model zoo")
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap prompts as a user turn (instruction-tuned checkpoints)",
    )
    ap.add_argument(
        "--eager",
        action="store_true",
        help="disable CUDA graphs (debugging; slower and off the default config)",
    )
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    result = evaluate_gsm8k(
        model_dir,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_gen_len=args.max_gen_len,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        use_chat_template=args.chat_template,
        use_cuda_graph=None if not args.eager else False,
    )

    print(f"\nGSM8K / rapid_llm — {model_dir}")
    print(result.report())

    payload = result.as_dict()
    payload["engine"] = "rapid_llm"
    payload["cuda_graph"] = "off" if args.eager else "on (default)"
    log = append_result_log("gsm8k", "rapid", payload)
    print(f"-> {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
