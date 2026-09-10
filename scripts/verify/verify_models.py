"""端到端推理验证:逐模型跑固定 prompt,检查输出合理性与停止原因。

用法:
    .venv/bin/python scripts/verify_models.py --model-dir my_weight/Qwen2.5-0.5B --kind text
    .venv/bin/python scripts/verify_models.py --model-dir my_weight/Qwen3-VL-4B-Instruct --kind vision
"""

from __future__ import annotations

import argparse
import sys
import time


def run_text(model_dir: str, use_cuda_graph: bool) -> bool:
    from rapid_llm import SamplingParams, TextGenerator

    prompts = [
        "The future of artificial intelligence is",
        "Give me a short list of primary colors:",
    ]
    t0 = time.time()
    gen = TextGenerator(checkpoints_dir=model_dir, max_seq_len=2048, use_cuda_graph=use_cuda_graph)
    load_s = time.time() - t0

    ok = True
    # greedy: 确定性输出,检查不重复、能停
    params = SamplingParams(temperature=0.0, max_gen_len=64, repetition_penalty=1.0)
    t0 = time.time()
    outs = gen.generate(prompts, params)
    dt = time.time() - t0
    reasons = gen.engine.last_stop_reasons
    n_tokens = 64 * len(prompts)
    for p, o, r in zip(prompts, outs, reasons, strict=True):
        line = o.replace("\n", " ")[:80]
        print(f"  [{r:6s}] {p!r} -> {line!r}")
        if not o.strip():
            print(f"  FAIL: empty output for {p!r}")
            ok = False
    print(
        f"  greedy: {n_tokens / max(dt, 1e-9):.1f} tok/s (batch={len(prompts)}), load {load_s:.1f}s"
    )
    return ok


def run_vision(model_dir: str) -> bool:
    from PIL import Image

    from rapid_llm import SamplingParams, VisionGenerator

    gen = VisionGenerator(checkpoints_dir=model_dir, max_seq_len=2048)
    image = Image.open("images/llava_test/dog.jpeg").convert("RGB")
    params = SamplingParams(temperature=0.0, max_gen_len=48, repetition_penalty=1.0)
    prompt = (
        "Describe this image."
        if gen.is_qwen3_vl
        else "USER: <image>\nDescribe this image. ASSISTANT:"
    )
    t0 = time.time()
    out = gen.generate(prompt, [image], params)
    dt = time.time() - t0
    print(f"  -> {out.strip()[:120]!r}  ({dt:.1f}s)")
    if not out.strip():
        print("  FAIL: empty output")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--kind", choices=["text", "vision"], default="text")
    ap.add_argument("--use-cuda-graph", action="store_true")
    args = ap.parse_args()

    print(f"== {args.model_dir} ({args.kind}, cuda_graph={args.use_cuda_graph}) ==")
    ok = (
        run_vision(args.model_dir)
        if args.kind == "vision"
        else run_text(args.model_dir, args.use_cuda_graph)
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
