"""精度金标准:记录/比对 greedy 逐 token 输出,保证优化不改变数值语义。

用例定义来自 ``tests/golden/cases.py``——与 pytest 侧共用同一份来源,避免脚本
录制的用例和测试回放的用例漂移。本脚本是 **录制工具**;日常校验由
``tests/golden/test_token_parity.py`` 在 pytest 中自动完成。

用法::

    # 录制单个 checkpoint 的基线
    .venv/bin/python scripts/golden_tokens.py \
        --save tests/golden/data/Qwen2.5-0.5B.json

    # 多模型批量重录
    .venv/bin/python scripts/golden_tokens.py --batch-save \
        --models my_weight/Qwen2.5-0.5B my_weight/Qwen3-0.6B

    # 含量化路径的录制
    .venv/bin/python scripts/golden_tokens.py \
        --save tests/golden/data/Qwen3-0.6B_int8.json \
        --model-dir my_weight/Qwen3-0.6B --quantization int8

    # 手动比对(可选;pytest 已覆盖)
    .venv/bin/python scripts/golden_tokens.py --check tests/golden/data/Qwen2.5-0.5B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rapid_llm import SamplingParams, TextGenerator
from tests.golden.cases import (
    CASES,
    MAX_GPU_NUM_BLOCKS,
    MAX_SEQ_LEN,
    PENALTIES,
    QUANT_CASES,
    QUANT_SCHEMES,
    case_key,
)


def collect(
    model_dir: str,
    use_cuda_graph: bool,
    quantization: str | None = None,
    include_quant_cases: bool = False,
) -> dict[str, list[str]]:
    """跑完所有 (用例 x repetition_penalty) 组合,返回 key -> 输出文本列表。"""
    gen = TextGenerator(
        checkpoints_dir=model_dir,
        max_seq_len=MAX_SEQ_LEN,
        max_gpu_num_blocks=MAX_GPU_NUM_BLOCKS,
        use_cuda_graph=use_cuda_graph,
        quantization=quantization,
    )
    out: dict[str, list[str]] = {}

    # Standard text cases
    cases = CASES
    if include_quant_cases and quantization:
        cases = QUANT_CASES

    for name, prompts, max_gen_len in cases:
        for penalty in PENALTIES:
            params = SamplingParams(
                temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=penalty
            )
            key = case_key(name, penalty, scheme=quantization or "")
            out[key] = gen.generate(prompts, params)
    return out


def collect_all_schemes(
    model_dir: str, use_cuda_graph: bool
) -> dict[str, list[str]]:
    """Record baselines for the fp16 path and all runtime quantisation schemes."""
    # fp16 baseline (standard CASES)
    results = collect(model_dir, use_cuda_graph)

    # Quantisation-specific cases for each runtime scheme
    for scheme in QUANT_SCHEMES:
        try:
            scheme_results = collect(
                model_dir, use_cuda_graph=False, quantization=scheme, include_quant_cases=True
            )
            results.update(scheme_results)
        except Exception as e:
            print(f"WARNING: scheme {scheme!r} failed: {e}", file=sys.stderr)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B")
    ap.add_argument("--save", help="录制基线到该 JSON 路径")
    ap.add_argument("--check", help="与该 JSON 基线比对")
    ap.add_argument("--cuda-graph", action="store_true", help="用 CUDA graph 路径采集")
    ap.add_argument("--quantization", default=None, help="运行时量化方案 (int8/fp8/smoothquant)")
    ap.add_argument(
        "--batch-save", action="store_true",
        help="多模型批量重录 (需配合 --models)",
    )
    ap.add_argument(
        "--models", nargs="+", default=None,
        help="多模型目录列表 (配合 --batch-save 使用)",
    )
    ap.add_argument(
        "--all-schemes", action="store_true",
        help="录制 fp16 + 所有运行时量化方案的基线",
    )
    ap.add_argument(
        "--output-dir", default="tests/golden/data",
        help="批量录制时输出目录 (默认 tests/golden/data)",
    )
    args = ap.parse_args()

    if args.batch_save:
        models = args.models or [args.model_dir]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            model_name = Path(model).name
            print(f"Recording golden for {model_name}...")
            if args.all_schemes:
                got = collect_all_schemes(model, args.cuda_graph)
                path = output_dir / f"{model_name}_all.json"
            else:
                got = collect(model, args.cuda_graph, quantization=args.quantization)
                suffix = f"_{args.quantization}" if args.quantization else ""
                path = output_dir / f"{model_name}{suffix}.json"
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1) + "\n")
            print(f"  saved {len(got)} cases -> {path}")
        return 0

    if not args.save and not args.check:
        ap.error("需要 --save, --check 或 --batch-save 之一")

    if args.all_schemes:
        got = collect_all_schemes(args.model_dir, args.cuda_graph)
    else:
        got = collect(args.model_dir, args.cuda_graph, quantization=args.quantization)

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, ensure_ascii=False, indent=1) + "\n")
        print(f"saved {len(got)} cases -> {path}")
        return 0

    want = json.loads(Path(args.check).read_text())
    bad = 0
    for name in want:
        if got.get(name) != want[name]:
            bad += 1
            print(f"MISMATCH {name}")
            for i, (a, b) in enumerate(zip(want[name], got.get(name, []), strict=False)):
                if a != b:
                    print(f"   seq{i} want: {a[:100]!r}")
                    print(f"   seq{i} got : {b[:100]!r}")
    print(f"{'FAIL' if bad else 'PASS'}: {len(want) - bad}/{len(want)} cases identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
