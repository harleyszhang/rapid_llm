"""单层 harness 命令行入口:只跑模型的一层,查正确性、耗时与派发结果。

不需要整份权重也能跑:``--weights random`` 只要 ``config.json``,``--weights mirror``
把 transformers 同层的随机权重镜像进来并做逐 token 数值比对——671B 模型的一层是单卡
对象,MLA / 新路由这类改动因此能在真机上先验证再谈整网。

用法::

    # 只测耗时与派发(随机权重)
    .venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B --layer 0

    # 与 transformers 同层比对(无需 checkpoint 权重)
    .venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B \
        --layer 3 --weights mirror --tolerance 2e-2

    # 用真实权重跑,并按 decode 形态加压
    .venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B \
        --layer 3 --weights checkpoint --batch 4 --seq-len 512 --decode-steps 32

    # 只列出该层在 checkpoint 里的 key,不读张量
    .venv/bin/python scripts/layer_harness.py --model-dir my_weight/Qwen3-0.6B \
        --layer 3 --list-keys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WEIGHT_CHOICES = ("random", "mirror", "checkpoint")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True, help="HuggingFace checkpoint 目录")
    ap.add_argument("--layer", type=int, default=0, help="层索引,负数从末尾数")
    ap.add_argument(
        "--weights",
        choices=WEIGHT_CHOICES,
        default="random",
        help="random: 随机初始化;mirror: 从 transformers 同层镜像并比对;checkpoint: 读真实权重",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="给定则作为门禁:相对误差超过它就以非零码退出(仅 mirror / checkpoint 比对时生效)",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="checkpoint 权重下也建 transformers 同层做比对(会覆写成镜像权重)",
    )
    ap.add_argument("--list-keys", action="store_true", help="只列出该层的 checkpoint key")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    from rapid_llm.tools.harness import HFLayerReference, SingleLayerHarness, layer_keys

    harness = SingleLayerHarness.from_pretrained(
        args.model_dir, args.layer, device=args.device, max_seq_len=args.max_seq_len
    )

    if args.list_keys:
        prefix = harness.checkpoint_prefix()
        keys = list(layer_keys(args.model_dir, prefix))
        print(f"{len(keys)} key(s) under {prefix}")
        for key in keys:
            print(f"  {key}")
        return 0 if keys else 1

    # mirror 与 --compare 都要建参照层;它同时是权重来源,所以先于 harness 侧填权重。
    reference = None
    if args.weights == "mirror" or args.compare:
        reference = HFLayerReference(harness.config, harness.layer_index, device=args.device)

    if args.weights == "checkpoint":
        harness.load_checkpoint(args.model_dir)
    elif args.weights == "random":
        harness.randomise()

    report = harness.run(
        batch=args.batch,
        seq_len=args.seq_len,
        decode_steps=args.decode_steps,
        iters=args.iters,
        reference=reference,
    )
    print(report.render())

    if args.tolerance is None:
        return 0
    diffs = [d for d in (report.prefill_diff, report.decode_diff) if d is not None]
    if not diffs:
        print(f"FAIL: --tolerance given but nothing was compared (--weights={args.weights})")
        return 1
    worst = max(d.rel for d in diffs)
    verdict = "PASS" if worst <= args.tolerance else "FAIL"
    print(f"{verdict}: worst relative difference {worst:.3e} vs tolerance {args.tolerance:.3e}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
