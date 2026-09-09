"""Unified CLI for benchmark suites and QK-RMSNorm A/B runs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.suites import qk_norm


@dataclass(frozen=True)
class Model:
    name: str
    compare_configs: tuple[tuple[int, int], ...] = ()
    compare_args: tuple[str, ...] = ()
    tp_configs: tuple[tuple[int, int], ...] = ()
    tp_args: tuple[str, ...] = ()
    e2e_pool: int | None = None
    vision: bool = False


MODELS = {
    model.name: model
    for model in (
        Model("Qwen1.5-0.5B", ((8, 128), (16, 256)), e2e_pool=40960),
        Model("Qwen3-MoE-Tiny", ((8, 128), (16, 256)), e2e_pool=40960),
        Model("Qwen2.5-1.5B", ((8, 128), (16, 256)), e2e_pool=40960),
        Model("Qwen2.5-1.5B-Instruct", ((8, 128), (16, 256))),
        Model("Qwen3-0.6B", ((8, 128), (16, 256)), e2e_pool=40960),
        Model("Qwen3-1.7B", ((8, 128), (16, 256)), e2e_pool=40960),
        Model("Qwen2.5-3B", ((8, 128), (16, 256)), e2e_pool=40960),
        Model("Llama-3.2-3B-Instruct", ((8, 128), (16, 256)), e2e_pool=40960),
        Model(
            "Qwen3-0.6B-FP8",
            ((8, 128), (16, 256)),
            ("--hf-dtype", "auto"),
            e2e_pool=40960,
        ),
        Model(
            "Qwen3-8B",
            ((8, 128),),
            ("--engine", "rapid_llm", "--max-gpu-num-blocks", "16384"),
            ((16, 128),),
            e2e_pool=16384,
        ),
        Model(
            "Meta-Llama-3.1-8B-Instruct",
            ((8, 128),),
            ("--engine", "rapid_llm", "--max-gpu-num-blocks", "16384"),
            ((16, 128),),
        ),
        Model(
            "Qwen3-30B-A3B-Instruct-2507-FP8",
            tp_configs=((8, 128), (16, 128)),
            tp_args=("--engine", "rapid_llm"),
        ),
        Model(
            "Qwen3-14B-AWQ",
            ((8, 128), (16, 128), (16, 256)),
            ("--engine", "rapid_llm"),
            e2e_pool=40960,
        ),
        Model("llava-1.5-7b-hf", vision=True),
        Model("Qwen3-VL-4B-Instruct", vision=True),
    )
}

COMPARE_STEPS = (
    *(("batch", name) for name in (
        "Qwen1.5-0.5B",
        "Qwen3-MoE-Tiny",
        "Qwen2.5-1.5B",
        "Qwen2.5-1.5B-Instruct",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen2.5-3B",
        "Llama-3.2-3B-Instruct",
        "Qwen3-0.6B-FP8",
        "Qwen3-8B",
        "Meta-Llama-3.1-8B-Instruct",
    )),
    ("tp", "Qwen3-8B"),
    ("tp", "Meta-Llama-3.1-8B-Instruct"),
    ("tp", "Qwen3-30B-A3B-Instruct-2507-FP8"),
    ("batch", "Qwen3-14B-AWQ"),
    ("vision", "llava-1.5-7b-hf"),
    ("vision", "Qwen3-VL-4B-Instruct"),
)
E2E_ORDER = (
    "Qwen1.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Qwen3-0.6B",
    "Qwen3-0.6B-FP8",
    "Qwen3-1.7B",
    "Qwen3-8B",
    "Qwen3-14B-AWQ",
    "Qwen3-MoE-Tiny",
    "Llama-3.2-3B-Instruct",
)


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _python_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else _REPO_ROOT / path)


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else _REPO_ROOT / path


class Runner:
    def __init__(self, python: str, dry_run: bool) -> None:
        self.python = _python_path(python)
        self.dry_run = dry_run

    @staticmethod
    def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT)
        if extra:
            env.update(extra)
        return env

    def show(self, command: list[str]) -> None:
        print(f"DRY-RUN {shlex.join(command)}")

    def logged(
        self,
        command: list[str],
        log_path: Path,
        timeout: float,
        *,
        append: bool,
        extra_env: dict[str, str] | None = None,
    ) -> bool:
        if self.dry_run:
            self.show(command)
            print(f"        log: {log_path} ({'append' if append else 'overwrite'})")
            return True
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a" if append else "w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=_REPO_ROOT,
                    env=self._env(extra_env),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def tail(
        self,
        command: list[str],
        lines: int,
        extra_env: dict[str, str] | None = None,
    ) -> bool:
        if self.dry_run:
            self.show(command)
            return True
        try:
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=self._env(extra_env),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            print(error, file=sys.stderr)
            return False
        output = (completed.stdout + completed.stderr).splitlines()
        print("\n".join(output[-lines:]))
        return completed.returncode == 0

    def require_cuda(self) -> None:
        if self.dry_run:
            return
        available = subprocess.run(
            [self.python, "-c", "import torch; raise SystemExit(not torch.cuda.is_available())"],
            cwd=_REPO_ROOT,
            env=self._env(),
            capture_output=True,
            check=False,
        )
        if available.returncode == 0:
            return
        version = subprocess.run(
            [self.python, "-c", "import torch; print(torch.__version__)"],
            cwd=_REPO_ROOT,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or "no torch"
        raise SystemExit(f"CUDA 不可用: {self.python} (torch {version}) 的构建与本机驱动不匹配。")

    def gpu_count(self) -> int:
        if self.dry_run:
            return 2
        result = subprocess.run(
            [self.python, "-c", "import torch; print(torch.cuda.device_count())"],
            cwd=_REPO_ROOT,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        return int(result.stdout.strip() or 0) if result.returncode == 0 else 0


def _checkpoint(model: Model) -> Path:
    return _REPO_ROOT / "my_weight" / model.name


def _exists(model: Model, runner: Runner) -> bool:
    if runner.dry_run or _checkpoint(model).is_dir():
        return True
    print(f"[{_clock()}] SKIP  {model.name} (my_weight/ 下无此 checkpoint)")
    return False


def _run_compare_batch(model: Model, out: Path, runner: Runner) -> None:
    for batch, gen_len in model.compare_configs:
        command = [
            runner.python,
            "examples/benchmark.py",
            "--model",
            str(_checkpoint(model).relative_to(_REPO_ROOT)),
            "--batch-size",
            str(batch),
            "--gen-len",
            str(gen_len),
            "--iters",
            "2",
            *model.compare_args,
        ]
        print(f"[{_clock()}] START {model.name} b{batch} g{gen_len} {' '.join(model.compare_args)}")
        ok = runner.logged(command, out / f"{model.name}.b{batch}.log", 1500, append=True)
        print(f"[{_clock()}] {'  OK' if ok else 'FAIL'}  {model.name} b{batch} g{gen_len}")


def _run_compare_tp(model: Model, out: Path, runner: Runner, gpu_count: int) -> None:
    if gpu_count < 2:
        print(f"[{_clock()}] SKIP  {model.name} tp2 (需要 2 张卡,本机 {gpu_count})")
        return
    for batch, gen_len in model.tp_configs:
        command = [
            runner.python,
            "examples/benchmark.py",
            "--model",
            str(_checkpoint(model).relative_to(_REPO_ROOT)),
            "--batch-size",
            str(batch),
            "--gen-len",
            str(gen_len),
            "--iters",
            "2",
            "--tensor-parallel-size",
            "2",
            *model.tp_args,
        ]
        print(f"[{_clock()}] START {model.name} tp2 b{batch} g{gen_len}")
        ok = runner.logged(command, out / f"{model.name}.tp2.b{batch}.log", 2400, append=True)
        print(f"[{_clock()}] {'  OK' if ok else 'FAIL'}  {model.name} tp2 b{batch} g{gen_len}")


def _run_compare_vision(model: Model, out: Path, runner: Runner) -> None:
    command = [
        runner.python,
        "examples/benchmark_vision.py",
        "--model",
        str(_checkpoint(model).relative_to(_REPO_ROOT)),
        "--num-requests",
        "8",
        "--gen-len",
        "128",
        "--iters",
        "2",
    ]
    print(f"[{_clock()}] START vision {model.name}")
    ok = runner.logged(command, out / f"{model.name}.vision.log", 1500, append=True)
    print(f"[{_clock()}] {'  OK' if ok else 'FAIL'}  vision {model.name}")


def _compare(args: argparse.Namespace, runner: Runner) -> int:
    args.out = _rooted(args.out)
    if not runner.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    runner.require_cuda()
    gpu_count = runner.gpu_count()
    for kind, name in COMPARE_STEPS:
        model = MODELS[name]
        if not _exists(model, runner):
            continue
        if kind == "batch":
            _run_compare_batch(model, args.out, runner)
        elif kind == "tp":
            _run_compare_tp(model, args.out, runner, gpu_count)
        else:
            _run_compare_vision(model, args.out, runner)
    print(f"[{_clock()}] ALL DONE -> {args.out}")
    return 0


def _e2e(args: argparse.Namespace, runner: Runner) -> int:
    if not runner.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    runner.require_cuda()
    for name in E2E_ORDER:
        model = MODELS[name]
        if not _exists(model, runner):
            continue
        command = [
            runner.python,
            "-m",
            "rapid_llm.benchmark.one_batch",
            "--model",
            str(_checkpoint(model).relative_to(_REPO_ROOT)),
            "--batch-sizes",
            "8",
            "--verify",
            "--engine-arg",
            f"max_gpu_num_blocks={model.e2e_pool}",
            "--tag",
            name,
            "--log-dir",
            str(args.out),
        ]
        print(f"[{_clock()}] START {name} (kv pool {model.e2e_pool} tokens)")
        ok = runner.logged(command, args.out / f"{name}.log", 1200, append=False)
        print(f"[{_clock()}] {'  OK' if ok else 'FAIL'}  {name}")
    print(f"[{_clock()}] ALL DONE -> {args.out}")
    return 0


def _models(args: argparse.Namespace, runner: Runner) -> int:
    if not runner.dry_run:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        runner.python,
        "-m",
        "rapid_llm.benchmark.models",
        "--zoo",
        str(args.zoo),
        "--vllm-python",
        _python_path(args.vllm_python),
        "--log-dir",
        str(args.log_dir),
        *args.model_args,
    ]
    if runner.dry_run:
        runner.show(command)
        return 0
    log_path = args.log_dir / f"models_suite_run_{datetime.now():%Y%m%d_%H%M%S}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            env=runner._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
    return process.wait()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=os.environ.get("PYTHON", ".venv/bin/python"))
    parser.add_argument("--dry-run", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="rapid_llm versus HuggingFace model suite")
    _add_common(compare)
    compare.add_argument("--out", type=Path, default=Path("/tmp/models_bench"))
    compare.set_defaults(_handler=_compare)

    e2e = subparsers.add_parser("e2e", help="eager versus CUDA-graph one-batch suite")
    _add_common(e2e)
    e2e.add_argument("--out", type=Path, default=Path("/tmp/e2e"))
    e2e.set_defaults(_handler=_e2e)

    models = subparsers.add_parser("models", help="planned modelzoo suite")
    _add_common(models)
    models.add_argument(
        "--zoo",
        type=Path,
        default=Path(os.environ.get("MODELZOO", "/mnt/otto-temp/modelzoo_with_full_weights")),
    )
    models.add_argument(
        "--vllm-python",
        default=os.environ.get(
            "VLLM_PYTHON",
            os.environ.get("PY_VLLM", "/mnt/otto-temp/zhanghonggao.zhg/vllm/.venv/bin/python"),
        ),
    )
    models.add_argument("--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "docs/benchmark_logs")))
    models.set_defaults(_handler=_models)

    qk = subparsers.add_parser("qk-norm", help="QK-RMSNorm fusion A/B suite")
    qk_subparsers = qk.add_subparsers(dest="qk_command", required=True)
    qk_run = qk_subparsers.add_parser("run", help="run one fused or baseline matrix")
    _add_common(qk_run)
    qk_run.add_argument("variant", choices=("fused", "baseline"))
    qk_run.add_argument("out", type=Path)
    qk_run.add_argument("--scope", choices=("single", "parallel", "all"), default="all")
    qk_run.add_argument(
        "--weight-root",
        type=Path,
        default=Path(os.environ.get("WEIGHT_ROOT", "/mnt/otto-temp/modelzoo_with_full_weights")),
    )
    qk_run.set_defaults(_handler=qk_norm.run_matrix)

    summarize = qk_subparsers.add_parser("summarize", help="recompute ratios from archived JSONs")
    summarize.add_argument(
        "--directory",
        type=Path,
        default=_REPO_ROOT / "docs/benchmark_logs/qk_norm",
    )
    summarize.set_defaults(_handler=lambda args, _: qk_norm.summarize(args.directory), dry_run=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if args.command != "models":
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        args.model_args = unknown
    elif args.command == "models":
        args.model_args = []
    for name in ("out", "log_dir", "zoo", "weight_root", "directory"):
        if hasattr(args, name):
            setattr(args, name, _rooted(getattr(args, name)))
    runner = Runner(getattr(args, "python", ".venv/bin/python"), getattr(args, "dry_run", False))
    return args._handler(args, runner)


if __name__ == "__main__":
    raise SystemExit(main())
