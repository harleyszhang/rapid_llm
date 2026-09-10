"""Modelzoo suite driver — preflight, planning, and subprocess orchestration.

Every serving-ready checkpoint under the modelzoo root runs the short/medium/
long scenarios with three engine arms, one subprocess per arm so VRAM never
overlaps: ``rapid_llm`` and ``transformers`` in this repo's venv, ``vllm`` in
its own venv (not importable next to rapid_llm) with ``PYTHONPATH`` back here.

The preflight is the suite's honesty check, before anything runs: shard
completeness (an ``*.incomplete`` shard is an interrupted download), registry
support for the checkpoint's ``model_type``, quantization reachability (fp8
loads on rapid_llm/vllm but not native transformers; compressed-tensors has no
rapid_llm loader), and the memory budget — the smallest TP keeping
``weights/TP + KV/TP + activations <= 0.92 x HBM`` per GPU, per scenario.
Models whose every scenario fits one GPU add a DP2 point on the medium tier; a
MoE model whose long tier needs TP2 adds a TP2+EP2 A/B there. ``--dry-run``
prints per-model verdicts and the plan, runs nothing — and each arm is one
``rapid_llm.benchmark.offline_throughput`` subprocess, so a plan row is
reproducible by hand.

Usage:
    .venv/bin/python -m rapid_llm.benchmark.models \
        --zoo /mnt/otto-temp/modelzoo_with_full_weights \
        --vllm-python /mnt/otto-temp/zhanghonggao.zhg/vllm/.venv/bin/python \
        --log-dir docs/benchmark_logs --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .utils import gpu_tag, print_row_table, write_json_log

#: Repo root: the suite runs its arms from here, and hands it to vLLM's venv as
#: ``PYTHONPATH`` so that interpreter can reach ``rapid_llm.benchmark`` (a lazy,
#: CUDA-free facade) without importing rapid_llm's engine.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# scenario ladder + memory model
# --------------------------------------------------------------------------- #

BATCH = 8
ITERS = 2
KV_MARGIN_TOKENS = 768  # per-request slack over input + output
ACTIVATION_GB = 4.0  # coarse per-GPU activation/workspace reserve
UTILIZATION = 0.92  # fraction of one GPU's HBM the suite may claim
#: DP2 keeps a full replica per GPU; past this weight a two-replica card has
#: no KV/activation headroom left, so no DP2 point is offered.
DP2_WEIGHT_GB = 40.0

SCENARIOS = {
    "short": {"input_len": 128, "output_len": 32, "max_seq_len": 4096},
    "medium": {"input_len": 4096, "output_len": 128, "max_seq_len": 8192},
    "long": {"input_len": 32768, "output_len": 256, "max_seq_len": 33792},
}

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def gpu_profile() -> tuple[int, float, str]:
    """(visible GPU count, per-GPU HBM GB, device name)."""
    import torch

    count = torch.cuda.device_count()
    if not count:
        return 0, 0.0, "no-cuda"
    props = torch.cuda.get_device_properties(0)
    return count, props.total_memory / 1e9, props.name


def kv_bytes_per_token(cfg: dict) -> int:
    """2 (K and V) x layers x kv-heads x head-dim x 2 bytes (bf16 KV, the
    conservative upper bound — an fp8 KV cache only shrinks it)."""
    layers = cfg.get("num_hidden_layers", 0)
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads", 0)
    head_dim = cfg.get("head_dim")
    if head_dim is None and cfg.get("num_attention_heads"):
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    return 2 * layers * kv_heads * (head_dim or 0) * 2


def quant_method(cfg: dict) -> str | None:
    q = cfg.get("quantization_config") or {}
    return q.get("quant_method") or q.get("fmt") or None


def check_shards(path: Path) -> tuple[set, set, list, list]:
    """Declared shards vs what the directory actually holds.

    Returns ``(declared, complete, missing, incomplete)`` — a single-file
    checkpoint has no index and an empty declared set.
    """
    idx = path / "model.safetensors.index.json"
    declared: set = set()
    if idx.exists():
        declared = set(json.loads(idx.read_text())["weight_map"].values())
    complete, incomplete = set(), []
    for f in path.glob("*.safetensors*"):
        if f.name.endswith(".incomplete"):
            incomplete.append(f.name)
        elif f.is_file():
            complete.add(f.name)
    return declared, complete, sorted(declared - complete), incomplete


def weight_size_gb(path: Path, declared: set, complete: set) -> float:
    names = (declared & complete) if declared else complete
    total = 0.0
    for name in names:
        f = path / name
        if f.is_file():
            total += f.stat().st_size / 1e9
    return total


def effective_lengths(max_pos: int, scenario: dict) -> tuple[int, int, int]:
    """(input, output, max_seq_len) after clamping to the checkpoint's context.

    A checkpoint whose ``max_position_embeddings`` cannot hold the tier's
    window (0.5B @ 32768 vs the long tier's 33792) takes the shrunken input
    the plan records; the effective values land in the JSON either way.
    """
    inp, out, msl = scenario["input_len"], scenario["output_len"], scenario["max_seq_len"]
    if max_pos and msl + KV_MARGIN_TOKENS > max_pos:
        msl, inp = max_pos, max_pos - 1024
    return inp, out, msl


def decide_tp(weight_gb: float, kv_tokens: int, kv_bpt: int, gpu_count: int, budget_gb: float):
    """Smallest TP keeping weights/TP + KV/TP + activations within the budget."""
    for tp in (1, 2, 4):
        if tp > gpu_count:
            break
        per_gpu = weight_gb / tp + kv_bpt * kv_tokens / 1e9 / tp + ACTIVATION_GB
        if per_gpu <= budget_gb:
            return tp
    return None


# --------------------------------------------------------------------------- #
# preflight + planning
# --------------------------------------------------------------------------- #


@dataclass
class ModelPlan:
    path: Path
    name: str
    model_type: str
    max_pos: int
    quant: str | None
    weight_gb: float
    status: str  # "run" | "skip"
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    kv_bpt: int = 0
    arms: list[str] = field(default_factory=list)
    tp: dict = field(default_factory=dict)  # scenario -> tp
    lengths: dict = field(default_factory=dict)  # scenario -> (input, output, msl)

    @property
    def label(self) -> str:
        return self.name


def discover(root: Path) -> list[Path]:
    """Directories holding a config.json: root level plus one level down."""
    found: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "config.json").exists():
            found.append(p)
        for q in sorted(p.iterdir()):
            if q.is_dir() and (q / "config.json").exists():
                found.append(q)
    return found


def plan_model(path: Path, zoo_root: Path, gpu_count: int, budget_gb: float) -> ModelPlan:
    cfg = json.loads((path / "config.json").read_text())
    model_type = cfg.get("model_type", "?")
    max_pos = cfg.get("max_position_embeddings") or cfg.get("max_sequence_length") or 0
    quant = quant_method(cfg)
    declared, complete, missing, incomplete = check_shards(path)
    weight_gb = weight_size_gb(path, declared, complete)
    plan = ModelPlan(
        path=path,
        name=str(path.relative_to(zoo_root)),
        model_type=model_type,
        max_pos=max_pos,
        quant=quant,
        weight_gb=weight_gb,
        status="skip",
        kv_bpt=kv_bytes_per_token(cfg),
    )

    if missing or incomplete:
        plan.reason = "incomplete shard download"
        plan.evidence = {
            "declared_shards": len(declared),
            "present": len(declared) - len(missing),
            "missing": missing[:3],
            "n_missing": len(missing),
            "n_incomplete": len(incomplete),
        }
        return plan

    from rapid_llm.models.registry import ModelRegistry

    supported = {t.lower() for t in ModelRegistry.supported_types()}
    if model_type.lower() not in supported:
        plan.reason = f"ModelRegistry does not support model_type {model_type!r}"
        plan.evidence = {"supported": sorted(supported)}
        return plan
    if quant is not None and quant not in ("fp8",):
        plan.reason = f"no rapid_llm loader for checkpoint quantization {quant!r}"
        plan.evidence = {"quantization_config": cfg.get("quantization_config")}
        return plan
    if weight_gb > gpu_count * budget_gb:
        plan.reason = f"weights {weight_gb:.0f}GB exceed the {gpu_count} x {budget_gb:.0f}GB budget"
        return plan

    plan.status = "run"
    plan.arms = ["rapid_llm"]
    if quant is None:
        plan.arms.append("transformers")  # native bf16 loading only
    if quant in (None, "fp8"):
        plan.arms.append("vllm")
    for name, sc in SCENARIOS.items():
        lengths = effective_lengths(max_pos, sc)
        kv_tokens = BATCH * (lengths[0] + lengths[1] + KV_MARGIN_TOKENS)
        tp = decide_tp(weight_gb, kv_tokens, plan.kv_bpt, gpu_count, budget_gb)
        if tp is None:
            plan.status = "skip"
            plan.reason = f"no TP in (1, 2, 4) fits the {name} tier within budget"
            plan.tp, plan.lengths, plan.arms = {}, {}, []
            return plan
        plan.tp[name], plan.lengths[name] = tp, lengths
    return plan


def wants_dp2_point(plan: ModelPlan) -> bool:
    """Every tier fits TP1 and a full replica leaves KV/activation headroom."""
    return (
        plan.status == "run"
        and plan.weight_gb <= DP2_WEIGHT_GB
        and all(tp == 1 for tp in plan.tp.values())
    )


def wants_ep2_point(plan: ModelPlan) -> bool:
    """The EP A/B lives where the baseline is already TP2 (long tier)."""
    return plan.status == "run" and plan.tp.get("long") == 2 and "moe" in plan.model_type


def print_plan(plans: list[ModelPlan], gpu_count: int, gpu_gb: float) -> None:
    print(f"\n=== preflight ({gpu_count} x {gpu_name_tag(gpu_gb)}) ===")
    print_row_table(
        ["status", "model", "type", "quant", "size", "note"],
        [6, 44, 11, 18, 8, 58],
        [
            [
                p.status,
                p.name,
                p.model_type,
                p.quant or "-",
                f"{p.weight_gb:.1f}GB",
                p.reason or f"arms: {', '.join(p.arms)}; kv {p.kv_bpt / 1024:.0f}KB/tok",
            ]
            for p in plans
        ],
    )

    print("\n=== plan (batch 8, iters 2, greedy, random-ids) ===")
    print_row_table(
        ["model", "scenario", "input", "output", "msl", "tp", "arms"],
        [44, 8, 7, 7, 7, 3, 34],
        [
            [
                p.name,
                sc,
                str(p.lengths[sc][0]),
                str(p.lengths[sc][1]),
                str(p.lengths[sc][2]),
                str(p.tp[sc]),
                ",".join(p.arms),
            ]
            for p in plans
            if p.status == "run"
            for sc in p.tp
        ],
    )

    dp2 = [p.name for p in plans if wants_dp2_point(p)]
    ep2 = [p.name for p in plans if wants_ep2_point(p)]
    print("\n=== extension points ===")
    print(f"DP2 scaling (medium, 64 prompts, TP1 vs TP2 vs DP2): {', '.join(dp2) or '-'}")
    print(f"TP2+EP2 A/B (long): {', '.join(ep2) or '-'}")


def dedupe_escapes(plans: list[ModelPlan]) -> list[ModelPlan]:
    """Collapse a model's escaped-name copy (``Qwen2___5`` = ``Qwen2.5``): the
    modelzoo holds some repos under both names, and they are the same weights."""

    def unescape(name: str) -> str:
        return name.replace("___", ".").replace("__", "_")

    key = lambda p: f"{p.path.parent.name}/{unescape(p.path.name)}"  # noqa: E731
    kept: dict[str, ModelPlan] = {}
    for p in plans:
        current = kept.get(key(p))
        if current is None or ("___" in current.path.name and "___" not in p.path.name):
            kept[key(p)] = p
    return [p for p in plans if kept.get(key(p)) is p]


def gpu_name_tag(gpu_gb: float) -> str:
    return f"{gpu_gb:.0f}GB GPU"


# --------------------------------------------------------------------------- #
# subprocess orchestration
# --------------------------------------------------------------------------- #


def arm_tp(plan: ModelPlan, scenario: str, engine: str) -> int:
    tp = plan.tp[scenario]
    if engine == "transformers" and plan.weight_gb > 40:
        # HF has no memory manager: a wide bf16 checkpoint shards across the
        # visible GPUs (device_map=auto) instead of gambling one card.
        return max(tp, 2)
    return tp


def bench_cmd(
    python: str,
    json_out: Path,
    plan: ModelPlan,
    lengths: tuple,
    *,
    engine: str,
    tp: int,
    num_prompts: int,
    iters: int,
    data_parallel: int = 1,
    extra: tuple = (),
) -> list[str]:
    inp, out, msl = lengths
    cmd = [
        python,
        "-m",
        "rapid_llm.benchmark.offline_throughput",
        "--model",
        str(plan.path),
        "--engine",
        engine,
        "--dataset-name",
        "random-ids",
        "--random-input-len",
        str(inp),
        "--random-output-len",
        str(out),
        "--max-seq-len",
        str(msl),
        "--num-prompts",
        str(num_prompts),
        "--iters",
        str(iters),
        "--greedy",
        "--gpu-mem-util",
        str(UTILIZATION),
        "--json-out",
        str(json_out),
    ]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    if data_parallel > 1:
        cmd += ["--data-parallel-size", str(data_parallel)]
        # Each replica hosts a resident engine: size its ceiling to the share
        # of the batch it will receive instead of the serving default.
        cmd += ["--max-num-seqs", str(max(1, num_prompts // data_parallel))]
    return cmd + list(extra)


GPU_BASE_MIB = 0  # idle VRAM per GPU before the suite; set once in main()


def gpu_used_mibs() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        return [int(x) for x in out.split() if x.isdigit()]
    except Exception:
        return []  # no nvidia-smi / transient failure


def wait_gpu_drain(threshold_mib: int = 2048, max_wait_s: int = 150) -> None:
    """Block until the driver reports every GPU back near its idle level.

    A just-finished arm's 0.92-util memory pool can linger in the driver for
    a minute or two after process exit; launching the next arm — especially
    TP, where NCCL buffers and graph capture need clean cards — into the
    half-drained GPU dies natively. Wait it out instead of failing arms
    spuriously (observed: TP2 arm crash 15s in right after a TP1 arm).

    The floor is the suite-start baseline + slack, not an absolute number:
    on shared hosts VRAM held by processes outside this container (or leaked
    by them) never comes back, and waiting on it would stall every arm.
    """
    limit = max(threshold_mib, GPU_BASE_MIB + 512)
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        used = gpu_used_mibs()
        if not used or max(used) <= limit:
            return
        time.sleep(3)


def run_arm(cmd: list[str], env: dict, timeout_s: int) -> tuple[int, str, str, float]:
    """Run one arm; return (returncode, stdout tail, stderr tail, wall s)."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout_s, cwd=REPO_ROOT
        )
    except subprocess.TimeoutExpired:
        wait_gpu_drain()
        return 1, "", f"timeout after {timeout_s}s", time.monotonic() - t0
    # Native crash stacks (torch/CUDA) run 40+ frames; keep enough of stderr
    # that the first faulting frame survives the truncation.
    tail = lambda s: (s or "")[-4000:]  # noqa: E731
    wait_gpu_drain()
    return proc.returncode, tail(proc.stdout), tail(proc.stderr), time.monotonic() - t0


def read_arm_result(json_out: Path, engine: str) -> dict:
    payload = json.loads(json_out.read_text())
    results = payload.get("results", {})
    return results.get(engine, results)


def run_one(
    plan: ModelPlan,
    engine: str,
    tp: int,
    lengths: tuple,
    json_out: Path,
    args,
    *,
    num_prompts: int,
    data_parallel: int = 1,
    extra: tuple = (),
) -> dict:
    python = args.vllm_python if engine == "vllm" else sys.executable
    env = dict(os.environ)
    if engine == "vllm":
        env["PYTHONPATH"] = str(REPO_ROOT)  # never import rapid_llm in vLLM's venv
    if engine == "transformers":
        # 32k-prefill HF generate fragments the caching allocator (17 GiB
        # reserved-but-unallocated observed on 4B long); expandable segments
        # absorb the fragmentation so the KV+logits peak actually fits.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = bench_cmd(
        python,
        json_out,
        plan,
        lengths,
        engine=engine,
        tp=tp,
        num_prompts=num_prompts,
        iters=args.iters,
        data_parallel=data_parallel,
        extra=extra,
    )
    print(f"  [{engine}] {' '.join(cmd)}", flush=True)
    rc, out_tail, err_tail, wall = run_arm(cmd, env, args.timeout_s)
    if rc == 0 and json_out.exists():
        result = read_arm_result(json_out, engine)
        result["wall_s"] = round(wall, 1)
        print(f"  [{engine}] done in {wall:.0f}s: tps={result.get('tps', '?')}", flush=True)
        return result
    err = err_tail or out_tail or f"exit {rc}"
    print(f"  [{engine}] FAILED ({wall:.0f}s): {err}", flush=True)
    return {"error": err, "cmd": cmd}


def run_scenario(plan: ModelPlan, scenario: str, args, tmp_dir: Path) -> dict:
    lengths = plan.lengths[scenario]
    arms: dict = {}
    for engine in plan.arms:
        tp = arm_tp(plan, scenario, engine)
        out = tmp_dir / f"{plan.label.replace('/', '_')}_{scenario}_{engine}.json"
        arms[engine] = run_one(plan, engine, tp, lengths, out, args, num_prompts=args.batch)
    return arms


def run_dp2_point(plan: ModelPlan, args, tmp_dir: Path) -> dict:
    """Throughput scaling point: same 64-prompt batch through TP1/TP2/DP2."""
    lengths = plan.lengths["medium"]
    point: dict = {"num_prompts": 64, "scenario": "medium"}
    for label, tp, dp in (("tp1", 1, 1), ("tp2", 2, 1), ("dp2", 1, 2)):
        out = tmp_dir / f"{plan.label.replace('/', '_')}_scaling_{label}.json"
        point[label] = run_one(
            plan,
            "rapid_llm",
            tp,
            lengths,
            out,
            args,
            num_prompts=64,
            data_parallel=dp,
        )
    base = point["tp1"].get("tps") or 0
    for label in ("tp2", "dp2"):
        tps = point[label].get("tps") or 0
        point[f"{label}_vs_tp1"] = round(tps / base, 2) if base else None
    return point


def run_ep2_point(plan: ModelPlan, args, tmp_dir: Path) -> dict:
    """TP2 vs TP2+EP2 on the long tier — single-variable feature A/B."""
    lengths = plan.lengths["long"]
    point: dict = {"scenario": "long"}
    out = tmp_dir / f"{plan.label.replace('/', '_')}_ep2.json"
    point["tp2_ep2"] = run_one(
        plan,
        "rapid_llm",
        2,
        lengths,
        out,
        args,
        num_prompts=args.batch,
        extra=("--engine-arg", "enable_expert_parallel=true"),
    )
    return point


def sanitize(name: str) -> str:
    return name.replace("/", "_")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", default="/mnt/otto-temp/modelzoo_with_full_weights")
    parser.add_argument(
        "--vllm-python",
        default="/mnt/otto-temp/zhanghonggao.zhg/vllm/.venv/bin/python",
        help="Interpreter owning the vllm arm (runs as its own subprocess)",
    )
    parser.add_argument("--log-dir", default="docs/benchmark_logs")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--scenarios",
        default="short,medium,long",
        help="Comma-separated subset of the ladder to run",
    )
    parser.add_argument("--models", default="", help="Substring filter on model directory name")
    parser.add_argument(
        "--timeout-s", type=int, default=3600, help="Wall-clock cap per arm subprocess"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print verdicts, plan and commands; run nothing"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios {unknown}; known: {sorted(SCENARIOS)}")

    gpu_count, gpu_gb, gpu_name = gpu_profile()
    if not gpu_count:
        raise SystemExit("no CUDA device visible")
    global GPU_BASE_MIB
    GPU_BASE_MIB = max(gpu_used_mibs() or [0])
    budget_gb = gpu_gb * UTILIZATION
    print(f"gpu: {gpu_count} x {gpu_name} ({gpu_gb:.1f}GB, budget {budget_gb:.1f}GB/GPU)")

    plans = [plan_model(p, Path(args.zoo), gpu_count, budget_gb) for p in discover(Path(args.zoo))]
    before = len(plans)
    plans = dedupe_escapes(plans)
    if len(plans) < before:
        print(f"collapsed {before - len(plans)} escaped-name duplicate model directories")
    print_plan(plans, gpu_count, gpu_gb)

    log_dir = Path(args.log_dir)
    tmp_dir = log_dir / "_arms"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return 0

    selected = [
        p for p in plans if p.status == "run" and (not args.models or args.models in p.name)
    ]
    if args.models:
        print(
            f"model filter {args.models!r}: running {len(selected)} of "
            f"{sum(p.status == 'run' for p in plans)} planned models"
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suite: dict = {
        "gpu": {
            "count": gpu_count,
            "name": gpu_name,
            "gb": round(gpu_gb, 1),
            "budget_gb_per_gpu": round(budget_gb, 1),
        },
        "batch": args.batch,
        "iters": args.iters,
        "scenarios": scenarios,
        "models": {},
    }

    for plan in selected:
        print(f"\n{'=' * 78}\n{plan.name}\n{'=' * 78}", flush=True)
        model_out: dict = {"arms": plan.arms, "tp": plan.tp, "scenarios": {}}
        for scenario in scenarios:
            if scenario not in plan.tp:
                continue
            print(f"\n--- {scenario} ---", flush=True)
            arms = run_scenario(plan, scenario, args, tmp_dir)
            model_out["scenarios"][scenario] = {
                "effective_input_len": plan.lengths[scenario][0],
                "effective_max_seq_len": plan.lengths[scenario][2],
                "results": arms,
            }
            write_json_log(
                log_dir / f"bench_{sanitize(plan.name)}_{scenario}_b{args.batch}_{stamp}.json",
                {
                    "model": plan.name,
                    "model_type": plan.model_type,
                    "quant": plan.quant,
                    "weight_gb": round(plan.weight_gb, 1),
                    "scenario": scenario,
                    "effective": {
                        "input_len": plan.lengths[scenario][0],
                        "max_seq_len": plan.lengths[scenario][2],
                    },
                    "tp_by_arm": {e: arm_tp(plan, scenario, e) for e in plan.arms},
                    "batch": args.batch,
                    "iters": args.iters,
                },
                arms,
            )
        if "medium" in model_out["scenarios"] and wants_dp2_point(plan):
            print("\n--- DP2 scaling point ---", flush=True)
            model_out["dp2_scaling"] = run_dp2_point(plan, args, tmp_dir)
        if "long" in model_out["scenarios"] and wants_ep2_point(plan):
            print("\n--- TP2+EP2 point ---", flush=True)
            model_out["ep2_ab"] = run_ep2_point(plan, args, tmp_dir)
        suite["models"][plan.name] = model_out

    skipped = [
        {
            "model": p.name,
            "type": p.model_type,
            "quant": p.quant,
            "weight_gb": round(p.weight_gb, 1),
            "reason": p.reason,
            "evidence": p.evidence,
        }
        for p in plans
        if p.status == "skip"
    ]
    suite["skipped"] = skipped
    write_json_log(
        log_dir / f"models_suite_{gpu_tag()}_{stamp}.json",
        {k: v for k, v in suite.items() if k != "models"},
        suite["models"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
