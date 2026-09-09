"""The overlap policy benches, one CLI.

    python -m benchmarks.overlap.policies --policy sbo [--graph]
    python -m benchmarks.overlap.policies --policy ep_tbo --batches 128 256
    python -m benchmarks.overlap.policies --policy ep_matrix
    python -m benchmarks.overlap.policies --policy prefill
    python -m benchmarks.overlap.policies --policy scaling --batches 32 128 256
    python -m benchmarks.overlap.policies --policy matrix [--part a|b|all]

* **sbo** — single-batch overlap: the shared MLP on a side stream beside the
  routed path's dispatch exchange. ``--graph`` runs both arms under a captured
  graph (EP keeps its graphs now), which removes the Python launch floor the
  eager arms sit on; the timeline round always runs eager, because a captured
  graph bakes the timeline's CUDA events.
* **ep_tbo** — two-batch overlap on EP MoE at large batch, the shape SGLang's
  TBO targets: a big dispatch/combine all-to-all whose wire time the other
  micro-batch's compute hides.
* **ep_matrix** — EP on/off × TBO on/off plus a graphed reference, to separate
  the overlap's effect from the launch floor.
* **prefill** — prefill TBO and the SM budget (``NCCL_MAX_CTAS``), the two
  pieces sglang gets from DeepEP.
* **scaling** — TBO across batch and model size, to find where (if anywhere) the
  doubled kernel count stops costing more than the all-reduce it hides.
* **matrix** — the L1×L2×L3 combination matrix on one model, then the
  recommended mix model by model.

Every arm reports TTFT / TPOT / TPS / TGS (per-GPU throughput). Offline
inference: all prompts submitted at once, no serving queue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.overlap.arms import (
    Arm,
    compare,
    l1_switch,
    l3_switch,
    make_arm,
    run_arm,
    run_arms,
    sbo_switch,
    sm_budget,
    tbo_switch,
    timeline_overlap,
)
from rapid_llm.benchmark import (
    BenchResult,
    environment,
    report_agreement,
    require_gpus,
    timestamped_log_path,
    write_json_log,
)

V2_LITE = "my_weight/DeepSeek-V2-Lite"

#: EP's captured a2a buffers make each graph far larger than a dense TP one, and
#: these benches build several EP engines in one process whose private graph
#: pools do not fully release between arms. Capping the KV pool leaves the
#: headroom the captures need; the bench prompts are short, so the cap is never
#: the binding constraint on the workload.
EP_ENGINE = {
    "tensor_parallel_size": 2,
    "enable_expert_parallel": True,
    "max_seq_len": 1024,
    "max_gpu_num_blocks": 65536,
    "max_num_seqs": 64,
}


def bench_sbo(args) -> dict:
    """Shared MLP beside the dispatch exchange, eager or graphed."""
    engine = {**EP_ENGINE, "use_cuda_graph": args.graph}
    if args.graph:
        # Lazy capture seeds a pair upfront and captures the bench batches on
        # demand during warmup, which is before the measured region — so the
        # timed steps replay rather than capture.
        engine = {**engine, "cuda_graph_lazy": True, "max_seq_len": 512}

    summary: dict[str, dict] = {}
    timeline_arms: list[Arm] = []
    for batch in args.batches:
        print(
            f"\n=== SBO {V2_LITE} TP=2 EP=2 batch={batch} gen={args.gen} "
            f"{'graph' if args.graph else 'eager'} ===",
            flush=True,
        )
        arms = [
            make_arm("sbo_off", sbo_switch(False), engine=engine, batch=batch, gen_len=args.gen),
            make_arm("sbo_on", sbo_switch(True), engine=engine, batch=batch, gen_len=args.gen),
        ]
        summary[str(batch)] = compare(
            run_arms(V2_LITE, arms, repeat=args.repeat), "sbo_off", ["sbo_on"]
        )
        timeline_arms.append(arms[1])

    print("\n=== timeline: shared MLP vs dispatch exchange overlap ===", flush=True)
    evidence = timeline_overlap(
        V2_LITE,
        timeline_arms[0],
        left=lambda r: r.name == "sbo.shared_mlp",
        right=lambda r: r.stream == "comm" and r.name.startswith("ep.dispatch"),
    )
    print(evidence, flush=True)
    return {
        "batches": summary,
        "timeline": evidence,
        "cuda_graph": args.graph,
        "note": (
            "SBO moves the shared MLP onto an alternate stream so it computes "
            "while the routed path's dispatch exchange is on the wire. The "
            "eager arms sit on the Python launch floor, so the exchange is a "
            "small share of a CPU-bound step; --graph removes that floor. The "
            "timeline always runs eager because a captured graph bakes the "
            "CUDA events. Greedy divergences between the arms are the bf16 "
            "reduction-order noise the EP arms already show."
        ),
    }


def bench_ep_tbo(args) -> dict:
    """Two-batch overlap on EP MoE at large batch, both arms graphed."""
    engine = {
        **EP_ENGINE,
        "use_cuda_graph": True,
        "cuda_graph_lazy": True,
        "max_seq_len": 512,
        "max_gpu_num_blocks": 32768,
    }
    summary: dict[str, dict] = {}
    for batch in args.batches:
        print(
            f"\n=== EP+TBO {V2_LITE} TP=2 EP=2 graph batch={batch} gen={args.gen} ===", flush=True
        )
        arms = [
            make_arm("tbo_off", tbo_switch(False), engine=engine, batch=batch, gen_len=args.gen),
            make_arm(
                "tbo_on",
                tbo_switch(True, min_rows=args.min_rows),
                engine=engine,
                batch=batch,
                gen_len=args.gen,
            ),
        ]
        summary[str(batch)] = compare(
            run_arms(V2_LITE, arms, repeat=args.repeat), "tbo_off", ["tbo_on"]
        )
    return {
        "batches": summary,
        "note": (
            "EP MoE + large batch + TBO, both arms under a captured graph: a big "
            "dispatch/combine all-to-all whose wire time TBO hides behind the "
            "other micro-batch's compute, with the launch floor removed by "
            "replay. min_rows is forced low so the interleave is captured at "
            "the bench batches."
        ),
    }


def bench_ep_matrix(args) -> dict:
    """EP on/off × TBO on/off, eager, with a graphed reference per batch."""
    summary: dict[str, dict] = {}
    for batch in args.batches:
        print(f"\n=== EP matrix {V2_LITE} TP=2 batch={batch} gen={args.gen} ===", flush=True)
        arms = [
            make_arm(
                f"ep={'on' if ep else 'off'} tbo={'on' if tbo else 'off'}",
                tbo_switch(tbo),
                engine={
                    "tensor_parallel_size": 2,
                    "use_cuda_graph": False,
                    "max_seq_len": 2048,
                    "max_num_seqs": 512,
                    "enable_expert_parallel": ep,
                },
                batch=batch,
                gen_len=args.gen,
            )
            for ep in (False, True)
            for tbo in (False, True)
        ]
        arms.append(
            make_arm(
                "graph_reference",
                tbo_switch(False),
                engine={
                    "tensor_parallel_size": 2,
                    "use_cuda_graph": True,
                    "max_seq_len": 2048,
                    "max_num_seqs": 64,
                },
                batch=batch,
                gen_len=args.gen,
            )
        )
        rows = run_arms(V2_LITE, arms)
        summary[str(batch)] = compare(rows, arms[0].label, [a.label for a in arms[1:]])
    return {
        "batches": summary,
        "note": (
            "Four eager arms plus a graphed TP2 reference per batch. The eager "
            "quadruple sits on the Python launch floor the reference escapes, "
            "so eager TBO — and eager EP a2a — pays its scheduling cost against "
            "a floor the graph never sees."
        ),
    }


def bench_prefill(args) -> dict:
    """Prefill TBO and the exchange's SM budget, on long prompts."""
    engine = {
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "use_cuda_graph": False,
        "max_seq_len": 2048,
    }
    summary: dict[str, dict] = {"environment": environment(), "results": {}}
    for batch in args.batches:
        print(
            f"\n=== prefill TBO batch={batch} prompt={args.prompt_len} gen={args.gen_len} EP2 ===",
            flush=True,
        )
        arms = [
            make_arm(
                label,
                tbo_switch(tbo),
                sm_budget(budget),
                engine={**engine, "max_num_seqs": max(256, batch)},
                batch=batch,
                prompt_chars=args.prompt_len,
                gen_len=args.gen_len,
            )
            for label, tbo, budget in (
                ("baseline", False, False),
                ("prefill_tbo", True, False),
                ("sm_budget", False, True),
                ("tbo+sm_budget", True, True),
            )
        ]
        rows = run_arms(V2_LITE, arms)
        summary["results"][f"batch{batch}"] = compare(
            rows,
            "baseline",
            ["prefill_tbo", "sm_budget", "tbo+sm_budget"],
            metric="ttft_ms",
        )
    summary["workload"] = {
        "model": V2_LITE,
        "batch_sizes": args.batches,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "tensor_parallel_size": 2,
        "expert_parallel": True,
        "greedy": True,
    }
    summary["framework_flags"] = {
        "RAPID_LLM_TBO": "per arm",
        "RAPID_LLM_SBO": "per arm",
        "NCCL_MAX_CTAS": "20 when the SM budget is on",
        "use_cuda_graph": "False (EP + prefill)",
    }
    return summary


def bench_scaling(args) -> dict:
    """TBO across batch and model size: eager and graphed, off and on."""
    summary: dict[str, dict] = {"environment": environment(), "models": {}, "results": {}}
    arm_specs = {
        "eager_off": (False, False),
        "eager_on": (True, False),
        "graph_off": (False, True),
        "graph_on": (True, True),
    }
    for model_dir in args.models:
        name = Path(model_dir).name
        summary["models"][name] = model_facts(model_dir)
        print(f"\n=== model {name}: {summary['models'][name]} ===", flush=True)
        for batch in args.batches:
            print(
                f"\n--- batch={batch} prompt={args.prompt_len} gen={args.gen_len} tp={args.tp} ---",
                flush=True,
            )
            engine = {
                "tensor_parallel_size": args.tp,
                "max_seq_len": 2048,
                "max_num_seqs": max(256, batch),
                "max_gpu_num_blocks": args.kv_blocks,
            }
            rows: dict[str, tuple[BenchResult, list[str]]] = {}
            for label in args.arms:
                tbo, graph = arm_specs[label]
                arm = make_arm(
                    label,
                    tbo_switch(tbo, min_rows=args.min_rows),
                    engine={**engine, "use_cuda_graph": graph},
                    batch=batch,
                    prompt_chars=args.prompt_len,
                    gen_len=args.gen_len,
                )
                try:
                    rows[label] = run_arm(model_dir, arm)
                except Exception as exc:  # a missing shape must not kill the sweep
                    print(f"  {label:12s} FAILED: {type(exc).__name__}: {exc}", flush=True)
                    continue
                from benchmarks.overlap.arms import print_row

                print_row(label, rows[label][0], args.tp)
            present = [label for label in args.arms if label in rows]
            base = "graph_off" if "graph_off" in present else present[0]
            summary["results"][f"{name}_b{batch}"] = compare(
                rows, base, [label for label in present if label != base], tp=args.tp
            )
    summary["workload"] = {
        "batch_sizes": args.batches,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "tensor_parallel_size": args.tp,
        "kv_blocks": args.kv_blocks,
        "greedy": True,
    }
    return summary


def model_facts(model_dir: str) -> dict:
    """The architecture numbers that decide whether halving M stays efficient."""
    from rapid_llm.models.config import ModelConfig

    cfg = ModelConfig.from_pretrained(model_dir, 2048)
    return {
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
    }


# --------------------------------------------------------------------------- #
# The L1 x L2 x L3 matrix: combinations on one model, then model by model
# --------------------------------------------------------------------------- #
COMBO_CKPT = "my_weight/Qwen2.5-1.5B-Instruct"
MODEL_DIRS = {
    "qwen2.5-1.5b": "my_weight/Qwen2.5-1.5B-Instruct",
    "deepseek-v2-lite": "my_weight/DeepSeek-V2-Lite",
    "deepseek-v3-4layers": "my_weight/DeepSeek-V3-4layers-MTP-BF16",
}
#: The trimmed V4 is built once into this scratch dir (config + random weights +
#: a borrowed tokenizer); ``--v4-rebuild`` forces a fresh seed. It is gitignored.
V4_WORKDIR = Path(__file__).resolve().parents[2] / "benchmarks" / ".v4_matrix_tmp"
TOKENIZER_SRC = "my_weight/Qwen2.5-1.5B-Instruct"
V4_VOCAB = 151936

#: L2 owns the row-parallel all-reduce site whenever it is on, so the ``l2l3``
#: and ``all`` cells must behave exactly like ``l2`` and ``l1l2`` — that
#: equivalence is the demotion test.
COMBOS = [
    ("baseline", False, False, False),
    ("l1", True, False, False),
    ("l2", False, True, False),
    ("l3", False, False, True),
    ("l1l2", True, True, False),
    ("l1l3", True, False, True),
    ("l2l3", False, True, True),
    ("all", True, True, True),
]

MATRIX_ENGINE = {
    "tensor_parallel_size": 2,
    "use_cuda_graph": False,
    "max_seq_len": 2048,
    "max_num_seqs": 32,
}


def matrix_cell(
    label: str, l1: bool, l2: bool, l3: bool, batch: int, gen: int, blocks: int | None = None
) -> Arm:
    """One matrix cell: the only differences are the three overlap switches."""
    return make_arm(
        label,
        l1_switch(l1),
        tbo_switch(l2),
        l3_switch(l3),
        engine={**MATRIX_ENGINE, **({"max_gpu_num_blocks": blocks} if blocks else {})},
        batch=batch,
        gen_len=gen,
    )


def run_combination_matrix(prompts_batch: int, gen: int) -> dict:
    """The eight on/off cells on one model."""
    arms = [matrix_cell(label, l1, l2, l3, prompts_batch, gen) for label, l1, l2, l3 in COMBOS]
    print(f"\n########## combination matrix ({COMBO_CKPT}) ##########")
    rows = run_arms(COMBO_CKPT, arms)
    texts = {label: texts for label, (_, texts) in rows.items()}
    report_agreement(texts["baseline"], [(k, v) for k, v in texts.items() if k != "baseline"])

    demotion_holds = texts["l2l3"] == texts["l2"] and texts["all"] == texts["l1l2"]
    if not demotion_holds:
        print("!! L3 failed to yield to L2: the demotion cells drifted from their twins")
    summary = compare(rows, "baseline", [label for label, *_ in COMBOS[1:]])
    summary["demotion_holds"] = demotion_holds
    return summary


def build_v4(workdir: Path) -> Path:
    """Config + random weights + borrowed tokenizer: a servable trimmed V4."""
    import json

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from benchmarks.models.bench_deepseek_v4 import CONFIG as V4_BODY

    workdir.mkdir(parents=True, exist_ok=True)
    body = {**V4_BODY, "vocab_size": V4_VOCAB, "tie_word_embeddings": False}
    (workdir / "config.json").write_text(json.dumps(body))
    for leaf in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (workdir / leaf).write_bytes((Path(TOKENIZER_SRC) / leaf).read_bytes())

    torch.manual_seed(0)
    hf = DeepseekV4ForCausalLM(AutoConfig.for_model("deepseek_v4", **body))
    state = {key: value.detach().clone() for key, value in hf.state_dict().items()}
    save_file(state, str(workdir / "model.safetensors"), metadata={"format": "pt"})
    print(f"built trimmed V4 at {workdir}")
    return workdir


def run_model_matrix(batch: int, gen: int, v4_rebuild: bool) -> dict:
    """Baseline vs the recommended mix (L1+L3) on every TP-2 model."""
    v4_dir = str(build_v4(V4_WORKDIR) if v4_rebuild or not V4_WORKDIR.exists() else V4_WORKDIR)
    summary: dict[str, dict] = {}
    print("\n########## model matrix: baseline vs l1l3 ##########")
    for name, model_dir in [*MODEL_DIRS.items(), ("deepseek-v4-trimmed", v4_dir)]:
        print(f"\n=== {name} ===", flush=True)
        blocks = 4096 if name == "deepseek-v4-trimmed" else None
        rows = run_arms(
            model_dir,
            [
                matrix_cell("baseline", False, False, False, batch, gen, blocks),
                matrix_cell("l1l3", True, False, True, batch, gen, blocks),
            ],
        )
        summary[name] = {"model_dir": model_dir, **compare(rows, "baseline", ["l1l3"])}
    return summary


def bench_matrix(args) -> dict:
    results: dict = {}
    if args.part in ("a", "all"):
        results["combination_matrix"] = run_combination_matrix(args.batch, args.gen)
    if args.part in ("b", "all"):
        results["model_matrix"] = run_model_matrix(args.batch, args.gen, args.v4_rebuild)
    results["workload"] = {
        "batch": args.batch,
        "gen": args.gen,
        "tensor_parallel": 2,
        "cuda_graph": False,
    }
    return results


POLICIES = {
    "sbo": bench_sbo,
    "ep_tbo": bench_ep_tbo,
    "ep_matrix": bench_ep_matrix,
    "prefill": bench_prefill,
    "scaling": bench_scaling,
    "matrix": bench_matrix,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=sorted(POLICIES), default="sbo")
    ap.add_argument("--models", nargs="+", default=[V2_LITE])
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--batch", type=int, default=16, help="matrix workload batch")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=32, help="prefill bench generation")
    ap.add_argument("--prompt-len", type=int, default=512, help="long: prefill dominates")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--kv-blocks", type=int, default=65536)
    ap.add_argument("--min-rows", type=int, default=8, help="TBO activation floor override")
    ap.add_argument("--arms", nargs="+", default=["eager_off", "eager_on", "graph_off", "graph_on"])
    ap.add_argument("--graph", action="store_true", help="SBO arms under a captured graph")
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    ap.add_argument("--v4-rebuild", action="store_true", help="rebuild the trimmed V4 seed")
    ap.add_argument("--repeat", type=int, default=2, help="runs per arm, best kept")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    require_gpus(args.tp)
    if args.json is None:
        args.json = str(
            timestamped_log_path(
                Path(__file__).resolve().parents[2] / "docs" / "benchmark_logs",
                f"overlap_{args.policy}",
            )
        )

    write_json_log(args.json, vars(args), POLICIES[args.policy](args))
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
