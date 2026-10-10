"""Decode context parallelism (DCP) against tensor parallelism of the same width.

DCP keeps an attention-TP group's KV heads on every rank of a DCP group and
deals the *positions* round-robin (``p % dcp``); each rank decodes against its
share and the partials are merged through their log-sum-exp. Its claim is a
decode one, so this benchmark measures long-context decode and owes its reader
the whole environment:

* every arm runs CUDA graphs (``--eager`` flips them all); ``tp1`` is the
  no-communication ceiling, and each DCP arm is read against the plain TP arm
  of the *same* width -- the one variable is where the history lives. Decode
  graphs cover histories up to the largest captured bucket (4096), so
  ``bs1-8k`` decodes eager in every arm; the ``replays`` column shows it;
* prompts prefill in one pass by default (``--chunk-size 0``): under DCP a
  continuation chunk cannot use the chunked prefill kernel and runs one row
  per token on the decode path, which ``--chunk-size 512`` measures;
* four scenarios sweep batch and context (bs1 x 3.5k / 8k, bs8 x 3.5k, bs32 x
  1k): DCP's attention reads ``1/dcp`` of the history but its merge traffic is
  per query, so the trade moves with both;
* metrics are per-request (TTFT from each request's own first token, TPOT from
  its own span) and offline (every prompt submitted at once, the engine drains);
* parity is held against ``tp1`` -- N cards against one -- on greedy decoding
  with ``logprobs=2``, by the rule ``tests/distributed/test_dcp_engine``
  enforces: a fork must swap two top-2 candidates both engines decided by at
  most :data:`_TIE_GAP`, and the chosen tokens' logprobs along the agreed
  prefix (reported as the logit-level drift) must stay within the same bound;
* communication is its own section: the Q all-gather plus the merge
  collectives, captured into a CUDA graph at each scenario's decode shape and
  timed per layer, next to the decode kernel itself at TP and at DCP geometry.
  Graph replays bypass Python, so the engine's collective accounting cannot
  see them; the microbenchmark is what can.

Usage:
    python benchmarks/parallelism/bench_context_parallel.py --model my_weight/Qwen3-0.6B
    python benchmarks/parallelism/bench_context_parallel.py --model my_weight/Qwen3-0.6B \
        --arms tp2,tp2_dcp2 --scenarios bs8-3.5k --no-comm    # a quick smoke
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.benchmark import (
    peak_mem_gb,
    print_row_table,
    print_run_header,
    require_gpus,
    run_in_spawned_process,
    run_requests,
    sampling_params,
    timestamped_log_path,
    write_json_log,
)

#: ``(label, tp, dcp, merge backend)`` in run order: the single-GPU ceiling, then
#: each width's TP arm immediately before its DCP arms.
ARMS: list[tuple[str, int, int, str]] = [
    ("tp1", 1, 1, "ag_rs"),
    ("tp2", 2, 1, "ag_rs"),
    ("tp2_dcp2", 2, 2, "ag_rs"),
    ("tp2_dcp2_a2a", 2, 2, "a2a"),
    ("tp4", 4, 1, "ag_rs"),
    ("tp4_dcp2", 4, 2, "ag_rs"),
    ("tp4_dcp4", 4, 4, "ag_rs"),
]

SCENARIOS: list[dict[str, Any]] = [
    {"label": "bs1-3.5k", "batch": 1, "prompt_tokens": 3584},
    {"label": "bs1-8k", "batch": 1, "prompt_tokens": 8192},
    {"label": "bs8-3.5k", "batch": 8, "prompt_tokens": 3584},
    {"label": "bs32-1k", "batch": 32, "prompt_tokens": 1024},
]

#: One engine per arm covers every scenario: 8k + generation fits the sequence
#: cap, and 32 x (1k + 128) / 8 x (3.5k + 128) fit the cache rows. Rows, not
#: tokens: a DCP rank stores ``block_size / dcp`` rows per block, so the same
#: rows hold ``dcp`` times the tokens (reported, not used).
BUILD = {"max_seq_len": 8448, "max_num_seqs": 32, "kv_tokens": 40960}

#: See ``tests/distributed/test_dcp_engine``: the margin (nats) under which a
#: reordered softmax sum may take a step either way.
_TIE_GAP = 0.5

PARITY_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at",
    "The largest planet in our solar system is",
    "Python is a language that",
    "Machine learning is",
]

#: Weight load, KV profile, graph capture and four warmed scenarios.
_PROBE_TIMEOUT_S = 1800.0

#: Collective microbenchmark: calls captured per graph, replays timed.
_COMM_CALLS_PER_GRAPH = 20
_COMM_REPLAYS = 20


# --------------------------------------------------------------------------- #
# workload (parent process, so every arm gets the same strings)
# --------------------------------------------------------------------------- #
def _synthetic_prompt(tokenizer, target_tokens: int, seed: int) -> str:
    """Seeded ordinary-vocabulary ids, re-tokenised and trimmed to ``target_tokens``."""
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 30000) for _ in range(target_tokens + 64)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return tokenizer.decode(tokenizer.encode(text)[:target_tokens], skip_special_tokens=True)


def _prompt_stats(tokenizer, prompts: list[str]) -> dict[str, float]:
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    return {"min": min(lengths), "mean": round(statistics.mean(lengths), 1), "max": max(lengths)}


def _metrics(run) -> dict[str, Any]:
    ttfts, tpots = run.ttfts_ms(), run.per_request_tpot_ms()
    return {
        "ttft_ms": round(statistics.mean(ttfts), 2) if ttfts else 0.0,
        "ttft_ms_max": round(max(ttfts), 2) if ttfts else 0.0,
        "tpot_ms": round(statistics.mean(tpots), 3) if tpots else 0.0,
        "total_s": round(run.total_s, 3),
        "gen_tokens": run.gen_tokens,
        "tps": round(run.gen_tokens / run.total_s, 1) if run.total_s else 0.0,
    }


def _record(completion) -> dict[str, Any]:
    records = completion.logprobs or ()
    return {
        "text": completion.text,
        "tokens": [r.token_id for r in records],
        "chosen": [float(r.logprob) for r in records],
        "top": [list(r.top_token_ids) for r in records],
        "top_logprobs": [[float(lp) for lp in r.top_logprobs] for r in records],
        "gaps": [
            float(r.top_logprobs[0] - r.top_logprobs[1]) if len(r.top_logprobs) >= 2 else None
            for r in records
        ],
    }


# --------------------------------------------------------------------------- #
# end-to-end: one arm, one spawned process
# --------------------------------------------------------------------------- #
def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    try:
        from rapid_llm import SamplingParams
        from rapid_llm.engine.scheduler import Scheduler

        started = time.perf_counter()
        engine = Scheduler.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=BUILD["max_seq_len"],
            max_gpu_num_blocks=BUILD["kv_tokens"],
            max_num_seqs=BUILD["max_num_seqs"],
            use_cuda_graph=spec["graph"],
            tensor_parallel_size=spec["tp"],
            decode_context_parallel_size=spec["dcp"],
            dcp_comm_backend=spec["backend"],
            # 0: one pass per prompt on the prefill kernel (the token budget
            # covers the longest prompt); else chunks of that many tokens.
            enable_chunked_prefill=spec["chunk_size"] > 0,
            max_chunk_size=spec["chunk_size"] or BUILD["max_seq_len"],
            max_num_batched_tokens=max(8192, BUILD["max_seq_len"]),
        )
        build_s = time.perf_counter() - started
        try:
            runner = engine.engine.model_runner
            manager = runner._graph_manager
            buffer = runner.kv_cache_manager.gpu_kv_buffer
            rows = buffer[0].shape[0]
            row_bytes = sum(layer[0].numel() * layer.element_size() for layer in buffer)
            parity = SamplingParams(
                temperature=0.0,
                max_gen_len=spec["parity_gen_len"],
                repetition_penalty=1.0,
                stop_on_repeat=False,
                logprobs=2,
            )
            report: dict[str, Any] = {
                "arm": spec["arm"],
                "tp": spec["tp"],
                "dcp": spec["dcp"],
                "backend": spec["backend"],
                "graph": spec["graph"],
                "build_s": round(build_s, 1),
                "graphs_installed": manager is not None,
                "graph_parity": None if manager is None else manager.parity_error,
                "cache_rows_per_rank": rows,
                "kv_token_capacity": rows * spec["dcp"],
                "kv_bytes_per_token_per_rank": row_bytes / spec["dcp"],
                "parity": [_record(o.outputs[0]) for o in engine.generate(PARITY_PROMPTS, parity)],
                "scenarios": [],
            }
            for scenario in spec["scenarios"]:
                prompts = scenario["prompts"]
                # Same shape as the measured run: autotuned tiles are per shape.
                engine.generate(prompts, sampling_params(8))
                torch.cuda.reset_peak_memory_stats()
                replays = None if manager is None else manager.replays
                run = run_requests(engine, prompts, sampling_params(spec["gen_len"]))
                report["scenarios"].append(
                    {
                        "scenario": scenario["label"],
                        "metrics": _metrics(run),
                        "graph_replays": None if manager is None else manager.replays - replays,
                        "peak_mem_gib": round(peak_mem_gb(), 2),
                    }
                )
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _run(target, spec: dict[str, Any], label: str) -> Any:
    status, payload = run_in_spawned_process(target, spec, _PROBE_TIMEOUT_S)
    if status == "timeout":
        raise SystemExit(f"{label} produced nothing in {_PROBE_TIMEOUT_S:.0f}s")
    if status == "error":
        raise SystemExit(f"{label} failed:\n{payload}")
    return payload


# --------------------------------------------------------------------------- #
# communication: the DCP collectives at decode shape, one rank group
# --------------------------------------------------------------------------- #
def _graph_time_us(step, calls: int, replays: int) -> float:
    """Mean time of one ``step()`` call, captured ``calls`` times into a graph."""
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            step()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * calls)


def _comm_rank(rank: int, spec: dict[str, Any], results) -> None:
    """One rank of a ``tp``-wide group: time the DCP collectives and the decode kernel."""
    try:
        # The ranks leave through ``abandon_parallel`` (graph-captured
        # collectives can wedge ``destroy_process_group``); NCCL's heartbeat
        # monitor would report the store host's exit as a crash.
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        from rapid_llm.distributed import parallel_state as ps
        from rapid_llm.distributed.dcp import merge_dcp_attention
        from rapid_llm.kernels.ops.attention.flashdecoding import flash_decoding

        torch.cuda.set_device(rank)
        tp, dcp = spec["tp"], spec["dcp"]
        ps.init_tensor_parallel(
            rank=rank,
            world_size=tp,
            master_port=spec["port"],
            decode_context_parallel_size=dcp,
            dcp_comm_backend=spec["backend"],
        )
        geometry = spec["geometry"]
        q_heads = geometry["heads"] // tp
        # KV heads split tp / dcp ways; a DCP group shares them.
        kv_heads = max(1, geometry["kv_heads"] * dcp // tp)
        dim = geometry["head_dim"]
        dcp_rank = ps.get_decode_context_parallel_rank()
        scale = dim**-0.5
        rows: list[dict[str, Any]] = []
        for shape in spec["shapes"]:
            batch, context = shape["batch"], shape["context"]
            local = (context - dcp_rank + dcp - 1) // dcp
            q = torch.randn(batch, q_heads, dim, device="cuda", dtype=torch.bfloat16)
            cache = torch.randn(
                batch * local + 1, kv_heads, dim, device="cuda", dtype=torch.bfloat16
            )
            # The engine's table: global positions as columns, this rank's own
            # every ``dcp``-th, read through the same strided view.
            columns = local * dcp
            table = torch.zeros(batch, columns, dtype=torch.int32, device="cuda")
            table[:, dcp_rank::dcp] = 1 + torch.arange(
                batch * local, dtype=torch.int32, device="cuda"
            ).view(batch, local)
            req_idx = torch.arange(batch, dtype=torch.int32, device="cuda")
            local_len = torch.full((batch,), local, dtype=torch.int32, device="cuda")
            entry: dict[str, Any] = dict(shape)
            if dcp > 1:
                group_q = ps.decode_context_parallel_all_gather(q, dim=1)
                outputs = torch.randn_like(group_q)
                lse = torch.randn(batch, group_q.shape[1], device="cuda", dtype=torch.float32)
                gather = functools.partial(ps.decode_context_parallel_all_gather, q, dim=1)
                merge = functools.partial(merge_dcp_attention, outputs, lse)
                kernel = functools.partial(
                    flash_decoding,
                    group_q,
                    cache,
                    cache,
                    scale,
                    table[:, dcp_rank::dcp],
                    req_idx,
                    local_len,
                    local,
                    return_lse=True,
                )
                entry["gather_us"] = _graph_time_us(gather, _COMM_CALLS_PER_GRAPH, _COMM_REPLAYS)
                entry["merge_us"] = _graph_time_us(merge, _COMM_CALLS_PER_GRAPH, _COMM_REPLAYS)
                # Bytes this rank sends per layer on a ring: an all-gather sends
                # its own share to each of ``dcp - 1`` peers (bf16 Q, fp32 LSE); a
                # reduce-scatter / all-to-all sends ``(dcp - 1) / dcp`` of its
                # input (fp32-weighted output under ag_rs, bf16 output + LSE under a2a).
                q_bytes = q.numel() * q.element_size()
                lse_bytes = lse.numel() * 4
                if spec["backend"] == "ag_rs":
                    merge_bytes = lse_bytes * (dcp - 1) + outputs.numel() * 4 * (dcp - 1) / dcp
                else:
                    merge_bytes = (outputs.numel() * 2 + lse_bytes) * (dcp - 1) / dcp
                entry["wire_bytes"] = int(q_bytes * (dcp - 1) + merge_bytes)
            else:
                kernel = functools.partial(
                    flash_decoding, q, cache, cache, scale, table, req_idx, local_len, local
                )
                entry["gather_us"] = entry["merge_us"] = 0.0
                entry["wire_bytes"] = 0
            entry["kernel_us"] = _graph_time_us(kernel, _COMM_CALLS_PER_GRAPH, _COMM_REPLAYS)
            rows.append(entry)
        if rank == 0:
            results.put(("ok", rows))
        ps.abandon_parallel()
    except Exception:
        results.put(("error", f"rank {rank}:\n{traceback.format_exc()}"))


def _comm_probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Launch the ``tp`` ranks of one group and forward rank 0's measurements."""
    try:
        context = mp.get_context("spawn")
        inner = context.Queue()
        procs = [
            context.Process(target=_comm_rank, args=(rank, spec, inner), daemon=False)
            for rank in range(spec["tp"])
        ]
        for proc in procs:
            proc.start()
        status, payload = inner.get(timeout=_PROBE_TIMEOUT_S)
        for proc in procs:
            proc.join(timeout=60.0)
            if proc.is_alive():  # pragma: no cover - a communicator that will not die
                proc.terminate()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put((status, payload))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _is_mutual_tie(want: dict, got: dict, step: int) -> bool:
    """Both decided ``step`` within the tie gap, by swapping the same two candidates."""
    if step >= min(len(want["tokens"]), len(got["tokens"])):
        return False
    if max(want["gaps"][step] or 0.0, got["gaps"][step] or 0.0) > _TIE_GAP:
        return False
    priced = dict(zip(got["top"][step], got["top_logprobs"][step], strict=True))
    ours = want["tokens"][step]
    return (
        ours in priced
        and got["tokens"][step] in want["top"][step]
        and abs(priced[ours] - want["chosen"][step]) <= _TIE_GAP
    )


def _parity_vs(reference: list[dict], other: list[dict]) -> dict[str, Any]:
    """Tie-licensed forks and chosen-token logprob drift of one arm against ``tp1``."""
    same = total = 0
    worst_margin: float | None = None
    first_fork: int | None = None
    licensed = True
    drift = 0.0
    for want, got in zip(reference, other, strict=True):
        step = next(
            (
                i
                for i, (a, b) in enumerate(zip(want["tokens"], got["tokens"], strict=False))
                if a != b
            ),
            min(len(want["tokens"]), len(got["tokens"])),
        )
        for i in range(step):
            drift = max(drift, abs(want["chosen"][i] - got["chosen"][i]))
        same += step
        total += len(want["tokens"])
        if step == len(want["tokens"]) == len(got["tokens"]):
            continue
        first_fork = step if first_fork is None else min(first_fork, step)
        licensed = licensed and _is_mutual_tie(want, got, step)
        margin = want["gaps"][step] if step < len(want["gaps"]) else None
        if margin is not None:
            worst_margin = margin if worst_margin is None else max(worst_margin, margin)
    agreement = same / total if total else 1.0
    verdict = "identical" if first_fork is None else "tie-licensed" if licensed else "SUSPECT"
    if drift > _TIE_GAP:
        verdict = "SUSPECT (drift)"
    return {
        "first_fork": first_fork,
        "worst_fork_margin": worst_margin,
        "agreement": round(agreement, 4),
        "max_logprob_drift": round(drift, 4),
        "verdict": verdict,
    }


def _entry(report: dict[str, Any], scenario: str) -> dict[str, Any]:
    return next(e for e in report["scenarios"] if e["scenario"] == scenario)


def _comm_row(comm: dict[str, list], arm: str, scenario: dict[str, Any]) -> dict | None:
    for row in comm.get(arm) or []:
        if row["batch"] == scenario["batch"] and row["context"] == scenario["context"]:
            return row
    return None


def _print(reports: list[dict], scenarios: list[dict], comm: dict, layers: int) -> None:
    by_arm = {r["arm"]: r for r in reports}
    tp1 = by_arm.get("tp1")
    for scenario in scenarios:
        label, tokens = scenario["label"], scenario["prompt_stats"]
        print(
            f"\nscenario {label}: batch {scenario['batch']}, prompt "
            f"{tokens['min']}-{tokens['mean']}-{tokens['max']} tok (min-mean-max), "
            f"gen {scenario['gen_len']}, offline"
        )
        base = _entry(tp1, label)["metrics"]["tps"] if tp1 else 0.0
        rows = []
        for report in reports:
            entry = _entry(report, label)
            metrics, width = entry["metrics"], report["tp"]
            peer = by_arm.get(f"tp{width}")
            peer_tpot = _entry(peer, label)["metrics"]["tpot_ms"] if peer else 0.0
            rows.append(
                [
                    report["arm"],
                    str(width),
                    f"{metrics['ttft_ms']:.0f}",
                    f"{metrics['tpot_ms']:.2f}",
                    f"{metrics['tps']:.0f}",
                    f"{metrics['tps'] / width:.0f}",
                    f"{metrics['tps'] / (base * width):.1%}" if base else "-",
                    f"{metrics['tpot_ms'] / peer_tpot:.2f}x"
                    if peer_tpot and report["dcp"] > 1
                    else "-",
                    f"{entry['peak_mem_gib']:.1f}",
                    "eager" if entry["graph_replays"] is None else str(entry["graph_replays"]),
                ]
            )
        print_row_table(
            [
                "config",
                "GPUs",
                "TTFT ms",
                "TPOT ms",
                "TPS",
                "TGS",
                "eff vs tp1",
                "TPOT/same-TP",
                "peak GiB",
                "replays",
            ],
            [14, 5, 9, 9, 8, 7, 11, 13, 9, 8],
            rows,
        )
        comm_rows = []
        for report in reports:
            row = _comm_row(comm, report["arm"], scenario)
            if row is None:
                continue
            step_ms = (row["gather_us"] + row["merge_us"]) * layers / 1000.0
            tpot = _entry(report, label)["metrics"]["tpot_ms"]
            comm_rows.append(
                [
                    report["arm"],
                    f"{row['kernel_us']:.1f}",
                    f"{row['gather_us']:.1f}",
                    f"{row['merge_us']:.1f}",
                    f"{row['wire_bytes'] / 1024:.1f}",
                    f"{step_ms:.3f}",
                    f"{step_ms / tpot:.1%}" if tpot else "-",
                ]
            )
        if comm_rows:
            print(
                f"  communication at decode shape (batch {scenario['batch']}, history "
                f"{scenario['context']}), per layer, CUDA-graph microbenchmark"
            )
            print_row_table(
                [
                    "config",
                    "attn kernel us",
                    "Q gather us",
                    "merge us",
                    "sent KiB",
                    f"comm/step ms (x{layers})",
                    "of TPOT",
                ],
                [14, 15, 12, 10, 10, 22, 9],
                comm_rows,
            )

    print(
        "\nKV cache (same row count per rank in every arm; under GQA a DCP row holds"
        " dcp x the heads, so bytes/token match the TP arm of the same width)"
    )
    print_row_table(
        ["config", "rows/rank", "token capacity", "KV bytes/token/rank", "graph parity"],
        [14, 11, 16, 21, 14],
        [
            [
                r["arm"],
                str(r["cache_rows_per_rank"]),
                str(r["kv_token_capacity"]),
                f"{r['kv_bytes_per_token_per_rank']:.0f}",
                "-" if r["graph_parity"] is None else f"{r['graph_parity']:.1e}",
            ]
            for r in reports
        ],
    )

    if tp1:
        print(f"\nparity vs tp1 ({len(PARITY_PROMPTS)} prompts, greedy, tie gap {_TIE_GAP} nat)")
        print_row_table(
            ["config", "first fork", "worst margin", "agreement", "max |dlogprob|", "verdict"],
            [14, 11, 13, 10, 15, 20],
            [
                [
                    r["arm"],
                    "none" if p["first_fork"] is None else f"step {p['first_fork']}",
                    "-" if p["worst_fork_margin"] is None else f"{p['worst_fork_margin']:.3f}",
                    f"{p['agreement']:.1%}",
                    f"{p['max_logprob_drift']:.4f}",
                    p["verdict"],
                ]
                for r in reports
                if (p := r.get("parity_vs_tp1"))
            ],
        )
    print(
        "\nTGS = TPS / GPUs; eff vs tp1 = TPS / (tp1 TPS x GPUs); TPOT/same-TP = this arm's "
        "TPOT over the plain TP arm of the same width. comm/step = (Q gather + merge) x "
        "layers: what DCP adds on top of the all-reduces TP pays in both arms."
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Decode context parallelism vs TP")
    parser.add_argument("--model", required=True, help="dense GQA checkpoint directory")
    parser.add_argument("--arms", default=",".join(arm[0] for arm in ARMS))
    parser.add_argument("--scenarios", default=",".join(s["label"] for s in SCENARIOS))
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--parity-gen-len", type=int, default=32)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="prefill chunk tokens; 0 prefills each prompt in one pass",
    )
    parser.add_argument("--eager", action="store_true", help="run every arm without graphs")
    parser.add_argument("--no-comm", action="store_true", help="skip the collective section")
    parser.add_argument("--log-dir", default="docs/benchmark_logs/parallel")
    args = parser.parse_args()

    known = {arm[0]: arm for arm in ARMS}
    arm_names = [name for name in args.arms.split(",") if name]
    if unknown := [name for name in arm_names if name not in known]:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(known)}")
    scenario_names = [name for name in args.scenarios.split(",") if name]
    scenarios = [dict(s) for s in SCENARIOS if s["label"] in scenario_names]
    if len(scenarios) != len(scenario_names):
        raise SystemExit(f"unknown scenario in {scenario_names}")
    arms = [known[name] for name in arm_names]
    require_gpus(max(tp for _, tp, _, _ in arms))

    from transformers import AutoTokenizer

    config = json.loads((Path(args.model) / "config.json").read_text())
    heads = config["num_attention_heads"]
    geometry = {
        "heads": heads,
        "kv_heads": config.get("num_key_value_heads", heads),
        "head_dim": config.get("head_dim") or config["hidden_size"] // heads,
    }
    layers = config["num_hidden_layers"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for scenario in scenarios:
        scenario["prompts"] = [
            _synthetic_prompt(tokenizer, scenario["prompt_tokens"], seed=1000 + index)
            for index in range(scenario["batch"])
        ]
        scenario["prompt_stats"] = _prompt_stats(tokenizer, scenario["prompts"])
        scenario["gen_len"] = args.gen_len
        # Mid-generation history: the decode shape the comm section times.
        scenario["context"] = scenario["prompt_tokens"] + args.gen_len // 2

    print_run_header(
        args.model,
        {
            "mode": "offline (all prompts submitted at once)",
            "arms": " ".join(arm_names),
            "graphs": "off (--eager)" if args.eager else "on (decode histories <= 4096)",
            "prefill": f"chunks of {args.chunk_size}" if args.chunk_size else "one pass per prompt",
            "gen_len": args.gen_len,
            "geometry": f"{geometry['heads']} q / {geometry['kv_heads']} kv heads x "
            f"{geometry['head_dim']}, {layers} layers",
            "engine": f"max_seq {BUILD['max_seq_len']}, max_num_seqs {BUILD['max_num_seqs']}, "
            f"{BUILD['kv_tokens']} cache rows/rank",
        },
    )

    reports = []
    for label, tp, dcp, backend in arms:
        spec = {
            "arm": label,
            "model": args.model,
            "tp": tp,
            "dcp": dcp,
            "backend": backend,
            "graph": not args.eager,
            "chunk_size": args.chunk_size,
            "gen_len": args.gen_len,
            "parity_gen_len": args.parity_gen_len,
            "scenarios": [{"label": s["label"], "prompts": s["prompts"]} for s in scenarios],
        }
        print(f"\n[{label}] tp={tp} dcp={dcp} backend={backend} - building", flush=True)
        started = time.perf_counter()
        reports.append(_run(_probe, spec, label))
        print(f"[{label}] finished in {time.perf_counter() - started:.0f}s", flush=True)
        time.sleep(2.0)  # let the previous probe's GPUs go before the next build

    comm: dict[str, list] = {}
    if not args.no_comm:
        shapes = [{"batch": s["batch"], "context": s["context"]} for s in scenarios]
        for index, (label, tp, dcp, backend) in enumerate(arms):
            if tp == 1:
                continue
            spec = {
                "tp": tp,
                "dcp": dcp,
                "backend": backend,
                "port": 29650 + index,
                "geometry": geometry,
                "shapes": shapes,
            }
            print(f"[comm {label}] timing", flush=True)
            comm[label] = _run(_comm_probe, spec, f"comm {label}")
            time.sleep(2.0)

    reference = next((r["parity"] for r in reports if r["arm"] == "tp1"), None)
    for report in reports:
        report["comm"] = comm.get(report["arm"])
        if reference is not None and report["arm"] != "tp1":
            report["parity_vs_tp1"] = _parity_vs(reference, report["parity"])

    _print(reports, scenarios, comm, layers)

    if args.log_dir:
        path = timestamped_log_path(args.log_dir, f"context_parallel_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "mode": "offline (all prompts submitted at once, no serving queue)",
                "graphs": not args.eager,
                "chunk_size": args.chunk_size,
                "build": BUILD,
                "geometry": geometry,
                "layers": layers,
                "arms": {name: {"tp": tp, "dcp": dcp, "backend": b} for name, tp, dcp, b in arms},
                "scenarios": [{k: v for k, v in s.items() if k != "prompts"} for s in scenarios],
                "tie_gap_nats": _TIE_GAP,
                "metric_basis": "per-request TTFT/TPOT; comm from a CUDA-graph microbenchmark "
                "of the DCP collectives at each scenario's mid-generation decode shape",
            },
            reports,
        )


if __name__ == "__main__":
    main()
