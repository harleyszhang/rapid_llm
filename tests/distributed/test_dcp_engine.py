"""End-to-end decode context parallelism: sharding the history must not move the answer.

Under DCP every rank of an attention-TP group keeps the same KV heads and a
``1/dcp`` share of the positions (``p % dcp``); each decodes against its share
and the partial outputs are merged through their log-sum-exp. That merge is an
identity of the softmax, so a DCP engine is held against the plain TP engine of
the same width -- the one variable is where the history lives.

Design follows ``test_tp_engine``: each configuration is a spawned, non-daemonic
*probe* (it spawns the TP followers itself) that builds one engine, answers a
fixed prompt set greedily with ``logprobs=2``, and reports. A divergence must
account for itself, from both sides: at the step two engines first disagree,
each must have decided by at most :data:`_TIE_GAP`, each must rank the other's
token as its runner-up, and both must price the reference's token alike --
a reordered sum can only swap two candidates the weights left level. Before
that step the chosen tokens' logprobs must agree to the same bound, so an
engine degenerated into noise cannot pass one tie at a time. A wrong rank
offset, a dropped share or an unmasked filler row moves logprobs by whole nats
instead.

What is compared, and why each needs its own probe:

* **DCP vs TP** at TP=2 (both checkpoints) and TP=4 (``dcp`` 2 and 4, where
  ``dcp < tp`` puts two KV shards in each DCP group).
* **MLA** (DCP's capacity case: the latent has no head axis, so TP replicates
  it and DCP is the only cut) on a synthetic DeepSeek-V2 checkpoint, eager and
  graphed, since no MLA checkpoint ships with the repo.
* **a2a vs ag_rs**: the two merge backends move different tensors through
  different collectives; both are held against TP.
* **graph vs eager** under DCP: the merge collectives are recorded into the
  decode graph and the replay serves. Not byte-identical to eager; see
  :func:`test_dcp_decode_graphs_install_and_replay_up_to_ties`.
* **a prefix-cache hit** under DCP resumes from a block boundary with every
  resumed chunk on the EXTEND path (the chunked kernel cannot read a strided
  share), and must answer what the cold prompt did.

The long prompt is chunked (:data:`_MAX_CHUNK`) so the continuation chunks run,
and spans many blocks so every rank owns history in every block.

Usage:
    pytest tests/distributed/test_dcp_engine.py    # skips below 2 GPUs
"""

from __future__ import annotations

import json
import queue as queue_module
import shutil
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from rapid_llm import SamplingParams
from rapid_llm.executor.cuda_graph import TP_GRAPH_PARITY_ATOL
from tests.conftest import checkpoint_candidates, checkpoint_problem
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

#: Cache rows per rank, sequence cap and slots: room for the long prompt and a
#: full batch beside a follower's copy of the weights.
_KV_TOKENS = 4096
_MAX_SEQ_LEN = 512
_MAX_NUM_SEQS = 8

#: Small enough that the long prompt prefills in three chunks.
_MAX_CHUNK = 128

#: Checkpoint load, KV profile, rendezvous and (graph probes) a capture grid.
_PROBE_TIMEOUT_S = 900.0

#: See ``test_tp_engine``: logprobs are the instrument that measures a tie.
_GREEDY = SamplingParams(
    temperature=0.0,
    max_gen_len=24,
    repetition_penalty=1.0,
    stop_on_repeat=False,
    logprobs=2,
)

#: Margin (nats) below which a differently ordered sum may take a step either
#: way; the bound ``test_tp_engine`` measured for a reassociated all-reduce. The
#: LSE merge is the same kind of reassociation, of the softmax denominator.
_TIE_GAP = 0.5

#: ``(case, prompts)``: one row, unequal rows in one prefill, a full batch, and a
#: chunked prompt whose history spans 17+ blocks.
_LONG = "The history of the Roman Empire spans many centuries, and " * 25
_CASES: list[tuple[str, list[str]]] = [
    ("single", ["The capital of France is"]),
    ("mixed", ["Hi", "The history of the Roman Empire spans many centuries, and"]),
    (
        "batch6",
        [
            "One plus one equals",
            "The sun rises in the",
            "Water boils at",
            "Machine learning is",
            "The largest planet is",
            "Python is a language that",
        ],
    ),
    ("long", [_LONG]),
]
_GROUPINGS = ("batched", "alone")

#: A two-layer dense DeepSeek-V2 (MLA, V2-Lite's ``q_lora_rank=None``) at test
#: size, on the default checkpoint's vocabulary so its tokenizer serves. Four
#: query heads split two per TP rank; the 64 + 64 latent row is what each rank
#: caches.
_SYNTHETIC_MLA: dict[str, Any] = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 2,
    "kv_lora_rank": 64,
    "q_lora_rank": None,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "initializer_range": 0.1,
}

#: Random weights at the default 0.02 give flat attention and flat logits, where
#: every step is a tie and the comparison proves nothing. At 0.1 with the head
#: scaled 5x the median step is decided by 1.25 nats, and perturbing only the
#: latent projection by 10% fails the comparison at the first token.
_SYNTHETIC_HEAD_SCALE = 5.0


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Build one engine, answer every case, report answers, graph and cache facts."""
    try:
        from rapid_llm.engine.scheduler import Scheduler

        engine = Scheduler.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_TOKENS,
            max_num_seqs=_MAX_NUM_SEQS,
            max_chunk_size=_MAX_CHUNK,
            use_cuda_graph=spec["graph"],
            tensor_parallel_size=spec["tp"],
            decode_context_parallel_size=spec["dcp"],
            dcp_comm_backend=spec["backend"],
            enable_prefix_cache=spec["prefix_cache"],
            kv_cache_dtype=spec["kv_cache_dtype"],
        )
        try:
            runner = engine.engine.model_runner
            manager = runner._graph_manager
            slots = engine._executor._worker._slot_batch
            report: dict[str, Any] = {
                "kv_row": list(runner.kv_row),
                "kv_dtype": str(runner.kv_cache_manager.gpu_kv_buffer[0].dtype),
                "cache_rows": runner.kv_cache_manager.gpu_num_blocks,
                "kv_blocks": engine.num_kv_blocks,
                "block_size": slots.block_size,
                "rows_per_block": slots.rows_per_block,
                "installed": manager is not None,
                "parity": None if manager is None else manager.parity_error,
            }
            if spec["prefix_cache"]:
                # Cold, then the same prompt again: the second resumes from the
                # blocks the first committed.
                report["cold"] = _answer(engine, [_LONG])[0]
                report["hit"] = _answer(engine, [_LONG])[0]
            else:
                report["batched"] = {name: _answer(engine, prompts) for name, prompts in _CASES}
                report["alone"] = {
                    name: [_answer(engine, [prompt])[0] for prompt in prompts]
                    for name, prompts in _CASES
                }
            report["replays"] = 0 if manager is None else manager.replays
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _answer(engine, prompts: list[str]) -> list[dict[str, Any]]:
    return [_record(output.outputs[0]) for output in engine.generate(prompts, _GREEDY)]


def _record(completion) -> dict[str, Any]:
    """Text, token ids, their logprobs, the top-2 per step and the chosen token's margin."""
    records = completion.logprobs or ()
    return {
        "text": completion.text,
        "tokens": [record.token_id for record in records],
        "logprobs": [float(record.logprob) for record in records],
        "top": [list(record.top_token_ids) for record in records],
        "top_logprobs": [[float(lp) for lp in record.top_logprobs] for record in records],
        "gaps": [
            float(r.top_logprobs[0] - r.top_logprobs[1])
            if len(r.top_logprobs) >= 2
            else float("inf")
            for r in records
        ],
    }


def _run_probe(
    model: Path,
    *,
    tp: int,
    dcp: int = 1,
    backend: str = "ag_rs",
    graph: bool = False,
    prefix_cache: bool = False,
    kv_cache_dtype: str = "auto",
) -> dict[str, Any]:
    """Run one probe to completion, surfacing its traceback as this test's failure."""
    spec = {
        "model": str(model),
        "tp": tp,
        "dcp": dcp,
        "backend": backend,
        "graph": graph,
        "prefix_cache": prefix_cache,
        "kv_cache_dtype": kv_cache_dtype,
    }
    context = mp.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=_probe, args=(spec, results), daemon=False)
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_PROBE_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(f"probe {spec} produced nothing in {_PROBE_TIMEOUT_S:.0f}s")
        if status == "error":
            pytest.fail(f"probe {spec} failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()


def _named_checkpoint(name: str) -> Path:
    """The checkpoint called ``name``, never the conftest's fallback substitute."""
    path = checkpoint_candidates(f"my_weight/{name}")[0]
    if path.name != name or checkpoint_problem(path) is not None:
        pytest.skip(f"needs the {name} checkpoint")
    return path


@pytest.fixture(scope="module", params=["default", "Qwen3-0.6B"])
def tp2(request, model_dir: Path) -> dict[str, dict[str, Any]]:
    """TP=2 reports per configuration, for the default checkpoint and for Qwen3.

    The two cover different head geometry: Qwen2.5-0.5B has 2 KV heads (one per
    TP rank, so a DCP group holds both), Qwen3-0.6B has 8.
    """
    model = model_dir if request.param == "default" else _named_checkpoint(request.param)
    return {
        "tp": _run_probe(model, tp=2),
        "dcp": _run_probe(model, tp=2, dcp=2),
        "a2a": _run_probe(model, tp=2, dcp=2, backend="a2a"),
        "graph": _run_probe(model, tp=2, dcp=2, graph=True),
    }


@pytest.fixture(scope="module")
def tp4() -> dict[str, dict[str, Any]]:
    """TP=4 on Qwen3-0.6B (8 KV heads; the default checkpoint's 14 query heads do not split 4 ways)."""
    model = _named_checkpoint("Qwen3-0.6B")
    return {
        "tp": _run_probe(model, tp=4),
        "dcp2": _run_probe(model, tp=4, dcp=2),
        "dcp4": _run_probe(model, tp=4, dcp=4),
    }


def _write_synthetic_mla(path: Path, tokenizer_dir: Path) -> None:
    """Seeded HF DeepSeek-V2 weights in bf16, beside ``tokenizer_dir``'s tokenizer."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, DeepseekV2Config, DeepseekV2ForCausalLM

    tokenizer_config = AutoConfig.from_pretrained(tokenizer_dir)
    body = dict(
        _SYNTHETIC_MLA,
        vocab_size=tokenizer_config.vocab_size,
        bos_token_id=tokenizer_config.bos_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
    )
    (path / "config.json").write_text(json.dumps(body))
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(DeepseekV2Config(**body)).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(_SYNTHETIC_HEAD_SCALE)
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (tokenizer_dir / name).exists():
            shutil.copy(tokenizer_dir / name, path / name)


@pytest.fixture(scope="module")
def mla(tmp_path_factory, model_dir: Path) -> dict[str, dict[str, Any]]:
    """TP=2 reports on the synthetic MLA checkpoint: plain, DCP eager, DCP graphed."""
    pytest.importorskip("transformers.models.deepseek_v2")
    path = tmp_path_factory.mktemp("synthetic_mla")
    _write_synthetic_mla(path, model_dir)
    return {
        "tp": _run_probe(path, tp=2),
        "dcp": _run_probe(path, tp=2, dcp=2),
        "graph": _run_probe(path, tp=2, dcp=2, graph=True),
    }


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _entries() -> list[tuple[str, int]]:
    return [(name, index) for name, prompts in _CASES for index in range(len(prompts))]


def _first_token_difference(one: list[int], two: list[int]) -> int:
    for step, (a, b) in enumerate(zip(one, two, strict=False)):
        if a != b:
            return step
    return min(len(one), len(two))


def _logprob_of(record: dict[str, Any], step: int, token: int) -> float | None:
    """What ``record`` priced ``token`` at ``step``, if it is among its top-2."""
    for candidate, logprob in zip(record["top"][step], record["top_logprobs"][step], strict=True):
        if candidate == token:
            return logprob
    return None


def _check_pair(want: dict[str, Any], got: dict[str, Any]) -> tuple[int, float, str | None]:
    """``(fork, prefix drift, problem)`` for one answer against its reference.

    ``problem`` is ``None`` when the answers agree, or fork on a swap of two
    candidates both engines left within :data:`_TIE_GAP` and priced alike.
    """
    fork = _first_token_difference(want["tokens"], got["tokens"])
    drift = max(
        (abs(a - b) for a, b in zip(want["logprobs"][:fork], got["logprobs"][:fork], strict=True)),
        default=0.0,
    )
    if drift > _TIE_GAP:
        return fork, drift, f"chosen logprobs drift {drift:.3f} before any fork"
    if fork == len(want["tokens"]) == len(got["tokens"]):
        return fork, drift, None
    if fork >= min(len(want["tokens"]), len(got["tokens"])):
        return fork, drift, f"one answer stops at token {fork}, the other continues"
    ours, theirs = want["tokens"][fork], got["tokens"][fork]
    ours_there, theirs_here = _logprob_of(got, fork, ours), _logprob_of(want, fork, theirs)
    if want["gaps"][fork] > _TIE_GAP or got["gaps"][fork] > _TIE_GAP:
        return (
            fork,
            drift,
            (
                f"fork at token {fork} decided by {want['gaps'][fork]:.3f} (ref) and "
                f"{got['gaps'][fork]:.3f} (got)"
            ),
        )
    if ours_there is None or theirs_here is None:
        return fork, drift, f"fork at token {fork} is not a swap of the top-2"
    if abs(ours_there - want["logprobs"][fork]) > _TIE_GAP:
        return (
            fork,
            drift,
            (
                f"fork at token {fork}: the engines price the reference's token at "
                f"{want['logprobs'][fork]:.3f} and {ours_there:.3f}"
            ),
        )
    return fork, drift, None


def _assert_only_ties_diverge(ref: dict[str, Any], other: dict[str, Any], label: str) -> None:
    """Every answer agrees with the reference or forks on a mutual tie (module docstring)."""
    agreed = total = 0
    worst_drift = 0.0
    forks: list[str] = []
    for name, index in _entries():
        for grouping in _GROUPINGS:
            want, got = ref[grouping][name][index], other[grouping][name][index]
            assert got["text"].strip(), f"{label}: {grouping} {name}/{index} came back empty"
            fork, drift, problem = _check_pair(want, got)
            agreed += fork
            total += len(want["tokens"])
            worst_drift = max(worst_drift, drift)
            assert problem is None, (
                f"{label}: {grouping} case {name!r} prompt {index}: {problem}\n"
                f"  ref: {want['text']!r}\n  got: {got['text']!r}"
            )
            if fork < len(want["tokens"]):
                forks.append(f"{grouping} {name}/{index}@{fork} (gap {want['gaps'][fork]:.3f})")
    print(
        f"\n{label}: identical tokens {agreed}/{total}, prefix logprob drift "
        f"<= {worst_drift:.4f}; forks: {forks or 'none'}"
    )


# --------------------------------------------------------------------------- #
# Numerics: DCP answers what TP answers
# --------------------------------------------------------------------------- #
@needs_gpus(2)
@pytest.mark.parametrize("arm", ["dcp", "a2a"], ids=["ag_rs", "a2a"])
def test_dcp2_only_diverges_from_tp2_on_ties(tp2, arm):
    """Both merge backends reproduce TP=2 up to ties, in both batch shapes."""
    _assert_only_ties_diverge(tp2["tp"], tp2[arm], f"tp2 vs tp2+dcp2[{arm}]")


@needs_gpus(4)
@pytest.mark.parametrize("arm", ["dcp2", "dcp4"])
def test_dcp_on_four_ranks_only_diverges_from_tp4_on_ties(tp4, arm):
    """``dcp < tp`` (two KV shards per DCP group) and ``dcp == tp`` (one)."""
    _assert_only_ties_diverge(tp4["tp"], tp4[arm], f"tp4 vs tp4+{arm}")


@needs_gpus(2)
@pytest.mark.parametrize("arm", ["dcp", "graph"], ids=["eager", "graph"])
def test_mla_dcp2_only_diverges_from_tp2_on_ties(mla, arm):
    """The latent decode's strided share and merge, eager and captured."""
    if arm == "graph":
        graph = mla["graph"]
        assert graph["installed"] and graph["replays"] > 0
        assert graph["parity"] is not None and graph["parity"] <= TP_GRAPH_PARITY_ATOL
    _assert_only_ties_diverge(mla["tp"], mla[arm], f"mla tp2 vs tp2+dcp2[{arm}]")


@needs_gpus(2)
def test_an_mla_dcp_rank_caches_half_the_bytes_per_token(mla):
    """Where DCP pays: TP replicates the latent row, so DCP's 1/dcp position
    share is a straight 1/dcp of the bytes -- twice the tokens in the same rows."""
    tp, dcp = mla["tp"], mla["dcp"]
    assert dcp["kv_row"] == tp["kv_row"]
    assert dcp["rows_per_block"] * 2 == tp["rows_per_block"]
    assert dcp["cache_rows"] == tp["cache_rows"]
    assert dcp["kv_blocks"] == 2 * tp["kv_blocks"]


@needs_gpus(2)
def test_dcp_decode_graphs_install_and_replay_up_to_ties(tp2):
    """The merge collectives capture: graphs pass the parity gate, serve, and match eager.

    Up to ties, not bytes (unlike ``test_tp_cuda_graph``): under DCP every
    resumed chunk runs EXTEND, which replays the decode graph one row per
    token, and ``flash_decoding`` sizes its KV partitions from
    ``max_actual_seq_len``. Eager passes the step's real maximum, a graph the
    one its bucket baked in, so the two split a long history differently and
    sum the partials in a different order. Measured on Qwen3-0.6B: the long
    prompt's first token flips on a 0.0625-nat tie.
    """
    graph, eager = tp2["graph"], tp2["dcp"]
    assert graph["installed"], "DCP captured no decode graphs (or the parity gate retired them)"
    assert graph["parity"] is not None and graph["parity"] <= TP_GRAPH_PARITY_ATOL
    assert graph["replays"] > 0, "graphs were installed but no decode step replayed one"
    assert not eager["installed"]
    _assert_only_ties_diverge(eager, graph, "tp2+dcp2 eager vs graph")


@needs_gpus(2)
def test_a_prefix_cache_hit_under_dcp_answers_what_the_cold_prompt_did(model_dir):
    """The hit resumes mid-prompt; its chunks all extend over the strided shares.

    The cold pass prefilled its first chunk with the prefill kernel and the hit
    re-reads those rows through EXTEND, so a tie may flip -- nothing else may.
    """
    report = _run_probe(model_dir, tp=2, dcp=2, prefix_cache=True)
    cold, hit = report["cold"], report["hit"]
    _, _, problem = _check_pair(cold, hit)
    assert problem is None, f"{problem}\n  cold: {cold['text']!r}\n  hit:  {hit['text']!r}"


@needs_gpus(2)
def test_an_fp8_cache_under_dcp_only_diverges_from_tp2_on_ties():
    """The e4m3 rows and their scales travel through the strided share too.

    Both arms store fp8, so the comparison isolates DCP from the cache's own
    rounding.
    """
    model = _named_checkpoint("Qwen3-0.6B")
    tp = _run_probe(model, tp=2, kv_cache_dtype="fp8")
    dcp = _run_probe(model, tp=2, dcp=2, kv_cache_dtype="fp8")
    assert tp["kv_dtype"] == dcp["kv_dtype"] == "torch.uint8"
    _assert_only_ties_diverge(tp, dcp, "fp8 cache: tp2 vs tp2+dcp2")


# --------------------------------------------------------------------------- #
# Layout: what each rank actually holds
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_a_dcp_rank_caches_dcp_times_the_heads_for_a_dcp_th_of_the_positions(tp2):
    """The geometry behind the capacity claim, stated as it is measured.

    A DCP group shares its KV heads, so each row is ``dcp`` times wider, and a
    16-position block takes ``16 / dcp`` rows. Per token per rank the bytes are
    therefore *equal* to plain TP whenever TP already split the KV heads (both
    checkpoints here): DCP only saves memory where TP has to replicate KV heads
    (``tp > num_kv_heads``) or there are none to split (MLA).
    """
    tp, dcp = tp2["tp"], tp2["dcp"]
    assert dcp["kv_row"][0] == 2 * tp["kv_row"][0]
    assert dcp["kv_row"][1] == tp["kv_row"][1]
    assert dcp["rows_per_block"] * 2 == dcp["block_size"] == tp["rows_per_block"]
    # Same rows per rank => twice the token blocks, at the same bytes per token.
    assert dcp["cache_rows"] == tp["cache_rows"]
    assert dcp["kv_blocks"] == 2 * tp["kv_blocks"]
    bytes_per_token = {
        arm: r["kv_row"][0] * r["kv_row"][1] * r["rows_per_block"] / r["block_size"]
        for arm, r in (("tp", tp), ("dcp", dcp))
    }
    assert bytes_per_token["tp"] == bytes_per_token["dcp"]
