"""Shared plumbing for the per-dataset accuracy benches.

Each dataset directory under ``benchmarks/eval/`` holds two arms:
``bench_rapid_vllm.py`` drives the rapid_llm engine (CUDA graphs on by default
— the optimised mode) and ``bench_hf.py`` drives transformers as the baseline.
Both arms build prompts and score completions through the same dataset-local
``common.py``, so the two accuracies differ by engine only. This module holds
what every arm would otherwise repeat: checkpoint resolution, the evidence log
append, HF model loading, batched HF generation, and token-level continuation
log-likelihoods — the HellaSwag scoring primitive, implemented once per engine
so the tokenisation boundary is defined in exactly one place per side.

Usage:
    from benchmarks.eval._common import resolve_model_dir, append_result_log
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - keeps rapid_llm off the module import path
    from rapid_llm import LLM

#: Repository root, so the benches run both as ``python -m`` and as plain files.
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Shared model zoo consulted when a named checkpoint is not at the given path.
#: Same convention as ``tests/conftest.py``'s ``RAPID_LLM_MODELZOO``; the
#: default matches the machine this suite was baselined on.
DEFAULT_MODELZOO = "/data/shared/llm_weights"

#: Evidence JSONs live under git (``docs/benchmark_logs/``), unlike the
#: ``benchmarks/logs/`` scratch dir which .gitignore excludes.
LOG_DIR = REPO_ROOT / "docs" / "benchmark_logs" / "accuracy"

_DOWNLOAD_TIMEOUT_S = 120


def resolve_model_dir(reference: str) -> Path:
    """The checkpoint a CLI named, as a directory that exists.

    A path that already resolves is used as-is; a bare name is then looked up
    under the model zoo and ``my_weight/``, in that order. Failing with the
    list of places tried beats failing with a bare ``FileNotFoundError`` — the
    usual cause is a zoo path this machine does not mount.
    """
    path = Path(reference).expanduser()
    tried = [path if path.is_absolute() else REPO_ROOT / path]
    if tried[0].is_dir():
        return tried[0]

    zoo = os.environ.get("RAPID_LLM_MODELZOO", DEFAULT_MODELZOO)
    for extra in (Path(zoo).expanduser() / path.name, REPO_ROOT / "my_weight" / path.name):
        tried.append(extra)
        if extra.is_dir():
            return extra
    raise FileNotFoundError(f"checkpoint {reference!r} not found; tried: {[str(t) for t in tried]}")


def append_result_log(dataset: str, engine: str, payload: dict) -> Path:
    """Write one run's numbers to ``docs/benchmark_logs/accuracy/`` and return the path.

    The file name carries the dataset, the engine and a timestamp, matching the
    directory's existing convention; the doc tables cite these files, so a
    number in ``docs/eval_models.md`` always has a JSON behind it.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"accuracy_{dataset}_{engine}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def fetch_jsonl(url: str, subdir: str, name: str) -> Path:
    """Download ``url`` into the eval cache as ``subdir/name`` unless cached.

    Mirrors :func:`tests.evals.dataset.fetch` but takes a full URL — the
    benchmark's sources (HellaSwag's GitHub, SuperGLUE's fbaipublicfiles) do
    not share GSM8K's base URL, and ``tests/evals`` should not grow knobs for
    benchmarks it does not run.
    """
    target = Path.home() / ".cache" / "rapid_llm" / "evals" / subdir / name
    if target.is_file() and target.stat().st_size > 0:
        return target

    import requests

    try:
        response = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".part")
        with open(staging, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        staging.replace(target)
    except Exception as exc:  # network, DNS, HTTP status, disk
        raise RuntimeError(
            f"cannot obtain {name!r} from {url}: {type(exc).__name__}: {exc}\n"
            f"Pre-seed the cache at {target}."
        ) from exc
    return target


# --------------------------------------------------------------------- #
# HuggingFace arm                                                       #
# --------------------------------------------------------------------- #


def load_hf(model_dir: str | Path) -> tuple[torch.nn.Module, object]:
    """Load a causal LM for the baseline arm, in the checkpoint's own dtype.

    The rapid_llm engine allocates weights in ``config.torch_dtype`` too, so
    ``torch_dtype="auto"`` keeps the two arms on the same arithmetic — the
    comparison is engine-vs-engine, not fp16-vs-bf16.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto").eval().cuda()
    return model, tokenizer


def hf_generate(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    chat_template: bool = False,
) -> tuple[list[str], int]:
    """Greedy-generate one completion per prompt; returns ``(texts, tokens)``.

    Left padding, so every sequence's newest token sits at the same column and
    a static batch generates in lockstep. Transformers has no stop strings, so
    stopping is the caller's job (:func:`tests.evals.runner.truncate_at_stop`),
    exactly like the rapid arm's post-hoc truncation.
    """
    if chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
    tokenizer.padding_side = "left"

    texts: list[str] = []
    total_tokens = 0
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(
            model.device
        )
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        total_tokens += int(new_tokens.shape[1] * new_tokens.shape[0])
        texts.extend(tokenizer.decode(row, skip_special_tokens=True) for row in new_tokens.tolist())
    return texts, total_tokens


def continuation_ids(tokenizer, prefix: str, continuation: str) -> tuple[list[int], list[int]]:
    """Tokenise a (prefix, continuation) pair the way lm-eval-harness does.

    Each side is encoded on its own and the ids are concatenated, so a BPE
    merge can never straddle the boundary and shift it mid-continuation. Both
    engines score continuations on exactly these ids — the guarantee the
    HellaSwag comparison rests on.
    """
    return (
        tokenizer.encode(prefix, add_special_tokens=True),
        tokenizer.encode(continuation, add_special_tokens=False),
    )


def hf_continuation_logprobs(
    model: torch.nn.Module,
    tokenizer,
    pairs: Sequence[tuple[list[int], list[int]]],
    *,
    batch_size: int,
) -> list[list[float]]:
    """Per-token log-probs of each continuation's ids under ``model``.

    Right padding (the score is read off each sequence's own positions, so
    alignment to the longest row would only invite off-by-ones). Row ``t-1``'s
    distribution scores token ``t``; the continuation occupies the trailing
    ``len(cont)`` positions of the concatenated ids. The log-softmax runs
    per-sequence on just the scored slice in fp32 — a whole-batch fp32
    softmax over ``[batch, len, vocab]`` would transiently allocate several GB.
    """
    pad = tokenizer.pad_token_id
    results: list[list[float]] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        seqs = [p + c for p, c in chunk]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        attention = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for row, seq in enumerate(seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            attention[row, : len(seq)] = 1
        with torch.inference_mode():
            logits = model(input_ids.cuda(), attention_mask=attention.cuda()).logits
        for row, (prefix, cont) in enumerate(chunk):
            if not cont:
                results.append([])
                continue
            total = len(prefix) + len(cont)
            targets = torch.tensor(seqs[row][len(prefix) : total], device=logits.device)
            scored = torch.log_softmax(logits[row, len(prefix) - 1 : total - 1].float(), dim=-1)
            results.append(scored.gather(-1, targets.view(-1, 1)).view(-1).tolist())
    return results


# --------------------------------------------------------------------- #
# rapid_llm arm                                                         #
# --------------------------------------------------------------------- #


def build_rapid_llm(
    model_dir: str | Path,
    *,
    max_seq_len: int,
    batch_size: int,
    use_cuda_graph: bool | None = None,
) -> LLM:
    """Build the offline :class:`rapid_llm.LLM` sized for one eval batch.

    ``max_gpu_num_blocks`` is pinned to ``batch_size * max_seq_len`` (the
    :func:`tests.evals.runner.kv_cache_tokens` argument): the engine's default
    profiling takes 90% of the device, which starves the prefill logits —
    several GB at ``batch x prompt_len x vocab`` — exactly when prompt
    log-probs need them.
    """
    from rapid_llm import LLM

    return LLM(
        model=str(model_dir),
        max_seq_len=max_seq_len,
        max_gpu_num_blocks=batch_size * max_seq_len,
        use_cuda_graph=use_cuda_graph,
    )


def rapid_continuation_logprobs(
    llm,
    pairs: Sequence[tuple[list[int], list[int]]],
    *,
    batch_size: int,
) -> list[list[float]]:
    """Per-token log-probs of each continuation, via the engine's prompt logprobs.

    The pair ids are fed straight to :meth:`generate_text` — bypassing the
    string path keeps the token boundary identical to the HF arm's. One decode
    step is the engine's minimum (``max_gen_len >= 1``); its cost is noise
    next to the prefill that produces the scores.
    """
    from rapid_llm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_gen_len=1,
        repetition_penalty=1.0,
        stop_on_repeat=False,
        prompt_logprobs=0,
    )
    results: list[list[float]] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        seqs = [p + c for p, c in chunk]
        llm.generate_text(seqs, params)
        records = llm.last_prompt_logprobs or []
        for row, (_prefix, cont) in enumerate(chunk):
            tail = records[row][len(records[row]) - len(cont) :] if cont else []
            results.append([r.logprob for r in tail])
    return results


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return ``(result, seconds)`` — the arms' shared clock."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
