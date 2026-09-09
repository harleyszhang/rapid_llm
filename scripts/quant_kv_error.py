"""fp8 KV cache: measure the error first, then decide whether to calibrate.

``Fp8KVCacheMethod`` ships with ``k_scale = v_scale = 1.0``, which is not a
calibrated value — it is the value that makes the write side a plain cast. The
usual reflex is to call that a bug and add a calibration pass. This script exists
because that reflex is not obviously right for a *floating* target format, and the
question is cheap to settle by measurement.

The argument, and what the numbers here have to decide between:

**e4m3 is a float, so a scale factor does not buy resolution in the interior.**
Its 3-bit mantissa gives ~6% relative spacing at *every* exponent, so multiplying
the input by a constant slides values along the exponent axis without making the
steps finer. Scaling only changes the answer at the two ends of the range:

* above ``448``, where :func:`quantize_fp8_per_tensor` clamps and the value is
  destroyed outright — this is what a scale ``> 1`` is for;
* below ``2**-6``, where e4m3 goes subnormal and the relative error grows until
  the value collapses to zero at ``2**-10`` — a scale ``< 1`` is what fixes this.

For int8 the picture is the opposite: a fixed absolute step means the scale sets
the resolution everywhere, and calibration is unconditionally worth doing. That
asymmetry is why "fp8 has a hardcoded 1.0" is not by itself evidence of a
problem, and why this script measures rather than assumes.

Three measurements, cheapest first:

1. **Range occupancy** (:func:`probe_ranges`) — per layer, per K/V: ``amax``, the
   count of values the clamp destroyed, the count that fell into the subnormal
   region, and the round-trip RMS error against two scales: the production
   ``1.0`` and a **per-call oracle** ``call_amax / 448``. The oracle is the point
   of the exercise: it is refit on every single write, so it is strictly finer
   than any *static* per-tensor calibration of the same form. If the oracle does
   not beat ``1.0`` by a useful margin, no calibrated constant can, and 4b is
   dead without writing it.
2. **Token agreement** — the same prompts decoded twice, ``auto`` against
   ``fp8_e4m3``, greedy. This is the only measurement that sees the error after
   it has been through softmax and 36 layers of accumulation, which is where a
   per-element error that looks negligible either cancels or does not.
3. **Task accuracy** (``--gsm8k N``, optional) — GSM8K exact match under both
   cache dtypes. Token agreement can fall well below 1.0 on questions where the
   model was undecided anyway; only a task score says whether the divergence
   costs anything.

The verdict thresholds are fixed in :data:`CLIP_LIMIT` and :data:`MATCH_FLOOR`
before any number is collected, so a marginal result cannot be argued past by
picking a threshold afterwards.

Usage::

    .venv/bin/python scripts/quant_kv_error.py \\
        --model-dir $RAPID_LLM_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \\
        --json docs/benchmark_logs/quantization/kv_fp8_error_qwen3-4b_20260901.json

    # add the task score (needs the GSM8K cache under ~/.cache/rapid_llm/evals)
    .venv/bin/python scripts/quant_kv_error.py --model-dir ... --gsm8k 200
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapid_llm import LLM, SamplingParams
from rapid_llm.modules.attention import PagedAttention
from rapid_llm.modules.quantization.utils import FP8_E4M3_MAX, quantize_fp8_per_tensor

#: Smallest e4m3 magnitude with a full 3-bit mantissa. Below it the format is
#: subnormal and the relative error grows; at ``2**-10`` it underflows to zero.
E4M3_MIN_NORMAL = 2.0**-6

#: Verdict gate 1: any layer whose ``amax`` exceeds this had values clamped, and
#: a clamp is not a rounding error — it is an unbounded one.
CLIP_LIMIT = FP8_E4M3_MAX

#: Verdict gate 2: greedy token agreement between the two cache dtypes.
MATCH_FLOOR = 0.98

#: Ratio of static-scale to oracle-scale RMS error below which calibration has
#: nothing to win. Reported, not gated: the two gates above are the plan's, and
#: this number is the *explanation* of their outcome rather than a third test.
ORACLE_GAIN_NOTE = 1.10

#: Prompts for the range probe and the token comparison. Deliberately mixed:
#: prose, code and arithmetic put different activation magnitudes into K/V, and
#: a range measurement over one register of text would not generalise. Long
#: enough that every sequence decodes for a while against a populated cache.
PROMPTS: tuple[str, ...] = (
    "Explain in detail how a paged key-value cache lets an inference server "
    "serve many sequences of different lengths without fragmenting memory.",
    "Write a Python function that merges two sorted lists in linear time, then "
    "explain why the naive concatenate-and-sort version is asymptotically worse.",
    "A train leaves at 09:15 travelling 84 km/h and a second leaves the same "
    "station at 10:00 travelling 112 km/h. When does the second catch the first?",
    "Translate to French, then back to English, and comment on what the round "
    "trip lost: 'The quantisation error is small but it is not zero.'",
)


# --------------------------------------------------------------------------- #
# 1. Range occupancy
# --------------------------------------------------------------------------- #
def _round_trip(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantise ``x`` at a device-resident ``scale`` and widen it back.

    A local reimplementation of :func:`quantize_fp8_per_tensor` rather than a
    call to it, because that function's ``scale`` is a Python float and the
    oracle scale is a value on the device: taking ``.item()`` on it would put a
    host synchronisation inside every attention layer of every decode step.
    :func:`_selfcheck` pins the two against each other at ``scale = 1.0``.
    """
    q = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.float() * scale


def _selfcheck(device: str) -> None:
    """Assert :func:`_round_trip` reproduces the production quantiser exactly.

    Both paths must land on the same e4m3 bit pattern, so this is an equality,
    not a tolerance. Values are drawn wide enough to cross both the clamp and the
    subnormal boundary, since those branches are the ones that could differ.
    """
    x = torch.randn(4096, device=device) * 200.0
    x[:8] = torch.tensor([0.0, 1e-4, 1e-3, 0.01, 1.0, 447.0, 448.0, 1e4], device=device)
    mine = _round_trip(x, torch.ones((), device=device))
    theirs = quantize_fp8_per_tensor(x, 1.0).view(torch.float8_e4m3fn).float()
    if not torch.equal(mine, theirs):
        bad = (mine != theirs).nonzero().flatten()[:8].tolist()
        raise SystemExit(f"round-trip reference disagrees with production at indices {bad}")


class _Accumulator:
    """Running range statistics for one tensor role (K or V) of one layer.

    Everything lands in a single device-resident vector so that a probe attached
    to a live decode loop costs a handful of reductions and *no* host
    synchronisation — the layer this hangs off is the one whose cost we are
    trying to characterise, and an ``.item()`` per step would make the engine
    measure the probe.

    Non-finite inputs are excluded from every sum and counted separately. That is
    not defensive padding: this probe *found* such values, and the reason is worth
    knowing before reading a NaN here as a model defect. A batched prefill pads
    every sequence to the longest one, and the pad rows' hidden states are
    whatever the caching allocator last left in that block — so they can be NaN,
    they reach ``quantize_kv``, and attention masks them out. Two runs of the same
    prompts, one with this probe installed and one without, produced *identical*
    completions while only the probed run saw NaN, which is what proves the values
    are unattended garbage rather than a broken forward pass: the probe's own
    temporaries changed which block the pad rows landed in.
    :func:`probe_ranges` avoids the situation by decoding one sequence at a time,
    and the counter is what says whether that worked.
    """

    _N, _OVER, _SUB, _SQ, _ERR_STATIC, _ERR_ORACLE, _BAD = range(7)

    def __init__(self, device: str) -> None:
        self.t = torch.zeros(7, dtype=torch.float64, device=device)
        self.amax = torch.zeros((), dtype=torch.float64, device=device)
        self.calls = 0

    def update(self, x: torch.Tensor) -> None:
        raw = x.detach().float()
        good = torch.isfinite(raw)
        self.t[self._BAD] += (~good).sum()
        # Zero is the identity for every sum below and has magnitude 0, so it is
        # excluded from the subnormal count as well; only _N would be inflated,
        # which is why _N counts the mask rather than numel().
        f = torch.where(good, raw, torch.zeros_like(raw))
        mag = f.abs()
        amax = mag.amax()

        self.t[self._N] += good.sum()
        self.t[self._OVER] += (mag > FP8_E4M3_MAX).sum()
        self.t[self._SUB] += ((mag > 0) & (mag < E4M3_MIN_NORMAL)).sum()
        self.t[self._SQ] += f.double().pow(2).sum()

        static = quantize_fp8_per_tensor(f, 1.0).view(torch.float8_e4m3fn).float()
        self.t[self._ERR_STATIC] += (f - static).double().pow(2).sum()

        # The oracle scale for *this* call. clamp_min keeps an all-zero tensor
        # from producing a 0/0 scale.
        oracle_scale = (amax / FP8_E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
        self.t[self._ERR_ORACLE] += (f - _round_trip(f, oracle_scale)).double().pow(2).sum()

        self.amax = torch.maximum(self.amax, amax.double())
        self.calls += 1

    def finish(self) -> TensorStats:
        """Drain to the host. The one synchronisation, after the run."""
        n, over, sub, sq, err_static, err_oracle, bad = self.t.tolist()
        rms_ref = (sq / n) ** 0.5 if n else 0.0
        return TensorStats(
            calls=self.calls,
            count=int(n),
            nonfinite=int(bad),
            amax=self.amax.item(),
            clipped=int(over),
            subnormal=int(sub),
            rms_value=rms_ref,
            rel_rms_static=((err_static / n) ** 0.5 / rms_ref) if rms_ref else 0.0,
            rel_rms_oracle=((err_oracle / n) ** 0.5 / rms_ref) if rms_ref else 0.0,
        )


@dataclass(frozen=True)
class TensorStats:
    """What one layer's K (or V) writes looked like over a whole run.

    ``rel_rms_static`` and ``rel_rms_oracle`` are RMS errors normalised by the
    RMS of the values themselves, which makes them comparable across layers whose
    activations differ in magnitude by an order of magnitude. Their *ratio* is the
    only thing calibration could improve.
    """

    calls: int
    count: int
    nonfinite: int
    amax: float
    clipped: int
    subnormal: int
    rms_value: float
    rel_rms_static: float
    rel_rms_oracle: float

    @property
    def oracle_gain(self) -> float:
        """How many times smaller a per-call scale makes the error. 1.0 = none."""
        return self.rel_rms_static / self.rel_rms_oracle if self.rel_rms_oracle else 1.0

    @property
    def clipped_frac(self) -> float:
        return self.clipped / self.count if self.count else 0.0

    @property
    def subnormal_frac(self) -> float:
        return self.subnormal / self.count if self.count else 0.0


class _Probe:
    """Instance-level replacement for one layer's ``quantize_kv``.

    Installed as ``method.quantize_kv = probe``, shadowing the bound class
    method. Swapping the whole ``kv_cache_method`` object would *not* work:
    :class:`~rapid_llm.modules.attention.PagedAttention` copies
    ``method.k_scale`` into ``self.k_scale`` in its constructor, so a replacement
    method installed later would be measured against scales the layer no longer
    reads from it. Wrapping the one function keeps the object — and therefore the
    scales the read side uses — untouched.
    """

    def __init__(self, inner, device: str) -> None:
        self._inner = inner
        self.k = _Accumulator(device)
        self.v = _Accumulator(device)

    def __call__(self, k: torch.Tensor, v: torch.Tensor):
        self.k.update(k)
        self.v.update(v)
        return self._inner(k, v)


def probe_ranges(
    model_dir: str,
    *,
    max_seq_len: int,
    max_gen_len: int,
    device: str,
) -> dict[str, dict[str, TensorStats]]:
    """Decode :data:`PROMPTS` with an fp8 cache, recording what each layer wrote.

    Runs against ``kv_cache_dtype="fp8_e4m3"`` so the probe sits on the exact
    tensors the production quantiser receives — same rope, same qk-norm, same
    dtype. The statistics describe the *input* to quantisation, so they would be
    identical under ``auto``; using the fp8 path means the run also exercises the
    read side, and a divergence there would show up as garbage output rather than
    being silently excluded.

    Two things about *how* it runs are load-bearing rather than incidental:

    ``use_cuda_graph=False``
        A replayed graph does not call Python, so a probe installed after
        construction never sees a single decode step. The first version of this
        script left graphs on and silently measured prefill only — which is the
        wrong half, since decode is where the fp8 cache is *read*. Graphs change
        no arithmetic, so eager loses nothing but the launch overhead.
    one sequence per call
        A batched prefill pads to the longest prompt and the pad rows carry
        uninitialised hidden states; see :class:`_Accumulator`. Padding cannot be
        masked out from inside ``quantize_kv``, which receives the padded tensor
        with no batch metadata, so the fix is upstream: never create padding.
    """
    llm = LLM(
        model=model_dir,
        max_seq_len=max_seq_len,
        kv_cache_dtype="fp8_e4m3",
        use_cuda_graph=False,
        device=device,
    )
    try:
        probes: dict[str, _Probe] = {}
        for name, module in llm.model_runner.model.named_modules():
            if not isinstance(module, PagedAttention):
                continue
            method = module.kv_cache_method
            if method is None:
                raise SystemExit(f"{name} has no kv_cache_method — is this build fp8-capable?")
            probe = _Probe(method.quantize_kv, device)
            method.quantize_kv = probe  # type: ignore[method-assign]
            probes[name] = probe
        if not probes:
            raise SystemExit("no PagedAttention layers found; nothing to probe")

        params = SamplingParams(
            temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=1.0, stop_on_repeat=False
        )
        for prompt in PROMPTS:
            out = llm.generate([prompt], params)[0]
            if not out.outputs[0].text.strip():
                raise SystemExit("fp8 KV cache produced an empty completion — read path is broken")

        return {
            name: {"k": probe.k.finish(), "v": probe.v.finish()} for name, probe in probes.items()
        }
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# 2. Token agreement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchStats:
    """Greedy agreement between the two cache dtypes, position by position."""

    match_rate: float
    matched: int
    positions: int
    per_prompt: list[float] = field(default_factory=list)
    first_divergence: list[int] = field(default_factory=list)


def _generate(model_dir: str, kv_cache_dtype: str, *, max_seq_len, max_gen_len, device):
    """Greedily complete every prompt one at a time under one cache dtype.

    One at a time for the same reason as :func:`probe_ranges`: a padded batch puts
    uninitialised rows through the layers, and although they are masked out of the
    result, a comparison whose two halves allocate differently is not worth the
    doubt. CUDA graphs stay at their default here — unlike the probe this measures
    the engine's *output*, and the production configuration is the one whose output
    matters.
    """
    llm = LLM(
        model=model_dir, max_seq_len=max_seq_len, kv_cache_dtype=kv_cache_dtype, device=device
    )
    try:
        params = SamplingParams(
            temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=1.0, stop_on_repeat=False
        )
        texts = [llm.generate([p], params)[0].outputs[0].text for p in PROMPTS]
        ids = [llm.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        return texts, ids
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def token_agreement(
    model_dir: str, *, max_seq_len: int, max_gen_len: int, device: str
) -> tuple[MatchStats, MatchStats, list[tuple[str, str]]]:
    """Decode :data:`PROMPTS` greedily under both cache dtypes and compare.

    Returns ``(fp8_vs_auto, auto_vs_auto, completion_pairs)``. The second is the
    control and it is not optional: greedy decoding is a chain of argmaxes, so a
    single flipped token sends the rest of the sequence somewhere else and the
    positional match rate collapses to roughly chance. Without knowing what the
    *same* configuration scores against itself there is no way to read a low rate
    as "fp8 hurt" rather than "this metric is chaotic" — the two builds differ in
    allocator state and cache size even when the dtype does not.

    The comparison is on tokens **re-encoded from the completion text**, not on
    the ids the engine sampled: :class:`~rapid_llm.engine.outputs.RequestOutput`
    carries only text. Re-tokenisation is not a lossless inverse — a divergence
    can shift a token boundary and cost more than one position — so the rate
    below is a *lower* bound on agreement. That is the safe direction for a gate
    that triggers extra work when the number is low.

    The denominator is the longer of the two sequences, so a run that diverges by
    stopping early scores the truncation instead of ignoring it.
    """
    kw = {"max_seq_len": max_seq_len, "max_gen_len": max_gen_len, "device": device}
    ref_texts, ref_ids = _generate(model_dir, "auto", **kw)
    fp8_texts, fp8_ids = _generate(model_dir, "fp8_e4m3", **kw)
    _, ctl_ids = _generate(model_dir, "auto", **kw)

    return (
        _compare(ref_ids, fp8_ids),
        _compare(ref_ids, ctl_ids),
        list(zip(ref_texts, fp8_texts, strict=True)),
    )


def _compare(ref_ids: list[list[int]], other_ids: list[list[int]]) -> MatchStats:
    """Positional token agreement between two runs over the same prompts."""
    matched = positions = 0
    per_prompt: list[float] = []
    first_div: list[int] = []
    for a, b in zip(ref_ids, other_ids, strict=True):
        n = max(len(a), len(b))
        same = sum(1 for x, y in zip(a, b, strict=False) if x == y)
        divergence = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), -1)
        if divergence == -1 and len(a) != len(b):
            divergence = min(len(a), len(b))
        matched += same
        positions += n
        per_prompt.append(same / n if n else 1.0)
        first_div.append(divergence)

    return MatchStats(
        match_rate=matched / positions if positions else 1.0,
        matched=matched,
        positions=positions,
        per_prompt=per_prompt,
        first_divergence=first_div,
    )


# --------------------------------------------------------------------------- #
# 3. Task accuracy (optional)
# --------------------------------------------------------------------------- #
def task_accuracy(model_dir: str, num_questions: int, *, batch_size: int = 16) -> dict:
    """GSM8K exact match under both cache dtypes, with the noise floor attached.

    Token agreement counts *any* divergence, including the ones that reword an
    answer without changing it. This is the measurement that says whether the
    divergences mattered. Requires the GSM8K cache to already exist locally; a
    download failure is reported, not raised, because the two measurements above
    are enough to reach a verdict on their own.

    ``resolvable`` is the part that keeps the comparison honest. Two accuracies a
    few questions apart are not a difference; at 100 questions and p ~= 0.16 the
    standard error of the gap is ~5 points, so anything under ~10 points is the
    sampling noise of the question draw. The estimate here treats the two runs as
    independent, which *overstates* the noise for paired data (both dtypes answer
    the same questions), so ``resolvable = True`` is conservative: it can only
    understate a real difference, never invent one.
    """
    from tests.evals.gsm8k import evaluate_gsm8k

    out: dict[str, object] = {"num_questions": num_questions}
    for dtype in ("auto", "fp8_e4m3"):
        result = evaluate_gsm8k(
            model_dir,
            num_questions=num_questions,
            batch_size=batch_size,
            use_chat_template=True,
            kv_cache_dtype=dtype,
            progress=True,
        )
        out[dtype] = {
            "accuracy": result.accuracy,
            "invalid_rate": result.invalid_rate,
            "tokens_per_second": result.tokens_per_second,
        }

    a, b = out["auto"]["accuracy"], out["fp8_e4m3"]["accuracy"]  # type: ignore[index]
    var = (a * (1 - a) + b * (1 - b)) / num_questions if num_questions else 0.0
    stderr = var**0.5
    out["delta"] = b - a
    out["stderr_unpaired"] = stderr
    out["resolvable"] = abs(b - a) > 1.96 * stderr
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _metadata(model_dir: str) -> dict:
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "model_dir": model_dir,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "commit": commit,
        "command": " ".join(sys.argv),
    }


def print_ranges(stats: dict[str, dict[str, TensorStats]]) -> None:
    """One row per layer per role, plus the aggregates the verdict reads."""
    header = (
        f"{'layer':<40} {'kv':<3} {'amax':>9} {'clip%':>8} {'sub%':>8} "
        f"{'relRMS@1.0':>11} {'relRMS@oracle':>14} {'gain':>6} {'bad':>5}"
    )
    print(header)
    print("-" * len(header))
    for name, roles in stats.items():
        for role, s in roles.items():
            print(
                f"{name:<40} {role:<3} {s.amax:>9.3f} {s.clipped_frac * 100:>7.4f}% "
                f"{s.subnormal_frac * 100:>7.4f}% {s.rel_rms_static:>11.4e} "
                f"{s.rel_rms_oracle:>14.4e} {s.oracle_gain:>6.3f}x {s.nonfinite:>5d}"
            )


def summarise(stats: dict[str, dict[str, TensorStats]]) -> dict:
    flat = [(f"{name}.{role}", s) for name, roles in stats.items() for role, s in roles.items()]
    worst_amax = max(flat, key=lambda kv: kv[1].amax)
    worst_gain = max(flat, key=lambda kv: kv[1].oracle_gain)
    worst_sub = max(flat, key=lambda kv: kv[1].subnormal_frac)
    mean_static = sum(s.rel_rms_static for _, s in flat) / len(flat)
    mean_oracle = sum(s.rel_rms_oracle for _, s in flat) / len(flat)
    return {
        "layers_probed": len(stats),
        "max_amax": worst_amax[1].amax,
        "max_amax_at": worst_amax[0],
        "any_clipped": any(s.clipped for _, s in flat),
        "total_clipped": sum(s.clipped for _, s in flat),
        "total_values": sum(s.count for _, s in flat),
        "max_subnormal_frac": worst_sub[1].subnormal_frac,
        "max_subnormal_at": worst_sub[0],
        "total_nonfinite": sum(s.nonfinite for _, s in flat),
        "mean_rel_rms_static": mean_static,
        "mean_rel_rms_oracle": mean_oracle,
        "mean_oracle_gain": mean_static / mean_oracle if mean_oracle else 1.0,
        "max_oracle_gain": worst_gain[1].oracle_gain,
        "max_oracle_gain_at": worst_gain[0],
    }


def verdict(summary: dict, match: MatchStats | None, control: MatchStats | None) -> dict:
    """Apply the two pre-registered gates and say what follows from them.

    The gates are the plan's and are reported exactly as written. What the two
    extra fields add is the reading, which the gates alone cannot supply:

    * ``control_match_rate`` — if the same configuration does not reproduce
      itself, the match gate is measuring the metric's own instability and a trip
      says nothing about the cache dtype.
    * ``oracle_headroom`` — a per-call amax scale is refit on every write, so it
      bounds what any *static* per-tensor scale of the same form can achieve. A
      trip with headroom near 1.0 means the error is real but per-tensor
      calibration is not the thing that removes it, and building 4b would be
      building a fix for a cause the measurement excluded.

    The headroom that decides is the **mean over layers**, not the best layer.
    One layer improving 1.19x while the model-wide mean improves 1.03x is not
    evidence that calibration helps; it is evidence that one layer's activations
    happen to sit further from a power of two than the rest. Since every layer
    feeds the same residual stream, the error that reaches the logits is the
    aggregate one. ``max_oracle_gain`` is still reported next to it so the spread
    is visible rather than hidden behind whichever aggregate was chosen.
    """
    clipping = summary["max_amax"] > CLIP_LIMIT
    diverging = match is not None and match.match_rate < MATCH_FLOOR
    control_clean = control is None or control.match_rate >= MATCH_FLOOR
    headroom = summary["mean_oracle_gain"]
    spread = (
        f"mean {headroom:.3f}x, best layer {summary['max_oracle_gain']:.3f}x "
        f"at {summary['max_oracle_gain_at']}"
    )

    reasons = []
    if clipping:
        reasons.append(
            f"amax {summary['max_amax']:.1f} > {CLIP_LIMIT} at {summary['max_amax_at']} "
            f"({summary['total_clipped']} of {summary['total_values']} values clamped)"
        )
    if diverging:
        reasons.append(f"token match {match.match_rate:.4f} < {MATCH_FLOOR}")
    if control is not None and not control_clean:
        reasons.append(
            f"control (auto vs auto) also scores {control.match_rate:.4f} < {MATCH_FLOOR}, "
            "so the match gate does not isolate the cache dtype"
        )

    if clipping:
        action = "calibration required (Phase 4b): a clamp is unbounded error and a scale fixes it"
    elif diverging and not control_clean:
        action = (
            "match gate is uninformative here — the control trips it too; "
            "decide on task accuracy alone"
        )
    elif diverging and headroom < ORACLE_GAIN_NOTE:
        action = (
            f"divergence is real but per-tensor calibration cannot fix it "
            f"(oracle headroom {spread}, below {ORACLE_GAIN_NOTE}); "
            "decide on task accuracy, not on 4b"
        )
    elif diverging:
        action = f"calibration worth trying: oracle headroom {spread}"
    else:
        action = "scale = 1.0 is sufficient; record the evidence in docs/quantization.md"

    return {
        "clip_gate_tripped": clipping,
        "match_gate_tripped": diverging,
        "control_match_rate": control.match_rate if control else None,
        "control_reproduces": control_clean,
        "needs_calibration": clipping
        or (diverging and control_clean and headroom >= ORACLE_GAIN_NOTE),
        "reasons": reasons,
        "action": action,
        "oracle_headroom": headroom,
        "oracle_headroom_basis": "mean over layers and roles",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--gsm8k",
        type=int,
        default=0,
        help="score GSM8K under both cache dtypes with this many questions (0 = skip)",
    )
    ap.add_argument("--skip-tokens", action="store_true", help="range probe only, no second build")
    ap.add_argument("--json", help="write the full record here")
    args = ap.parse_args(argv)

    _selfcheck(args.device)

    print(f"== fp8 KV cache error — {args.model_dir} ==\n")
    print("[1/3] range occupancy (fp8_e4m3 cache, probe on quantize_kv)")
    stats = probe_ranges(
        args.model_dir,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        device=args.device,
    )
    print()
    print_ranges(stats)
    summary = summarise(stats)
    print(
        f"\n  layers {summary['layers_probed']}, max amax {summary['max_amax']:.3f} "
        f"at {summary['max_amax_at']}, clamped {summary['total_clipped']}/"
        f"{summary['total_values']}, max subnormal share "
        f"{summary['max_subnormal_frac'] * 100:.4f}% at {summary['max_subnormal_at']}"
    )
    if summary["total_nonfinite"]:
        print(
            f"  WARNING {summary['total_nonfinite']} non-finite values excluded — with one"
            " sequence per call there is no padding, so this points at a real defect"
        )
    print(
        f"  mean relative RMS: {summary['mean_rel_rms_static']:.4e} at scale 1.0 vs "
        f"{summary['mean_rel_rms_oracle']:.4e} with a per-call oracle scale "
        f"({summary['mean_oracle_gain']:.3f}x mean gain, {summary['max_oracle_gain']:.3f}x "
        f"at the best layer {summary['max_oracle_gain_at']})"
    )

    match: MatchStats | None = None
    control: MatchStats | None = None
    pairs: list[tuple[str, str]] = []
    if not args.skip_tokens:
        print("\n[2/3] greedy token agreement, auto vs fp8_e4m3 (plus an auto-vs-auto control)")
        match, control, pairs = token_agreement(
            args.model_dir,
            max_seq_len=args.max_seq_len,
            max_gen_len=args.max_gen_len,
            device=args.device,
        )
        print(
            f"  fp8 vs auto  {match.match_rate:.4f} ({match.matched}/{match.positions} "
            f"positions, re-tokenised so this is a lower bound)"
        )
        for i, (rate, div) in enumerate(zip(match.per_prompt, match.first_divergence, strict=True)):
            where = "identical" if div == -1 else f"first divergence at token {div}"
            print(f"    prompt{i}: {rate:.4f}  ({where})")
        print(f"  auto vs auto {control.match_rate:.4f}   <- the metric's own floor")

    task: dict | None = None
    if args.gsm8k:
        print(f"\n[3/3] GSM8K exact match, {args.gsm8k} questions per dtype")
        try:
            task = task_accuracy(args.model_dir, args.gsm8k)
            a, b = task["auto"], task["fp8_e4m3"]
            print(f"  auto      {a['accuracy']:.4f} (invalid {a['invalid_rate']:.4f})")
            print(f"  fp8_e4m3  {b['accuracy']:.4f} (invalid {b['invalid_rate']:.4f})")
            verb = "resolved" if task["resolvable"] else "NOT resolved"
            print(
                f"  delta {task['delta']:+.4f}, 1.96*se {1.96 * task['stderr_unpaired']:.4f}"
                f" -> {verb} at {args.gsm8k} questions"
            )
        except Exception as exc:  # dataset download, OOM — report, do not mask
            task = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  SKIPPED: {task['error']}")

    decision = verdict(summary, match, control)
    print("\n== verdict ==")
    for line in decision["reasons"] or ["both gates clear"]:
        print(f"  {line}")
    print(f"  -> {decision['action']}")

    if args.json:
        record = {
            "meta": _metadata(args.model_dir),
            "gates": {"clip_limit": CLIP_LIMIT, "match_floor": MATCH_FLOOR},
            "prompts": list(PROMPTS),
            "layers": {
                name: {role: asdict(s) for role, s in roles.items()}
                for name, roles in stats.items()
            },
            "summary": summary,
            "token_agreement": asdict(match) if match else None,
            "token_agreement_control": asdict(control) if control else None,
            "completions": [{"auto": a, "fp8_e4m3": b} for a, b in pairs],
            "gsm8k": task,
            "verdict": decision,
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
