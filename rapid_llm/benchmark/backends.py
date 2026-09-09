"""The measured systems: one ABC, one factory — and the only place an engine is driven.

Every engine the benchmarks compare — rapid_llm's four entry points plus the
HF and vLLM baselines — adapts into a :class:`Backend`, so a scenario script
measures systems without knowing their constructors. One implementation per
engine on purpose: when the offline runner carried its own copies of the same
three drive loops, the two disagreed about warm-up, reported rounds and what
``greedy`` means — and the numbers stopped being comparable.

:meth:`Backend.measure_rows` is the primitive (per-row output lengths,
``iters`` rounds, median reported); :meth:`Backend.measure` is the
uniform-length shorthand; arm-specific extras come out of ``details()``.

Usage:
    from rapid_llm.benchmark import make_backend, build_arm, LiteBackend
"""

from __future__ import annotations

import contextlib
import statistics
import time
from abc import ABC, abstractmethod

import torch

from .datasets import DatasetRow
from .metrics import BenchResult, pctl, run_requests, steps_to_result
from .utils import count_gen_tokens, footprint_stats, free_gpu, median_round
from .workloads import SAMPLE_KW, sampling_params


def uniform_rows(prompts: list[str], output_len: int) -> list[DatasetRow]:
    """One row per prompt, every row run to the same output length.

    ``prompt_len`` stays 0: nothing here re-tokenizes, and the field is only an
    input-token accounting column for the dataset-driven runners.
    """
    return [DatasetRow(prompt=p, prompt_len=0, output_len=output_len) for p in prompts]


class Backend(ABC):
    """One measured system.

    A subclass answers two questions: how a row set becomes a
    :class:`BenchResult`, and how to tear itself down. ``texts()`` serves the
    accuracy comparisons — the same run's output, so nothing is measured twice.
    """

    @abstractmethod
    def measure_rows(
        self,
        rows: list[DatasetRow],
        *,
        greedy: bool = True,
        iters: int = 1,
        warmup_rows: list[DatasetRow] | None = None,
    ) -> BenchResult:
        """Run the workload ``iters`` times and return the median round's metrics.

        Args:
            rows: The workload — one prompt and its output length per request.
            greedy: Greedy decoding (the benchmark default) or :data:`SAMPLE_KW`.
            iters: Timed rounds; the median one by wall clock is reported.
            warmup_rows: Rows for the short warm-up round, ``rows`` by default.
                A caller running with the prefix cache on must narrow this, or
                the warm-up writes the measured prompts into the cache and every
                row then reports a hit it did not earn.
        """

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool = True) -> BenchResult:
        """Uniform-length shorthand over :meth:`measure_rows` (one round)."""
        return self.measure_rows(uniform_rows(prompts, max_gen_len), greedy=greedy)

    def details(self) -> dict:
        """Extras of the last run that are not metrics: footprint, basis, flags."""
        return {}

    @property
    def runner(self):
        """This backend's ``ModelRunner`` (memory and KV capacity come from it)."""
        raise NotImplementedError(f"{type(self).__name__} holds no ModelRunner")

    def texts(self) -> list[str]:
        """The completions of the last measurement; empty for text-less backends."""
        return []

    def timeline_summary(self) -> str:
        """The engine's CUDA-event region table, the overlap benches' evidence.

        Empty for backends with no engine timeline (HF, vLLM).
        """
        return ""

    def close(self) -> None:
        """Return the GPU memory. Required before building a second backend in the
        same process, or its KV budget profiles as zero."""
        free_gpu()


class LiteBackend(Backend):
    """Single-process rapid_llm: a stream callback per step splits TTFT from TPOT.

    The batch advances in lockstep, so ``gen_tokens = steps * batch``. Constructor
    arguments pass straight through to ``TextGenerator`` (no defaults here, so this
    never overrides its own).
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, **gen_kwargs):
        from rapid_llm import TextGenerator

        self._gen = TextGenerator(
            checkpoints_dir=model_dir,
            use_cuda_graph=use_cuda_graph,
            **gen_kwargs,
        )
        self._texts: list[str] = []

    @property
    def generator(self):
        """The bare generator: ``one_batch --verify`` calls ``generate()`` directly."""
        return self._gen

    @property
    def runner(self):
        """``ModelRunner``: memory footprint and KV pool capacity are read from it."""
        return self._gen.engine.model_runner

    def measure_rows(self, rows, *, greedy=True, iters=1, warmup_rows=None) -> BenchResult:
        prompts = [row.prompt for row in rows]
        # Lockstep: the whole batch advances together, so one output length
        # governs the run and a per-row ladder has nothing to bite on.
        max_gen_len = max(row.output_len for row in rows)
        warmup = [row.prompt for row in (warmup_rows if warmup_rows is not None else rows)]

        # Warm up autotune + allocator so the measured run is steady state.
        for _ in range(2):
            list(self._gen.stream(warmup, sampling_params(8)))

        rounds = [self._one_round(prompts, max_gen_len, greedy) for _ in range(iters)]
        result, self._texts = median_round(rounds, key=lambda r: r[0].total_s)
        return result

    def _one_round(self, prompts, max_gen_len, greedy) -> tuple[BenchResult, list[str]]:
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        step_texts: list[list[str]] = []
        step_ends: list[float] = []
        for deltas in self._gen.stream(prompts, sampling_params(max_gen_len, greedy)):
            step_ends.append(time.perf_counter())
            step_texts.append(list(deltas))
        torch.cuda.synchronize()
        total = time.perf_counter() - t_start

        texts = ["".join(step[i] for step in step_texts) for i in range(len(prompts))]
        result = steps_to_result(
            step_ends, t_start=t_start, total_s=total, batch=len(prompts)
        )
        return result, texts

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        del self._gen
        super().close()


class EngineBackend(Backend):
    """Continuous-batching engine: submit the whole batch, then time each step.

    The only single-process path that can drive ``tp > 1`` — its executor
    broadcasts each step's plan to the follower ranks — and the only one that
    reports a per-request latency distribution, because the engine stamps each
    request's own first-token and finish times.
    """

    def __init__(self, model_dir: str, *, tensor_parallel_size: int = 1, **engine_kwargs):
        from rapid_llm.engine import ContinuousBatchingEngine

        self._engine = ContinuousBatchingEngine.from_pretrained(
            model_dir, tensor_parallel_size=tensor_parallel_size, **engine_kwargs
        )
        self.tensor_parallel_size = tensor_parallel_size
        self._engine_args = dict(engine_kwargs)
        self._texts: list[str] = []
        self._run = None

    @property
    def engine(self):
        """The wrapped engine: the scheduler benches reach for its knobs."""
        return self._engine

    @property
    def runner(self):
        return self._engine.engine.model_runner

    def measure_rows(self, rows, *, greedy=True, iters=1, warmup_rows=None) -> BenchResult:
        prompts = [row.prompt for row in rows]
        params = [sampling_params(row.output_len, greedy) for row in rows]
        warmup = [row.prompt for row in (warmup_rows if warmup_rows is not None else rows)]

        run_requests(self._engine, warmup, sampling_params(8))  # autotune + allocator
        rounds = [run_requests(self._engine, prompts, params) for _ in range(iters)]
        self._run = median_round(rounds, key=lambda r: r.total_s)
        self._texts = self._run.texts
        return self._run.result(len(rows))

    def details(self) -> dict:
        out: dict = {"engine_args": dict(self._engine_args)}
        window = self._run.steady_state() if self._run is not None else None
        if window:
            out["steady_state_output_tps"] = window[2] / (window[1] - window[0])
            out["steady_state_window_s"] = window[1] - window[0]
        # A test double's runner has no cache manager; the footprint is a
        # reporting extra, never a reason to lose a measured row.
        with contextlib.suppress(Exception):
            out["footprint"] = footprint_stats(self.runner)
        return out

    def texts(self) -> list[str]:
        return self._texts

    def timeline_summary(self) -> str:
        return self._engine.timeline_summary()

    def close(self) -> None:
        self._engine.shutdown()
        del self._engine
        super().close()


class DPBackend(Backend):
    """Data-parallel replicas: a throughput basis, not a latency one.

    DP multiplies aggregate throughput, not single-request latency — the batch
    wall time is the slowest replica's — so this arm fills ``total_s`` and
    ``gen_tokens`` and leaves the per-request percentile fields at zero, which
    :class:`BenchResult` documents as "not measured".
    """

    def __init__(self, model_dir: str, *, data_parallel_size: int, max_num_seqs: int = 0,
                 **engine_kwargs):
        from rapid_llm import DataParallelEngine

        self._engine = DataParallelEngine(
            model=model_dir,
            data_parallel_size=data_parallel_size,
            max_num_seqs=max_num_seqs,
            **engine_kwargs,
        )
        self.data_parallel_size = data_parallel_size
        self._texts: list[str] = []

    def measure_rows(self, rows, *, greedy=True, iters=1, warmup_rows=None) -> BenchResult:
        prompts = [row.prompt for row in rows]
        # One params object for the batch: the replica router hands whole
        # requests out, so there is no per-row plan to express here.
        params = sampling_params(max(row.output_len for row in rows), greedy)
        warmup = [row.prompt for row in (warmup_rows if warmup_rows is not None else rows)]

        self._engine.generate(warmup, sampling_params(8))
        rounds = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.monotonic()
            outputs = self._engine.generate(prompts, params)
            torch.cuda.synchronize()
            rounds.append((time.monotonic() - t0, [o.text for o in outputs]))
        wall, self._texts = median_round(rounds, key=lambda r: r[0])

        tokens = count_gen_tokens(self._texts, self._engine.tokenizer)
        return BenchResult(
            ttft_ms=0.0,
            tpot_ms=0.0,
            total_s=wall,
            steps=0,
            batch=len(rows),
            gen_tokens=tokens,
        )

    def details(self) -> dict:
        return {
            "data_parallel_size": self.data_parallel_size,
            "basis": "throughput (batch wall = slowest replica)",
        }

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        self._engine.shutdown()
        del self._engine
        super().close()


class VisionBackend(Backend):
    """Multimodal measurement: ``VisionGenerator`` serves one request at a time.

    rapid_llm's multimodal path is serial (the processor takes one request), so
    the batch becomes the number of serial requests: TTFT is the mean of each
    request's first-token latency, TPOT the mean of all decode-step intervals, TPS
    the aggregate throughput of the whole serial loop. Images are resized to a
    fixed 672x672, which pins the visual-token count of a dynamic-resolution tower
    (Qwen3-VL). Decode steps still replay a CUDA graph — the visual tokens are
    already in the KV cache by then, so what is captured and replayed is a plain
    text step.
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, image_path: str, **gen_kwargs):
        from PIL import Image

        from rapid_llm import VisionGenerator
        from rapid_llm.models.config import read_model_type

        self._image = Image.open(image_path).convert("RGB").resize((672, 672), Image.BICUBIC)
        self._gen = VisionGenerator(
            checkpoints_dir=model_dir, use_cuda_graph=use_cuda_graph, **gen_kwargs
        )
        # llava wants an explicit <image> marker plus vicuna turns; Qwen3-VL's
        # preparer (like HF's chat template) inserts the visual placeholder itself,
        # so a plain question goes straight in.
        self._is_llava = read_model_type(model_dir) == "llava"
        self._texts: list[str] = []

    @property
    def runner(self):
        return self._gen.engine.model_runner

    def _wrap(self, prompt: str) -> str:
        return f"USER: <image>\n{prompt} ASSISTANT:" if self._is_llava else prompt

    def measure_rows(self, rows, *, greedy=True, iters=1, warmup_rows=None) -> BenchResult:
        prompts = [row.prompt for row in rows]
        params = sampling_params(max(row.output_len for row in rows), greedy)
        warmup = [row.prompt for row in (warmup_rows if warmup_rows is not None else rows)]

        # Warm up autotune + graph capture so the measured run is steady state.
        for _ in range(2):
            list(self._gen.stream(self._wrap(warmup[0]), [self._image], sampling_params(8)))

        rounds = [self._one_round(prompts, params) for _ in range(iters)]
        result, self._texts = median_round(rounds, key=lambda r: r[0].total_s)
        return result

    def _one_round(self, prompts, params) -> tuple[BenchResult, list[str]]:
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        req_ttfts: list[float] = []
        step_deltas: list[float] = []
        texts: list[str] = []
        for prompt in prompts:
            req_start = time.perf_counter()
            first = True
            prev = 0.0
            pieces: list[str] = []
            for delta in self._gen.stream(self._wrap(prompt), [self._image], params):
                now = time.perf_counter()
                if first:
                    req_ttfts.append(now - req_start)
                    first = False
                else:
                    step_deltas.append(now - prev)
                prev = now
                pieces.append(delta)
            texts.append("".join(pieces))
        torch.cuda.synchronize()
        total = time.perf_counter() - t_start

        # Every request contributed its first token plus len(step deltas per
        # request) decode tokens; deltas were only collected after each first.
        result = BenchResult(
            ttft_ms=(statistics.mean(req_ttfts) if req_ttfts else 0.0) * 1000,
            tpot_ms=(statistics.mean(step_deltas) * 1000) if step_deltas else 0.0,
            tpot_p50_ms=(statistics.median(step_deltas) * 1000) if step_deltas else 0.0,
            total_s=total,
            steps=len(req_ttfts) + len(step_deltas),
            batch=len(prompts),
            gen_tokens=len(req_ttfts) + len(step_deltas),
        )
        return result, texts

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        del self._gen
        super().close()


def checkpoint_dtype(model_dir: str) -> torch.dtype:
    """The dtype the checkpoint's ``config.json`` declares (``torch_dtype``/``dtype``).

    Both engines load weights at the config's dtype, so the HF baseline must too:
    running HF in fp16 against a bf16 checkpoint measures "dtype change + engine
    change" together, not the engine difference. Falls back to fp16 when the field
    is absent (transformers' own historical default).
    """
    from transformers import AutoConfig

    declared = getattr(AutoConfig.from_pretrained(model_dir), "dtype", None)
    if isinstance(declared, str):  # older transformers returns a string
        declared = getattr(torch, declared, None)
    return declared if isinstance(declared, torch.dtype) else torch.float16


def dtype_tag(dtype: torch.dtype) -> str:
    """Short dtype name for row labels: ``torch.bfloat16`` -> ``bf16``."""
    return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, str(dtype))


#: What an arm without per-request timestamps can honestly claim. The runners
#: print the batch-level means for these rows instead of empty percentiles.
_BATCH_BASIS = "batch-level (no per-request timing)"


class HFBackend(Backend):
    """HF transformers: ``generate`` has no per-step callback, so TTFT is a separate run.

    Weights load at the checkpoint's declared dtype (see :func:`checkpoint_dtype`),
    the same precision rapid_llm uses. Under greedy, ``min_new_tokens ==
    max_gen_len`` forbids early EOS so the batch runs exactly ``max_gen_len`` steps
    (matching rapid_llm's lockstep); under sampling, early EOS is allowed, ``steps``
    is the longest sequence's, and ``gen_tokens`` counts non-pad tokens.
    """

    def __init__(self, model_dir: str, attn: str = "sdpa", *, tensor_parallel_size: int = 1):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dtype = checkpoint_dtype(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict = {"dtype": self.dtype, "attn_implementation": attn}
        if tensor_parallel_size > 1:
            # HF has no memory manager: let accelerate shard the checkpoint over
            # the visible GPUs rather than gambling one card.
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        if tensor_parallel_size == 1:
            self.model = self.model.cuda()
        self.model.eval()
        self._last_gen: torch.Tensor | None = None

    def measure_rows(self, rows, *, greedy=True, iters=1, warmup_rows=None) -> BenchResult:
        prompts = [row.prompt for row in rows]
        # ``generate`` runs one lockstep batch, so the longest row governs it.
        max_out = max(row.output_len for row in rows)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        gen_cfg: dict = {"pad_token_id": self.tokenizer.pad_token_id}
        if greedy:
            gen_cfg["do_sample"] = False
        else:
            gen_cfg.update(do_sample=True, **SAMPLE_KW)

        # Warm up cudnn/autotune so the measured run is steady state.
        for _ in range(2):
            self.model.generate(
                **inputs,
                min_new_tokens=8,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_cfg = dict(gen_cfg)
        if greedy:
            full_cfg["min_new_tokens"] = max_out  # lockstep: exactly max_out steps
        rounds = [self._one_round(inputs, gen_cfg, full_cfg, max_out) for _ in range(iters)]
        ttft, total, out = median_round(rounds, key=lambda r: r[1])

        prompt_len = inputs["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        self._last_gen = gen
        steps = gen.shape[1]
        return BenchResult(
            ttft_ms=ttft * 1000,
            tpot_ms=(total - ttft) / (steps - 1) * 1000 if steps > 1 else 0.0,
            total_s=total,
            steps=steps,
            batch=len(rows),
            gen_tokens=int((gen != self.tokenizer.pad_token_id).sum()),
        )

    def _one_round(self, inputs, gen_cfg, full_cfg, max_out) -> tuple[float, float, torch.Tensor]:
        # TTFT: one-token run, prefill plus the first sampled token.
        torch.cuda.synchronize()
        t0 = time.monotonic()
        self.model.generate(**inputs, **gen_cfg, min_new_tokens=1, max_new_tokens=1)
        torch.cuda.synchronize()
        ttft = time.monotonic() - t0

        torch.cuda.synchronize()
        t0 = time.monotonic()
        out = self.model.generate(**inputs, **full_cfg, max_new_tokens=max_out)
        torch.cuda.synchronize()
        return ttft, time.monotonic() - t0, out

    def details(self) -> dict:
        # HF generate exposes no per-request timing: TTFT is a separate 1-token
        # round and TPOT a batch-level mean, so the percentile fields stay 0 and
        # the runners print the means with a marker.
        return {"dtype": str(self.dtype), "basis": _BATCH_BASIS}

    def texts(self) -> list[str]:
        if self._last_gen is None:
            return []
        return [self.tokenizer.decode(row, skip_special_tokens=True) for row in self._last_gen]

    def sample_text(self, limit: int = 120) -> str:
        """Decode the first row of the last run, for eyeball-checking output."""
        rows = self.texts()
        return rows[0][:limit] if rows else ""

    def close(self) -> None:
        del self.model
        super().close()


class VLLMBackend(Backend):
    """vllm offline ``LLM``: batch API with no per-step callback, so TTFT is a
    separate one-token run — the same pattern :class:`HFBackend` uses, which is
    what keeps the two external baselines' TTFT columns comparable.

    Weights load at the checkpoint's declared dtype (the precision rapid_llm uses
    too). vllm's own CUDA-graph capture stays on (its default), matching
    rapid_llm's graph rows. ``ignore_eos`` runs every sequence to its full length
    — sglang's offline convention, and the only way a fixed output length is a
    fact rather than an upper bound. Percentiles come from vLLM's own request
    stats, so this arm reports a distribution the HF arm cannot.
    """

    def __init__(
        self,
        model_dir: str,
        max_model_len: int = 0,
        gpu_util: float = 0.90,
        *,
        tensor_parallel_size: int = 1,
        ignore_eos: bool = True,
        **llm_kwargs,
    ):
        from vllm import LLM

        self.dtype = checkpoint_dtype(model_dir)
        self.ignore_eos = ignore_eos
        kwargs = dict(llm_kwargs)
        kwargs.setdefault("model", model_dir)
        kwargs.setdefault("dtype", str(self.dtype).split(".")[-1])
        kwargs.setdefault("gpu_memory_utilization", gpu_util)
        kwargs.setdefault("tensor_parallel_size", tensor_parallel_size)
        # The offline LLM constructor flips this to True by default, which zeroes
        # the per-request stats read in :meth:`measure_rows`.
        kwargs.setdefault("disable_log_stats", False)
        if max_model_len:
            kwargs.setdefault("max_model_len", max_model_len)
        self.llm = LLM(**kwargs)
        self._last_texts: list[str] = []

    def measure_rows(self, rows, *, greedy=True, iters=1, warmup_rows=None) -> BenchResult:
        from vllm import SamplingParams

        prompts = [row.prompt for row in rows]
        warmup = [row.prompt for row in (warmup_rows if warmup_rows is not None else rows)]
        base = (
            {"temperature": 0.0, "top_p": 1.0}
            if greedy
            else {"temperature": SAMPLE_KW["temperature"], "top_p": SAMPLE_KW["top_p"]}
        )
        per_row = [
            SamplingParams(max_tokens=row.output_len, ignore_eos=self.ignore_eos, **base)
            for row in rows
        ]

        # Warm up capture/autotune so the measured run is steady state.
        for _ in range(2):
            self.llm.generate(warmup, SamplingParams(max_tokens=8, temperature=0.0),
                              use_tqdm=False)

        rounds = [self._one_round(prompts, per_row) for _ in range(iters)]
        ttft, total, outputs = median_round(rounds, key=lambda r: r[1])

        self._last_texts = [o.outputs[0].text for o in outputs]
        steps = max(len(o.outputs[0].token_ids) for o in outputs)
        result = BenchResult(
            ttft_ms=ttft * 1000,
            tpot_ms=(total - ttft) / (steps - 1) * 1000 if steps > 1 else 0.0,
            total_s=total,
            steps=steps,
            batch=len(rows),
            gen_tokens=sum(len(o.outputs[0].token_ids) for o in outputs),
        )
        self._fill_percentiles(result, outputs)
        return result

    def _one_round(self, prompts, per_row):
        from vllm import SamplingParams

        # TTFT: one-token run — prefill plus the first sampled token.
        torch.cuda.synchronize()
        t0 = time.monotonic()
        self.llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
        torch.cuda.synchronize()
        ttft = time.monotonic() - t0

        torch.cuda.synchronize()
        t0 = time.monotonic()
        outputs = self.llm.generate(prompts, per_row, use_tqdm=False)
        torch.cuda.synchronize()
        return ttft, time.monotonic() - t0, outputs

    @staticmethod
    def _fill_percentiles(result: BenchResult, outputs) -> None:
        """Per-request percentiles from vLLM's own v1 request stats.

        All-monotonic engine-core timestamps: TTFT is queue + prefill, TPOT the
        mean decode gap. Absent stats leave the fields at zero rather than
        inventing a distribution.
        """
        ttfts, tpots = [], []
        for o in outputs:
            m = o.metrics
            n = len(o.outputs[0].token_ids)
            if m and getattr(m, "first_token_ts", 0.0):
                ttfts.append((m.first_token_ts - m.queued_ts) * 1000)
                if n > 1 and getattr(m, "last_token_ts", 0.0) > m.first_token_ts:
                    tpots.append((m.last_token_ts - m.first_token_ts) / (n - 1) * 1000)
        if ttfts:
            result.ttft_p50_ms, result.ttft_p99_ms = pctl(ttfts, 50), pctl(ttfts, 99)
        if tpots:
            result.tpot_p50_ms, result.tpot_p99_ms = pctl(tpots, 50), pctl(tpots, 99)

    def details(self) -> dict:
        return {"dtype": str(self.dtype), "ignore_eos": self.ignore_eos}

    def texts(self) -> list[str]:
        return self._last_texts

    def sample_text(self, limit: int = 120) -> str:
        return self._last_texts[0][:limit] if self._last_texts else ""

    def close(self) -> None:
        del self.llm
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        super().close()


def make_backend(
    model_dir: str,
    *,
    use_cuda_graph: bool = True,
    tensor_parallel_size: int = 1,
    continuous: bool = False,
    image_path: str | None = None,
    max_gpu_num_blocks: int | None = None,
    **engine_kwargs,
) -> Backend:
    """Pick the rapid_llm measurement strategy for a checkpoint and parallelism.

    ``tp > 1`` goes to :class:`EngineBackend` (only the continuous-batching path
    broadcasts each step's plan), and so does ``continuous=True``: the overlap
    benches need the continuous engine on one GPU as well, because the
    copy-stream overlap and its CUDA-event timeline live in the worker rather
    than in ``TextGenerator``. A multimodal checkpoint goes to
    :class:`VisionBackend`; everything else to :class:`LiteBackend`.

    ``max_gpu_num_blocks`` is not forwarded to the multimodal path: visual tokens
    make each request's KV demand a function of image resolution, so reusing a text
    workload's pool size there only OOMs — let the engine profile it.
    """
    from rapid_llm.models.config import read_model_type
    from rapid_llm.models.registry import ModelRegistry

    if ModelRegistry.resolve(read_model_type(model_dir)).is_multimodal:
        if not image_path:
            raise ValueError(f"{model_dir} is a multimodal checkpoint and needs image_path")
        return VisionBackend(model_dir, use_cuda_graph, image_path, **engine_kwargs)
    if tensor_parallel_size > 1 or continuous:
        return EngineBackend(
            model_dir,
            tensor_parallel_size=tensor_parallel_size,
            use_cuda_graph=use_cuda_graph,
            max_gpu_num_blocks=max_gpu_num_blocks,
            **engine_kwargs,
        )
    return LiteBackend(
        model_dir,
        use_cuda_graph=use_cuda_graph,
        max_gpu_num_blocks=max_gpu_num_blocks,
        **engine_kwargs,
    )


#: The cross-engine comparison's arms, in report order.
ARMS = ("rapid_llm", "transformers", "vllm")


def build_arm(name: str, args) -> Backend:
    """Construct one arm of the cross-engine comparison from parsed CLI args.

    Each branch imports its engine inside the backend's constructor, which is
    what lets ``--engine vllm`` run under vLLM's own venv without ever importing
    rapid_llm's engine (``PYTHONPATH=<repo root>`` reaches only the lazy facade).

    Args:
        name: One of :data:`ARMS`.
        args: The offline runner's namespace (``model``, ``tensor_parallel_size``,
            ``data_parallel_size``, ``max_seq_len``, ``max_num_seqs``,
            ``engine_arg``, ``vllm_arg``, ``gpu_mem_util``, ``disable_ignore_eos``).
    """
    if name == "rapid_llm":
        engine_kwargs = dict(args.engine_arg)
        engine_kwargs.setdefault("use_cuda_graph", True)
        # Suite parity with the vLLM defaults this comparison runs against
        # (vLLM enables prefix caching by default; rapid_llm does not).
        engine_kwargs.setdefault("enable_prefix_cache", True)
        if args.max_seq_len:
            engine_kwargs.setdefault("max_seq_len", args.max_seq_len)
        if args.data_parallel_size > 1:
            return DPBackend(
                args.model,
                data_parallel_size=args.data_parallel_size,
                max_num_seqs=args.max_num_seqs or 0,
                **{k: v for k, v in engine_kwargs.items() if k != "enable_prefix_cache"},
            )
        if args.max_num_seqs:
            engine_kwargs.setdefault("max_num_seqs", args.max_num_seqs)
        return EngineBackend(
            args.model, tensor_parallel_size=args.tensor_parallel_size, **engine_kwargs
        )
    if name == "transformers":
        return HFBackend(args.model, tensor_parallel_size=args.tensor_parallel_size)
    if name == "vllm":
        return VLLMBackend(
            args.model,
            max_model_len=args.max_seq_len,
            gpu_util=args.gpu_mem_util,
            tensor_parallel_size=args.tensor_parallel_size,
            ignore_eos=not args.disable_ignore_eos,
            **dict(args.vllm_arg),
        )
    raise ValueError(f"unknown arm {name!r}; known: {ARMS}")


__all__ = [
    "ARMS",
    "Backend",
    "DPBackend",
    "EngineBackend",
    "HFBackend",
    "LiteBackend",
    "VLLMBackend",
    "VisionBackend",
    "build_arm",
    "checkpoint_dtype",
    "dtype_tag",
    "make_backend",
    "uniform_rows",
]
