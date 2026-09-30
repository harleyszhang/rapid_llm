"""User-facing ``LLM`` entry point — a vLLM-style facade over the engine.

:class:`LLM` mirrors vLLM's offline API: construct with a checkpoint, call
``generate`` with prompts plus :class:`SamplingParams`, and receive
:class:`~rapid_llm.engine.outputs.RequestOutput` objects.

Usage:
    llm = LLM(model_dir)
    outs = llm.generate(prompts, sampling_params)
"""

from __future__ import annotations

from collections.abc import Iterator

from PIL import Image

from ..distributed.parallel_state import get_tensor_model_parallel_world_size
from ..models.config import read_model_type
from ..models.registry import ModelRegistry, ModelSpec
from .llm_engine import LLMEngine
from .multimodal import MultimodalPreparer
from .outputs import CompletionOutput, RequestOutput
from .sampler import SamplingParams


def _resolve_spec(model: str) -> ModelSpec:
    """Resolve the checkpoint's architecture before the engine is built.

    The decision whether to build a multimodal preparer (and whether CUDA
    graphs are safe) must happen *before* ``LLMEngine.__init__`` runs, so the
    engine's own config load cannot be reused; the extra read costs a few KB.
    """
    return ModelRegistry.resolve(read_model_type(model))


class LLM(LLMEngine):
    """Generate completions for text or vision-language prompts.

    Args:
        model: HuggingFace checkpoint directory (``config.json`` plus ``*.safetensors``).
        tokenizer: Tokenizer location; defaults to ``model``.
        max_seq_len: Context bound; also caps the KV cache.
        max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
        device: Torch device string.
        use_cuda_graph: Capture decode CUDA graphs. ``None`` (default) enables
            them — the graph only replays decode steps, which are multimodal-free
            (vision tokens already live in the KV cache), so vision towers and
            DeepStack hooks never appear inside a capture. Tensor parallelism is
            included: a captured region then contains the blocks' all-reduce, and
            the graphs are only installed after the startup checks in
            :meth:`~rapid_llm.executor.model_runner.ModelRunner.enable_cuda_graph`
            pass on every rank. Set ``RAPID_LLM_TP_CUDA_GRAPH=0`` to force eager
            decoding there instead.
        quantization: Runtime weight quantisation (``"int8"``); ``None`` keeps
            the checkpoint's native format (fp16 or auto-detected fp8).
        tensor_parallel_size: Number of GPUs this replica's weights are split over.
        enable_expert_parallel: Split MoE experts whole-across-ranks over the
            TP group (vLLM semantics); decode keeps its CUDA graphs (EP
            defaults to lazy capture for the larger a2a buffers). The group
            state itself is set by whoever rendezvoused this process (the DP
            controller or ``from_pretrained``'s launcher) — this flag only
            drives the graph decision and is stored for introspection.
        data_parallel_size: Accepted only as ``1``. DP replicates the whole model
            across processes, which cannot be done from inside one of them; use
            :class:`~rapid_llm.engine.data_parallel.DataParallelController`, which owns
            the replicas and exposes the same ``generate``.
        kv_cache_dtype: KV-cache element type — ``"auto"`` (fp16) or an fp8
            spelling (``"fp8"`` / ``"fp8_e4m3"``), halving the cache footprint.
        cuda_graph_lazy: O13 lazy graph capture — seed pair at startup, the
            remaining ``(batch, bucket)`` shapes on first use. Trades a one-off
            ~0.5–1 s stall per new shape for a seconds-scale cold start.
        hf_overrides: Fields applied over the checkpoint's ``config.json``
            (vLLM ``--hf-overrides`` semantics), e.g.
            ``{"num_hidden_layers": 1}`` to run a trimmed stack.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool | None = None,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        kv_cache_dtype: str = "auto",
        cuda_graph_lazy: bool = False,
        hf_overrides: dict[str, object] | None = None,
    ) -> None:
        if data_parallel_size != 1:
            raise ValueError(
                f"LLM is a single model replica and cannot host "
                f"data_parallel_size={data_parallel_size}; use "
                f"DataParallelController(model=..., data_parallel_size={data_parallel_size}) "
                f"instead — it spawns one LLM per replica and routes requests to them"
            )

        # A group this process did not join is a group it cannot drive: the
        # followers wait for broadcast plans, and this class's generate loop never
        # sends any. Left unchecked the argument was silently ignored and the run
        # went single-GPU — which reads as a working TP configuration in a
        # benchmark table, so it has to be an error rather than a warning.
        if tensor_parallel_size > 1 and get_tensor_model_parallel_world_size() == 1:
            raise ValueError(
                f"LLM cannot start a tensor-parallel group: its generate loop does not "
                f"broadcast plans to follower ranks. Use "
                f"Scheduler.from_pretrained(model=..., "
                f"tensor_parallel_size={tensor_parallel_size}) instead, or construct LLM "
                f"inside a process that has already joined the group"
            )

        spec = _resolve_spec(model)
        if use_cuda_graph is None:
            use_cuda_graph = True
        # Architectures whose forward mutates Python-side per-step state (V4's
        # per-layer rolling caches) cannot be replayed from a capture.
        if use_cuda_graph and not spec.supports_cuda_graph:
            use_cuda_graph = False

        super().__init__(
            checkpoints_dir=model,
            tokenizer_path=tokenizer,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_lazy=cuda_graph_lazy,
            hf_overrides=hf_overrides,
        )

        # Strategy: only multimodal checkpoints get a preparer (and pay for the
        # AutoProcessor import/load).
        self.multimodal = MultimodalPreparer(self, model) if spec.is_multimodal else None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def is_qwen3_vl(self) -> bool:
        """Whether the loaded model is Qwen3-VL (mrope position-id path)."""
        return self.multimodal is not None and self.multimodal.is_qwen3_vl

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        images: list[Image.Image] | None = None,
    ) -> list[RequestOutput]:
        """Generate one completion per prompt; returns vLLM-shaped outputs.

        Args:
            prompts: A single prompt or a batch.
            sampling_params: Defaults to :class:`SamplingParams` defaults.
            images: Images shared by every prompt (multimodal models only).
        """
        params = sampling_params or SamplingParams()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)

        if images is not None:
            completions, reasons = self._generate_multimodal(prompts, images, params)
        else:
            token_ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
            completions = LLMEngine.generate_text(self, token_ids, params)
            reasons = self.last_stop_reasons or ["length"] * len(prompts)

        n = len(prompts)
        output_lps = self.last_output_logprobs or [None] * n
        prompt_lps = self.last_prompt_logprobs or [None] * n
        return [
            RequestOutput(
                prompt=p,
                outputs=[CompletionOutput(0, text, reason, logprobs=out_lp)],
                prompt_logprobs=prompt_lp,
            )
            for p, text, reason, out_lp, prompt_lp in zip(
                prompts, completions, reasons, output_lps, prompt_lps, strict=True
            )
        ]

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        images: list[Image.Image] | None = None,
    ) -> Iterator[list[str]]:
        """Yield incremental text per prompt at each decode step.

        Multimodal streaming serves one prompt per call (the processor path is
        single-request); text streaming is batched.
        """
        params = sampling_params or SamplingParams()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)

        if images is not None:
            preparer = self._require_multimodal()
            if len(prompts) != 1:
                raise ValueError("multimodal streaming accepts exactly one prompt")
            token_ids, mm_inputs, position_ids = preparer.prepare(prompts[0], images)
            yield from LLMEngine.generate(
                self, [token_ids], params, position_ids=position_ids, multi_modal_inputs=mm_inputs
            )
            return

        token_ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        yield from LLMEngine.generate(self, token_ids, params)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _require_multimodal(self) -> MultimodalPreparer:
        if self.multimodal is None:
            raise ValueError(
                "images were provided but the loaded model is text-only; "
                "use a vision-language checkpoint (llava / qwen3_vl)"
            )
        return self.multimodal

    def _generate_multimodal(
        self, prompts: list[str], images: list[Image.Image], params: SamplingParams
    ) -> tuple[list[str], list[str | None]]:
        """Generate per prompt with shared images (the processor path is single-request)."""
        preparer = self._require_multimodal()
        completions: list[str] = []
        reasons: list[str | None] = []
        out_lps: list = []
        prompt_lps: list = []
        for prompt in prompts:
            token_ids, mm_inputs, position_ids = preparer.prepare(prompt, images)
            completions.extend(
                LLMEngine.generate_text(
                    self,
                    [token_ids],
                    params,
                    position_ids=position_ids,
                    multi_modal_inputs=mm_inputs,
                )
            )
            # Per-call engine results are overwritten each iteration — collect
            # as we go, the same reason stop reasons are.
            reasons.append(self.last_stop_reasons[0] if self.last_stop_reasons else None)
            out_lps.append(self.last_output_logprobs[0] if self.last_output_logprobs else None)
            prompt_lps.append(self.last_prompt_logprobs[0] if self.last_prompt_logprobs else None)
        # Publish the per-prompt aggregates where generate() reads them.
        self.last_output_logprobs = out_lps if any(lp is not None for lp in out_lps) else None
        self.last_prompt_logprobs = prompt_lps if any(lp is not None for lp in prompt_lps) else None
        return completions, reasons
