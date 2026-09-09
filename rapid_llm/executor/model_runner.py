"""Model runner: builds the model, sizes the KV cache, runs each forward step.

``ModelRunner.build`` is the one-call constructor (config, loader, KV
blocks); the instance then owns the KV buffers, the request-to-token
table and ``forward`` for both phases.

Usage:
    runner = ModelRunner.build(checkpoints_dir, max_seq_len)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ..batch_overlap.two_batch_overlap import model_forward_maybe_tbo, tbo_policy
from ..distributed.parallel_state import (
    divide,
    expert_parallel_enabled,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce_min,
    tensor_model_parallel_ranks_agree,
)
from ..kernels import update_kv_index
from ..kernels.dispatcher import step_prepare_for
from ..models.config import ModelConfig
from ..models.registry import ModelRegistry, ModelSpec
from ..utils.logger import get_logger
from .attention_metadata import AttentionMetadata
from .cuda_graph import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQ_LEN_BUCKETS,
    TP_GRAPH_PARITY_ATOL,
    CUDAGraphManager,
    estimate_capture_workspace,
)
from .kv_cache_manager import KVCacheManager, MemoryProfiler
from .loader import DefaultModelLoader, ModelLoader
from .slot_batch import SlotBatch

logger = get_logger(__name__)

#: Set to ``0`` to keep the pre-TP-graph behaviour: eager decode when ``tp_size > 1``.
#: A kill-switch, not a config field — the failure it guards is a hang, and someone
#: meeting one needs a way out that does not involve editing code.
_TP_GRAPH_ENV = "RAPID_LLM_TP_CUDA_GRAPH"


class ModelRunner:
    """Owns the model, the KV-cache memory manager, and the per-step attention state."""

    def __init__(
        self,
        checkpoints_dir: str,
        config: ModelConfig,
        spec: ModelSpec,
        model: nn.Module,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
        cuda_graph_lazy: bool = False,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.config = config
        self.spec = spec
        self.model = model
        self.device = device

        # ModelConfig already normalises the geometry and unwraps a VLM's text config.
        self.num_layers = config.num_layers
        if config.is_mla:
            # MLA caches one latent vector per token: no head axis to shard, so every
            # TP rank holds the full row (as vLLM does).
            self.kv_row = (1, config.kv_lora_rank + config.qk_rope_head_dim)
        elif config.model_type == "deepseek_v4":
            # V4's kv_proj emits one head_dim-wide latent row per token — the
            # same no-head-axis geometry as MLA. The layers manage their own
            # sliding-window/compressor state on top; this row is the engine
            # side's reservation shape, never sharded.
            self.kv_row = (1, config.head_dim)
        else:
            # Heads are dealt across TP ranks, so this rank caches only its own K/V.
            kv_heads = divide(
                config.num_kv_heads, get_tensor_model_parallel_world_size(), "key/value heads"
            )
            self.kv_row = (2 * kv_heads, config.head_dim)
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        # Paged KV element type: fp16, or uint8 holding e4m3 for an fp8 cache.
        kv_dtype = config.kv_cache_torch_dtype

        if max_gpu_num_blocks is None:
            # Withhold decode-graph workspace from the KV budget (capture OOMs once
            # the cache fills the card); lazy mode withholds only the seed pair.
            reserved = (
                estimate_capture_workspace(self.max_seq_len, lazy=cuda_graph_lazy)
                if use_cuda_graph
                else 0
            )
            profiler = MemoryProfiler(
                num_layers=self.num_layers,
                kv_row=self.kv_row,
                dtype=kv_dtype,
                device=device,
                reserved_bytes=reserved,
            )
            # Every rank must agree: a rank's cache rows derive from its capacity, and
            # differing capacities would write the same token to different rows.
            max_gpu_num_blocks = tensor_model_parallel_all_reduce_min(
                profiler.available_kv_blocks(model, self.vocab_size)
            )

        self.kv_cache_manager = KVCacheManager(
            num_layers=self.num_layers,
            kv_row=self.kv_row,
            gpu_num_blocks=max_gpu_num_blocks,
            dtype=kv_dtype,
            device=device,
        )
        # Block-table rows = concurrency ceiling. Paging decoupled this from cache
        # size (a slot's rows are pages it holds), so the table is sized by in-flight
        # requests. One row above the largest batch covers graphs + filler; decided
        # here because a capture bakes this tensor's pointer.
        self.max_request_num = max(DEFAULT_BATCH_SIZES) + 1
        # Request -> KV-row map; row i holds the cache rows of ``b_req_idx == i``.
        # Written at prefill, extended by ``update_kv_index`` each decode step; under
        # continuous batching it is a block table the scheduler fills.
        self.b_req_tokens_table = torch.zeros(
            (self.max_request_num, self.max_seq_len), dtype=torch.int32, device=device
        )

        self.atten_info = AttentionMetadata()
        self.atten_info.kv_buffer = self.kv_cache_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.b_req_tokens_table

        # Set by :meth:`enable_cuda_graph`; when set, :meth:`forward` replays graphs.
        self._graph_manager: CUDAGraphManager | None = None
        # Set by :meth:`enable_slot_kv_cache` for continuous batching.
        self._slot_batch: SlotBatch | None = None

    # ------------------------------------------------------------------ build #
    @classmethod
    def build(
        cls,
        checkpoints_dir: str,
        max_seq_len: int,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
        loader: ModelLoader | None = None,
        quantization: str | None = None,
        kv_cache_dtype: str = "auto",
        cuda_graph_lazy: bool = False,
        hf_overrides: dict[str, object] | None = None,
    ) -> ModelRunner:
        """Load config + weights and return a ready-to-run runner.

        Args:
            checkpoints_dir: HF checkpoint dir (``config.json`` + ``*.safetensors``).
            max_seq_len: Sequence-length bound; also bounds the KV cache.
            max_gpu_num_blocks: Manual KV size in tokens; profiled when ``None``.
            device: Torch device string.
            use_cuda_graph: Reserve capture workspace when profiling, so a later
                :meth:`enable_cuda_graph` does not OOM.
            loader: Weight-loading strategy (default
                :class:`~rapid_llm.executor.loader.DefaultModelLoader`); inject a
                fake in tests.
            quantization: Quantisation for an fp16 checkpoint (``"int8"``); fp8
                checkpoints carry their own and ignore this.
            kv_cache_dtype: ``"auto"`` (fp16) or ``"fp8"``/``"fp8_e4m3"``, which
                halves the cache so twice as many tokens fit.
            cuda_graph_lazy: Withhold only the seed pair's workspace (O13); pair with
                :meth:`enable_cuda_graph`'s ``lazy`` flag.
            hf_overrides: Fields over the checkpoint's ``config.json`` (vLLM
                ``--hf-overrides``), e.g. ``{"num_hidden_layers": 1}``.
        """
        config = ModelConfig.from_pretrained(
            checkpoints_dir, max_seq_len, kv_cache_dtype, hf_overrides=hf_overrides
        )
        spec = ModelRegistry.resolve(config.model_type)
        model = (loader or DefaultModelLoader()).load_model(
            config, spec.load_class(), checkpoints_dir, device, quantization
        )
        return cls(
            checkpoints_dir,
            config,
            spec,
            model,
            max_gpu_num_blocks,
            device,
            use_cuda_graph,
            cuda_graph_lazy=cuda_graph_lazy,
        )

    # --------------------------------------------------------- kv allocation #
    def _init_req_tokens_table(
        self, b_req_idx, b_seq_len, alloc_index, max_prompt_len
    ) -> torch.Tensor:
        """Record which cache rows each prefill sequence occupies.

        The model flattens the padded ``[batch, max_prompt_len]`` grid row-major, so
        sequence ``i``'s ``j``-th token is at ``i * max_prompt_len + j`` and its K/V
        land in ``alloc_index`` at the same offset. The table must use that padded
        layout — a packed layout would point sequence ``i`` at rows written by
        sequence ``i - 1``'s tail, corrupting every sequence after the first. One
        masked gather + ``index_put_``, not a per-sequence Python loop (which cost a
        host round-trip and a launch per row).

        Returns:
            ``b_start_loc``: start offset of each sequence in the flattened batch.
        """
        n = b_seq_len.shape[0]
        rows = torch.arange(n, device=self.device)
        cols = torch.arange(max_prompt_len, device=self.device)
        # [n, max_prompt_len] — True where the padded column is a real token.
        valid = cols.unsqueeze(0) < b_seq_len.view(-1, 1)
        # Flattened source offsets into the padded allocation, row-major.
        src = alloc_index[(rows * max_prompt_len).unsqueeze(1) + cols.unsqueeze(0)]

        table = self.atten_info.b_req_tokens_table
        table.index_put_(
            (
                b_req_idx.view(-1, 1).expand(n, max_prompt_len)[valid],
                cols.unsqueeze(0).expand(n, max_prompt_len)[valid],
            ),
            src[valid],
        )
        return (rows * max_prompt_len).to(torch.int32)

    def prefill_alloc_kv_cache(self, max_prompt_len, actual_prompt_lens, b_req_idx) -> torch.Tensor:
        """Reserve cache rows for the whole prompt batch.

        Vision tokens need no special handling: the prompt already has one token per
        vision patch (the processor expanded ``<image>``), so this covers them.
        """
        batch_size = len(actual_prompt_lens)
        self.atten_info.b_req_idx = b_req_idx
        self.atten_info.cur_select_index = self.kv_cache_manager.alloc_kvcache_index(
            max_prompt_len * batch_size
        )
        self.atten_info.b_seq_len = actual_prompt_lens
        self.atten_info.max_actual_seq_len = max_prompt_len
        self.atten_info.is_prefill = True
        # One-shot prefills start at position zero, so clear the chunked routing fields.
        self.atten_info.b_prefix_len = None
        self.atten_info.b_kv_base = None
        self.atten_info.max_chunk_len = 0
        self.atten_info.b_start_loc = self._init_req_tokens_table(
            b_req_idx, actual_prompt_lens, self.atten_info.cur_select_index, max_prompt_len
        )
        return self.atten_info.cur_select_index

    def decode_alloc_kv_cache(self, batch_size) -> torch.Tensor:
        """Reserve one cache row per sequence for the next decode step.

        ``update_kv_index`` writes at ``b_seq_len - 1``, so ``b_seq_len`` must be
        incremented *before* the kernel launches (doing it after overwrote the last
        prompt token's mapping, giving non-deterministic completions once a second
        request shared the runner).
        """
        self.atten_info.cur_select_index = self.kv_cache_manager.alloc_kvcache_index(batch_size)
        self.atten_info.b_seq_len += 1
        self.atten_info.max_actual_seq_len += 1
        self.atten_info.is_prefill = False
        update_kv_index(
            self.atten_info.b_req_tokens_table,
            self.atten_info.b_req_idx,
            self.atten_info.b_seq_len,
            self.atten_info.cur_select_index,
        )
        return self.atten_info.cur_select_index

    # ------------------------------------------------------------- inference #
    def enable_slot_kv_cache(self) -> SlotBatch:
        """Switch the KV cache to the paged block-table layout continuous batching needs.

        Idempotent, and mutually exclusive with the one-shot path: the returned
        :class:`~rapid_llm.executor.slot_batch.SlotBatch` reads rows from the block
        table the scheduler fills, while ``*_alloc_kv_cache`` allocate rows themselves.
        Call *after* :meth:`enable_cuda_graph` so padding sees the captured grid.
        """
        if self._slot_batch is None:
            self._slot_batch = SlotBatch(self)
        return self._slot_batch

    def graph_batch_size(self, batch_size: int) -> int:
        """Batch size to submit so a decode step lands on a captured graph.

        ``batch_size`` unchanged when graphs are off or the batch exceeds anything
        captured (the step then runs eager).
        """
        if self._graph_manager is None:
            return batch_size
        return self._graph_manager.pad_to(batch_size) or batch_size

    def enable_cuda_graph(
        self,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        seq_len_buckets: tuple[int, ...] = DEFAULT_SEQ_LEN_BUCKETS,
        *,
        lazy: bool = False,
        tbo: bool = False,
    ) -> None:
        """Capture decode graphs for the given ``(batch, seq_len_bucket)`` grid.

        Multimodal models are supported: a capture only ever replays a decode
        step, and by then the vision tokens are ordinary KV-cache rows — the
        vision tower and DeepStack hooks run during prefill, which stays eager.
        """

        if torch.device(self.device).type != "cuda":
            logger.info("CUDA Graph is unavailable on %s; using eager execution", self.device)
            return
        if self._graph_manager is not None:
            return  # idempotent

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and os.environ.get(_TP_GRAPH_ENV, "1") == "0":
            logger.warning(
                "%s=0: CUDA Graph disabled under tensor parallelism; running eager.",
                _TP_GRAPH_ENV,
            )
            return
        # A captured graph replays the Python side of a kernel call verbatim, so
        # a decode backend that assembles per-step inputs on the host would bake
        # the capture-time lengths in and silently attend stale rows. vLLM runs
        # the same gate (AttentionCGSupport) before its first capture.
        from ..kernels.dispatcher import unsafe_for_graph

        for module in self.model.modules():
            bind = getattr(module, "_bind_kernels", None)
            if bind is not None:
                bind("cuda")

        unsafe = unsafe_for_graph("attention.decode", device_type="cuda") + unsafe_for_graph(
            "attention.mla_decode", device_type="cuda"
        )
        if unsafe:
            raise ValueError(
                "CUDA graph capture refused: the selected attention decode "
                f"backend(s) {', '.join(unsafe)} assemble per-step inputs on the "
                "host and would replay capture-time lengths forever. Pin the "
                "native kernel (RAPID_LLM_ATTENTION_DECODE_BACKEND=native) or "
                "run without graphs."
            )

        seq_len_buckets = tuple(b for b in seq_len_buckets if b <= self.max_seq_len)
        if not seq_len_buckets:
            logger.warning(
                "max_seq_len=%d is smaller than every requested bucket; skipping capture",
                self.max_seq_len,
            )
            return

        # The table has only max_request_num rows; a larger batch would index past it
        # and corrupt the CUDA context.
        batch_sizes = tuple(b for b in batch_sizes if b <= self.max_request_num)
        if not batch_sizes:
            logger.warning(
                "max_request_num=%d is smaller than every requested batch size; skipping capture",
                self.max_request_num,
            )
            return

        logger.info(
            "Capturing CUDA graphs for batch_sizes=%s seq_len_buckets=%s",
            batch_sizes,
            seq_len_buckets,
        )
        manager = CUDAGraphManager(
            self.model,
            kv_buffer=self.kv_cache_manager.gpu_kv_buffer,
            b_req_tokens_table=self.b_req_tokens_table,
            batch_sizes=batch_sizes,
            seq_len_buckets=seq_len_buckets,
            device=self.device,
            lazy=lazy,
            step_factory=self._tbo_step_factory() if tbo else None,
        )
        captured = True
        try:
            if lazy:
                manager.capture_seed()
            else:
                manager.capture_all()
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as exc:
            # A failed capture may leave a half-open graph; dropping the manager
            # is safe because replay state is only installed on success. An
            # allocation failure inside ``capture_end`` surfaces as a generic
            # CUDA ``AcceleratorError`` ("out of memory") rather than
            # ``OutOfMemoryError``; EP's a2a buffers make each graph far larger
            # than the dense estimate the KV profiler reserved, so capturing a
            # full EP grid beside a profiled KV pool lands here. Anything that
            # is not an OOM is a real capture bug and must not be swallowed.
            if (
                not isinstance(exc, torch.cuda.OutOfMemoryError)
                and "out of memory" not in str(exc).lower()
            ):
                raise
            logger.warning("CUDA graph capture ran out of memory; falling back to eager decode")
            captured = False

        if tp_size > 1 and not self._tp_graphs_are_safe(manager, captured):
            manager.discard()
            return
        if not captured:
            return

        self._graph_manager = manager

    def _tp_graphs_are_safe(self, manager: CUDAGraphManager, captured: bool) -> bool:
        """Whether this rank's captured graphs may serve traffic. Same answer everywhere.

        Two checks, in this order because the first is what makes the second
        well-defined:

        1. **Grid agreement.** Every rank contributes a fingerprint of what it
           captured, or ``0`` if it captured nothing. Ranks that disagree all
           return ``False`` together — an asymmetric grid means one rank replays
           where another runs eager, and the replayed all-reduce then waits
           forever. A grid of ``0`` fails even when unanimous, so passing this
           establishes that every rank *has* graphs and the parity check below is
           entered by all of them or none.
        2. **Numerical parity.** Each rank compares every captured graph against
           an eager step on the same synthetic input and keeps the graphs only if
           all ranks are within :data:`TP_GRAPH_PARITY_ATOL`. Reduced with a
           minimum so one rank's failure retires the graphs everywhere; a group
           where half the ranks replay is the hang again.

        Both results are functions of a collective's output, which is why every
        rank branches the same way without a second round of agreement.
        """
        fingerprint = manager.grid_fingerprint() if captured else 0
        if not tensor_model_parallel_ranks_agree(fingerprint) or fingerprint == 0:
            logger.warning(
                "tensor-parallel ranks captured different CUDA graph grids "
                "(this rank: %d); dropping graphs on every rank and decoding eager",
                fingerprint,
            )
            return False

        error = manager.max_parity_error(self.vocab_size)
        # Phrased as ``error <= tol`` rather than ``not error > tol`` so that a NaN
        # difference — the signature of a graph reading freed memory — fails the
        # gate instead of slipping through a negated comparison.
        local_ok = error <= TP_GRAPH_PARITY_ATOL
        if tensor_model_parallel_all_reduce_min(int(local_ok)) != 1:
            logger.warning(
                "CUDA graph replay disagrees with eager decode by %.3e (tolerance %.1e) "
                "on at least one rank; dropping graphs and decoding eager. "
                "Per graph: %s",
                error,
                TP_GRAPH_PARITY_ATOL,
                manager.parity_errors,
            )
            return False

        logger.info(
            "TP CUDA graphs verified: worst graph-vs-eager logit difference %.3e (tolerance %.1e)",
            error,
            TP_GRAPH_PARITY_ATOL,
        )
        return True

    @property
    def uses_cuda_graph(self) -> bool:
        """Whether at least one decode CUDA graph is installed."""
        return self._graph_manager is not None

    def release_cuda_graph(self) -> None:
        """Drop every captured graph so a process-group teardown cannot block.

        The follower teardown calls this before destroying the TP group: a
        captured region can hold NCCL kernels registered with the communicator,
        and ``destroy_process_group`` would wait on them forever. Eager engines
        (no manager) pass through untouched.
        """
        if self._graph_manager is not None:
            self._graph_manager.discard()
            self._graph_manager = None

    def _run_tbo(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionMetadata,
        *,
        prefill: bool = False,
    ) -> torch.Tensor:
        """The interleave itself, through the batch_overlap entry.

        One implementation behind both arms — the eager one
        (:meth:`forward_tbo`, metadata installed by ``begin_decode``) and the
        captured one (:meth:`_tbo_step`, metadata from the graph's persistent
        surface). They differ only in where the metadata comes from, so the
        captured arm cannot drift from the eager reference it is tested
        against.

        ``prefill`` splits by sequence with a token-balanced cut and runs the
        prefill op stream; a decode step splits by row.
        """
        return model_forward_maybe_tbo(
            self.model,
            enable_tbo=True,
            input_ids=input_ids,
            position_ids=position_ids,
            atten_info=atten_info,
            prefill=prefill,
        )

    def _tbo_step(self) -> Callable[[torch.Tensor, torch.Tensor, AttentionMetadata], torch.Tensor]:
        """The step shape a TBO graph records: split, interleave, concat."""

        def step(
            input_ids: torch.Tensor, position_ids: torch.Tensor, atten_info: AttentionMetadata
        ) -> torch.Tensor:
            return self._run_tbo(input_ids, position_ids, atten_info)

        return step

    def _tbo_step_factory(self) -> Callable[[int], Callable | None]:
        """Per-batch decision on which step shape a captured graph records."""
        policy = tbo_policy()
        world_size = get_tensor_model_parallel_world_size()
        expert_parallel = expert_parallel_enabled()

        def factory(batch_size: int) -> Callable | None:
            if policy.capture_eligible(
                world_size=world_size, batch=batch_size, expert_parallel=expert_parallel
            ):
                return self._tbo_step()
            return None

        return factory

    def forward_maybe_tbo(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *, enable_tbo: bool
    ) -> torch.Tensor:
        """One decode step through the batch_overlap entry, overlapped or not.

        sglang's shape: the caller (the worker, from :func:`tbo_policy`) answers
        the policy question and hands it over as ``enable_tbo``; this method and
        the module it delegates to own the execution. ``False`` runs the same
        op stream serially (sglang's ``_model_forward_non_tbo``), so a decode
        step's overlapped and plain arms share one definition of a layer.
        The metadata must already be installed for the whole step
        (``begin_decode``), which the decode path does before calling either arm.

        Args:
            input_ids: ``[rows, 1]`` the step's token ids.
            position_ids: ``[rows, 1]`` absolute position per row.
            enable_tbo: The policy's verdict for this step.

        Returns:
            ``[rows, 1, vocab]`` logits, rows in batch order -- the same
            shape :meth:`forward` returns for the step.
        """
        if self._graph_manager is not None:
            replayed = self._graph_manager.try_replay(input_ids, position_ids, self.atten_info)
            if replayed is not None:
                return replayed

        prepare = step_prepare_for("attention.decode") if input_ids.is_cuda else None
        if prepare is not None:
            prepare(self.atten_info, self)
        return model_forward_maybe_tbo(
            self.model,
            enable_tbo=enable_tbo,
            input_ids=input_ids,
            position_ids=position_ids,
            atten_info=self.atten_info,
        )

    def forward_tbo(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """One decode step in two interleaved halves (L2 two-batch overlap).

        Splits the step with :class:`~rapid_llm.batch_overlap.two_batch_overlap.TboSplitter`
        -- narrow views of the inputs plus per-half attention metadata over
        the shared paged KV cache -- and runs the halves through the
        batch_overlap entry's TBO arm: :func:`model_forward_maybe_tbo`
        (``two_batch_overlap``) with ``enable_tbo=True``. The metadata must
        already be installed for the whole step (``begin_decode``), which
        the decode path does before choosing between this and
        :meth:`forward_maybe_tbo`'s plain arm.

        Args:
            input_ids: ``[rows, 1]`` the step's token ids.
            position_ids: ``[rows, 1]`` absolute position per row.

        Returns:
            ``[rows, 1, vocab]`` logits, rows in batch order -- the same
            shape :meth:`forward` returns for the step.
        """
        return self._run_tbo(input_ids, position_ids, self.atten_info)

    def forward_tbo_prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One prefill pass in two interleaved halves.

        Splits the grid by *sequence* with sglang's token-balanced cut, so a
        batch of uneven prompts does not hand one half most of the work. The
        prefill op stream alternates strictly (no lead) and hides the shared
        MLP behind the return exchange.

        Args:
            input_ids: ``[num_seqs, max_prompt_len]`` the padded prompt grid.
            position_ids: Same shape, absolute positions.
            logits_positions: Per-sequence position whose logits to keep.

        Returns:
            The same shape :meth:`forward` returns for a prefill.
        """
        out = self._run_tbo(input_ids, position_ids, self.atten_info, prefill=True)
        if logits_positions is not None:
            # Each half dropped its padded rows, so the kept positions are the
            # caller's, re-indexed against the concatenated halves.
            return out[:, logits_positions, :] if out.dim() == 3 else out
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        multi_modal_inputs: dict[str, Any] | None = None,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one model step.

        Dispatches to a captured graph when the step is a decode (``seq_len == 1``)
        whose ``(batch_size, max_actual_seq_len)`` matches a captured bucket; else eager.

        Args:
            input_ids: ``[batch, seq_len]`` token ids.
            position_ids: Absolute positions for this step.
            multi_modal_inputs: Processor outputs for a multimodal prefill.
            logits_positions: Optional ``[batch]`` positions whose logits are wanted.
                Given (prefill), the model gathers those hidden states *before* the
                lm_head GEMM and returns ``[batch, vocab]``, skipping seq_len-1 vocab
                projections. ``None`` (decode/replay) returns full logits.
        """
        if self._graph_manager is not None and multi_modal_inputs is None:
            # A decode step with no vision payload: eligible for replay on text and
            # multimodal alike (the latter's vision tokens already sit in the cache).
            replayed = self._graph_manager.try_replay(input_ids, position_ids, self.atten_info)
            if replayed is not None:
                return replayed

        # Eager decode step: give the winning backend its once-per-step shot at
        # hoisting per-layer host work (index assembly, wrapper planning) out
        # of the layer loop — the role vLLM's build_metadata plays. Prefill
        # passes and multimodal prefills never run it; the graph path needs no
        # hook because its selected rows are all graph_safe by the gate above.
        if input_ids.is_cuda and multi_modal_inputs is None and input_ids.shape[-1] == 1:
            prepare = step_prepare_for("attention.decode")
            if prepare is not None:
                prepare(self.atten_info, self)

        if self.spec.is_multimodal:
            return self.model(
                input_ids, position_ids, self.atten_info, multi_modal_inputs, logits_positions
            )

        return self.model(
            input_ids, position_ids, self.atten_info, logits_positions=logits_positions
        )
