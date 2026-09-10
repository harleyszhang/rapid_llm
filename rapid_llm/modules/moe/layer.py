"""MoE layer (the orchestrator): a five-stage routed sparse FFN.

:class:`SparseMoeBlock` is rapid_llm's ``FusedMoE`` equivalent. It wires the
three seams the refactor split out -- :class:`~rapid_llm.modules.moe.router.TopKRouter`
(stage 1), a :class:`~rapid_llm.modules.moe.token_dispatcher.base.BaseDispatcher`
(stages 2 & 5), and a :class:`~rapid_llm.modules.moe.moe_runner.MoeRunner`
(stages 3-4) -- into one forward pass:

1. route     -- ``router.forward`` turns tokens into ``(weights, ids)``.
2. dispatch  -- the dispatcher shuffles tokens to the ranks owning their experts.
3-4. run     -- the runner's grouped GEMM over the received batch.
5. combine   -- the dispatcher reduces the results back to ``[tokens, hidden]``.
finalize     -- routed scaling / shared expert / deferred-AR fence / TP all-reduce.

Under DP-attention stages 2-5 take one of two pooled shapes instead, both
described in ``distributed.dp_attention``: the a2a rectangle, or the default
AgRs ragged pool, whose exchange *is* the dispatch/combine (no dispatcher
object involved).

Two route families, dispatched on the HF ``topk_method``: greedy top-k (Qwen3-MoE,
DeepSeek-V2-Lite) and :func:`grouped_topk` (the group-limited selection DeepSeek-V2
and the biased ``noaux_tc`` routing DeepSeek-V2.5+/V3 ship).

Usage:
    moe = SparseMoeBlock(config, quant)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...batch_overlap import CommStreamPool, current_deferred_ar
from ...batch_overlap.single_batch_overlap import SboFlags, sbo_alt_stream
from ...distributed.dp_attention import (
    DPMetadata,
    current_dp_metadata,
    dp_combine,
    dp_dispatch,
    dp_gather,
    dp_scatter,
)
from ...distributed.parallel_state import (
    divide,
    expert_parallel_enabled,
    get_ep_rank,
    get_ep_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from ...kernels.ops.moe.ep_dispatch import ep_local_ids
from ...models.config import ModelConfig
from ..mlp import FusedMLP
from ..quantization import QuantizationConfig, RawParameter, UnquantizedFusedMoEMethod
from .moe_runner import MoeRunner, MoeRunnerConfig
from .router import TopKRouter
from .token_dispatcher import (
    AllToAllDispatcher,
    DispatchHandle,
    StandardDispatcher,
    get_dispatcher,
)
from .utils import MoeA2ABackend, get_moe_a2a_backend


@dataclass
class MoEOpContext:
    """Per-micro-batch state the EP op stream threads between its ops.

    One context per half per layer invocation; the TBO strategy owns the
    lifetime, the block's ``op_*`` methods only read/write these fields.
    """

    weights: torch.Tensor | None = None
    ids: torch.Tensor | None = None
    handle: DispatchHandle | None = None
    shared: torch.Tensor | None = None
    local_ids: torch.Tensor | None = None
    local_weights: torch.Tensor | None = None
    leading_shape: tuple[int, ...] | None = None


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Reads the HF MoE fields (``num_experts``, ``num_experts_per_tok``,
    ``moe_intermediate_size``, ``norm_topk_prob``); DeepSeek configs also drive the shared
    expert, ``routed_scaling_factor`` and the routing family (``topk_method``: ``greedy``
    vs the grouped ``noaux_tc``/``group_limited_greedy``). The router stays in the model
    dtype: it is ``num_experts x hidden``, small enough to be free and precise enough to
    matter (a wrong top-k pick costs far more than a rounded weight).

    The three stage seams are composed in :meth:`__init__`: a :class:`TopKRouter`
    (stage 1), a dispatcher resolved from the active :class:`MoeA2ABackend`
    (stages 2 & 5), and a :class:`MoeRunner` (stages 3-4). ``gate_weight`` and the
    expert storage stay registered on the block so checkpoint loading and the
    ``isinstance(module, SparseMoeBlock)`` weight scan are unchanged.
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # Routing fields read defensively: configs outside DeepSeek (Qwen3-MoE, test
        # doubles) do not carry them.
        self.scoring_func = str(getattr(config, "scoring_func", "softmax") or "softmax")
        # ``greedy`` is plain top-k; the grouped methods first pick which expert groups
        # a token may draw from.
        self.topk_method = str(getattr(config, "topk_method", "greedy") or "greedy")
        self.n_group = int(getattr(config, "n_group", 1) or 1)
        self.topk_group = int(getattr(config, "topk_group", 1) or 1)
        self.hidden_size = config.hidden_size
        # Expert placement, vLLM's ``--enable-expert-parallel`` semantics: EP
        # owns whole experts ``[offset, offset + num_local)`` and does *not*
        # TP-split the intermediate (the two expert splits are mutually
        # exclusive); without EP every expert is TP-sliced along it.
        self.ep_enabled = expert_parallel_enabled() and get_ep_world_size() > 1
        if self.ep_enabled:
            ep_world = get_ep_world_size()
            self.num_local_experts = divide(config.num_experts, ep_world, "experts per EP rank")
            self.expert_offset = get_ep_rank() * self.num_local_experts
            self.moe_intermediate_size = config.moe_intermediate_size
        else:
            self.num_local_experts = config.num_experts
            self.expert_offset = 0
            self.moe_intermediate_size = divide(
                config.moe_intermediate_size,
                get_tensor_model_parallel_world_size(),
                "MoE intermediate",
            )
        self.quant = quant
        # The model dtype drives every unquantised tensor this block owns (the router and
        # the expert storage the quant method allocates).
        self.dtype = config.dtype

        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
        )

        self.gate_e_score_correction_bias: nn.Parameter | None = None
        if self.topk_method == "noaux_tc":
            self.gate_e_score_correction_bias = RawParameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )

        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedFusedMoEMethod()
        )
        self.experts = nn.ParameterDict(self.quant_method.create_weights(self))

        for param in self.experts.values():
            param.weight_loader = self._expert_loader

        self.shared_experts: FusedMLP | None = None
        if config.n_shared_experts > 0:
            self.shared_experts = FusedMLP(
                config,
                quant,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )

        # Stage 1: the router (gate_weight stays on the block; the router reads it).
        self.router = TopKRouter(
            top_k=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            scoring_func=self.scoring_func,
            topk_method=self.topk_method,
            n_group=self.n_group,
            topk_group=self.topk_group,
        )

        # Stages 2 & 5: the dispatch/combine seam. ``ALL_TO_ALL`` when EP is on
        # without DP-attention; ``ALLGATHER_REDUCESCATTER`` -- the DP-attention
        # default -- builds *no* dispatcher, because its exchange lives in
        # ``distributed.dp_attention`` and the forward branches on the backend;
        # ``NONE`` otherwise. The dispatcher is stateless apart from the
        # placement, so one instance serves every forward; handles carry the
        # per-call buffers.
        self.a2a_backend = get_moe_a2a_backend()
        self.dispatcher: AllToAllDispatcher | None = (
            get_dispatcher(
                self.a2a_backend,
                self.num_experts,
                self.num_local_experts,
                self.expert_offset,
            )
            if self.a2a_backend is MoeA2ABackend.ALL_TO_ALL
            else None
        )
        # The non-EP passthrough seam: dispatch/combine are identity, the TP
        # all-reduce over the expert split stays in :meth:`forward`.
        self.std_dispatcher = StandardDispatcher()

        # Stages 3-4: the expert-compute runner (Triton grouped GEMM).
        self.runner = MoeRunner(
            MoeRunnerConfig(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                expert_offset=self.expert_offset,
                hidden_size=self.hidden_size,
                top_k=self.top_k,
                moe_intermediate_size=self.moe_intermediate_size,
            ),
            self.a2a_backend,
        )

    def _expert_loader(self, param, loaded, shard_id) -> torch.Tensor:
        """Fill one expert's slice of a stacked parameter; return the view written.

        ``shard_id`` is ``(expert_index, projection)`` with gate=0, up=1,
        down=2. gate/up share one stacked tensor fused along dim 1, so each
        fills its half of the expert's slice, TP-sharded along the incoming
        rows; down fills a whole slice, sharded along the columns. Quantised
        scale grids follow the same rule — their axes count scale blocks, so
        the same proportional narrow applies. A checkpoint that ships experts
        already stacked carries no shard id. Without EP it requires one rank;
        under EP its expert axis is sliced to the local rank before copying.

        Under EP the parameter is indexed by *local* expert and holds whole
        experts (no TP narrow): ids outside ``[expert_offset, expert_offset +
        num_local_experts)`` live on other ranks and are skipped — the loader
        contract wants the written view back, so they get an empty one.
        """
        if shard_id is None:
            if self.ep_enabled:
                loaded = loaded.narrow(0, self.expert_offset, self.num_local_experts)
            elif get_tensor_model_parallel_world_size() > 1:
                raise ValueError(
                    "a checkpoint with pre-stacked experts cannot be TP-sharded on "
                    "load; use the per-expert layout or enable expert parallelism"
                )
            if param.shape != loaded.shape:
                raise ValueError(
                    f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                    f"parameter of shape {tuple(param.shape)}"
                )
            param.data.copy_(loaded)
            return param.data

        expert_index, proj = shard_id
        if self.ep_enabled:
            local = expert_index - self.expert_offset
            if not 0 <= local < self.num_local_experts:
                return param.data[:0]
            view = param.data[local]
            if proj < 2:
                half = view.shape[0] // 2
                view = view.narrow(0, proj * half, half)
            if view.shape != loaded.shape:
                raise ValueError(
                    f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                    f"parameter view of shape {tuple(view.shape)}"
                )
            view.copy_(loaded)
            return view
        view = param.data[expert_index]

        if proj < 2:
            half = view.shape[0] // 2
            view = view.narrow(0, proj * half, half)
            dim = 0
        else:
            dim = 1

        world_size = get_tensor_model_parallel_world_size()

        if world_size > 1:
            size = loaded.shape[dim] // world_size
            loaded = loaded.narrow(dim, get_tensor_model_parallel_rank() * size, size)

        if view.shape != loaded.shape:
            raise ValueError(
                f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                f"parameter view of shape {tuple(view.shape)}"
            )
        view.copy_(loaded)
        return view

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: per-token expert ids and weights (HF-compatible ordering).

        Delegates to :meth:`TopKRouter.forward` and unpacks its :class:`TopKOutput`
        into the historical ``(weights, ids)`` pair the forward paths and the op
        stream expect; weights are in ``x.dtype``.
        """
        topk = self.router.forward(x, self.gate_weight, self.gate_e_score_correction_bias)
        return topk.topk_weights, topk.topk_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)

        # Under DP-attention the rows above are one DP rank's slice of the step,
        # so the routed stage runs over the pooled batch instead (the routing
        # itself included -- a token's experts must be picked from the same
        # weights on whichever rank ends up computing them).
        metadata = current_dp_metadata()
        if self.a2a_backend is MoeA2ABackend.ALLGATHER_REDUCESCATTER:
            if metadata is None:
                # AgRs has no a2a handle to fall back on, and the dense path
                # would index this rank's local experts with global ids.
                raise RuntimeError(
                    "the AgRs MoE backend is selected but no DP-attention region is "
                    "active: there is no all-to-all handle to fall back on, and the "
                    "dense path would index local experts with global routing ids"
                )
            out, shared = self._forward_dp_attention_agrs(x, metadata)
        elif self.a2a_backend is MoeA2ABackend.ALL_TO_ALL:
            if metadata is not None:
                out, shared = self._forward_dp_attention_a2a(x, metadata)
            else:
                # EP: tokens travel to the ranks owning their experts and the
                # results travel back — the combine already lands every token's
                # full routed sum here, so no all_reduce follows (unlike the TP
                # expert split below, where each rank only holds a partial sum).
                weights, ids = self._route(x)
                out, shared = self._forward_ep(x, ids, weights)
        else:
            # Dense TP path expressed on the standard seam: route, then the
            # passthrough dispatch/combine bracket the grouped GEMM (both
            # identity -- every rank already holds all tokens), then the TP
            # all-reduce over the expert split. Equivalent to the pre-refactor
            # ``quant_method.apply`` + ``tensor_model_parallel_all_reduce``.
            topk = self.router.forward(x, self.gate_weight, self.gate_e_score_correction_bias)
            disp = self.std_dispatcher.dispatch(x, topk)
            local_out = self._run_experts(disp.hidden_states, disp.topk_ids, disp.topk_weights)
            out = self.std_dispatcher.combine(local_out)
            # Each rank's routed partial sum joined the all_reduce; the shared MLP's
            # down_proj is row-parallel and reduces on its own, and summing after is
            # the same total (all_reduce(a) + all_reduce(b) == all_reduce(a + b)).
            out = tensor_model_parallel_all_reduce(out)
            shared = self.shared_experts(x) if self.shared_experts is not None else None
        if shared is not None:
            # The shared MLP's down_proj defers its all-reduce under a
            # deferred-AR context (TBO), so ``shared`` is a promise this very
            # sum consumes — fence before the read, the discipline the next
            # stage's layernorm applies for the dense stack.
            ar = current_deferred_ar()
            if ar is not None:
                ar.fence_pending_reads()
            out = out + shared
        return out.reshape(*leading_shape, self.hidden_size)

    def _forward_dp_attention_a2a(
        self, x: torch.Tensor, metadata: DPMetadata
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The a2a pooling contract: pad, pool, route the pool, all-to-all.

        The original pairing of DP-attention with EP, kept for A/B against the
        default :meth:`_forward_dp_attention_agrs`. Attention ran ``dp``
        batches independently, so without pooling each rank would dispatch
        ``1/dp`` of the step's tokens over an EP group ``dp`` times wider than
        a replica's -- every expert seeing a ``dp**2`` smaller share of the
        work, which is the opposite of what a widened expert split is for.
        Gathering first means one exchange carries the whole step and each
        expert gets the full batch.

        Returns ``(routed_out, shared_out)`` on this rank's own rows, so
        :meth:`forward` keeps owning the deferred-all-reduce fence and the sum.
        """
        assert self.dispatcher is not None
        pooled = dp_gather(x, metadata)
        weights, ids = self._route(pooled)
        handle, local_x, local_ids, local_weights = self.dispatcher.dispatch(pooled, ids, weights)
        routed = self.dispatcher.combine(
            handle, self._run_experts(local_x, local_ids, local_weights)
        )
        # The shared expert stays on this rank's own tokens: it is a dense MLP,
        # so pooling would have every rank compute the whole step's shared half
        # and throw ``dp - 1`` of it away.
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        return dp_scatter(routed, metadata), shared

    def _forward_dp_attention_agrs(
        self, x: torch.Tensor, metadata: DPMetadata
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The AgRs contract: route locally, pool routing + hidden, reduce-scatter.

        vLLM's default pairing (``allgather_reducescatter``). The route runs on
        this rank's own rows and :func:`dp_dispatch` carries the hidden states,
        weights and ids together into the ragged pool, so every rank holds the
        whole step's routing after one exchange -- no pad rows, no separate
        permute round trip. Pool rows outside this rank's expert window are
        masked to ``-1`` and skipped by the grouped GEMM; :func:`dp_combine`
        reduce-scatters the expert output back to this rank's rows.

        Returns ``(routed_out, shared_out)`` on this rank's own rows, so
        :meth:`forward` keeps owning the deferred-all-reduce fence and the sum.
        """
        weights, ids = self._route(x)
        pool_x, pool_weights, pool_ids = dp_dispatch(x, weights, ids, metadata)
        # Rebase the pooled global ids to this rank's window; -1 marks rows the
        # grouped GEMM skips. The Triton kernel does it in one launch; on CPU
        # it is the sub/compare/where chain the a2a dispatcher's non-fused
        # branch runs.
        flat_ids = pool_ids.reshape(-1)
        if flat_ids.is_cuda:
            local_ids = ep_local_ids(
                flat_ids,
                expert_offset=self.expert_offset,
                num_local=self.num_local_experts,
            )
        else:
            local = flat_ids - self.expert_offset
            local_ids = torch.where(
                (local >= 0) & (local < self.num_local_experts),
                local,
                torch.full_like(local, -1),
            )
        local_ids = local_ids.view_as(pool_ids)
        routed = dp_combine(self._run_experts(pool_x, local_ids, pool_weights), metadata)
        # The shared expert stays on this rank's own tokens: it is a dense MLP,
        # so pooling would have every rank compute the whole step's shared half
        # and throw ``dp - 1`` of it away.
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        return routed, shared

    def _forward_ep(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The EP routed path; with SBO the shared MLP runs beside the exchange.

        Returns ``(routed_out, shared_out)`` so :meth:`forward` keeps owning
        the deferred-all-reduce fence and the final sum.

        Without SBO the shared MLP runs after both exchanges; with SBO it
        moves to the alternate stream and computes while the dispatch
        exchange is on the wire, the fence collecting it before the sum --
        one batch, no second half to interleave against: the gap TBO cannot
        cover and SBO exists for.
        """
        assert self.dispatcher is not None
        rows = x.shape[0]
        if (
            not x.is_cuda
            or self.shared_experts is None
            or not SboFlags.enable_dispatch_shared_overlap(rows)
        ):
            handle, local_x, local_ids, local_weights = self.dispatcher.dispatch(x, ids, weights)
            out = self.dispatcher.combine(
                handle, self._run_experts(local_x, local_ids, local_weights)
            )
            shared = self.shared_experts(x) if self.shared_experts is not None else None
            return out, shared

        alt = sbo_alt_stream(x.device)
        main = torch.cuda.current_stream(x.device)
        handle = self.dispatcher.dispatch_a(x, ids, weights)
        # The shared MLP reads ``x``, which the main stream produced.
        alt.wait_stream(main)
        # Recorded on its own stream label so the timeline can show the region
        # intersecting the dispatch exchange's comm region — near-free when the
        # timeline is off, which is every run that is not collecting evidence.
        timeline = CommStreamPool.for_device(x.device).timeline
        with torch.cuda.stream(alt), timeline.region("sbo.shared_mlp", "sbo"):
            shared = self.shared_experts(x)
        # Its output is summed on the main stream: mark the block so the
        # allocator cannot recycle it while that stream is still reading.
        shared.record_stream(main)
        local_x, local_ids, local_weights = self.dispatcher.dispatch_b(handle)
        local_out = self._run_experts(local_x, local_ids, local_weights)
        self.dispatcher.combine_a(handle, local_out)
        out = self.dispatcher.combine_b(handle)
        # The sum in :meth:`forward` reads the shared MLP's output.
        main.wait_stream(alt)
        return out, shared

    def op_gate(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Flatten to ``[tokens, hidden]``, route, stash ids/weights in ``ctx``.

        The op stream is a flat token pipeline — ``dispatch_a`` permutes rows and
        ``combine_b`` returns ``[tokens, hidden]`` — but the caller (the TBO decode
        path) carries ``[rows, seq, hidden]``. Flatten here exactly as
        :meth:`forward` brackets its own reshape; :meth:`op_combine_b` restores.
        """
        ctx.leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)
        if current_dp_metadata() is not None:
            # Halves are cut from each rank's local batch, so peer ranks need
            # not agree on half count or size; a pooled exchange would hang,
            # not corrupt. Refuse the pairing until the splitter pools first.
            raise RuntimeError(
                "two-batch overlap and DP-attention cannot share a step: the halves "
                "are cut per rank, which the pooled expert exchange cannot follow"
            )
        ctx.weights, ctx.ids = self._route(x)  # ids are global
        return x

    def op_dispatch_a(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Post the dispatch exchange; ``ctx.handle`` carries the fence events."""
        assert self.dispatcher is not None and ctx.ids is not None
        ctx.handle = self.dispatcher.dispatch_a(x, ctx.ids, ctx.weights)
        return x

    def op_shared_experts(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Run the shared MLP while the dispatch exchange is on the wire."""
        ctx.shared = self.shared_experts(x) if self.shared_experts is not None else None
        return x

    def op_dispatch_b(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Fence the dispatch; swap ``x`` for this rank's expert batch.

        Returns ``[ep*cap, hidden]`` — from here until :meth:`op_combine_b`
        the stream carries the permuted local batch, not the token batch.
        """
        assert self.dispatcher is not None and ctx.handle is not None
        local_x, ctx.local_ids, ctx.local_weights = self.dispatcher.dispatch_b(ctx.handle)
        return local_x

    def op_experts(self, local_x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Grouped GEMM over the received batch with unit routing weights
        (the sender applies the real ones in :meth:`op_combine_b`)."""
        return self._run_experts(local_x, ctx.local_ids, ctx.local_weights)

    def op_combine_a(self, local_out: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Post the return exchange for this rank's expert results."""
        assert self.dispatcher is not None and ctx.handle is not None
        self.dispatcher.combine_a(ctx.handle, local_out)
        return local_out

    def op_combine_b(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Fence the return exchange; weighted token sum plus the shared MLP.

        The result is complete on every rank — EP's routed path needs no
        all-reduce. The shared MLP's deferred all-reduce (if any) is fenced
        here, where the sum consumes its promise.
        """
        assert self.dispatcher is not None and ctx.handle is not None
        out = self.dispatcher.combine_b(ctx.handle)
        if ctx.shared is not None:
            ar = current_deferred_ar()
            if ar is not None:
                ar.fence_pending_reads()
            out = out + ctx.shared
        # Restore the leading dims :meth:`op_gate` flattened, so the next layer's
        # attention stage (and the final head) see ``[rows, seq, hidden]`` again.
        return out.reshape(*(ctx.leading_shape or ()), self.hidden_size)

    def _run_experts(
        self,
        local_x: torch.Tensor,
        local_ids: torch.Tensor,
        local_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Stages 3-4: run the local experts over the received batch.

        Delegates to the :class:`MoeRunner` when the block owns one; the runner
        resolves the Triton core (and its identity permutes) for the block's
        ``(a2a_backend, runner_backend)``. A block built by ``object.__new__``
        (the custom-method test that only sets ``quant_method``) has no runner,
        so this falls back to the same call the runner core would make.
        """
        runner = getattr(self, "runner", None)
        if runner is not None:
            return runner.run(self, local_x, local_ids, local_weights)
        return self.quant_method.apply(self, local_x, local_weights, local_ids)

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Convert loaded fp16 expert weights to the requested scheme, in place."""
        if self.quant is not None:
            return
        method = quant.get_quant_method(self)
        method.quantize_from_fp16(self, quant)
        # Set quant before the hook: GPTQ bits=8 reads self.quant.bits inside
        # process_weights_after_loading to pick the repack kernel.
        self.quant = quant
        self.quant_method = method
        method.process_weights_after_loading(self)
