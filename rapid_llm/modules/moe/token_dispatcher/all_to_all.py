"""All-to-all EP token dispatcher (the a2a path of stages 2 & 5).

:class:`AllToAllDispatcher` is the structural analog of sglang's ``DeepEPDispatcher``
(the a2a EP path with the same ``dispatch_a``/``dispatch_b`` + ``combine_a``/``combine_b``
two-phase split), not its ``StandardDispatcher`` (which is the non-EP passthrough).
It is kept transport-honest — the wire op is NCCL ``all_to_all_single``, not the DeepEP
library — and backward-compatible: the six methods (``dispatch``/``combine`` plus the
four phase methods) and the ``AllToAllDispatcher`` name are what ``models/base.py``'s
TBO op stream and ``tests/{modules,distributed}`` import directly.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from ....batch_overlap import CommStreamPool
from ....distributed.parallel_state import get_ep_group
from ..utils import MoeA2ABackend
from .base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
    register_dispatcher,
)


@dataclass
class DispatchHandle:
    """Everything ``dispatch_a`` learned that the later phases need.

    Attributes:
        rows: Tokens in the dispatching batch (before top-k expansion).
        top_k: Routing slots per token.
        cap: Per-destination capacity of the exchange (``rows * top_k``).
        ep_size: Ranks in the EP group; the buffers are ``ep_size * cap`` rows.
        order: ``[ep*cap]`` permutation — flat slot index of each sorted row.
        send_pos: ``[n]`` positions of the sorted rows in the send buffer.
        flat_weights: ``[n]`` routing weights, kept on the sender.
        recv_x / recv_ids: Dispatch receive buffers (rows and expert ids).
        recv_out: Combine receive buffer, allocated by ``combine_a``.
        events: Comm-stream events outstanding phases must fence on.
    """

    rows: int
    top_k: int
    cap: int
    ep_size: int
    order: torch.Tensor
    send_pos: torch.Tensor
    flat_weights: torch.Tensor
    recv_x: torch.Tensor
    recv_ids: torch.Tensor
    recv_out: torch.Tensor | None = None
    events: list[torch.cuda.Event] = field(default_factory=list)


class AllToAllDispatchOutput(NamedTuple):
    """Dispatch result in the a2a layout (stage 2 → stage 3).

    Field order matches the historical ``dispatch`` return tuple
    ``(handle, local_x, local_ids, local_weights)`` so existing unpacking keeps
    working; :attr:`format` tags it for the runner.
    """

    handle: DispatchHandle
    recv_x: torch.Tensor
    local_ids: torch.Tensor
    local_weights: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ALL_TO_ALL


class AllToAllCombineInput(NamedTuple):
    """Combine input in the a2a layout (stage 4 → stage 5)."""

    handle: DispatchHandle
    local_out: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ALL_TO_ALL


@register_dispatcher(MoeA2ABackend.ALL_TO_ALL)
class AllToAllDispatcher(BaseDispatcher):
    """Two-phase EP dispatch/combine over the shared comm stream.

    Args:
        num_experts: Global routed-expert count (routing ids live in
            ``[0, num_experts)``).
        num_local_experts: Experts this rank owns.
        expert_offset: Global id of this rank's first expert.
        capacity_factor: Multiplicative slack over the mean load. Opt-in:
            ``None`` (the default, from an unset ``RAPID_EP_CAPACITY_FACTOR``)
            keeps ``cap == n`` -- no drops, no crash -- so the safe win is the
            -1 GEMM skip alone; set it to bound the padded buffer at
            ``mean * factor`` (pays off as ep_size grows).
        capacity_slack: Additive rows over ``ceil(mean * factor)`` so tiny
            batches keep ``cap == n``; defaults to ``RAPID_EP_CAPACITY_SLACK`` (8).
    """

    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        expert_offset: int,
        capacity_factor: float | None = None,
        capacity_slack: int | None = None,
    ) -> None:
        if num_experts % num_local_experts != 0:
            raise ValueError(
                f"{num_experts} experts do not split into groups of {num_local_experts}"
            )
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.expert_offset = expert_offset
        # Receive capacity = mean load (n / ep_size) * factor + slack, sglang's
        # DeepEP sizing rather than the rows*top_k worst case. Opt-in: unset it
        # (the default) keeps cap == n -- no drops, no crash -- so the safe win is
        # the -1 GEMM skip alone. When set, the multiplicative factor is the
        # load-imbalance headroom; the additive slack keeps tiny (decode) batches
        # at cap == n. Both are host constants, so cap stays a pure function of
        # (rows, top_k, ep_size) and CUDA-graph safe. The comm/buffer savings
        # scale with ep_size (mean is n/ep_size), so this pays off most at high
        # EP degree; at ep_size 2 real routing skew leaves little safe headroom.
        env_factor = os.environ.get("RAPID_EP_CAPACITY_FACTOR")
        self.capacity_factor = (
            capacity_factor
            if capacity_factor is not None
            else float(env_factor)
            if env_factor is not None
            else None
        )
        self.capacity_slack = (
            capacity_slack
            if capacity_slack is not None
            else int(os.environ.get("RAPID_EP_CAPACITY_SLACK", "8"))
        )

    # ------------------------------------------------------------------ #
    # dispatch: tokens out, per-expert batches in
    # ------------------------------------------------------------------ #

    def dispatch_a(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> DispatchHandle:
        """Permute routing slots by destination rank and post the exchange.

        Args:
            x: ``[rows, hidden]`` token embeddings.
            topk_ids: ``[rows, top_k]`` global expert ids.
            topk_weights: ``[rows, top_k]`` routing weights; these stay local.

        Returns:
            A handle carrying the buffers and the fence events; finish the
            phase with :meth:`dispatch_b`.
        """
        if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError("EP dispatch expects x [tokens, hidden] and matching 2-D routing")
        if topk_ids.shape[0] != x.shape[0]:
            raise ValueError("routing rows must match token rows")
        # Reading a CUDA predicate here would synchronize every MoE layer and
        # is illegal during graph capture. CUDA ids come from the router.
        if (
            x.device.type == "cpu"
            and topk_ids.numel()
            and (torch.any(topk_ids < 0) or torch.any(topk_ids >= self.num_experts))
        ):
            raise ValueError(f"expert ids must be in [0, {self.num_experts})")
        rows, hidden = x.shape
        k = topk_ids.shape[1]
        n = rows * k
        group = get_ep_group()
        ep_size = self._ep_size(group)
        # ``dest = id // num_local`` must land in [0, ep_size), which holds
        # exactly when the placement tiles the group; otherwise the send
        # buffer index goes out of bounds.
        if ep_size * self.num_local_experts != self.num_experts:
            raise ValueError(
                f"EP group of {ep_size} cannot host {self.num_experts} experts in "
                f"groups of {self.num_local_experts}"
            )
        # Receive capacity per destination rank: the mean load grown by a slack
        # factor, sglang's DeepEP sizing rather than the rows*top_k worst case.
        # A rank almost never owns every routing slot, so padding the exchange
        # to the worst case burns all-to-all bandwidth and, once the pads reach
        # the grouped GEMM as -1 rows, still costs the align a scan over dead
        # slots. mean+slack keeps the useful work (valid tokens * top_k)
        # unchanged while shrinking the padded buffer that crosses the wire.
        # cap is a pure function of (n, ep_size) -- static per (rows, top_k) --
        # so the path stays CUDA-graph capturable. Opt-in: an unset factor keeps
        # cap == n (no drops, no crash), so the safe default win is the -1 GEMM
        # skip alone; setting the factor trades a small drop risk for a smaller
        # padded buffer, which pays off as ep_size (and thus mean = n/ep_size) grows.
        if self.capacity_factor is None:
            cap = n
        else:
            cap = min(
                n, math.ceil(n / ep_size * self.capacity_factor) + self.capacity_slack
            )
        buf = ep_size * cap

        flat_ids = topk_ids.reshape(-1)
        dest = torch.div(flat_ids, self.num_local_experts, rounding_mode="floor")
        # Stable sort keeps slots of one token in routing order within a
        # destination segment — deterministic across ranks.
        sorted_dest, order = torch.sort(dest, stable=True)
        # Position of each row inside its destination segment: sort is stable,
        # so segment start = first index with the same dest (searchsorted left).
        seg_start = torch.searchsorted(sorted_dest, sorted_dest, side="left")
        pos_in_seg = torch.arange(n, device=x.device) - seg_start
        # Rows past their destination's cap spill to a trash slot (the extra row
        # of an over-allocated buffer) instead of corrupting the next segment;
        # the trash row is sliced off before the exchange, so those tokens never
        # cross the wire and combine reads a zero back for them.
        keep = pos_in_seg < cap
        send_pos = torch.where(keep, sorted_dest * cap + pos_in_seg, buf)
        self._detect_overflow(keep, cap, rows, k, ep_size, x.device)

        # Pad slots carry -1 (fused_moe drops any id outside [0, num_local))
        # and zero data: they cross the wire but never reach the grouped GEMM,
        # and combine gathers back only the ``send_pos`` rows.
        send_x = torch.zeros(buf + 1, hidden, dtype=x.dtype, device=x.device)
        send_ids = torch.full((buf + 1,), -1, dtype=flat_ids.dtype, device=x.device)
        # ``order[i]`` is the flat slot (token*k + j) placed at sorted position
        # ``i``; // k folds it back to its token row.
        send_x[send_pos] = x[order // k]
        send_ids[send_pos] = flat_ids[order]
        # Drop the trash row; the exchange sees only the ep_size*cap real slots.
        send_x = send_x[:buf]
        send_ids = send_ids[:buf]

        pool = CommStreamPool.for_device(x.device)
        recv_x = torch.empty_like(send_x)
        recv_ids = torch.empty_like(send_ids)
        events = [
            e
            for e in (
                pool.all_to_all_async(recv_x, send_x, group=group, label="ep.dispatch.x"),
                pool.all_to_all_async(recv_ids, send_ids, group=group, label="ep.dispatch.ids"),
            )
            if e is not None
        ]
        return DispatchHandle(
            rows=rows,
            top_k=k,
            cap=cap,
            ep_size=ep_size,
            order=order,
            send_pos=send_pos,
            flat_weights=topk_weights.reshape(-1).to(x.dtype),
            recv_x=recv_x,
            recv_ids=recv_ids,
            events=events,
        )

    def dispatch_b(self, handle: DispatchHandle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fence the dispatch exchange; return this rank's expert batch.

        Returns:
            ``(local_x, local_ids, local_weights)``: ``[ep*cap, hidden]`` rows,
            ``[ep*cap, 1]`` local expert ids with pad rows marked -1 so the
            grouped GEMM skips them, and unit weights — the sender applies the
            real ones in :meth:`combine_b`.
        """
        self._fence(handle)
        # A real id always lands in [0, num_local) after the shift; anything
        # outside the window is a pad row, mapped back to -1 for fused_moe to
        # skip. Clamping to expert 0 instead would run every pad row as work.
        local = handle.recv_ids - self.expert_offset
        valid = (local >= 0) & (local < self.num_local_experts)
        local_ids = torch.where(valid, local, torch.full_like(local, -1))
        ones = torch.ones(
            handle.recv_x.shape[0], 1, dtype=handle.recv_x.dtype, device=handle.recv_x.device
        )
        return handle.recv_x, local_ids.reshape(-1, 1), ones

    # ------------------------------------------------------------------ #
    # combine: expert results back, weighted token sums out
    # ------------------------------------------------------------------ #

    def combine_a(self, handle: DispatchHandle, local_out: torch.Tensor) -> DispatchHandle:
        """Post the return exchange for ``local_out`` (``[ep*cap, hidden]``)."""
        pool = CommStreamPool.for_device(local_out.device)
        handle.recv_out = torch.empty_like(local_out)
        event = pool.all_to_all_async(
            handle.recv_out, local_out, group=get_ep_group(), label="ep.combine"
        )
        if event is not None:
            handle.events.append(event)
        return handle

    def combine_b(self, handle: DispatchHandle) -> torch.Tensor:
        """Fence the return exchange; reduce to ``[rows, hidden]``.

        Un-permutes the received results, scales each slot by the routing
        weight the sender kept, and sums the ``top_k`` slots per token. The
        output is complete on every rank — EP's routed path needs no
        all-reduce, unlike the TP expert split.
        """
        self._fence(handle)
        assert handle.recv_out is not None, "combine_b before combine_a"
        # A trailing zero row absorbs the trash slot: dropped (over-capacity)
        # rows carry ``send_pos == ep_size*cap``, so their gather lands on this
        # zero and contributes nothing. When nothing overflowed no send_pos hits
        # it, so the extra row is inert -- the append keeps a static shape
        # either way, unlike a boolean mask.
        recv_out = torch.cat(
            [handle.recv_out, handle.recv_out.new_zeros(1, handle.recv_out.shape[1])], dim=0
        )
        # Gather drops the padding rows of the sorted layout in one step.
        results_sorted = recv_out[handle.send_pos]
        results_flat = torch.empty_like(results_sorted)
        results_flat[handle.order] = results_sorted
        weighted = results_flat * handle.flat_weights.unsqueeze(-1)
        # A shaped sum over the top-k axis, not index_add_: CUDA atomics arrive
        # in run-varying order and re-round bf16 per arrival, a jitter layers
        # amplify into rejected parity runs. The row layout already groups each
        # token's slots contiguously, so this is a fixed-order fp32 accumulate.
        out = weighted.view(handle.rows, handle.top_k, -1).sum(dim=1)
        handle.events.clear()
        return out

    # ------------------------------------------------------------------ #
    # synchronous convenience: a + b back to back (non-TBO path)
    # ------------------------------------------------------------------ #

    def dispatch(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> AllToAllDispatchOutput:
        """:meth:`dispatch_a` + :meth:`dispatch_b` with the fence immediate.

        Returns an :class:`AllToAllDispatchOutput`; it unpacks in the historical
        ``(handle, local_x, local_ids, local_weights)`` order.
        """
        handle = self.dispatch_a(x, topk_ids, topk_weights)
        local_x, local_ids, local_weights = self.dispatch_b(handle)
        return AllToAllDispatchOutput(handle, local_x, local_ids, local_weights)

    def combine(self, handle: DispatchHandle, local_out: torch.Tensor | None = None) -> torch.Tensor:
        """:meth:`combine_a` + :meth:`combine_b` with the fence immediate.

        Accepts either the historical ``(handle, local_out)`` pair or a single
        :class:`AllToAllCombineInput` (the :class:`BaseDispatcher` seam).
        """
        if local_out is None:
            handle, local_out = handle.handle, handle.local_out
        return self.combine_b(self.combine_a(handle, local_out))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _ep_size(group) -> int:
        """Rank count of ``group``; 1 when there is no live process group."""
        if group is None:
            return 1
        import torch.distributed as dist

        return dist.get_world_size(group)

    @staticmethod
    def _fence(handle: DispatchHandle) -> None:
        """Order the compute stream after every outstanding exchange event."""
        if handle.events and torch.cuda.is_available():
            stream = torch.cuda.current_stream()
            for event in handle.events:
                stream.wait_event(event)

    def _detect_overflow(
        self, keep: torch.Tensor, cap: int, rows: int, k: int, ep_size: int, device: torch.device
    ) -> None:
        """Raise if the mean+slack capacity dropped any routed slot.

        A silent drop biases the routed output, so overflow is a hard error the
        operator fixes by raising ``capacity_factor``/``capacity_slack``. The
        check reads a CUDA predicate (``keep.all()``), a host sync that is
        illegal during graph capture -- so it is skipped there. In practice the
        default cap clears balanced routing with room to spare (see the module
        docstring); capture replays a shape already exercised in eager warmup.
        """
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return
        if bool(keep.all()):
            return
        dropped = int((~keep).sum().item())
        msg = (
            f"EP dispatch dropped {dropped}/{rows * k} routed slots past capacity {cap} "
            f"(rows={rows} top_k={k} ep_size={ep_size} factor={self.capacity_factor} "
            f"slack={self.capacity_slack}); raise RAPID_EP_CAPACITY_FACTOR/SLACK to fit the load"
        )
        # RAPID_EP_CAPACITY_WARN turns the hard error into a measurement-only
        # warning: the graph-capture warmup routes degenerate all-zero tokens
        # (every token to the same experts, ~all load on one rank), which is not
        # a real routing pattern -- warn mode lets a benchmark run past it and
        # observe how often *real* inference overflows a tight cap.
        if os.environ.get("RAPID_EP_CAPACITY_WARN", "0") == "1":
            print(f"[ep-capacity-warn] {msg}", file=sys.stderr, flush=True)
            return
        raise RuntimeError(msg)
