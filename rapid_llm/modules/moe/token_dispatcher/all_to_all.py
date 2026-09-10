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
from ....kernels.ops.moe.ep_dispatch import ep_combine, ep_dispatch_place, ep_local_ids
from ..utils import MoeA2ABackend
from .base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
    register_dispatcher,
)

#: Default per-destination capacity headroom over the mean load (n/ep_size);
#: ``None`` = unbounded cap == n. Measured on this 2-GPU NVLink box (ep2,
#: graph): decode a2a messages are 0.85 vs 1.0 MB, a ~0.1 us/layer bandwidth
#: delta dwarfed by the collective's latency floor -- and NCCL's graph-captured
#: algorithm choice for the smaller message ran 4-9% *slower* end to end
#: (bs16-2k TPOT 17.1 vs 16.5 ms). Capacity pays off where the wire is
#: bandwidth-bound (high ep_size, big batches, slower interconnect).
_DEFAULT_CAPACITY_FACTOR: float | None = None


@dataclass
class DispatchHandle:
    """Everything ``dispatch_a`` learned that the later phases need.

    Attributes:
        rows: Tokens in the dispatching batch (before top-k expansion).
        top_k: Routing slots per token.
        cap: Per-destination capacity of the exchange (``rows * top_k``).
        ep_size: Ranks in the EP group; the buffers are ``ep_size * cap`` rows.
        order: ``[ep*cap]`` permutation — flat slot index of each sorted row;
            ``None`` when the fused placement path ran (it needs no
            permutation: combine gathers by ``send_pos`` directly).
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
    order: torch.Tensor | None
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
        capacity_factor: Multiplicative headroom over the mean load,
            sglang's DeepEP-LL ``num_max_dispatch_tokens_per_rank`` sizing:
            ``cap = min(n, ceil(n/ep_size * factor) + slack)`` instead of the
            rows*top_k worst case (which pads the wire to ep_size x the mean).
            Opt-in via ``RAPID_EP_CAPACITY_FACTOR``: measured ep2+graph decode
            is latency-bound (see ``_DEFAULT_CAPACITY_FACTOR``), so the default
            ``None`` keeps ``cap == n``. ``<= 0`` (or a non-numeric value)
            also restores the unbounded cap. Slots past the cap drop to the
            trash row -- the drop-is-a-feature semantics every capacity-bound
            EP stack (DeepEP LL, vLLM EPMoE ``capacity_ratio``) ships.
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
        # DeepEP-LL sizing rather than the rows*top_k worst case. The mean and
        # the slack are host constants, so cap stays a pure function of
        # (rows, top_k, ep_size) and CUDA-graph safe. Opt-in (the measured
        # ep2 wire is latency-bound; see ``_DEFAULT_CAPACITY_FACTOR``): unset
        # keeps cap == n. Load imbalance beyond the factor drops past-cap
        # slots to the trash row (zero on combine): with a static NCCL exchange
        # and graph capture there is no room for DeepEP-normal's dynamic
        # per-expert counts, so the tight cap plus bounded drops is the
        # LL-mode trade. Raise the factor for skewed routers; ``<= 0`` restores
        # the unbounded cap == n.
        env_factor = os.environ.get("RAPID_EP_CAPACITY_FACTOR")
        if capacity_factor is not None:
            factor: float | None = capacity_factor
        elif env_factor is not None:
            try:
                factor = float(env_factor)
            except ValueError:  # e.g. "none"/"off"
                factor = None
        else:
            factor = _DEFAULT_CAPACITY_FACTOR
        self.capacity_factor = factor if factor is None or factor > 0 else None
        self.capacity_slack = (
            capacity_slack
            if capacity_slack is not None
            else int(os.environ.get("RAPID_EP_CAPACITY_SLACK", "8"))
        )
        # Fused Triton placement/combine (one kernel per phase instead of the
        # aten sort/scatter chain). Only the unbounded-cap path is fusible —
        # atomic placement cannot say *which* slots a cap would drop — so the
        # flag is always consulted together with ``capacity_factor is None``.
        self._fused = os.environ.get("RAPID_EP_FUSED_DISPATCH", "1") != "0"
        # combine_b's unit weights are constant per (rows, dtype, device); the
        # cache is warm from the pre-capture eager warmup, so graphs never
        # allocate it. Saves a fill kernel per layer per step.
        self._unit_cache: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}

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
        # factor, sglang's DeepEP-LL sizing rather than the rows*top_k worst
        # case. A rank almost never owns every routing slot, so padding the
        # exchange to the worst case burns all-to-all bandwidth at ep_size x
        # the mean and, once the pads reach the grouped GEMM as -1 rows, still
        # costs the align a scan over dead slots. mean+slack keeps the useful
        # work (valid tokens * top_k) unchanged while shrinking the padded
        # buffer that crosses the wire. cap is a pure function of (n, ep_size)
        # -- static per (rows, top_k) -- so the path stays CUDA-graph
        # capturable. Overflow past the cap drops to the trash row; see
        # :meth:`_detect_overflow` for the opt-in visibility knobs.
        if self.capacity_factor is None:
            cap = n
        else:
            cap = min(n, math.ceil(n / ep_size * self.capacity_factor) + self.capacity_slack)
        buf = ep_size * cap

        flat_ids = topk_ids.reshape(-1)
        # x and ids cross in ONE exchange: a per-row byte layout [hidden*esize
        # | 4B id | pad to 16B]. A second all_to_all for the ids alone would
        # carry ~1 KB yet pay the same latency floor as the 1 MB payload
        # (~16-25 us in-graph on this box), and it fires once per layer.
        # 16B row alignment keeps both views usable for vectorised access.
        esize = x.element_size()
        row_bytes = hidden * esize
        stride_bytes = row_bytes + (16 - row_bytes % 16) % 16 + 16
        if (
            self._fused
            and self.capacity_factor is None
            and ep_size <= 32
            and x.is_cuda
            and x.stride(1) == 1
        ):
            # Atomic-ticket placement (kernels.ops.moe.ep_dispatch): two
            # kernels — prep (counter clear + pad-id fill) and place&scatter —
            # do the sort/searchsorted/where chain and both indexed scatters'
            # work. cap == n here, so no slot overflows and the buffer needs
            # no trash row; pad rows keep the -1 id from the prep kernel and
            # their data is never read, so the payload is left uninitialised.
            payload = torch.empty(buf * stride_bytes, dtype=torch.uint8, device=x.device)
            rows2d = payload.view(buf, stride_bytes)
            send_pos = ep_dispatch_place(
                x,
                topk_ids,
                payload.view(x.dtype),
                rows2d[:, row_bytes : row_bytes + 4].view(torch.int32).squeeze(-1),
                num_local=self.num_local_experts,
                cap=cap,
                ep_size=ep_size,
            )
            order = None
        else:
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
            payload = torch.zeros((buf + 1) * stride_bytes, dtype=torch.uint8, device=x.device)
            rows2d = payload.view(buf + 1, stride_bytes)
            send_x = rows2d[:, :row_bytes].view(x.dtype)
            send_ids = rows2d[:, row_bytes : row_bytes + 4].view(torch.int32)
            # ``order[i]`` is the flat slot (token*k + j) placed at sorted position
            # ``i``; // k folds it back to its token row.
            send_x[send_pos] = x[order // k]
            send_ids[send_pos] = flat_ids[order].to(torch.int32).unsqueeze(-1)
        # (Fused path: the buffer holds exactly the ep_size*cap real slots.)
        send_flat = payload[: buf * stride_bytes]

        pool = CommStreamPool.for_device(x.device)
        recv_flat = torch.empty_like(send_flat)
        event = pool.all_to_all_async(recv_flat, send_flat, group=group, label="ep.dispatch")
        recv_rows = recv_flat.view(buf, stride_bytes)
        # recv_x keeps a row stride of stride_bytes/esize elements; the MoE
        # GEMM only requires a contiguous last dim, so no copy is needed.
        recv_x = recv_rows[:, :row_bytes].view(x.dtype)
        recv_ids = recv_rows[:, row_bytes : row_bytes + 4].view(torch.int32).squeeze(-1)
        events = [event] if event is not None else []
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
        buf = handle.recv_ids.shape[0]
        if handle.order is None:
            # Fused placement ran: one rebase kernel replaces the sub/compare/
            # and/where chain, and the unit weights come from the cache.
            local_ids = ep_local_ids(
                handle.recv_ids,
                expert_offset=self.expert_offset,
                num_local=self.num_local_experts,
            ).reshape(-1, 1)
            key = (buf, handle.recv_x.dtype, handle.recv_x.device)
            ones = self._unit_cache.get(key)
            if ones is None:
                ones = torch.ones(buf, 1, dtype=handle.recv_x.dtype, device=handle.recv_x.device)
                self._unit_cache[key] = ones
            return handle.recv_x, local_ids, ones
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
        if handle.order is None:
            # Fused placement: one gather-weight-sum kernel replaces the
            # cat/gather/scatter/mul/reduce chain. cap == n means no slot
            # overflowed, so there is no trash row to absorb.
            handle.events.clear()
            return ep_combine(
                handle.recv_out,
                handle.send_pos,
                handle.flat_weights,
                handle.rows,
                handle.top_k,
            )
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

    def combine(
        self, handle: DispatchHandle, local_out: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        """Optionally surface capacity drops: warn or raise, never silent.

        Overflow past the cap drops to the trash row by design (zero on
        combine) -- the trade every capacity-bound EP stack ships. Reading the
        ``keep`` predicate costs a host sync eager mode cannot pay per layer,
        so visibility is opt-in: ``RAPID_EP_CAPACITY_WARN=1`` prints a
        measurement-only warning, ``RAPID_EP_CAPACITY_STRICT=1`` raises so a
        skewed router cannot silently lose slots. Both skip graph capture
        (replays never run Python) and the impossible case ``cap >= n``.
        """
        strict = os.environ.get("RAPID_EP_CAPACITY_STRICT", "0") == "1"
        warn = os.environ.get("RAPID_EP_CAPACITY_WARN", "0") == "1"
        if not (strict or warn) or cap >= rows * k:
            return
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
        if strict:
            raise RuntimeError(msg)
        print(f"[ep-capacity-warn] {msg}", file=sys.stderr, flush=True)
