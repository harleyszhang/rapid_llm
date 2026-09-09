"""FlashInfer attention wrappers: both phases behind the native signatures.

Both phases share one lazily allocated workspace, one wrapper each (cleared via
``_reset_cache``); the functions mirror the native kernels' signatures so
dispatch can swap them in. Decode is NOT CUDA-graph compatible as written —
page indices come from Python-side slicing over live lengths and the wrapper is
planned on every call, so a capture would bake capture-time lengths in and
replay would attend stale rows forever — hence ``graph_safe=False``, and the
runner refuses capture while it is chosen
(:func:`~rapid_llm.kernels.dispatcher.unsafe_for_graph`).

The vLLM-shaped route to compatibility has three parts, two exist: fixed-shape
GPU ops for the page indices (:func:`paged_kv_indices_gpu`, never Python
slicing); a per-step plan hook (:func:`prepare_decode`, wired through
``KernelSpec.step_prepare`` and fed by the engine-side CPU length ledger) so
per-layer host work becomes per-step, sync-free, and every layer's call reduces
to ``run()``; and per-captured-batch wrappers whose plan inputs stay at stable
addresses across replays (FlashInfer's ``fast_decode_plan``). The third piece
is missing — that is what keeps ``graph_safe=False`` until a wheel shipping it
can be installed and verified.

Usage:
    out = prefill_attention(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

#: Workspace FlashInfer's wrappers plan into, allocated on first use.
_WORKSPACE_BYTES = 128 * 1024 * 1024
_prefill_wrapper = None
_decode_wrapper = None
_workspace: torch.Tensor | None = None

# Head layout and scales of the last legacy-path decode call. The prepare
# hook needs them for ``wrapper.plan`` but the step protocol only hands it
# ``(atten_info, runner)``, which knows nothing about heads — the first
# decode call teaches it, and from the second step on the hook can plan.
_last_decode_shape: tuple | None = None

#: Set by :func:`prepare_decode` once this step's plan is in; every layer's
#: decode call then reduces to ``run()`` until the next hook call resets it.
_planned_pending = False


def _reset_cache() -> None:
    """Drop cached wrappers and per-step plan state (tests swapping devices/fakes)."""
    global _prefill_wrapper, _decode_wrapper, _workspace
    global _last_decode_shape, _planned_pending
    _prefill_wrapper = None
    _decode_wrapper = None
    _workspace = None
    _last_decode_shape = None
    _planned_pending = False


def _get_workspace() -> torch.Tensor:
    global _workspace
    if _workspace is None:
        _workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    return _workspace


def _get_wrapper(which: str) -> Callable:
    global _prefill_wrapper, _decode_wrapper
    if which == "prefill":
        if _prefill_wrapper is None:
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

            _prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(_get_workspace(), "NHD")
        return _prefill_wrapper
    if _decode_wrapper is None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        _decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(_get_workspace(), "NHD")
    return _decode_wrapper


def paged_kv_indices_gpu(
    b_req_tokens_table: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    *,
    page_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact ``kv_indices``/``kv_indptr`` on the GPU, at fixed shapes.

    FlashInfer's decode wrapper wants the paged rows flattened into one
    contiguous ``indices`` vector with per-request windows delimited by
    ``indptr``. Building that with Python slicing (the eager path below) has a
    shape that moves every step — exactly what a CUDA graph cannot capture;
    this version keeps every tensor at a fixed shape, so the whole assembly is
    a recordable op sequence whose result follows the live contents of the
    persistent length/table buffers on every replay.

    Positions past a request's window scatter into one sink slot after the
    real rows, keeping ``scatter_`` fixed-shape; the sink is never read
    because every kernel-side window is bounded by ``indptr``. Same trick as
    vLLM's ``_copy_page_indices_kernel`` (cu_num_blocks-driven scatter), in
    plain torch. Caller invariant: ``b_seq_len[i] <= table width`` for every
    row, else a window would run past the sink.

    Returns:
        ``(kv_indices, kv_indptr, last_page_len)`` for
        ``BatchDecodeWithPagedKVCacheWrapper.plan``: ``kv_indices`` is
        ``batch * table_width + 1`` wide (the last slot is the sink),
        ``kv_indptr`` is ``batch + 1`` and ``last_page_len`` is ``batch`` wide.
    """
    if page_size != 1:
        raise NotImplementedError(f"page_size {page_size}: pack pages with a div/mod pass first")
    table_rows = b_req_tokens_table[b_req_idx]  # [batch, width] gather, fixed shape
    batch, width = table_rows.shape

    # Window starts: prefix sum of each request's page count (page size 1 ->
    # one page per cached row).
    pages = b_seq_len[:batch].to(torch.int64)
    indptr = torch.zeros(batch + 1, dtype=torch.int64, device=table_rows.device)
    torch.cumsum(pages, dim=0, out=indptr[1:])

    total = batch * width
    positions = indptr[:batch, None] + torch.arange(width, device=table_rows.device)
    in_window = torch.arange(width, device=table_rows.device)[None, :] < pages[:, None]
    safe = torch.where(in_window, positions, total)

    indices = torch.empty(total + 1, dtype=torch.int32, device=table_rows.device)
    indices[total] = 0  # the sink; read by nobody
    indices.scatter_(0, safe.flatten(), table_rows.flatten().to(torch.int32))
    last_page_len = torch.ones(batch, dtype=torch.int32, device=table_rows.device)
    return indices, indptr.to(torch.int32), last_page_len


@dataclass
class FlashInferDecodePlan:
    """One decode step's plan inputs, assembled once outside the layer path.

    Attention is layer-invariant within a step: every layer reads the same
    ``indptr``/``indices``/``last_page_len``, so the per-layer cost of
    rebuilding them is pure repetition. Produced by :func:`prepare_decode`
    and stored on ``AttentionMetadata.decode_plan``.
    """

    #: Host indptr from the engine-side CPU ledger — feeding it to ``plan()``
    #: keeps the wrapper from syncing the GPU lengths back (plan()'s ~91 us).
    indptr_cpu: torch.Tensor
    #: GPU ``kv_indices`` from :func:`paged_kv_indices_gpu` — fixed shape,
    #: no mask-select sync (~136 us of the old per-layer cost).
    kv_indices: torch.Tensor
    #: Host ``last_page_len``: page size 1 makes every page full.
    last_page_len_cpu: torch.Tensor


def prepare_decode(atten_info, runner) -> None:
    """Plan one decode step once, outside the per-layer call path.

    Wired through ``KernelSpec.step_prepare`` and called by the runner on
    every eager decode step with ``(atten_info, runner)`` — the role vLLM's
    ``build_metadata`` plays. Three sync-free pieces:

    - ``indptr`` comes from the host ledger (``atten_info.b_seq_len_cpu``,
      written by the engine loop) instead of a device sync; without a ledger
      (probe paths before any engine step) it falls back to one ``.cpu()``
      copy, which is legal here because no capture is in progress.
    - ``indices`` come from the fixed-shape GPU scatter
      (:func:`paged_kv_indices_gpu`), driven by the persistent buffers.
    - ``wrapper.plan`` runs once with the head layout recorded by the first
      legacy-path decode call. The very first step after startup has no
      record yet, so this function returns without planning and that one
      step runs the legacy path (which also records the layout); from the
      second step on every layer reduces to ``run()``.

    Graph capture stays refused for this backend (``graph_safe=False``): a
    planned wrapper's internal buffers move on every ``plan()`` call, so a
    replayed ``run()`` would read capture-time addresses. The remaining
    piece is FlashInfer's ``fast_decode_plan`` (see the module docstring).
    """
    global _planned_pending
    _planned_pending = False  # a failed/absent plan must not leak into this step

    batch = atten_info.b_req_idx.shape[0]
    lens_cpu = atten_info.b_seq_len_cpu
    if lens_cpu is None:
        # No engine ledger yet (parity probes, standalone model calls): sync
        # is acceptable outside a capture stream.
        lens_cpu = atten_info.b_seq_len.cpu()
    seq_lens = lens_cpu.reshape(-1)[:batch].to(torch.int32)

    indptr_cpu = torch.zeros(batch + 1, dtype=torch.int32)
    indptr_cpu[1:] = torch.cumsum(seq_lens.to(torch.int64), dim=0).to(torch.int32)
    kv_indices, _indptr_gpu, _last_gpu = paged_kv_indices_gpu(
        atten_info.b_req_tokens_table, atten_info.b_req_idx, atten_info.b_seq_len
    )
    atten_info.decode_plan = FlashInferDecodePlan(
        indptr_cpu=indptr_cpu,
        kv_indices=kv_indices,
        last_page_len_cpu=torch.ones(batch, dtype=torch.int32),
    )

    shape = _last_decode_shape
    if shape is None:
        # First decode after startup: the legacy path below runs once and
        # records the head layout; planning starts on the next step.
        return
    num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, sm_scale = shape
    wrapper = _get_wrapper("decode")
    wrapper.plan(
        indptr=indptr_cpu,
        indices=kv_indices,
        last_page_len=atten_info.decode_plan.last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=1,
        sm_scale=sm_scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    _planned_pending = True


def prefill_attention(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len):
    """Causal prefill over a packed ragged batch, via FlashInfer.

    Args follow :func:`~rapid_llm.kernels.ops.attention.flashattention2_nopad.
    flash_attention2_no_pad` exactly; ``max_seq_len`` only sizes the native
    kernel's grid and is re-derived here from the length vector, so callers
    cannot disagree with themselves.
    """
    _total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    # The ragged wrapper needs compact rows; the engine's prefill grid is
    # padded to the widest chunk (slot_batch.begin_prefill), so a batch of
    # unequal prompt lengths only agrees with this layout by accident. When
    # the declared packed total differs from the rows actually here, hand the
    # pass to the native kernel, which addresses each sequence at
    # b_start_loc and bounds it by b_seq_len — the exact padded contract.
    declared = int(b_start_loc[-1].item()) + int(b_seq_len[-1].item())
    if _total_q != declared:
        from ...ops.attention.flashattention2_nopad import flash_attention2_no_pad

        return flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len)
    qo_indptr = torch.cat(
        [b_start_loc.new_zeros(1), b_start_loc[1:], b_start_loc[-1:] + b_seq_len[-1:]]
    ).to(torch.int32)
    # Prefill attends over the same freshly projected rows: one kv chunk per
    # sequence, aligned with the query rows.
    kv_indptr = qo_indptr

    wrapper = _get_wrapper("prefill")
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    out = torch.empty_like(q)
    wrapper.run(q, k, v, out=out)
    return out


def decode_attention(
    q,
    k_cache,
    v_cache,
    qk_scale,
    b_req_tokens_table,
    b_req_idx,
    b_seq_len,
    max_actual_seq_len,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Decode attention, one token per sequence, via FlashInfer.

    Args follow :func:`~rapid_llm.kernels.ops.attention.flashdecoding.
    flash_decoding` exactly. The slot table is flattened into FlashInfer's
    ``indptr``/``indices`` (page_size 1: one cache row per page), and the
    fp8-KV dequantisation scales pass through as FlashInfer's per-tensor
    ``k_scale``/``v_scale`` on the run call.

    When :func:`prepare_decode` has already planned this step, every layer
    lands on the two-line tail: one ``run()`` against the shared plan. The
    legacy path below is the cold route — it also records the head layout
    the prepare hook needs for its ``plan()`` calls.
    """
    if _planned_pending:
        # This step's plan inputs were assembled once by prepare_decode and
        # are layer-invariant; only the run is left per layer.
        out = torch.empty_like(q)
        _get_wrapper("decode").run(
            q,
            # Zero-copy page views: [T, H, D] -> [T, 1, H, D] per pool.
            (k_cache.unsqueeze(1), v_cache.unsqueeze(1)),
            out=out,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return out

    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    seq_lens = b_seq_len[:batch].to(torch.int32)
    # Cache-row ids of every attended token, request order flattened.
    rows = [b_req_tokens_table[b_req_idx[i], : seq_lens[i]].to(torch.int32) for i in range(batch)]
    indices = torch.cat(rows)
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    torch.cumsum(seq_lens, dim=0, out=indptr[1:])
    # page_size == 1: every page is full, so the last page of each sequence
    # holds exactly one token.
    last_page_len = torch.ones(batch, dtype=torch.int32, device=q.device)

    global _last_decode_shape
    _last_decode_shape = (num_heads, num_kv_heads, head_dim, q.dtype, k_cache.dtype, qk_scale)

    wrapper = _get_wrapper("decode")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=1,
        sm_scale=qk_scale,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )
    out = torch.empty_like(q)
    wrapper.run(
        q,
        # Zero-copy page views: [T, H, D] -> [T, 1, H, D] per pool.
        (k_cache.unsqueeze(1), v_cache.unsqueeze(1)),
        out=out,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return out
