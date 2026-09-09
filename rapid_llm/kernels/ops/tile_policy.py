"""Tile-config resolution shared by the dense GEMM and MoE launchers.

Two jobs, both formerly copy-pasted into every quant kernel file:
:func:`resolve_tiles` is the autotune-store-first, heuristic-fallback pattern,
with an optional ``BLOCK_K`` convergence for formats whose k-tile has to cover
whole quantisation groups; :func:`tile_tier` names the device generation a
heuristic table forks on — ``TileTier.PRE_HOPPER`` instead of re-deriving
``sm_version(device_index) < (9, 0)`` at every branch.

``sm_version`` and ``has_native_fp8`` live here, not in ``quantization.w8a16``:
they are device queries, not a property of one numeric format, and every
sibling file was importing them from a format module it has nothing else to do
with.

Usage:
    cfg = resolve_tiles("w8a16_matmul", m=m, n=n, k=k, dtype_label="int8_block",
                        heuristic=lambda dev: _launch_config(m, dev),
                        device_index=x.device.index)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from enum import Enum, auto

import torch

from ..dispatcher.autotune import get_best_config


@functools.cache
def sm_version(device_index: int | None) -> tuple[int, int]:
    """Compute capability of ``device_index`` (current device when ``None``).

    Cached because the query is not free and the dense launchers consult it
    on every call to pick their tile table.
    """
    return torch.cuda.get_device_capability(device_index)


@functools.cache
def has_native_fp8(device_index: int | None) -> bool:
    """Whether this device has the fp8 MMA (sm89+)."""
    return sm_version(device_index) >= (8, 9)


class TileTier(Enum):
    """Device generation a heuristic tile table forks on.

    ``PRE_HOPPER`` is everything below sm90 — sm86 (A10) and sm89 (4090/L40)
    alike, since ``(8, 9) < (9, 0)``. Those parts carry ~100 KB of shared
    memory per SM against Hopper's 228 KB, so the tables measured on H100 do
    not merely run slower there: their wide tiers spill or fail to compile.
    Each launcher keeps one conservative table for this tier and defers to
    the autotune store for anything it was never swept on.
    """

    PRE_HOPPER = auto()
    HOPPER_UP = auto()


def tile_tier(device_index: int | None) -> TileTier:
    """Which heuristic table ``device_index`` reads from."""
    return TileTier.PRE_HOPPER if sm_version(device_index) < (9, 0) else TileTier.HOPPER_UP


def _converge_block_k(config: dict, multiple: int) -> dict:
    """Halve a tuned ``BLOCK_K`` until it covers whole quantisation groups.

    A store entry's k-tile is whatever the search measured; a format whose
    scales are grouped along k (int4's ``group_size``, nvfp4's 16-element
    blocks) additionally needs the tile to align with those groups. The
    kernels mask a ragged k-tail, so divisibility of K itself is not
    required — only the group alignment is.
    """
    block_k = config.get("BLOCK_K", multiple)
    while block_k > multiple and block_k % multiple != 0:
        block_k //= 2
    return {**config, "BLOCK_K": max(block_k, multiple)}


def resolve_tiles(
    op: str,
    *,
    m: int,
    n: int,
    k: int,
    dtype_label: str,
    heuristic: Callable[[int | None], dict],
    device_index: int | None,
    block_k_multiple: int | None = None,
) -> dict:
    """The tile config for one launch: tuned entry first, heuristic second.

    Args:
        op: Kernel family name for the autotune key (``"w8a16_matmul"``).
        m, n, k: GEMM extents; ``m`` is bucketed inside the lookup.
        dtype_label: Numeric-format label for the key. Launchers whose
            heuristic forks on more than the weight format (scale layout,
            W8A8 vs weight-only) encode the fork here, so one tuned entry
            never stands in for a path the search did not measure.
        heuristic: ``device_index -> config``, the device-tiered fallback
            table. Called only when the store has no entry.
        device_index: Device the launch runs on.
        block_k_multiple: When set, a tuned ``BLOCK_K`` is converged to a
            multiple of it (see :func:`_converge_block_k`).

    Returns:
        Tile config dict (``{"BLOCK_M": ..., "BLOCK_N": ..., ...}``).
    """
    config = get_best_config(op, m=m, n=n, k=k, dtype=dtype_label)
    if config is None:
        return heuristic(device_index)
    if block_k_multiple is not None:
        config = _converge_block_k(config, block_k_multiple)
    return config
