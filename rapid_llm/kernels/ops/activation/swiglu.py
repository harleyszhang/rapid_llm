"""SwiGLU activation as a fused Triton kernel.

``swiglu_forward`` fuses the silu gate with the elementwise multiply
over two input tensors; ``swiglu_forward_fused`` is the single-input
variant used by the fused-MLP path.

Usage:
    y = swiglu_forward(gate, up)
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    from ..utils import calculate_settings
    from .activations import silu
except ImportError:  # Triton is optional on CPU-only and macOS installs.
    triton = None
    silu = None
    calculate_settings = None

    class _TritonLanguageStub:
        constexpr = object()

    tl = _TritonLanguageStub()


_jit = triton.jit if triton is not None else lambda function: function


@_jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, row_stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * row_stride
    b_ptr += program_id * row_stride
    c_ptr += program_id * row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward(gate, up):
    """silu(gate) * up with the two projection halves as separate tensors.

    Args:
        gate: ``[..., n_cols]`` gate projection.
        up: Same shape as ``gate``.

    Returns:
        ``[..., n_cols]`` product, in the inputs' dtype.
    """
    if triton is None or not gate.is_cuda:
        return torch.nn.functional.silu(gate) * up

    ori_shape = gate.shape  # ori_shape is [batch_size, seq_len, hidden_size]

    n_cols = ori_shape[-1]
    gate = gate.view(-1, n_cols)
    up = up.view(-1, n_cols)
    c = torch.empty_like(gate)
    n_rows = gate.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        gate,
        up,
        c,
        c.stride(-2),  # c.stride(-2) = n_cols
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape)


@_jit
def _swiglu_forward_fused_kernel(
    x_ptr,
    c_ptr,
    row_stride,
    n_cols: tl.constexpr,
    LIMIT: tl.constexpr,
    APPLY_LIMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    # One fused row holds gate then up, so the up half starts n_cols elements
    # into the row; row_stride is the whole 2 * n_cols width.
    x_ptr += program_id * row_stride
    c_ptr += program_id * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    gate_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up_row = tl.load(x_ptr + n_cols + col_offsets, mask=mask, other=0)
    if APPLY_LIMIT:
        gate_row = tl.minimum(gate_row, LIMIT)
        up_row = tl.minimum(tl.maximum(up_row, -LIMIT), LIMIT)
        activated = silu(gate_row).to(c_ptr.dtype.element_ty)
    else:
        activated = silu(gate_row)
    c_row = activated * up_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def _swiglu_forward_fused(x, limit: float):
    if triton is None or not x.is_cuda:
        gate, up = x.chunk(2, dim=-1)
        if limit != math.inf:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return torch.nn.functional.silu(gate) * up

    ori_shape = x.shape  # [..., 2 * n_cols]
    n_cols = ori_shape[-1] // 2
    x = x.reshape(-1, ori_shape[-1])  # GEMM output is contiguous: no copy
    c = torch.empty(x.shape[0], n_cols, dtype=x.dtype, device=x.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_fused_kernel[(x.shape[0],)](
        x,
        c,
        x.stride(0),
        n_cols=n_cols,
        LIMIT=limit if limit != math.inf else 0.0,
        APPLY_LIMIT=limit != math.inf,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape[:-1], n_cols)


def swiglu_forward_fused(x):
    """SwiGLU over a packed ``[gate, up]`` projection."""
    return _swiglu_forward_fused(x, math.inf)


def swiglu_forward_fused_bounded(x, limit: float):
    """Bounded SwiGLU with activation rounding in the input dtype."""
    return _swiglu_forward_fused(x, limit)
