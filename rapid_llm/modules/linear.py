"""Linear layers: one place that knows about both quantisation and sharding.

Every layer is a :class:`LinearBase` subclass; the column/row/QKV variants
own their TP shard maths, and the weight layout (quant method, scales) is
delegated to the quantisation sub-package's method objects.

Usage:
    qkv = QKVParallelLinear(hidden_size, num_heads, num_kv_heads, head_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..batch_overlap import row_parallel_forward
from ..distributed.parallel_state import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from ..distributed.sequence_parallel import column_parallel_g, is_g_boundary
from .quantization import QuantizationConfig, UnquantizedLinearMethod


class LinearBase(nn.Module):
    """``y = x @ W.T (+ b)`` with the weight stored however the quant method says.

    Subclasses decide how ``input_size``/``output_size`` are split; :attr:`quant_method`
    owns the parameters and the multiply (composition, not subclassing — sharding and
    storage format are orthogonal). ``params_dtype`` follows vLLM's auto convention: the
    layer prescribes no precision, the model passes ``config.dtype``, and a direct
    instantiation falls back to ``torch.get_default_dtype()``. The resolved type is
    threaded into :meth:`create_weights` so a bf16 checkpoint never allocates fp16.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant = quant
        # The checkpoint's element type. The model passes ``config.dtype`` (auto);
        # ``None`` (a direct instantiation) defers to PyTorch's default dtype.
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.dtype = params_dtype
        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedLinearMethod()
        )
        self.quant_method.create_weights(self, input_size, output_size, params_dtype=params_dtype)
        self.bias = (
            nn.Parameter(torch.empty(output_size, dtype=params_dtype), requires_grad=False)
            if bias
            else None
        )
        # One rule fills every parameter this layer owns (weight, scale grids, bias
        # are all direct attributes), so the non-recursive iterator covers them.
        for param in self.parameters(recurse=False):
            param.weight_loader = self._weight_loader

    def apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        """The multiply itself, without any tensor-parallel communication."""
        return self.quant_method.apply(self, x, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inside a sequence-parallel region a marked layer is the *g* boundary:
        # this rank's token shard is all-gathered back to the full grid before
        # the GEMM, because attention reads across tokens. The mark decides, not
        # the class: ``ReplicatedLinear`` shares this method without being on the
        # residual path, and only the pass knows which projections bound a region.
        if is_g_boundary(self):
            return column_parallel_g(self, x)
        return self.apply_linear(x)

    def _weight_loader(
        self, param: torch.Tensor, loaded: torch.Tensor, shard_id=None
    ) -> torch.Tensor:
        """Fill ``param`` from one checkpoint tensor and return the view written.

        The terminal step every subclass lands on: check the tensor fits, copy, report
        what was written (the loader counts elements to verify coverage). Subclasses with
        a sharding rule override to compute the destination view and narrow ``loaded``,
        then call ``super()._weight_loader(view, loaded)``. ``shard_id`` names a block of
        a packed parameter (q/k/v, gate/up); ``None`` for unpacked.
        """
        if param.shape != loaded.shape:
            raise ValueError(
                f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                f"parameter view of shape {tuple(param.shape)}"
            )
        param.copy_(loaded)
        return param

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Replace a loaded 16-bit weight with its quantised form, in place.

        The ``--quantization <scheme>`` path: fp16 storage is dropped as each layer
        converts, so peak memory stays at the fp16 checkpoint's size. Already-quantised
        layers (an fp8 checkpoint) are left alone.
        """
        if self.quant is not None:
            return
        method = quant.get_quant_method(self)
        method.quantize_from_fp16(self, quant)
        # Set quant before the hook: a method whose kernel layout differs from what
        # quantize_from_fp16 produced (GPTQ bits=8 unpacks to int8 bytes) reads
        # self.quant.bits inside the hook.
        self.quant = quant
        self.quant_method = method
        # Same post-load hook the checkpoint path runs; a no-op for other methods.
        method.process_weights_after_loading(self)

    def extra_repr(self) -> str:
        fmt = self.quant.format if self.quant else "unquantized"
        return f"in={self.input_size}, out={self.output_size}, bias={self.bias is not None}, {fmt}"


class ReplicatedLinear(LinearBase):
    """Full weight on every rank; for layers too small or too awkward to split."""


class ColumnParallelLinear(LinearBase):
    """Splits the output features across ranks; no communication.

    Each rank returns its slice of the output feature dimension, which the next
    :class:`RowParallelLinear` consumes as its slice of the contracted dimension — that
    pairing keeps the all-reduce count at one per block. ``what`` names the split
    dimension for the error message.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        what: str = "output features",
    ) -> None:
        world_size = get_tensor_model_parallel_world_size()
        local_out = divide(output_size, world_size, what)
        _check_shard_alignment(quant, local_out, what)
        super().__init__(input_size, local_out, bias=bias, quant=quant, params_dtype=params_dtype)
        self.full_output_size = output_size

    def _weight_loader(self, param, loaded, shard_id=None):
        # ``shard_id`` selects a half of the packed gate/up pair (``FusedMLP`` builds its
        # fused projection from this class). The narrow is by proportion of the incoming
        # tensor, so scale grids (the same matrix at scale-block resolution) follow the
        # same rule as the weight.
        view = param.data
        if shard_id is not None:
            half = view.shape[0] // 2
            view = view.narrow(0, shard_id * half, half)

        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[0] // world_size
            loaded = loaded.narrow(0, get_tensor_model_parallel_rank() * size, size)

        return super()._weight_loader(view, loaded)


class QKVParallelLinear(LinearBase):
    """The query, key and value projections as one column-parallel weight.

    Three GEMMs over the same activation become one over ``[q | k | v]``: the activation
    is read once and one launch replaces three, which matters most on the memory-bound
    decode path where TP disables CUDA graphs and nothing hides launch overhead. Not
    simply ``ColumnParallelLinear(hidden, q + 2*kv)``: GQA gives more query than kv heads,
    and the two block boundaries must split independently (one cut of ``q + 2*kv`` would
    hand low ranks only query heads, high ranks only kv heads). The local layout is
    ``[q | k | v]`` with heads adjacent in a row, so :meth:`split` never copies (RoPE
    rotates q/k in place, the KV write addresses them by stride).

    Raises:
        ValueError: If either head count does not divide across the ranks, or a block's
            local width is not a whole number of scale blocks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        world_size = get_tensor_model_parallel_world_size()
        local_heads = divide(num_heads, world_size, "attention heads")
        local_kv_heads = divide(num_kv_heads, world_size, "key/value heads")
        q_size = local_heads * head_dim
        kv_size = local_kv_heads * head_dim
        # Checked per block, not on the total: a fused width can be a whole number of
        # scale blocks while the query block alone is not.
        _check_shard_alignment(quant, q_size, "query features")
        _check_shard_alignment(quant, kv_size, "key/value features")
        super().__init__(
            hidden_size, q_size + 2 * kv_size, bias=bias, quant=quant, params_dtype=params_dtype
        )
        self.num_heads = local_heads
        self.num_kv_heads = local_kv_heads
        self.head_dim = head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.full_output_size = (num_heads + 2 * num_kv_heads) * head_dim

    def split(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cut a fused output into ``(q, k, v)`` views along the last dimension.

        Views, not copies: each keeps the fused row stride, which every kernel
        downstream accepts.
        """
        return torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One GEMM, then three views — the call every attention block wants.

        Through :meth:`~LinearBase.forward` rather than straight to the multiply,
        so a sequence-parallel region's ``g`` boundary applies here too; the rows
        it returns are the full grid's, not the caller's shard.
        """
        return self.split(self.forward(x))

    def _weight_loader(self, param, loaded, shard_id=None):
        # ``shard_id`` is which of [q | k | v] arrived. Block boundaries come from this
        # layer's head geometry, scaled to the parameter's resolution: factor 1 for
        # weight/bias, ``group_n`` for a scale grid (rows count scale blocks).
        if shard_id is None:
            raise ValueError("qkv_proj is packed: a checkpoint tensor must name its block")
        factor = (self.q_size + 2 * self.kv_size) // param.shape[0]
        q_rows, kv_rows = self.q_size // factor, self.kv_size // factor
        offset, rows = ((0, q_rows), (q_rows, kv_rows), (q_rows + kv_rows, kv_rows))[shard_id]
        view = param.data.narrow(0, offset, rows)

        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[0] // world_size
            loaded = loaded.narrow(0, get_tensor_model_parallel_rank() * size, size)

        return super()._weight_loader(view, loaded)

    def extra_repr(self) -> str:
        fmt = self.quant.format if self.quant else "fp16"
        return (
            f"in={self.input_size}, q_heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, bias={self.bias is not None}, {fmt}"
        )


class RowParallelLinear(LinearBase):
    """Splits the contracted features across ranks and all-reduces the result.

    Each rank multiplies its slice of ``x`` by its slice of ``W``, so the output is a
    partial sum that :func:`tensor_model_parallel_all_reduce` completes. The collective is
    inserted after the local multiply, gated on ``reduce_results`` and world size (the same
    insertion point and flag vLLM's ``RowParallelLinear`` uses). A bias is rejected, not
    silently added ``world_size`` times (no supported projection has one).

    Args:
        input_size: Full contracted width, split ``world_size`` ways.
        output_size: Full output width (not split).
        quant: Quantisation layout, or ``None``.
        reduce_results: Whether ``forward`` all-reduces the partial sums. ``True``
            (o_proj/down_proj default) keeps the collective in the layer; ``False`` hands
            back the partial sum so a caller can time, batch or defer it (vLLM's
            ``ParallelLMHead`` uses the same escape hatch).
        what: Name of the dimension being split, for the error message.
        params_dtype: Parameter storage type; ``None`` defers to
            ``torch.get_default_dtype()`` (vLLM's auto convention).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        reduce_results: bool = True,
        params_dtype: torch.dtype | None = None,
        what: str = "input features",
    ) -> None:
        if bias:
            raise ValueError(
                "RowParallelLinear cannot carry a bias: it would be added once per rank"
            )
        world_size = get_tensor_model_parallel_world_size()
        local_in = divide(input_size, world_size, what)
        _check_shard_alignment(quant, local_in, what)
        super().__init__(local_in, output_size, quant=quant, params_dtype=params_dtype)
        self.full_input_size = input_size
        self.reduce_results = reduce_results

    def _weight_loader(self, param, loaded, shard_id=None):
        # The split is along the contracted dimension (columns of the weight and scale
        # grid alike); no packed form of this layer exists.
        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[1] // world_size
            loaded = loaded.narrow(1, get_tensor_model_parallel_rank() * size, size)
        return super()._weight_loader(param.data, loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The multiply is this layer's; *when and where the reduction happens*
        # is the comm-overlap module's — deferred under TBO, chunked under L3,
        # blocking otherwise. Dispatched rather than inlined here so the
        # policy lives in exactly one place.
        return row_parallel_forward(self, x)


def _check_shard_alignment(quant: QuantizationConfig | None, local_size: int, what: str) -> None:
    """Reject a split that would cut a quantisation scale block in half.

    Raises:
        ValueError: If ``local_size`` is not a whole number of scale blocks (the
            tensor-parallel size is too large for this checkpoint's block size).
    """
    if quant is not None and not quant.shard_is_aligned(local_size):
        raise ValueError(
            f"tensor-parallel shard of {what} is {local_size} channels, which is not a "
            f"multiple of the {quant.get_name()} scale block ({quant.group_n}x{quant.group_k}); "
            "use a smaller tensor_parallel_size"
        )
