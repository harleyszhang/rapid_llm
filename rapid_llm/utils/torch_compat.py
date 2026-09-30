"""Expose stable capability checks for optional PyTorch symbols.

Import-time resolution keeps version checks out of model and executor hot paths while
preserving native behavior whenever the installed PyTorch provides the capability.

Usage:
    if is_float8_e8m0fnu(tensor.dtype):
        tensor = tensor.float()
"""

from __future__ import annotations

import torch

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)
_ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", None)

TORCH_ACCELERATOR_ERRORS: tuple[type[BaseException], ...] = (
    (torch.cuda.OutOfMemoryError, _ACCELERATOR_ERROR)
    if isinstance(_ACCELERATOR_ERROR, type) and issubclass(_ACCELERATOR_ERROR, BaseException)
    else (torch.cuda.OutOfMemoryError,)
)


def is_float8_e8m0fnu(dtype: torch.dtype) -> bool:
    """Return whether ``dtype`` is PyTorch's optional e8m0 scale type."""
    return _FLOAT8_E8M0FNU is not None and dtype == _FLOAT8_E8M0FNU


def is_accelerator_oom(exc: BaseException) -> bool:
    """Recognize CUDA OOM errors across PyTorch exception hierarchies."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
