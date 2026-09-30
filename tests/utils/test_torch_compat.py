"""Regression tests for optional PyTorch capability detection.

The compatibility module must import and classify errors even when a PyTorch build
omits newer dtype and accelerator exception symbols.

Usage:
    pytest tests/utils/test_torch_compat.py
"""

from __future__ import annotations

import torch

from rapid_llm.modules.quantization.mxfp4 import e8m0_to_fp32
from rapid_llm.utils.torch_compat import (
    TORCH_ACCELERATOR_ERRORS,
    is_accelerator_oom,
    is_float8_e8m0fnu,
)


def test_uint8_e8m0_conversion_does_not_require_native_dtype():
    scale = torch.tensor([0, 126, 127, 128, 255], dtype=torch.uint8)

    actual = e8m0_to_fp32(scale)
    expected = torch.exp2(scale.to(torch.int32) - 127)

    torch.testing.assert_close(actual, expected)


def test_optional_e8m0_dtype_detection():
    native_dtype = getattr(torch, "float8_e8m0fnu", None)

    assert not is_float8_e8m0fnu(torch.uint8)
    if native_dtype is not None:
        assert is_float8_e8m0fnu(native_dtype)


def test_accelerator_error_tuple_is_safe_on_older_torch():
    error = torch.cuda.OutOfMemoryError("simulated OOM")

    assert isinstance(error, TORCH_ACCELERATOR_ERRORS)
    assert is_accelerator_oom(error)
    assert not is_accelerator_oom(RuntimeError("kernel launch failed"))
