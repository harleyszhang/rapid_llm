"""Model loading: a HuggingFace checkpoint in, a ready-to-run ``nn.Module`` out.

:class:`DefaultModelLoader` builds the model with empty parameters
(:func:`init_empty_parameters`), streams checkpoint weights through the
translator, then :func:`materialise_parameters` moves them to device.

Usage:
    model = DefaultModelLoader().load(...)
"""

from __future__ import annotations

import contextlib
import gc
import sys
import time
import traceback
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..models.config import ModelConfig
from ..modules.quantization import RUNTIME_SCHEMES, RawParameter, for_runtime_scheme
from ..utils.logger import get_logger
from .weight_utils import hf_weights_iterator

logger = get_logger(__name__)

#: Parameters take the checkpoint's own element type (``config.dtype``) — bf16
#: unless the checkpoint declares otherwise. Kept as a named constant only for
#: the rare caller that builds a model with no config at hand.
PARAM_DTYPE = torch.bfloat16


@contextlib.contextmanager
def init_empty_parameters():
    """Skeleton context: parameters allocate on the meta device, buffers do not.

    Mirrors ``accelerate.init_empty_weights(include_buffers=False)``. Buffers keep real
    storage because non-persistent ones (e.g. ``RotaryEmbedding.inv_freq``) are absent
    from checkpoints and computed, not loaded.
    """
    original = nn.Module.register_parameter

    def register_meta_parameter(module: nn.Module, name: str, param) -> None:
        original(module, name, param)
        if module._parameters.get(name) is None:
            return
        # Preserve the Parameter subclass and its attributes: ``requires_grad``,
        # and the ``weight_loader`` the owning layer bound to the parameter.
        existing = module._parameters[name]
        new = type(existing)(
            existing.to(torch.device("meta")), requires_grad=existing.requires_grad
        )
        new.__dict__.update(existing.__dict__)
        module._parameters[name] = new

    try:
        nn.Module.register_parameter = register_meta_parameter
        yield
    finally:
        nn.Module.register_parameter = original


def materialise_parameters(
    model: nn.Module, device: str | torch.device, dtype: torch.dtype = PARAM_DTYPE
) -> None:
    """Give every meta parameter real, uninitialised storage on ``device``.

    Only floating-point parameters are cast to ``dtype``; integer ones (some HF vision
    towers keep index tables as parameters) and :class:`RawParameter` quantisation
    parameters keep theirs (an 8-bit weight must not be widened, its fp32 scales not
    narrowed). Storage stays uninitialised because :func:`load_weights` overwrites and
    verifies it. ``requires_grad=False`` is load-bearing: copying into a leaf that
    requires grad is an in-place autograd error, and the copy fills fused parameters.
    """
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is None:
                continue
            keep_dtype = isinstance(param, RawParameter) or not param.is_floating_point()
            if keep_dtype:
                # Preserve dtype *and* the kernel-facing layout: scale grids
                # allocate as column-major views (quantization.base_config's
                # scale_parameter), and ``torch.empty(shape)`` would flatten
                # them back to row-major, undoing the allocation-time decision.
                storage = torch.empty_strided(
                    param.shape, param.stride(), dtype=param.dtype, device=device
                )
            else:
                storage = torch.empty(param.shape, dtype=dtype, device=device)
            new = type(param)(storage, requires_grad=False)
            # Attribute carrier as well as storage: the ``weight_loader`` bound at
            # construction must survive into the materialised parameter.
            new.__dict__.update(param.__dict__)
            module._parameters[name] = new


@runtime_checkable
class ModelLoader(Protocol):
    """Strategy seam: anything that can turn a checkpoint dir into a model."""

    def load_model(
        self,
        config: ModelConfig,
        model_cls: type[nn.Module],
        checkpoints_dir: str,
        device: str,
        quantization: str | None = None,
    ) -> nn.Module: ...


class DefaultModelLoader:
    """Loads HuggingFace weights directly, with no offline conversion step."""

    def load_model(
        self,
        config: ModelConfig,
        model_cls: type[nn.Module],
        checkpoints_dir: str,
        device: str,
        quantization: str | None = None,
    ) -> nn.Module:
        """Build ``model_cls`` and fill it from the checkpoint in ``checkpoints_dir``.

        Args:
            config: Parsed config, carrying ``max_seq_len`` and the checkpoint's weight
                format (``config.quant``).
            model_cls: Implementation class resolved from the registry.
            checkpoints_dir: HF checkpoint dir (``config.json`` + ``*.safetensors``).
            device: Torch device the model must end up on.
            quantization: Post-load quantisation for an fp16 checkpoint (see
                :data:`RUNTIME_SCHEMES`), or ``None``; ignored for an already-quantised
                checkpoint.
        """
        start = time.time()
        self._check_device(device)

        logger.info("Building %s skeleton on the meta device", model_cls.__name__)
        with init_empty_parameters():
            model = model_cls(config)
        try:
            materialise_parameters(model, device, dtype=config.dtype)

            # An already-quantised checkpoint is copied in byte for byte; only an
            # unquantised model wants the FP8 blocks widened on the way through.
            model.load_weights(
                hf_weights_iterator(
                    checkpoints_dir,
                    device,
                    dequantize_fp8=config.quant is None,
                    dequant_dtype=config.dtype,
                )
            )
            if quantization and config.quant is None:
                self._quantize(model, quantization)
            # Buffers were computed on the CPU while the skeleton was on meta.
            model.to(device).eval()
        except BaseException:
            # A failed load leaves the half-filled weights pinned in this
            # process in three ways: the module tree is a reference cycle the
            # generational GC has not run on yet; the traceback keeps every
            # frame it passed through — including the ``model.modules()``
            # iterator that holds the model — alive for as long as the caller
            # keeps the exception (pytest keeps it for the whole session); and
            # the caching allocator pools the freed blocks instead of handing
            # them back. Either way, the memory the next engine build in this
            # process needed stays occupied. Break every hold, then re-raise:
            # the re-raised traceback still shows the code path, only the
            # dead frames' locals are gone.
            del model
            tb = sys.exc_info()[2]
            if tb is not None:
                traceback.clear_frames(tb)
            # Order matters: the collect must run after the frames are
            # cleared, or the cycles it would break are still rooted in them.
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            raise
        logger.info("Loaded %s onto %s in %.2fs", model_cls.__name__, device, time.time() - start)
        return model

    @staticmethod
    def _quantize(model: nn.Module, quantization: str) -> None:
        """Apply a runtime quantisation request to a freshly loaded fp16 model.

        Raises:
            ValueError: On an unrecognised scheme name.
        """
        try:
            quant = for_runtime_scheme(quantization)
        except ValueError:
            raise ValueError(
                f"cannot quantise an fp16 checkpoint to {quantization!r} at load time; "
                f"supported schemes: {sorted(RUNTIME_SCHEMES)}, or point --model-dir "
                "at a checkpoint that ships pre-quantised weights"
            ) from None
        if not hasattr(model, "quantize_"):
            raise ValueError(f"{type(model).__name__} does not support runtime quantisation")
        logger.info("Quantising weights to %s (%s)", quantization, quant.get_name())
        model.quantize_(quant)

    @staticmethod
    def _check_device(device: str) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            # Copying into cuda parameters would otherwise fail deep inside torch with
            # a message that says nothing about drivers.
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "This usually means the installed torch build targets a newer CUDA "
                f"than the NVIDIA driver on this machine. Installed: torch=={torch.__version__} "
                f"(cuda={torch.version.cuda}). Fix by installing a torch build that matches "
                "the local driver, e.g. `uv pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124`."
            )
