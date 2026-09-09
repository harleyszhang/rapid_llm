"""Tests for the platform layer (ROADMAP A9).

``PlatformInfo`` / ``CapabilityRequirement`` construction and matching,
plus ``current_platform()`` agreeing with the torch-visible device.

Usage:
    pytest tests/platform/test_platform.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.platform import (
    CapabilityRequirement,
    CpuPlatform,
    Platform,
    PlatformInfo,
    capabilities_match,
    current_platform,
)

A10 = PlatformInfo("cuda", 8, 6, "NVIDIA A10")
H100 = PlatformInfo("cuda", 9, 0, "NVIDIA H100")
CPU = PlatformInfo()


class TestPlatformInfo:
    def test_compute_capability_present(self) -> None:
        assert A10.compute_capability == (8, 6)

    def test_compute_capability_minor_defaults_to_zero(self) -> None:
        assert PlatformInfo("cuda", 9).compute_capability == (9, 0)

    def test_compute_capability_absent_on_cpu(self) -> None:
        assert CPU.compute_capability is None

    def test_frozen(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            A10.arch_major = 9  # type: ignore[misc]


class TestCapabilityRequirement:
    def test_device_mismatch_rejects(self) -> None:
        assert not CapabilityRequirement("cuda").matches(CPU)

    def test_no_bounds_admits_any_cuda(self) -> None:
        assert CapabilityRequirement("cuda").matches(A10)
        assert CapabilityRequirement("cuda").matches(H100)

    def test_min_cc_is_inclusive(self) -> None:
        hopper = CapabilityRequirement("cuda", min_cc=(9, 0))
        assert hopper.matches(H100)
        assert not hopper.matches(A10)

    def test_max_cc_is_inclusive(self) -> None:
        pre_hopper = CapabilityRequirement("cuda", max_cc=(8, 9))
        assert pre_hopper.matches(A10)
        assert not pre_hopper.matches(H100)

    def test_sm_window(self) -> None:
        ada = CapabilityRequirement("cuda", min_cc=(8, 9), max_cc=(8, 9))
        assert ada.matches(PlatformInfo("cuda", 8, 9))

    def test_bounds_reject_missing_capability(self) -> None:
        assert not CapabilityRequirement("cuda", min_cc=(7, 5)).matches(CPU)


class TestCapabilitiesMatch:
    def test_or_semantics(self) -> None:
        reqs = [CapabilityRequirement("cuda", min_cc=(9, 0)), CapabilityRequirement("cpu")]
        assert capabilities_match(reqs, H100)
        assert capabilities_match(reqs, CPU)
        assert not capabilities_match(reqs, A10)

    def test_empty_means_everywhere(self) -> None:
        assert capabilities_match([], CPU)
        assert capabilities_match([], A10)


class TestCurrentPlatform:
    def test_singleton_is_stable(self) -> None:
        assert current_platform() is current_platform()

    def test_returns_a_platform(self) -> None:
        assert isinstance(current_platform(), Platform)

    def test_cpu_box_degrades_to_cpu_platform(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("asserts the CPU-only degradation path")
        plat = current_platform()
        assert isinstance(plat, CpuPlatform)
        assert plat.detect() == PlatformInfo()

    def test_cuda_box_detects_real_device(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("asserts the CUDA detection path")
        from rapid_llm.platform.cuda import CudaPlatform

        plat = current_platform()
        assert isinstance(plat, CudaPlatform)
        info = plat.detect()
        assert info.device_type == "cuda"
        assert info.compute_capability == torch.cuda.get_device_capability()
