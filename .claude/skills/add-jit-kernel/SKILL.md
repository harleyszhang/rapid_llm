---
name: add-jit-kernel
description: Step-by-step tutorial for adding a lightweight JIT-compiled CUDA kernel to rapid_llm (torch cpp_extension path), covering the one-time loader, KernelSpec registration, tests, and benchmarks. Use when adding a CUDA C++ kernel that Triton cannot express (inline PTX, special intrinsics, fine-grained memory control), when the kernel has no heavyweight C++ dependency such as CUTLASS, or when fast compile-test iteration on one custom CUDA op matters.
---

# Tutorial: Adding a JIT-Compiled CUDA Kernel

Walks through adding `scale(x, factor) = x * factor` (FP16/BF16/FP32) to show the complete workflow. JIT = CUDA C++ source compiled by `torch.utils.cpp_extension.load` on first use, cached afterwards. No build system, no wheel involvement.

## Path decision (choose before writing code)

| Path | When to use | Skill |
| --- | --- | --- |
| Triton (default) | Expressible in Triton; supplies the never-failing `native` floor row | `triton-kernel-writing` |
| JIT CUDA (this skill) | Needs CUDA C++: inline PTX, special intrinsics, cooperative groups, exotic memory patterns; no CUTLASS-class dependency; single op, fast iteration | `add-jit-kernel` |
| AOT extension | CUTLASS / large C++ project, shared C++ infra across kernels, must ship prebuilt in the wheel, JIT compile time unacceptable at first run | `add-sgl-kernel` |
| External library | flashinfer / deepgemm / deepep / flashmla already ships a proven implementation | `add-op-backend` |

Rule of thumb (sglang's rule, adapted): no heavyweight C++ dependency -> JIT; CUTLASS dependency or wheel-shipped -> AOT. A JIT row whose toolchain probe fails is filtered by dispatch and the `native` row wins — degrade-to-native is the existing design, not an error.

## Repository map

```
rapid_llm/kernels/
├── jit.py                        # ONE-TIME: cached load() wrapper + available() probe
├── csrc/<group>/<op>.cu          # per kernel: CUDA source, one file per op family
└── ops/<group>/<op>_jit.py       # per kernel: thin Python wrapper
rapid_llm/kernels/ops/<group>/__init__.py   # per kernel: KernelSpec row
tests/kernels/test_<op>_jit.py              # required, per `write-test`
benchmarks/kernels/...                      # required, per `kernel-microbenchmark`
```

CPU-only installs must keep importing the whole package (see `rules/kernel-conventions.md`): `jit.py` is only reached through the lazy KernelSpec `target`, never at package top level.

## Step 1 (once): the loader `rapid_llm/kernels/jit.py`

```python
"""Cached JIT compilation for CUDA sources under kernels/csrc/."""
from __future__ import annotations

import functools
import shutil
from pathlib import Path

import torch

_CSRC = Path(__file__).parent / "csrc"


def available() -> bool:
    """Toolchain probe for KernelSpec rows. Cheap: never compiles."""
    return torch.cuda.is_available() and shutil.which("nvcc") is not None


@functools.cache
def load_jit(name: str, source: str, *, extra_cuda_cflags: tuple[str, ...] = ("-O3",)):
    """Compile ``csrc/<source>`` on first use; return the extension module.

    torch's own extension cache makes repeat imports cheap; this cache avoids
    even the filesystem probe. The cache key is (name, source, flags) — bump
    `name` if you change flags for the same source.
    """
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("JIT kernels need a CUDA toolkit (CUDA_HOME unset)")
    return load(
        name=f"rapid_llm_jit_{name}",
        sources=[str(_CSRC / source)],
        extra_cuda_cflags=list(extra_cuda_cflags),
        verbose=False,
    )
```

## Step 2: the CUDA source `csrc/elementwise/scale.cu`

```cpp
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

template <typename T>
__global__ void scale_kernel(T* __restrict__ out, const T* __restrict__ in,
                             float factor, int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<T>(static_cast<float>(in[idx]) * factor);
  }
}

}  // namespace

// Validate HERE, not in Python — checks cost nothing next to a kernel launch.
void scale(at::Tensor& out, const at::Tensor& input, double factor) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(), "scale: CUDA tensors only");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(), "scale: contiguous only");
  TORCH_CHECK(out.sizes() == input.sizes() &&
              out.scalar_type() == input.scalar_type(),
              "scale: out must match input shape/dtype");

  const int64_t n = input.numel();
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  const at::cuda::OptionalCUDAGuard guard(at::device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                  input.scalar_type(), "scale", [&] {
    scale_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        static_cast<float>(factor), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scale", &scale); }
```

Conventions, hold for every `.cu` under `csrc/`:

- ASCII only; `const T* __restrict__` for read-only pointers; `int64_t` for numel-derived indexing (same big-table rule as the Triton side).
- `at::cuda::getCurrentCUDAStream()` + device guard — never a hardcoded stream or `cudaDeviceSynchronize`. This is what keeps the kernel CUDA-graph capture-safe.
- A kernel that calls `.item()`, syncs, or does host-side per-step assembly is **not** capture-safe; its spec row must say `graph_safe=False` (see `KernelSpec` docstring for why replay bakes host values in).
- `TORCH_CHECK` validation and dtype dispatch in C++; `C10_CUDA_KERNEL_LAUNCH_CHECK()` after every launch.
- Arch-gated features (SM90+ intrinsics): gate in Python before `load_jit` with a clear error, mirroring how `ops/tile_policy.py` exposes `sm_version`.

## Step 3: the wrapper `ops/elementwise/scale_jit.py`

```python
"""JIT CUDA scale: out = input * factor (fp16/bf16/fp32)."""
from __future__ import annotations

import torch

from rapid_llm.kernels.jit import load_jit


def scale(input: torch.Tensor, factor: float,
          out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    # Per-call path: allocate + one cached-module call. No validation here —
    # tensor invariants live in the C++ launcher, toolchain checks in jit.py.
    load_jit("scale", "elementwise/scale.cu").scale(out, input, factor)
    return out
```

Check placement (sglang's rule, adapted): compile-time facts -> `static_assert` in C++; tensor invariants -> `TORCH_CHECK` in the launcher; toolchain gating -> the cached factory; per-call Python does nothing but pick the module and allocate `out`. A wrapper that re-checks tensors costs interpreter time on every forward.

Register the module in `kernels/__init__.py` `_EXPORTS` like every new ops module.

## Step 4: the KernelSpec row in `ops/elementwise/__init__.py`

```python
register(
    KernelSpec(
        name="jit/scale",
        op="scale",
        backend="jit",
        target="rapid_llm.kernels.ops.elementwise.scale_jit:scale",
        available="rapid_llm.kernels.jit:available",
        dtypes=("bf16", "fp16", "fp32"),
        golden=GoldenRecord(verified=True, max_abs_diff=0.0, baseline="native/scale"),
        priority=UNMEASURED,
        graph_safe=True,
    )
)
```

- `backend="jit"` is the family label. The exactly-one-`native`-row-per-op rule is untouched: a JIT row can fail to build, so it must never be the floor.
- `available` False -> row filtered, `explain()` shows why, `native` wins. That fallback is the design; do not hard-fail engine startup on a missing toolchain.
- `golden`: record the measured diff against the native baseline after `tests/kernels/` passes; unverified rows are excluded from default dispatch.
- First dispatch triggers the real compile (seconds). Tests warm it; a broken build raises with the torch extension log — fix the toolchain, don't swallow the error.

## Step 5: tests (required)

`tests/kernels/test_scale_jit.py` per `write-test`: parametrize dtype x size (include tail sizes like 4097) x factor; compare against the `tests/reference.py` oracle, not against the Triton row; CPU-tensor and shape-mismatch inputs must raise. Use a module-scoped fixture that calls `load_jit` once so a broken toolchain fails fast with the build log.

## Step 6: benchmark (required)

Add a `benchmarks/kernels/` case through the `kernel-microbenchmark` harness — `bench` / `bench_stateful` / `bench_host`, cold-L2 rotation, no hand-rolled timers (kernel-conventions rule). Arms: the JIT row, the `native` row, and a torch reference. The measured latency feeds the row's `priority`; an unmeasured row stays `UNMEASURED` and loses ties.

## Troubleshooting

- **Build fails**: rerun with `verbose=True` in `load_jit`; the ninja log under `~/.cache/torch_extensions/` has the real error. Check nvcc version vs the CUDA torch wheel.
- **`available()` False on a CUDA box**: `CUDA_HOME`/`nvcc` missing — install the toolkit; the row correctly degrades to `native` meanwhile.
- **Illegal memory access**: `CUDA_LAUNCH_BLOCKING=1`, then `compute-sanitizer --tool memcheck python ...`.
- **Stale binary after editing flags**: bump the `name` argument or clear the torch extension cache dir.
- **CUDA graph capture breaks**: the kernel burns host state per call — set `graph_safe=False` on the row or move that work into the kernel.

## Cross-references

- `triton-kernel-writing` — the default kernel path; read it first and justify why Triton cannot express the op
- `add-sgl-kernel` — the AOT counterpart for CUTLASS-class kernels
- `add-op-backend` — external-library path; KernelSpec registration detail, dispatch correctness tests, `explain()`
- `kernel-microbenchmark` — the only sanctioned timing harness
- `write-test` — test admission rules
- `locate-numeric-divergence` — when the new row disagrees with the baseline
- `rules/kernel-conventions.md` — layering, indexing, and registration rules that bind `.cu` code too

## Summary of files

```
rapid_llm/kernels/jit.py                        # NEW once: cached loader + available()
rapid_llm/kernels/csrc/elementwise/scale.cu     # NEW: CUDA source
rapid_llm/kernels/ops/elementwise/scale_jit.py  # NEW: wrapper
rapid_llm/kernels/ops/elementwise/__init__.py   # NEW group (or MODIFIED): KernelSpec row
rapid_llm/kernels/__init__.py                   # MODIFIED: _EXPORTS entry
tests/kernels/test_scale_jit.py                 # NEW: tests
benchmarks/kernels/...                          # NEW: benchmark per harness
```
