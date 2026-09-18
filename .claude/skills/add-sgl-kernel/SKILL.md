---
name: add-sgl-kernel
description: Step-by-step tutorial for adding a heavyweight AOT CUDA/C++ kernel to rapid_llm's optional native extension (csrc/ tree + torch library registration + optional wheel build), covering first-time build scaffolding, KernelSpec registration, tests, and benchmarks. Use when a kernel depends on CUTLASS or another large C++ project, when C++ infrastructure is shared across many kernels, when the extension must ship prebuilt in the wheel, or when JIT compile time at first run is unacceptable.
---

# Tutorial: Adding an AOT CUDA/C++ Kernel

The AOT counterpart of `add-jit-kernel` — the flow sglang's `sgl-kernel` package uses, adapted to rapid_llm. Walks through `scale(x, factor) = x * factor` (FP16/BF16/FP32). rapid_llm has no prebuilt native extension yet, so the first AOT kernel also lands the scaffolding in this tutorial; later kernels repeat Steps 1-6 only.

## Path decision (choose before writing code)

| Path | When to use | Skill |
| --- | --- | --- |
| Triton (default) | Expressible in Triton; supplies the never-failing `native` floor row | `triton-kernel-writing` |
| JIT CUDA | CUDA C++ needed but no heavyweight C++ dependency; single op, fast iteration | `add-jit-kernel` |
| AOT extension (this skill) | CUTLASS / large C++ project, shared C++ infra, ship prebuilt in the wheel, no compile cost tolerated at first run | `add-sgl-kernel` |
| External library | flashinfer / deepgemm / deepep / flashmla already ships a proven implementation | `add-op-backend` |

Every new kernel ships with tests and a benchmark in the same PR — same bars as every other kernel path.

## Repository map (first kernel creates the tree)

```
csrc/                                   # NEW tree: AOT extension sources
├── include/rapid_llm_ext.h             # public C++ declarations + dispatch macros
├── elementwise/scale.cu                # kernel + launcher
└── common_extension.cc                 # TORCH_LIBRARY_FRAGMENT registration
setup.py                                # NEW: optional CUDAExtension, RAPID_LLM_BUILD_EXT=1
rapid_llm/kernels/aot.py                # NEW once: available() probe
rapid_llm/kernels/ops/<group>/<op>_aot.py       # thin wrapper over torch.ops.rapid_llm.*
rapid_llm/kernels/ops/<group>/__init__.py     # KernelSpec row (backend="aot")
tests/kernels/, benchmarks/kernels/           # same bars as every kernel
```

Hard constraint: `pyproject.toml` builds a pure-Python wheel and the README promises a working CPU-only install. The extension stays **optional**: built only under `RAPID_LLM_BUILD_EXT=1`, imported only behind the lazy KernelSpec `target`, and its row carries an `available` probe so CPU-only or extension-less installs degrade to the `native` row. Never import `rapid_llm._C` at package top level.

## Step 1: kernel + launcher in `csrc/elementwise/scale.cu`

Same body as the `add-jit-kernel` example, two differences: **no `PYBIND11_MODULE`** (registration lives in `common_extension.cc`, Step 3) and the launcher is declared in the public header (Step 2). The conventions carry over unchanged: `TORCH_CHECK` validation around the launch, `at::cuda::getCurrentCUDAStream()` + `OptionalCUDAGuard`, `AT_DISPATCH_FLOATING_TYPES_AND2`, `C10_CUDA_KERNEL_LAUNCH_CHECK()` after launch, ASCII only, `const T* __restrict__`, `int64_t` for numel-derived indexing. A kernel that syncs or assembles per-step host data is not CUDA-graph capture-safe — its spec row must say `graph_safe=False`.

Arch-gated code (SM90+ CUTLASS paths): enforce with `TORCH_CHECK` in the launcher and skip logic in tests.

## Step 2: declaration in `csrc/include/rapid_llm_ext.h`

```cpp
#pragma once
#include <torch/extension.h>

// elementwise
void scale(at::Tensor& out, const at::Tensor& input, double factor);
```

## Step 3: registration in `csrc/common_extension.cc`

```cpp
#include <torch/extension.h>
#include "rapid_llm_ext.h"

TORCH_LIBRARY_FRAGMENT(rapid_llm, m) {
  // elementwise
  m.def("scale(Tensor! out, Tensor input, float factor) -> ()");
  m.impl("scale", torch::kCUDA, &scale);
}
```

- `Tensor!` marks the mutable out-argument; the schema is what `torch.compile` and the dispatcher see, so keep it exact.
- Schema scalar types stay PyTorch types (`float`); the C++ signature still takes `double` — `torch::Library` requires it.
- One `TORCH_LIBRARY_FRAGMENT` per extension; new ops append `m.def`/`m.impl` pairs here.

## Step 4: build wiring `setup.py` (once)

```python
"""Optional native extension. Pure-Python install remains the default."""
import os

from setuptools import setup

ext_modules = []
cmdclass = {}
if os.environ.get("RAPID_LLM_BUILD_EXT") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules.append(
        CUDAExtension(
            name="rapid_llm._C",
            sources=[
                "csrc/common_extension.cc",
                "csrc/elementwise/scale.cu",  # keep sorted
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(ext_modules=ext_modules, cmdclass=cmdclass)
```

Build:

```bash
RAPID_LLM_BUILD_EXT=1 pip install -e .
# limit host load:
RAPID_LLM_BUILD_EXT=1 MAX_JOBS=2 pip install -e .
```

`TORCH_CUDA_ARCH_LIST` controls the gencode set; set it explicitly on build machines so the wheel is not pinned to the build GPU. When the source tree outgrows one list (CUTLASS generators, per-arch files), migrate to CMake — that is a deliberate trade, not the default.

## Step 5: availability probe `rapid_llm/kernels/aot.py` (once)

```python
"""Availability of the optional AOT extension for KernelSpec rows."""
from __future__ import annotations

import importlib.util


def available() -> bool:
    return importlib.util.find_spec("rapid_llm._C") is not None
```

## Step 6: Python wrapper + KernelSpec row

`ops/elementwise/scale_aot.py` — thin like every wrapper: allocate `out`, one call, return. Validation lives in the C++ launcher.

```python
def scale(input: torch.Tensor, factor: float,
          out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    torch.ops.rapid_llm.scale.default(out, input, factor)
    return out
```

Row in `ops/elementwise/__init__.py`:

```python
register(
    KernelSpec(
        name="aot/scale",
        op="scale",
        backend="aot",
        target="rapid_llm.kernels.ops.elementwise.scale_aot:scale",
        available="rapid_llm.kernels.aot:available",
        dtypes=("bf16", "fp16", "fp32"),
        golden=GoldenRecord(verified=True, max_abs_diff=0.0, baseline="native/scale"),
        priority=UNMEASURED,
    )
)
```

- `backend="aot"` is the family label; the exactly-one-`native`-row-per-op rule is untouched — an optional extension must never be the floor.
- Extension absent -> row filtered, `explain()` shows why, `native` wins. That fallback is what keeps CPU-only installs working.
- `golden` records the measured diff against the native baseline; unverified rows stay out of default dispatch. Register the module in `kernels/__init__.py` `_EXPORTS`.

## Step 7: tests and benchmark (required, same bars as JIT)

- Tests: `tests/kernels/test_scale_aot.py` per `write-test` — dtype x size (tail sizes) x factor, oracle is `tests/reference.py`, CPU/shape-mismatch inputs raise. Skip cleanly when the extension is absent so CPU-only CI stays green.
- Benchmark: `benchmarks/kernels/` via the `kernel-microbenchmark` harness; arms are the AOT row, the `native` row, and the JIT row if one exists. The result feeds `priority`; unmeasured rows lose ties.

## Step 8: validate

```bash
RAPID_LLM_BUILD_EXT=1 pip install -e .
pytest tests/kernels/test_scale_aot.py -q
python -m benchmarks.kernels.<...>          # the harness entry for the op
```

Also verify the negative path in CI: an install without `RAPID_LLM_BUILD_EXT` must import the package, filter the `aot/*` rows, and pass the full test suite on the `native` floor.

## Troubleshooting

- **Undefined symbol at import**: the `.cu` is missing from `sources` in `setup.py` — the build succeeded but never compiled the file.
- **`torch.ops.rapid_llm.scale` missing**: `m.def`/`m.impl` not in `common_extension.cc`, or the wheel was built without `RAPID_LLM_BUILD_EXT=1`.
- **Build OOM / too slow**: lower `MAX_JOBS`; CUTLASS TUs are heavy — budget one job per 2 GB RAM.
- **Wrong-arch wheel**: set `TORCH_CUDA_ARCH_LIST` explicitly on the build machine.
- **Async CUDA errors**: `CUDA_LAUNCH_BLOCKING=1`; memory errors: `compute-sanitizer --tool memcheck python ...`.

## Cross-references

- `add-jit-kernel` — the lightweight counterpart; try it first unless a heavyweight C++ dependency forces AOT
- `triton-kernel-writing` — the default kernel path
- `add-op-backend` — external-library path; KernelSpec registration detail, dispatch correctness tests, `explain()`
- `kernel-microbenchmark` — the only sanctioned timing harness
- `write-test` — test admission rules
- `locate-numeric-divergence` — when the new row disagrees with the baseline
- `rules/kernel-conventions.md` — layering, indexing, and registration rules that bind `.cu` code too

## Summary of files (first kernel)

```
csrc/elementwise/scale.cu                 # NEW: kernel + launcher
csrc/include/rapid_llm_ext.h              # NEW: declaration
csrc/common_extension.cc                  # NEW: TORCH_LIBRARY_FRAGMENT registration
setup.py                                  # NEW: optional CUDAExtension (RAPID_LLM_BUILD_EXT=1)
rapid_llm/kernels/aot.py                  # NEW once: available() probe
rapid_llm/kernels/ops/elementwise/scale_aot.py  # NEW: wrapper
rapid_llm/kernels/ops/elementwise/__init__.py   # NEW group (or MODIFIED): KernelSpec row
rapid_llm/kernels/__init__.py             # MODIFIED: _EXPORTS entry
tests/kernels/test_scale_aot.py           # NEW: tests
benchmarks/kernels/...                    # NEW: benchmark per harness
```
