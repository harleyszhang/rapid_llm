---
paths:
  - "rapid_llm/kernels/**/*.py"
  - "tests/kernels/**/*.py"
  - "benchmarks/kernels/**/*.py"
---

# Kernel 编写与注册规范

## 分层与导入边界

- `rapid_llm/kernels/` 分三层：`ops/` 放实现（一个文件一个算子族），`dispatcher/` 放注册与选择策略，`backend/` 放外部库行。实现文件顶层可以 `import triton`；`dispatcher/` 与 `ops/interfaces.py` 必须保持 torch-free（运行时只 `TYPE_CHECKING`），`kernels/__init__.py` 的惰性导出保证 CPU-only 安装能 import 整个包。新增实现模块要登记到 `_EXPORTS`；需要 CPU 兜底时同时补 `_CPU_OPS` 与 `backend/cpu.py`。
- 模块骨架固定：模块 docstring（职责 + 机制 + `Usage:`）→ `@torch.no_grad()` 公共包装（校验形状、算 grid、只传 stride 与标量）→ `_` 前缀的 `@triton.jit` kernel。

## 启动与索引

- 大表地址运算先 `.to(tl.int64)` 再乘 stride：vocab 表、专家权重、KV 池都算大表（`swiglu.py`、`vocab_embedding.py`、`fused_moe.py` 开头就是 cast）。KV scatter 目前接受 allocator 给的 int32 行号，仅在 `max_rows * row_stride_elements < 2**31` 时成立——扩大池子前先检查这个界限。
- `grid[1]` / `grid[2]` 不得超过 65535；随 token 数增长的维度放轴 0 或自行切块（`flashattention2_nopad` 的 `cdiv(max_seq_len, BLOCK_M)` 在轴 0）。
- 小 scatter kernel 写死 `num_warps=1, num_stages=1`（`update_kv_buffer`）；块 GEMM 的 tile / warps 一律从 `resolve_tiles` 拿，不新增 `@triton.autotune`。
- 包装函数不做 `.contiguous()`：按调用方传入的 stride 读写（生产输入是组合缓冲的 strided 视图）；确需连续时断言，让调用方在计时路径外付拷贝。

## 设备能力与回退

- 设备分档用现成查询：`sm_version` / `has_native_fp8`（`ops/tile_policy.py`）、`CapabilityRequirement`（spec 行）。不支持当前设备的 kernel 必须回退（torch 路径或原生 Triton 行）或被过滤——外部后端缺失时降级到原生行是既定设计，不硬失败。
- 量化格式的新 kernel 按 `TileTier` 分档给启发式表；在 A10（sm86）上能成立的设计才允许作为默认（H100 表在 sm86 上会溢出或编不出来）。

## 注册、dispatch 与验证

- 新 kernel 落地三件套：实现 + `KernelSpec` 行（走 dispatch 的必须）+ `tests/kernels/` 用例，同一个 PR 提交。行声明要如实：dtype / layout tags、shape 约束；被过滤的行要能在 `explain()` 里看到原因。
- 模块层在 `__init__` 缓存 `dispatch(...)` 结果（`modules/attention.py` 的 `_kv_write`），forward 里不重复 dispatch。
- 正确性先于性能：先过 `tests/kernels/`（对照 `tests/reference.py`），再谈延迟；benchmark 走 `kernel-microbenchmark` skill 的 harness，不在脚本里另起计时。
