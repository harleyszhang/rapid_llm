---
paths:
  - "**/*.py"
---

# 通用代码风格

新写或改动的 Python 代码默认遵循以下约定；有具体理由可以偏离，在 review 里点明即可。

- **与工具链对齐。** ruff 0.16.3 已锁定，`line-length = 100`，`target-version = "py313"`；提交前 `make lint`（ruff check + ruff format --check）必须零告警。每个模块写 `from __future__ import annotations`，类型标注用新式写法（`X | None`、`list[int]`）。
- **优先无状态。** 优先纯函数：输入进、输出出；只有真正需要对象状态的行为才做成实例方法。生命周期内固定的派生值在 `__init__` 里算好、存成有名字的属性（`LinearBase.__init__` 解析 `params_dtype`，`modules/attention.py` 在 init 时缓存 `_kv_write = dispatch(...)`）；forward 路径只读属性，不重新推导。
- **优先不可变。** 数据结构默认 frozen dataclass（`KernelSpec`、`ShapeConstraint`、`ScaleLayout` 都是）；确需可变再放开，更新用重绑而不是原地改。
- **函数保持小。** 单函数 ~100 行以内，超出拆成有名字的 helper；单文件 ~2k 行以内，超出按内聚边界拆模块。
- **编排函数读起来像伪代码。** 一个单元的主函数要短、要直白，细节下沉到有名字的 helper，让顶层流程一眼可见。
- **避免 mixin。** 不通过 mixin 类加行为；用显式组合（持有协作者并调用）或普通函数。
- **默认 protected。** 方法默认 `_name`；只公开调用方真正使用的名字，`__all__` 与公开面同步维护。
- **两个及以上参数优先关键字。** 调用用关键字；设计 API 时用 `*` 分隔（`LinearOp.__call__`、`resolve_tiles` 都这么做）。
- **传所需的值，不传大对象。** 给被调方它实际用到的值；确需传整个对象时保持只读——读字段、把结果返回给调用方赋值。
- **不破坏依赖分层。** 核心依赖只列在 pyproject 的 `dependencies`；triton 属于 `cuda` extra，CPU-only 安装必须能 import 整个包——这就是 `kernels/dispatcher/` 保持 torch-free、`kernels/__init__.py` 惰性导出的原因。新增第三方依赖先论证能否进可选 extra。
- **错误要带上下文。** 前置条件用显式检查 + 说清期望与实际的异常（`update_kv_buffer` 的形状断言是范式）；库代码不 print、不静默吞异常；防御式访问见 `no-getattr-defensive.md`。
