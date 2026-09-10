---
paths:
  - "**/*.py"
---

# 注释与 docstring 风格

注释按代码来 review，举证责任是反的：作者为注释的存在辩护，而不是 reviewer 为删除辩护。

## 每个文件都有模块 docstring

本仓库的签名惯例，三句话结构：一句话职责 → 原理与机制 → `Usage:` 代码块。

```python
"""Scatter freshly computed K/V rows into the paged KV buffer.

One Triton launch writes the selected rows of this step's K and V to
the cache positions ``select_index`` names — rows not selected are
never touched.

Usage:
    update_kv_buffer(k, v, select_index, kv_buffer)
"""
```

（范例：`rapid_llm/kernels/ops/kvcache/update_kv_buffer.py`；`ops/interfaces.py` 与 `dispatcher/spec.py` 的模块 docstring 把契约讲得更完整。）

## 注释该写什么

写读者从这一行看不出来的事实。删掉注释后问：恢复这个事实要付出多大代价？跨文件的事实代价最大，复述下一行的代价是零。

- **跨边界约束。** 与 kernel / 上游调用方 / checkpoint 布局共享的字段顺序、内存布局、调用次序。
- **名字带不动的单位与布局。** tokens / reqs / blocks / bytes / slots 的区分，张量的 shape / dtype / layout。先试着写进名字；名字被接口固定时才用注释。
- **魔数的出处。** 硬件约束、实测结果，或明说「随手选的、可以改」——最后一种最有价值，它告诉下一个人这个值可以动。
- **workaround 带锚点和退场条件。** 上游 issue 链接 + 何时能删；没有锚点的 workaround 是永生的。
- **像巧合的契约决策。** 哨兵值的含义、刻意的缺席、看似随意的顺序。

仓库里的好例子（`vocab_embedding.py`）：

```python
# An id another rank owns leaves the load mask empty, so ``other=0.0`` flows
# to the store: a zero row is this rank's contribution for that token. The
# negative pointer arithmetic for ``local_row == -1`` is never dereferenced
# because the mask is what guards the access, not the pointer value.
```

## 不写什么

- 复述下一行代码的散文；`# Step 1:` 式编号（流程需要编号时，抽有名字的 helper）。
- 注释掉的代码，任何长度。
- 变更史：「以前是这么做的，因为……」——过去做法的家在 `git log`；仍是活约束的旧故障，写成约束本身而不是故事。
- review 归属，或把 PR 号当 changelog。
- 私有 helper / override 的多行 `Args:` / `Returns:`——签名和类型标注已经承载名字与类型。

## 形式与约定

- **一到两行。** 真正复杂的约束可以放宽，但那是例外不是许可。
- **贴在它约束的那一行**，不要收集成函数顶部的前言——收集起来的注释最先过期。
- **改了行就负责它的注释**：更新或删除，不留孤儿；过期注释比没有注释更糟。
- **只用两种 tag。** `# NOTE:` 表示约束或陷阱（多数时候前缀是多余的，直接陈述事实即可）；`# TODO(owner):` 必须有 owner 或 issue 链接，无主 TODO 在 review 里拒绝。
- **中文注释保持原样。** 仓库保留早期教学材料里的中文注释和 docstring（ruff 的 RUF001–RUF003 已为此豁免）——不翻译、不顺手重写；新代码的注释用英文，与周围一致。
- **docstring 里的代码块会被 ruff format**（`docstring-code-format = true`），保持可格式化。

## docstring 写到哪一层

- **写**：dispatch 契约（`ops/interfaces.py` 的每个 `LogicalOp.__call__` 钉住 shape / dtype / layout）、kernel 包装函数、`KernelSpec` 字段语义、给第三方子类的扩展点、公共模块函数。
- **不写**：内部 helper、`_` 前缀私有方法、override。
