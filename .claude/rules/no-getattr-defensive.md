---
paths:
  - "**/*.py"
---

# 不用 getattr / hasattr 做防御式访问

`getattr(obj, "field", default)` / `hasattr(obj, "field")` 的过度防御会掩盖错误、拖垮类型检查：字段既然总在，防御式访问就是误导；真出 `AttributeError` 时又把它吞掉，字段改名后没有任何东西会报警。

优先两种写法：

1. **`isinstance` 收窄类型，再直接访问字段。**
2. **字段总是存在**（构造时显式设好，必要时设 `None`），调用侧用 `None` 判断：

   ```python
   self.residual = None  # __init__ 里先设好
   ...
   if residual is not None:
       ...
   ```

坏 / 好对比：

```python
revision=getattr(config, "revision", None),   # BAD：字段总在，getattr 会吞掉改名后的 AttributeError
revision=config.revision,                     # GOOD
```

**允许的例外**（要一眼能看出是刻意的）：

- 真正可选的扩展点 / hook 探测：`bind = getattr(module, "_bind_kernels", None)`（`executor/model_runner.py`）——对象可能合法地没有这个 hook。
- 按名字动态分派：`getattr(self, f"_step_{self._state}")`（`engine/tool_parser.py`）。
- 兼容外部对象的类型差异（如 HF `PretrainedConfig` 的不同版本）时，收进一个具名 helper 并写清原因。

范围：新增代码执行本规则；既有用法顺手迁移，不做顺路大扫除。
