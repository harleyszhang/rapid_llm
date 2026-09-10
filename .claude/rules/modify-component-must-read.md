---
paths:
  - "**"
---

# 改动组件前必读的技能

在改动下列组件之前，先读对应的 skill。

- **Kernel 实现**（`rapid_llm/kernels/**`，任何新增或修改的 `@triton.jit` kernel）→ [`triton-kernel-writing`](../../.claude/skills/triton-kernel-writing/SKILL.md)
- **Kernel 微基准**（`benchmarks/kernels/**` 的内核级性能数字）→ [`kernel-microbenchmark`](../../.claude/skills/kernel-microbenchmark/SKILL.md)
- **模型接入**（`rapid_llm/models/**`，新架构 / 新家族变体）→ [`add-model`](../../.claude/skills/add-model/SKILL.md)
- **量化方案**（`rapid_llm/modules/quantization/**`，新增格式 / 接新 checkpoint 家族）→ [`add-quant-method`](../../.claude/skills/add-quant-method/SKILL.md)
- **写测试**（`tests/**`，新增或移动用例）→ [`write-test`](../../.claude/skills/write-test/SKILL.md)
- **数值偏差排查**（输出不对、与参考对不上、golden 失败）→ [`locate-numeric-divergence`](../../.claude/skills/locate-numeric-divergence/SKILL.md)
- **引擎级基准与对外数字**（`benchmarks/engine/**`、`benchmarks/suites/**`、性能文档结论）→ [`benchmark-and-report`](../../.claude/skills/benchmark-and-report/SKILL.md)

说明：技能是按需加载的，这里列的是硬性入口；各技能的「In-repo examples / 引用文件」指向仓库内可直接对照的完整实现。
