# 文档

安装和最小示例见 [README](../README.zh.md)。使用 CPU 时先读 [CPU 支持](cpu.md)，不要照搬 GPU 的内存预算或性能参数。

## 使用与配置

- [CPU 支持](cpu.md)：安装、设备选择、已验证路径和限制。
- [优化特性与实测](optimization_features.md)：各项优化的作用、执行图、开关对照、复现命令和已知负收益。
- [连续批处理](continuous_batching.md)：请求调度、分页 KV、前缀复用和抢占。
- [在线服务](online_serving.md)：HTTP 接口、流式输出和进程模型。
- [张量并行](tensor_parallel.md)、[专家并行](expert_parallel.md)与[数据并行](data_parallel.md)：权重切分、MoE 路由和请求分发。文中的 GPU 实测不适用于 CPU。
- [量化](quantization.md)：存储格式和加载方式；CPU 支持范围以 CPU 文档为准。
- [模型评测](eval_models.md)与[性能测试](benchmark_models.md)：复现命令和测量口径。

## 历史资料

[框架介绍](introduction.md)、`release-*.md`、[ROADMAP](../ROADMAP.md) 和 `benchmark_logs/` 保留了实现演进与历史测量。其中的测试数量、支持范围和速度比对应当时版本，不作为当前能力承诺。源码和回归测试用于核对当前行为。

提交文档时保留复现命令、环境、原始数据路径和限制。性能结论应注明设备、模型、精度、输入输出长度、并发数及启用的优化；不要把单次结果概括成通用加速比。
