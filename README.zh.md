# RapidLLM

RapidLLM 是一个大模型推理框架，提供连续批处理、张量并行、数据并行、权重量化及可替换的 GPU 算子。CPU 后端使用 PyTorch，无需安装 Triton 即可进行开发和推理。

[English](README.md) · [文档目录](docs/README.md) · [优化特性与实测](docs/optimization_features.md) · [CPU 支持](docs/cpu.md)

## 安装

需要 Python 3.13 或更新版本。在仓库根目录执行：

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e '.[cuda]'
```

Linux 纯 CPU 环境先安装 CPU 版 PyTorch，再安装框架：

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

macOS 直接执行 `uv pip install --python .venv/bin/python -e .`，运行时指定 `device="cpu"`。GPU 推理需要与 NVIDIA 驱动兼容的 CUDA 版 PyTorch。

可选依赖包括：`serve`（HTTP 服务）、`eval`（评测）、`bench`（绘图）、`trace`（OTLP 导出）和 `flashinfer`。版本约束以 [pyproject.toml](pyproject.toml) 为准。

## 文本生成

模型目录需要包含 Hugging Face 的 `config.json`、tokenizer 文件和 safetensors 权重，也支持旧的 PyTorch `.bin` 权重。无需离线转换。

```python
from rapid_llm import LLM, SamplingParams

llm = LLM(
    "my_weight/Qwen2.5-0.5B",
    device="cpu",                 # GPU 推理改为 "cuda"
    max_seq_len=512,
    max_gpu_num_blocks=2048,       # KV token 行数；CPU 也使用这个参数
)
outputs = llm.generate("法国的首都是", SamplingParams(
    temperature=0.0, max_gen_len=32,
))
print(outputs[0].outputs[0].text)
```

Python API 用 `max_gen_len` 限制生成长度；HTTP API 对应字段为 `max_tokens`。CPU 上请求 CUDA Graph 会回退到普通前向执行。内存预算和设备限制见 [CPU 支持](docs/cpu.md)。

## 连续批处理与服务

请求可以在解码步骤之间加入和退出。调度器支持分块预填充、前缀复用，以及按需开启的重计算抢占。

```python
from rapid_llm import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-0.5B", device="cpu",
    max_seq_len=512, max_gpu_num_blocks=2048, max_num_seqs=4,
)
try:
    outputs = engine.generate(["你好", "介绍一下月球"], SamplingParams(max_gen_len=32))
    for output in outputs:
        print(output.outputs[0].text)
finally:
    engine.shutdown()
```

```bash
uv pip install --python .venv/bin/python -e '.[serve]'
.venv/bin/rapid-llm serve --model-dir my_weight/Qwen2.5-0.5B \
    --device cpu --max-seq-len 512 --max-gpu-num-blocks 2048 --port 8000
```

服务提供 `/v1/completions`、`/v1/chat/completions`、SSE 流式输出、`/health` 和 `/metrics`。参数及限制见 [在线服务](docs/online_serving.md)。

## 功能与设备

模型注册表包含 LLaMA、Qwen2、Qwen3、Qwen3-MoE、LLaVA、Qwen3-VL 和 DeepSeek 系列。注册了模型不代表所有 checkpoint、量化格式和设备组合都经过验证；缺少完整权重时，测试使用随机初始化的小模型。

| 功能 | 行为 |
| --- | --- |
| 张量并行 | 在多个 rank 间切分投影、注意力头和词表 |
| 专家并行 | 在 TP 组内分配完整 MoE 专家，并用 all-to-all 路由 token |
| 数据并行 | 在独立副本之间分配请求，可与 TP 组合 |
| CPU 并行 | 使用 Gloo，指定 `device="cpu"` |
| CUDA Graph | 捕获支持的 decode 形状；TP 各 rank 必须选择相同执行路径 |
| 量化 | 支持 INT8、FP8、INT4 等格式，具体范围取决于后端 |
| 通信重叠 | 使用 CUDA 流重叠上传、TP 归约或 EP 交换 |
| Tile-Signaling | 实验性 CUDA 生产者/消费者算子，不是 CPU 优化 |
| Kernel Autotune | 搜索 GPU 配置，并按设备保存结果 |

CPU 的 TP/DP 进程共享主机资源。增加进程可能增加内存占用、降低吞吐，需要在目标机器上实测。

## 性能与测试

比较性能时，需要固定硬件、batch、输入与输出长度、精度和功能开关。多开优化开关不一定更快。

- [优化原理、可视化与实测结果](docs/optimization_features.md)
- [模型性能](docs/benchmark_models.md)与[精度评测](docs/eval_models.md)
- [量化设计](docs/quantization.md)与[量化实测矩阵](docs/benchmark_logs/quant_matrix_20260901.md)
- [重叠实验记录](docs/release-v0.11.5.md)与[后续算子变更](docs/release-v0.12.0.md)

发布说明和 benchmark 日志描述的是当时的版本及测试环境，不作为当前 API 说明，也不能直接推断其他设备上的加速比。

## 代码目录

```text
rapid_llm/
├── engine/          # 生成、调度、采样、异步入口
├── executor/        # worker、模型加载、KV 存储、CUDA Graph
├── models/          # 模型注册、架构实现、权重映射
├── modules/         # 注意力、线性层、MoE、量化方法
├── kernels/
│   ├── ops/         # GPU 算子及接口
│   ├── backend/     # CPU 和外部 GPU 后端
│   └── dispatcher/  # 选择策略、配置缓存、自动调优
├── distributed/     # 进程组和集合通信
├── batch_overlap/   # CUDA 流调度及重叠策略
├── entrypoints/     # HTTP 协议和服务
└── tools/           # 检查、评测、观测工具
```

## 开发

```bash
uv pip install --python .venv/bin/python -e . --group dev
.venv/bin/python -m pytest tests/cpu -q
make test-cpu PYTHON=.venv/bin/python
make lint
```

CPU 集成测试会自行生成小型 checkpoint，不下载模型。GPU 测试需要 CUDA；依赖外部权重的测试会报告缺失原因。请使用安装依赖时的 Python 环境运行测试。

项目参考了 Transformers、vLLM、SGLang、LightLLM、Triton 等实现，链接与引用格式见 [英文 README](README.md#acknowledgements)。
