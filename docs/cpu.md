# CPU 支持

CPU 后端使用 PyTorch 执行模型计算，不需要 Triton。它用于本地推理、开发和回归测试；不是 GPU kernel 的模拟器，也不保证量化或多进程比普通单进程更快。

## 安装与运行

```bash
uv venv --python 3.13
# Linux：先安装不包含 CUDA 运行库的 PyTorch。macOS 可以省略这一步。
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

明确传入 `device="cpu"`。即使机器有 GPU，CPU 请求也使用 CPU 算子和 Gloo 通信，不依据 `torch.cuda.is_available()` 切换设备。

```python
from rapid_llm import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-0.5B",
    device="cpu",
    max_seq_len=512,
    max_num_seqs=4,
    max_gpu_num_blocks=2048,
)
try:
    results = engine.generate(["Hello"], SamplingParams(temperature=0, max_gen_len=32))
    print(results[0].outputs[0].text)
finally:
    engine.shutdown()
```

`max_gpu_num_blocks` 是保留的参数名，单位是 KV token 行，CPU 也使用它。CPU 默认分配 4096 行，不会探测可用主存。加载大模型前应设置明确预算：普通注意力的 KV 字节数约为 `行数 × 层数 × 2 × 本地 KV 头数 × head_dim × 每元素字节数`，另需权重、激活和临时缓冲区。MLA 的缓存布局不同。

## 执行原理

CPU 和 GPU 使用同一套模型、调度器、分页 KV 缓存和采样逻辑。模型先在 `meta` 设备搭建结构，再把 checkpoint 权重直接装入目标设备；运行时根据输入 tensor 的设备选择算子后端。因此 `device="cpu"` 不会导入或模拟 Triton kernel，而是调用等价的 PyTorch 实现。

一次推理仍分为 prefill 和 decode：prefill 批量写入每层 KV cache，decode 每步只追加新 token，并通过页表读取历史 KV。CPU 注意力按请求处理不同长度的序列，归一化、softmax、MoE 和量化反算通常使用 FP32 累加，再转回模型 dtype。低比特权重能减少存储，但当前 CPU 路径会先反量化或构造临时张量，不等价于原生低比特 GEMM。

并行语义也保持一致。TP、EP 和 DP 使用 Gloo；collective 在 CPU 上同步完成。CUDA Graph、CUDA stream overlap、Tile-Signaling 和 kernel autotune 没有 CPU 对应收益，框架会走 eager 或同步路径。CPU 后端的目标是让功能、接口和数值回归可用，不是预测 GPU kernel 的性能。

## 功能边界

| 路径 | CPU 行为 |
| --- | --- |
| RMSNorm、RoPE、KV 写入、GQA/MHA、MLA、MoE | PyTorch 实现；注意力按请求处理，支持分页解码 |
| 离线生成、连续批处理 | 使用同一套调度和采样接口 |
| 分块预填充、前缀缓存、重计算抢占 | 保留调度语义；缓存容量和功能组合限制仍然有效 |
| TP / DP | Gloo 多进程；通过 `device="cpu"` 选择 |
| CUDA Graph | 跳过捕获，使用 eager 执行 |
| TBO、CUDA 通信重叠、共享专家流重叠 | CPU 同步路径，不创建 CUDA stream/event |
| Tile-Signaling、GPU kernel autotune | 仅 GPU；CPU 不使用这些执行路径 |
| INT8、INT4、FP8、SmoothQuant | 已有线性层回退；反量化后计算，不是 CPU 原生低比特 GEMM |
| MXFP4 专家 | 解包后执行 PyTorch MoE；大型专家可能需要较多临时内存 |
| NVFP4 | 暂无 CPU 执行路径；运行时量化请改用 INT4 |

HTTP 服务需要 `serve` extra，启动时加 `--device cpu`。CPU 不自动转换为 MPS。外部 CUDA 后端以及直接导入的 Triton 实现模块仍要求对应 GPU 依赖。

## 验证与性能

```bash
uv pip install --python .venv/bin/python -e . --group dev
.venv/bin/python -m pytest tests/cpu tests/models/test_deepseek_v4.py -q
.venv/bin/python -m pytest -m 'not gpu and not weights'
```

测试生成本地随机小模型，不下载权重。LLaMA、Qwen2、Qwen3、Qwen3-MoE、DeepSeek-V2 的 prefill/decode logits 与同权重 Transformers 模型比较；DeepSeek-V4 另有模块与生成测试。这不替代完整 checkpoint、长上下文或全部量化组合的验证。

CPU 量化计算会产生反量化临时张量。先测普通权重，再决定是否量化；不要以权重体积下降推断延迟下降。多进程 TP/DP 共享主存带宽，每个进程还可能启动多条 PyTorch 线程，需一起调整线程数和并发数。CPU 上的结果不能用于判断 CUDA 路径是否加速。

模型前向微基准（默认生成随机小 LLaMA，不下载权重）：

```bash
.venv/bin/python benchmarks/engine/run.py cpu --threads 1 --repeats 30
.venv/bin/python benchmarks/engine/run.py cpu --model-dir my_weight/Qwen2.5-0.5B --prompt-length 128
```

该脚本测量 batch=1、固定上下文长度的 prefill/decode 中位延迟，不包含加载、调度、采样和分词。它不是服务端 TTFT 或端到端吞吐测试。
