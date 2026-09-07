# RapidLLM

RapidLLM is an LLM inference framework with continuous batching, tensor and data parallelism, quantized weights, and interchangeable GPU kernels. A PyTorch CPU backend supports local development and inference without Triton.

[中文](README.zh.md) · [Documentation](docs/README.md) · [Optimization guide](docs/optimization_features.md) · [CPU support](docs/cpu.md)

## Install

Requires Python 3.13 or newer. Install from the repository root:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e '.[cuda]'
```

For CPU-only Linux, install the CPU PyTorch wheel first, then omit the CUDA extra:

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

On macOS, `uv pip install --python .venv/bin/python -e .` installs without Triton. Select `device="cpu"` explicitly. For GPU execution, use a CUDA-enabled PyTorch build compatible with your NVIDIA driver.

Optional extras: `serve` for the HTTP server, `eval` for evaluation, `bench` for plots, `trace` for OTLP export, and `flashinfer` for that kernel backend. Dependency constraints are in [pyproject.toml](pyproject.toml).

## Generate text

Use a local Hugging Face checkpoint containing `config.json`, tokenizer files, and safetensors weights. Legacy PyTorch `.bin` weights are also accepted. No offline conversion is needed.

```python
from rapid_llm import LLM, SamplingParams

llm = LLM(
    "my_weight/Qwen2.5-0.5B",
    device="cpu",                 # use "cuda" for GPU inference
    max_seq_len=512,
    max_gpu_num_blocks=2048,       # KV token rows, also used on CPU
)
outputs = llm.generate("The capital of France is", SamplingParams(
    temperature=0.0, max_gen_len=32,
))
print(outputs[0].outputs[0].text)
```

The Python sampling limit is `max_gen_len`; HTTP requests use `max_tokens`. CUDA Graph requests on CPU fall back to eager execution. CPU memory, precision, and feature limits are described in [CPU support](docs/cpu.md).

## Continuous batching and serving

Requests can enter and leave between decoding steps. The scheduler supports chunked prefill, prefix reuse, and opt-in recompute preemption.

```python
from rapid_llm import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-0.5B", device="cpu",
    max_seq_len=512, max_gpu_num_blocks=2048, max_num_seqs=4,
)
try:
    outputs = engine.generate(["Hello", "Describe the moon"], SamplingParams(max_gen_len=32))
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

The server provides `/v1/completions`, `/v1/chat/completions`, SSE streaming, `/health`, and `/metrics`. See [serving](docs/online_serving.md) for request fields and deployment limits.

## Models and execution

The model registry includes LLaMA, Qwen2, Qwen3, Qwen3-MoE, LLaVA, Qwen3-VL, and DeepSeek families. Registration does not imply that every checkpoint, quantization format, and device combination has been validated. The tests use small random checkpoints where full weights are unavailable.

| Feature | Behavior |
| --- | --- |
| Tensor parallelism | Shards projections, attention heads, and the vocabulary across ranks |
| Expert parallelism | Assigns whole MoE experts within the TP group and routes tokens with all-to-all |
| Data parallelism | Routes requests between independent replicas; can combine with TP |
| CPU parallelism | Uses Gloo; select `device="cpu"` |
| CUDA Graph | Captures supported decode shapes; requires matching choices across TP ranks |
| Quantization | Weight-only INT8/FP8/INT4 and GPU-specific formats; support varies by backend |
| Communication overlap | Optional CUDA streams for uploads, TP reductions, or EP exchanges |
| Tile-Signaling | Experimental CUDA producer/consumer kernels; not a CPU optimization |
| Kernel Autotune | GPU configuration search and a persistent per-device cache |

CPU TP/DP workers share host resources. Increasing worker count may increase memory use and reduce throughput; measure it on the target machine.

## Performance and correctness

GPU benchmark results are workload-specific. Hardware, batch shape, prompt length, output length, precision, and enabled features must match before comparing results. Enabling more optimization switches does not necessarily improve performance.

- [Optimization explanations, diagrams, and measured results](docs/optimization_features.md)
- [Model benchmarks](docs/benchmark_models.md) and [evaluation](docs/eval_models.md)
- [Quantization](docs/quantization.md) and [recorded quantization matrix](docs/benchmark_logs/quant_matrix_20260901.md)
- [Overlap experiments](docs/release-v0.11.5.md) and [later kernel changes](docs/release-v0.12.0.md)

Release notes and benchmark logs describe the revision and environment measured at the time. They are not the current API reference or a guarantee of speedup on another device.

## Code layout

```text
rapid_llm/
├── engine/          # generation, scheduling, sampling, async front ends
├── executor/        # workers, model loading, KV storage, CUDA Graph
├── models/          # model registry, architecture code, checkpoint mapping
├── modules/         # attention, linear layers, MoE, quantization methods
├── kernels/
│   ├── ops/         # GPU operator implementations and contracts
│   ├── backend/     # CPU and external GPU backends
│   └── dispatcher/  # selection, configuration cache, autotuning
├── distributed/     # process groups and collective operations
├── batch_overlap/   # CUDA stream scheduling and overlap policies
├── entrypoints/     # HTTP protocol and server
└── tools/           # inspection, evaluation, observability
```

## Development

```bash
uv pip install --python .venv/bin/python -e . --group dev
.venv/bin/python -m pytest tests/cpu -q
make test-cpu PYTHON=.venv/bin/python
make lint
```

CPU integration tests generate their own small checkpoints. GPU tests require CUDA, and checkpoint-based tests report missing weights. Use the interpreter from the installed environment when running tests.

## Acknowledgements

The implementation draws on [Transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [LightLLM](https://github.com/ModelTC/lightllm), [Triton](https://github.com/triton-lang/triton), [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), [Kernl](https://github.com/ELS-RD/kernl), [Unsloth](https://github.com/unslothai/unsloth), and [LLaMA](https://github.com/meta-llama/llama-models).

## Citation

```bibtex
@misc{rapidllm-2023,
  author       = {RapidLLM AI team},
  title        = {RapidLLM},
  howpublished = {\url{https://github.com/harleyszhang/rapid_llm}},
  year         = {2023},
}
```
