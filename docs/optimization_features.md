# Optimization Features and Measurements

This page preserves the optimization explanations, diagrams, benchmark results, and reproduction commands that previously lived in `README.md`. The measurements belong to the releases and hardware named in each section. They are evidence for those configurations, not performance guarantees for the current revision or another GPU.

Current API and device limits are documented in [the documentation index](README.md). Raw benchmark data and release-specific methods remain under [`benchmark_logs/`](benchmark_logs/) and the linked release notes.

## Continuous Batching

The trace contains six requests and three slots. When a request finishes, its slot starts decoding a queued request on the next step.

![continuous batching](images/continuous_batching.gif)

Serve an OpenAI-compatible API:

```bash
pip install 'rapid-llm[serve]'
rapid-llm serve --model-dir my_weight/Qwen2.5-1.5B-Instruct --port 8000
```

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen2.5-1.5B-Instruct",
       "messages": [{"role": "user", "content": "Explain a GPU in one sentence."}],
       "max_tokens": 64}'
```

Or drive the engine directly — every prompt is an independent request, and each carries its own sampling parameters:

```python
from rapid_llm import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-1.5B-Instruct", max_num_seqs=16
)
engine.add_request("Name the capital of Japan.", SamplingParams(max_gen_len=32))
engine.add_request("Write a haiku about rain.", SamplingParams(temperature=0.8, max_gen_len=64))

while engine.has_unfinished_requests():
    for request in engine.step():
        print(f"[{request.request_id}] {request.delta}", end="", flush=True)
```

Asynchronously, with concurrent coroutines sharing one batch:

```python
import asyncio
from rapid_llm import AsyncLLMEngine, SamplingParams

async def main():
    async with AsyncLLMEngine.from_pretrained("my_weight/Qwen2.5-1.5B-Instruct") as engine:
        async def ask(prompt):
            async for chunk in engine.generate(prompt, SamplingParams(max_gen_len=64)):
                print(chunk.delta, end="", flush=True)
        await asyncio.gather(ask("Hello"), ask("Goodbye"))

asyncio.run(main())
```

## Chunked Prefill

Chunked prefill limits the amount of prompt work admitted to one scheduler step, allowing decode requests to run between chunks instead of waiting for a complete long prompt.

![chunked prefill](images/chunked_prefill.gif)

The v0.7 scheduler trace used one 2000-token prompt alongside four decode requests:

| `max_chunk_size` | Prefill steps | Peak prefill tokens per step | Decode requests per step |
| --- | ---: | ---: | ---: |
| disabled | 1 | 2000 | 4 |
| 512 | 4 | 512 | 4 |
| 256 | 8 | 256 | 4 |

At a 512-token chunk, worst-case prefill work per step fell by 3.9×. Reproduce the scheduler trace with `python -m scripts.visualize.gen_chunked_prefill_gif`; configuration and raw data are in [release v0.7.0](release-v0.7.0.md).

## Prefix Caching

Prefix caching hashes 16-token blocks as a chain, so a block key identifies the complete prefix leading to it. Requests with a shared system prompt reuse existing KV blocks and prefill only the unmatched tail.

![prefix caching](images/prefix_cache.gif)

In the v0.7 trace, four requests shared a 768-token prefix and each added a 32-token tail. The cold request prefilled 800 tokens; each later request prefilled 32, a 25× reduction. The cumulative hit rate reached 72%. Reproduce it with `python -m scripts.visualize.gen_prefix_cache_gif`; lifecycle and raw scheduler output are in [release v0.7.0](release-v0.7.0.md).

## Recompute Preemption

With `enable_preemption=True`, the scheduler can admit more requests than available slots. When a waiting request cannot obtain a slot, a decoding request releases its KV state and later recomputes it. A progress quota prevents the same request from being selected again before it produces another token.

![recompute preemption](images/preemption.gif)

The recorded case runs three requests on two slots. Each step advances one decoding request while the evicted request re-enters the queue; the trace verifies progress without starvation. Reproduce it with `python -m scripts.visualize.gen_preemption_gif`; the five-step schedule is in [release v0.7.0](release-v0.7.0.md).

## Tensor Parallelism

Where data parallelism replicates the model, **tensor parallelism** splits model weights across ranks, allowing checkpoints that exceed one card's capacity. Pass `--tensor-parallel-size N` and the engine spawns the extra ranks itself; one GPU stays in one process, so a breakpoint in the engine loop is still a breakpoint in the kernel.

```bash
# 30B MoE on 2x A10
python -m rapid_llm.cli chat \
    --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --tensor-parallel-size 2
```

```python
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

engine = ContinuousBatchingEngine.from_pretrained("my_weight/Qwen3-8B", tensor_parallel_size=2)
```

The engine never learns how many processes run its model: it hands a plan to an `Executor` (`UniProcExecutor` for one GPU, `MultiprocExecutor` for many) and gets sampled tokens back. Because the plan is pure data, driver and follower ranks run one code path rather than two — no mirror process re-deriving the batch from a broadcast prompt, which is what used to turn any disagreement into an NCCL hang. Plans travel on a CPU (gloo) group so the control plane never stages through GPU memory, while the vocabulary-parallel sampler exchanges **two scalars per row** instead of gathering logits, keeping per-step traffic independent of vocabulary size.

Every collective reports its payload to a **collective ledger**, making the control-plane and data-plane byte counts directly measurable:

![tensor parallel](images/tensor_parallel.gif)

```python
from rapid_llm.tools.observability import CollectiveStats

with CollectiveStats.collect() as stats:
    engine.step()
print(stats.report())          # per-op calls and bytes, split data / control plane
```

Recording is windowed, so the default path costs one `if`; windows nest, so a per-step window inside a whole-run window comes out of a single pass. Regenerate the GIF above with `python -m scripts.visualize.gen_collective_gif` — it drives a real `tp=2` engine and every byte in it is a measurement.

See [tensor_parallel.md](tensor_parallel.md) for the design, the sharding rules (including why QKV is split per segment under GQA), and what byte-exact parity between `tp=1` and `tp=2` can and cannot assert under fp16.

## CUDA Graph

Decode repeatedly launches the same operator graph for a bounded set of batch shapes. CUDA Graph records those launches once and replays them with one host submission, reducing Python and driver launch gaps without changing GPU arithmetic.

![eager and CUDA Graph launch timelines](images/cuda_graph_launch.gif)

The recorded H100 decode step contains 327 GPU kernels in both modes. Eager execution took 15.724 ms wall time for 1.038 ms of GPU work; graph replay took 1.228 ms for 1.017 ms of GPU work, about 13× lower wall time in that launch-bound case. The profiler method and launch-count rules are documented in [the optimization design](design/deep_optimization.md). TP graph additionally requires every rank to capture the same collective sequence and shape; unsupported model architectures remain eager.

## Expert Parallelism

Expert parallelism assigns whole MoE experts to ranks in the tensor-parallel group. Router outputs remain global expert ids; two all-to-all exchanges send token rows to the owner and return expert outputs to the source rank. Quantized expert weights and their scale/zero tensors are allocated only for the local expert range.

The CPU/Gloo suite verifies two-rank routing, full MoE forward parity, stacked checkpoint loading, and FP8/INT8 quantized execution. NCCL, CUDA Graph, TBO, and SBO tests require a multi-GPU runner; no GPU speedup is inferred from CPU results. Configuration and validation commands are in [expert parallelism](expert_parallel.md).

## Multi-head Latent Attention (v0.11)

DeepSeek-V2-Lite runs end-to-end (`rapid-llm serve --model-dir my_weight/DeepSeek-V2-Lite --tensor-parallel-size 2`): every token caches one 576-element latent row (512 lora + 64 rope) instead of per-head K and V, through the same `(dim,)` KV row every other model uses. Under TP the latent is **replicated, not sharded** — it is single-KV-head, so splitting it would leave no rank able to compute attention alone; the consequence is that a per-rank pool IS the whole-model pool, and the benchmark reports it under that convention.

On 2× A10 (batch=8, gen=128, eager decode): TTFT 64.8 ms, TPOT 63.01 ms; KV density **33.6k tokens/GiB** of pool memory vs 9.3k for Qwen3-1.7B's GQA on the same card — the 3.6× is the config-parsed 30.4 vs 112.0 KiB/token showing up in a real pool. `python benchmarks/models/bench_mla.py` reports the complete table: two models that are not the same size, labeled as such, with the latency columns run on one identical workload.

Accuracy is checked by a regression gate: `pytest tests/golden/test_deepseek_v2_tp2.py` compares greedy tokens and per-step logprobs against `transformers` on 2× A10, with drift budgets calibrated from a parity probe — the BOS investigation showed a single-layer max-abs threshold can flag a 1-ULP arithmetic tie as a hotspot, so the budget is what the noise floor actually measured, not a round number.

## DeepSeek-V4 (v0.11.5)

V4 has no public weights, so the end-to-end path is verified against a randomly-initialised trimmed checkpoint built from its `config.json`: mHC residual (Sinkhorn mixing), the Compressor + Lightning Indexer pair, SWA/CSA hybrid attention over a 512-dim latent KV, O-LoRA grouped projections and Hash MoE. Eight module-level tests plus a TP2 consistency test carry the numerics; `python benchmarks/models/bench_deepseek_v4.py` measures the speed side against `transformers`.

![DeepSeek-V4 trimmed vs transformers](images/deepseek_v4_speed.png)

Prefill closes to parity and passes it at seq 2048 (1.06×); decode is CPU-bound rather than kernel-bound — the compressor and indexer walk the batch row by row in Python, so a batch-32 step issues ~8.7k kernel launches for ~22 ms of GPU work. The chart includes this limitation. In the v0.11.5 snapshot, fp4 weights were not yet supported; current MXFP4 support is documented in [quantization](quantization.md) and [expert parallelism](expert_parallel.md).

## Cross-Stream Overlap (L1)

A continuous-batching step can hold up to three passes — prefill, extend, decode — and each pass needs its input tensors on the GPU. With L1 overlap (on by default, `RAPID_LLM_OVERLAP=0` to disable) the next pass's upload leaves on a dedicated copy stream while the current forward is still running, so the H2D transfer hides inside the compute instead of serialising behind it. The engine step harvests tokens once at the end rather than synchronising after every pass, which is what makes the overlap structurally possible at all.

![L1 cross-stream overlap](images/overlap_l1.gif)

The GIF is rendered from the engine's own CUDA-event timeline (`RAPID_LLM_OVERLAP_TIMELINE=1`): the extend forward fills the window on the compute stream while the next pass's upload lands inside it on the copy stream — the intersection records the overlap on a shared device clock. Measure both sides with `python -m benchmarks.overlap.levels --level l1 --timeline`; regenerate the picture with `python -m scripts.visualize.gen_overlap_l1_gif`.

## Decode Host-Overhead Cuts (v0.11.1)

Two things every layer did every decode step now happen once per engine: the MoE router's fp32 gate widen (a cast kernel per layer per step for a weight that is frozen after load) and the attention K/V half-view slicing (the paged layout packs K and V in one row; both kernels want halves, so each step cut two views of a buffer that never changes identity). Both are host-side costs, so the win grows with depth and shrinks with batch: Qwen3-30B-A3B eager TPOT **-3.5%** at batch 1, -2.6% at batch 8; graphs +2.0% throughput with byte-identical greedy output.

![router GEMM evolution](images/v0111_router_evolution.png)

The router kept evolving after the release: the cached fp32 widen gave way to vllm's tier-4 path — `torch.mm(x, gate_weight.T, out_dtype=fp32)`, a single bf16 tensor-core GEMM whose epilogue emits fp32 logits directly, dropping both the weight copy and the per-step activation widen. Operator-level (H100, topk parity verified before timing): 2.2× at decode, 5.28× at 2048 tokens, geomean 3.23×; e2e A/B on the same tree: graph TPOT another **-2.6%** / TPS +2.7%.

![e2e A/B TPOT](images/v0111_e2e_tpot_ab.png)

The same release fixed a TP=2 + captured-graph shutdown deadlock: `ncclCommAbort` on a graph-captured communicator parks in a futex, and the old teardown ordered the two ranks' aborts one after another instead of side by side. Teardown now rendezvouses every rank at a gloo barrier, destroys before joining followers, and carries a 15 s deadline whose last resort is abandoning the wedged group to die with the process. The graph × TP × quant cross-validation suite went from a 900 s timeout to 11/11 green.

![TP teardown timeline](images/v0111_teardown_timeline.png)

All three figures are generated from the shipped benchmark logs — `python -m scripts.visualize.gen_v0111_release_figs` re-renders them; numbers and method in [release-v0.11.1.md](release-v0.11.1.md).

## Compute–Communication Overlap (L2 / L3 / L4)

L1 covers the host↔device axis. Three more primitives hide communication and kernel boundaries, each behind its own switch, so a deployment adopts them one at a time and can attribute any change to a single flag.

![overlap axes](images/overlap_axes.png)

**L2 two-batch overlap** (`RAPID_LLM_TBO=1`, off by default) splits a TP decode step into two halves that ping-pong at layer-segment granularity: while half A's `o_proj` all-reduce is on the comm stream, half B's attention GEMMs hold the SMs.

![L2 two-batch overlap timeline](images/overlap_l2.gif)

The GIF is the engine's own CUDA-event timeline — two compute lanes for the halves, one comm lane for the deferred reductions, and the red band is their intersection on a single device clock. The overlap does happen: the benchmark's timeline counts 792 intersecting pairs totalling 65.5 ms, and the GIF shows one such window. Eager TBO still loses on 2× A10 PCIe (+134% TPOT), because an eager TP decode step costs ~27 ms of Python launch time that a graphed reference of the same load cuts to 6.2 ms — the primitive saves GPU time inside a step whose cost is CPU time. Graph-captured TBO is now implemented (the engine wires `enable_cuda_graph(tbo=True)` through `TboPolicy.capture_eligible`, per captured batch): replay is numerically identical to the eager interleave and the launch floor drops from 60 ms to ~10 ms — but on this dense 1.5B TP2 PCIe shape the interleave itself is net-negative (+47-61% TPOT vs a plain graph), because the all-reduce it can hide is ~3-5% of the step while the half-batch efficiency it pays is more. The switch stays off by default, and the full four-arm regression is published next to the evidence (`python -m benchmarks.overlap.levels --level l2 --timeline`).

**L3 chunked all-reduce** (`RAPID_LLM_COMM_OVERLAP=1`, off by default) splits one row-parallel GEMM by rows: chunk k's reduction goes on the wire the moment its GEMM lands, while chunk k+1 computes.

![L3 chunked all-reduce timeline](images/overlap_l3.gif)

Row independence is what makes this legal — the sum a chunk reduces is the sum the unsplit GEMM would have produced for those rows. `RAPID_LLM_L3_MIN_ROWS` (512) plus a 256-row-per-chunk floor keep small GEMMs on the blocking path, where one collective beats two. Prefill is where it earns: TTFT 33.25 → 33.07 ms on a TP2 chunked-prefill load, with 111 real overlaps recorded in the timeline.

**SBO single-batch overlap** (`RAPID_LLM_SBO=1`, off by default) covers the case TBO cannot: an EP decode step with only one batch, so no second half to ping-pong against. It overlaps inside the MoE layer instead — the dispatch exchange goes on the wire first, and the shared MLP moves onto an alternate compute stream, so it computes while the tokens travel. Without it the shared MLP runs *after* dispatch, experts and combine, hiding neither exchange. Verified on two ranks: identical output with the switch on and off, and the two regions intersecting on one device clock.

**L4 tile-signaling** (`rapid_llm/kernels/tile_signal.py`) works inside one device: a persistent Triton producer publishes each output tile with a release-semantics flag write, and a consumer kernel acquires that flag with a bounded spin, so tile k's SiLU·mul epilogue runs while tile k+1's GEMM is still computing.

![L4 tile-signaling timeline](images/overlap_l4.gif)

Nothing here touches the interconnect, so these numbers say nothing about NVLink: +8.0~13.7% on large shapes (4096×4480×1536: 5.85 → 5.05 ms) and a loss on small ones, where the persistent kernel's resident occupancy is pure overhead. Producer and consumer grids together are capped at the SM count, and a host-side watchdog backs the bounded spin up.

L2 and L3 aim at the same all-reduce, so only one can own it: `row_parallel_forward` dispatches passthrough → deferred (TBO) → chunked (L3) → blocking, and combination tests verify the priority and fallback behavior. The eight-cell matrix runs one workload with nothing but the switches moving — it is what caught a TBO numerical regression that every per-feature parity test had passed:

![overlap combination matrix](images/overlap_combination_matrix.png)

Full tables, the nsys kernel-level evidence, and the negative results sit in [release-v0.11.5.md](release-v0.11.5.md); `python -m scripts.visualize.gen_overlap_gifs` regenerates the three timelines above straight from a live engine.

## Kernel Dispatch and Autotune

Each logical operator selects an implementation through `KernelSpec` constraints: library availability, GPU capability, dtype, quantization scheme, shape, layout, and golden status. Rejected candidates retain a reason, so an override or fallback can be inspected instead of silently changing kernels.

![kernel backend selection and fallback](images/backend_registry.gif)

The GIF records the v0.8 backend registry selecting Triton on an A10, honoring an explicit PyTorch override, and falling back when a requested backend is unavailable. The registry has since moved to `rapid_llm/kernels/dispatcher/`; current selection behavior is covered by `tests/ops/test_dispatch.py`. Historical output is preserved in [release v0.8.0](release-v0.8.0.md).

Autotune searches candidate tile shapes offline and stores the best configuration by GPU, operator, shape bucket, and dtype. Startup performs a cache lookup; it does not search on the decode hot path. `RAPID_LLM_AUTOTUNE=0` selects the heuristic fallback. The v0.5 A10 collection covered 12 fused-MoE and attention shapes; the later dequant-fused MoE collection improved all 12 measured keys, with a 5.4% geometric-mean gain and a 37.8% best case. See [release v0.5.0](release-v0.5.0.md) and [release v0.12.0](release-v0.12.0.md).

## TP Collectives and RMSNorm Fusion

For eligible TP=2 small messages, the P2P all-reduce path reduces collective launch overhead and remains graph-capturable. The v0.12 A10 measurements reported 15–25 µs for the NCCL ring path and 5–8 µs for the P2P path; applicability depends on payload, topology, NCCL, and driver versions.

`fused_allreduce_rmsnorm` is a blocking composition API, not an in-collective fusion claim. The fused elementwise stage combines residual addition and RMSNorm after the reduction while preserving the TBO and L3 communication schedulers. On the recorded TP=2 A10 shapes, baseline and fused paths were both about 4.411 ms because communication dominated; no speedup was claimed. Current design, tests, and logs are in [release v0.12.0](release-v0.12.0.md) and `docs/benchmark_logs/kernels/fused_allreduce_rmsnorm_*.json`.

## Quantization

rapid_llm supports multiple weight quantization schemes (architecture aligned with [sglang](https://github.com/sgl-project/sglang)). See [quantization.md](quantization.md) for the full design and API.

|  Scheme  |  CLI Flag  |  Weight  |  Activation  |  Speedup vs HF  |
| -------- | ---------- | -------- | ------------ | --------------- |
| fp8 (checkpoint) | auto-detected | fp8-e4m3 | fp16 | 6.4× |
| int8 (runtime) | `--quantization int8` | int8 | fp16 | 6.3× |
| fp8 W8A8 (runtime) | `--quantization fp8` | fp8-e4m3 | fp8-e4m3 | 3.1× |
| int4 AWQ/GPTQ | auto-detected | int4 | fp16 | — |
| smoothquant | `--quantization smoothquant` | int8 | int8 | — |
| nvfp4 | `--quantization nvfp4` | fp4-e2m1 | bf16 | smallest weights, **slower than bf16** |
| fp8 KV cache | `--kv-cache-dtype fp8` | — | — | 2× KV capacity |

fp8 W8A8 covers MoE experts too: `fused_moe` quantises activations per token, worth
1.18× over fp16 at 512 tokens and **33% slower** at decode width, where the two extra
quantisation launches land on an already launch-bound layer. NVFP4 is weight-only
(sm90 has no fp4 MMA), so it buys memory and costs time. Both, with the measured
numbers behind them, are in [quantization.md](quantization.md); the full
2×H100 matrix — kernel, offline, online, TP/DP/graph/KV — is in
[benchmark_logs/quantization/quant_matrix_20260901.md](benchmark_logs/quantization/quant_matrix_20260901.md).

**FP8 checkpoint** (auto-detected from `config.json`):

```bash
python -m rapid_llm.cli chat --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

**Runtime int8 quantisation** (halve memory of any fp16 model):

```bash
python -m rapid_llm.cli chat --model-dir my_weight/Qwen2.5-0.5B --quantization int8
```

**True W8A8 fp8** (no weight dequantisation, per-token fp8 activations):

```bash
python -m rapid_llm.cli chat --model-dir my_weight/Qwen3-0.6B --quantization fp8
```

**FP8 KV cache** (halve decode memory footprint):

```bash
python -m rapid_llm.cli chat --model-dir my_weight/Qwen3-0.6B --kv-cache-dtype fp8
```

**Vision-language models** (Qwen3-VL / LLaVA, single GPU):

```bash
# Qwen3-VL vision chat
python -m rapid_llm.cli vl-chat \
    --model-dir /data/shared/llm_weights/Qwen3-VL-4B-Instruct \
    --image photo.jpg

# LLaVA with INT8 quantisation
python -m rapid_llm.cli vl-chat \
    --model-dir /data/shared/llm_weights/llava-hf/llava-1.5-7b-hf \
    --image photo.jpg --quantization int8
```

> `vl-chat` is single-GPU: tensor parallelism runs through the continuous-batching
> engine, which hosts text checkpoints only, so `--tensor-parallel-size > 1` returns an explicit error.

### Qwen3-0.6B Benchmark

Environment: (A10, batch=4, greedy)

How to run Benchmarks:

```bash
# Single model
python benchmarks/engine/run.py quant --model-dir /data/shared/llm_weights/Qwen3-0.6B \
    --schemes fp16 int8 fp8 --json docs/benchmark_logs/quantization/quant_Qwen3-0.6B.json

# All representative models (Qwen3-0.6B, 0.6B-FP8, VL-4B, 30B-MoE)
python benchmarks/engine/run.py quant --all
```

Quantization Benchmark Result (A10, Qwen3-0.6B, batch=4, seq_len=25, gen_len=64, greedy):

|  Config  |  Model Mem  |  KV Capacity  |  TPOT (ms)  |  TPS  |  vs HF fp16  |
| -------- | ----------- | ------------- | ----------- | ----- | ------------ |
| HF fp16 (baseline) | 1.17 GB | — | 28.19 | 141.7 | 1.0× |
| lite fp16 | 1.40 GB | 147,875 tok | 4.14 | 918.8 | **6.5×** |
| lite int8 | 0.99 GB | 141,549 tok | 4.16 | 904.1 | **6.4×** |
| lite int8-blockwise | 1.00 GB | 138,385 tok | 4.44 | 849.4 | **6.0×** |
| lite fp8 (W8A8) | 0.99 GB | 139,153 tok | 8.35 | 448.1 | **3.2×** |
| lite smoothquant (W8A8) | 0.99 GB | 135,642 tok | 3.70 | 983.8 | **6.9×** |

> Model Mem = model weights only; KV Capacity = max cached tokens (paged pool fills remaining GPU memory).
> Benchmark logs: [`benchmark_logs/`](benchmark_logs/)

Quantization benchmark visualization (Qwen3-0.6B, A10, all schemes vs HF fp16):

![quantization benchmark](images/quantization_benchmark.gif)

### Qwen3-VL-4B-Instruct Benchmark

A10, batch=4, seq_len=25, gen_len=64, greedy benchmark result:

|  Config  |  Model Mem  |  KV Capacity  |  TPOT (ms)  |  TPS  |
| -------- | ----------- | ------------- | ----------- | ----- |
| lite fp16 | 8.99 GB | 73,676 tok | 23.36 | 170.7 |
| lite int8 | 5.61 GB | 93,559 tok | 27.47 | 145.3 |
| lite int8-blockwise | 5.71 GB | 92,748 tok | 27.97 | 142.7 |
| lite fp8 (W8A8) | 5.61 GB | 93,345 tok | 59.25 | 67.4 |
| lite smoothquant (W8A8) | 5.61 GB | 93,559 tok | 34.00 | 117.5 |

> Vision tower stays fp16 (not quantised); only language model projections are quantised.

![Qwen3-VL-4B quantization benchmark](images/Qwen3-VL-4B-Instruct_quantization_benchmark.gif)

## Data Parallelism

Where tensor parallelism splits one model across GPUs, data parallelism replicates the whole model onto each GPU and routes the request stream between the replicas — for throughput once one card is saturated. Each prompt is dealt to a replica by a load balancer (round-robin or least-loaded), and the replicas decode their own batches concurrently. See [data_parallel.md](data_parallel.md) for the design (it mirrors vLLM's `DPEngineCoreProc` / `DPLBAsyncMPClient` and SGLang's `DataParallelController`) and the benchmarks.

![data parallel](images/data_parallel.gif)

```python
from rapid_llm import DataParallelEngine, SamplingParams

# Two whole-model replicas, one per GPU; requests routed round-robin.
with DataParallelEngine(model="my_weight/Qwen2.5-1.5B-Instruct", data_parallel_size=2) as engine:
    outputs = engine.generate(prompts, SamplingParams(temperature=0.0))
```

For serving, `rapid-llm serve --data-parallel-size 2 --load-balancer total_tokens` swaps in `AsyncDataParallelEngine`, which streams each request's chunks from whichever replica the balancer picks and aborts a request whose connection drops.

On 2× A10 (Qwen2.5-1.5B-Instruct): **weak scaling 2.00x** (100% linear, 1857 → 3716 tok/s) with byte-identical outputs, and **1.64x** on a fixed 256-prompt batch. Compose it with TP — `data_parallel_size=2, tensor_parallel_size=2` — on a 4-GPU box.

Each replica can also replay a captured decode graph. A replica is tp=1, so no collective is captured inside its graph and the replicas never lockstep through a replay — DP scaling and graph replay compose instead of fighting:

![DP x CUDA graph](images/dp_cuda_graph.png)

Qwen3-0.6B, batch 16 per replica, 128 steps: TPOT 25.9 → 5.2 ms per replica (**-80%**) and 618 → 6162 tok/s aggregate (**5.1×**) at DP2, with the +2.4 s capture cost and the per-GPU memory delta recorded in the log. `python benchmarks/parallelism/bench_data_parallel.py --mode graph --model my_weight/Qwen3-0.6B` reproduces it; `tests/engine/test_dp_cuda_graph.py` asserts both replicas hold captured graphs and agree greedily.

## MoE Dequant-Fused Grouped GEMM (v0.12)

MoE expert weights stay quantised (fp8/int8/int4/mxfp4) through the GEMM: the dequantisation runs inside the k-loop (`dequant_fp8e4m3` bit-trick for fp8, shift-and-mask for int4, e2m1 LUT for mxfp4), so there is no intermediate fp16 weight materialisation. On A10, where MoE decode is pure bandwidth-bound, the saved HBM read goes straight to TPOT.

This round also fixed two pre-existing kernel defects that the MXFP4 merge had introduced: int4 experts were misclassified as fp8 (the `_quant_mode` int4 branch was lost), and mxfp4's logical K was 4× too small (pack factor 2 applied where 8 was needed). Both are now correct — int4 and mxfp4 tests pass alongside bf16/fp8/int8.

![MoE dequant-fused grouped GEMM benchmark](images/moe_o4.gif)

The A10 autotune collect round (previously unmeasured — the PRE_HOPPER table was a guess) improved all 12 shape keys, best +37.8% (int4 M64), geomean +5.4%. Evidence in [release-v0.12.0.md](release-v0.12.0.md).

## Split-KV Adaptive Decode Attention (v0.12)

batch=1 decode has a structural weakness: without split-kv, one sequence's attention is pinned to a handful of SMs while the rest sit idle. At 8K context the KV read is ~750 MB/step — comparable to the MoE weight read — and the TPOT spike is real.

The fix splits the KV dimension into partial softmax lanes that recombine via the numerically invariant online-softmax. The partition count is looked up from an autotune table keyed on `(batch, seq_len)`: large batches already have enough row-parallel work, so split=1 avoids the combine cost; batch=1 with long context opens the split wide.

![Split-KV adaptive decode attention](images/splitkv_o8.gif)

A10 result: geomean 1.07×, best case (batch=1, seq=512) 1.80× (27.65 µs → 15.36 µs). The GIF shows the SM occupancy grid — fixed 128 partitions vs adaptive 32 partitions — and the per-shape speedup sweep. Evidence in [release-v0.12.0.md](release-v0.12.0.md).

## Paged KV + Radix Zero-Copy Sharing (v0.12)

KV memory is paged in 16-token blocks — the allocation unit is a block, not a contiguous token range. Three layers make it work:

1. **BlockPool** (`rapid_llm/engine/block_pool.py`): block-granularity allocation with reference counting, LRU eviction (doubly-linked FreeBlockQueue), and hash-indexed block lookup (`cached_block_hash_to_block`).
2. **PrefixCache** (`rapid_llm/engine/prefix_cache.py`): hash-chained prefix reuse — blake2b chain hash (`iter_block_hashes`) over token blocks, with allocate/commit/free/lookup lifecycle. Blocks with zero references are auto-reclaimed.
3. **SlotBatch** (`rapid_llm/executor/slot_batch.py`): block table indirection — `b_req_tokens_table[slot, pos] = block_id * 16 + offset` maps logical token positions to physical block addresses.

The consequence: requests sharing a system prompt prefill it once; later requests zero-copy reuse the cached blocks. 80 unit tests cover block allocation, prefix hit, and block-table address translation end-to-end.

## fp8 KV Precision Gate (v0.12)

fp8 KV cache (e4m3 + uint8 container + per-tensor scale) halves KV memory — but the quantisation error must be bounded. `tests/kernels/test_fp8_kv_accuracy.py` is the gate: 8 test scenarios covering decode/prefill shapes, heavy-tailed distributions, near-zero values, independent K/V quantisation, and scale sensitivity. The gate requires rel_err < 5% (normal) / < 10% (heavy-tailed) and cosine_sim > 0.999 — all 8 pass.

End-to-end benchmark (`python benchmarks/engine/run.py quant --kv-cache-dtype auto fp8_e4m3`): fp16 0.784s vs fp8 0.812s — **3.6% throughput cost for 2× KV capacity**. Evidence in `docs/benchmark_logs/quantization/fp8_kv_o14_*.json`.

## Ngram Speculative Decoding (v0.12)

Decode normally produces one token per step, leaving the GPU underutilised. For repetitive workloads (code, templates), much of the output repeats patterns the model has already seen. Speculative decoding exploits this: an n-gram proposer scans the prompt + generated text for repeated patterns and drafts the next few tokens; a single EXTEND forward verifies them all at once.

The proposer (`rapid_llm/engine/ngram_proposer.py`) searches from the longest n-gram down to bigram, returning the continuation that followed the latest match. Verification is greedy: `argmax(logits[j]) == draft[j+1]` — accepted tokens plus one bonus from the model's own distribution are appended per step.

Enabled via `LITE_LLAMA_SPECULATE=1` (off by default). The worker gains an `execute_verify` method that returns full `[tokens, vocab]` logits alongside sampled tokens; `ModelInput.return_logits` controls the path.

Repetitive workload (Qwen3-0.6B, batch=4, gen=32): steps 32→11 (**-65.6%**), wall time 0.153s→0.083s (**1.85× speedup**). Evidence in `docs/benchmark_logs/kernels/speculative_o5_*.json`.

## Observability

### Token Scores (`logprobs` / `prompt_logprobs`)

A sampled token on its own tells you what the model said, not how close the call was. `logprobs=k` returns the drawn token's log-probability together with the `k` most likely alternatives it outranked; `prompt_logprobs=k` does the same for every position of the prompt, which is what perplexity scoring and prompt debugging need. Both come out of the forward pass the request was already paying for — there is no second scoring pass — and both are off by default.

![logprobs and prompt_logprobs](images/logprobs.gif)

The GIF is a real Qwen3-0.6B run (`python -m scripts.visualize.gen_logprobs_gif`): position 1 of the prompt shows `' capital'` at -12.8, and the last generated token is a near tie — `' Italy'` at -1.74 beat `' France'` at -1.86, exactly the case a mean-logprob filter is there to catch.

```python
from rapid_llm import LLM, SamplingParams

llm = LLM(model="my_weight/Qwen3-0.6B")
output = llm.generate(["The capital of France is"], SamplingParams(logprobs=5, prompt_logprobs=5))[0]

for record in output.outputs[0].logprobs:      # one per generated token
    print(record.token_id, record.logprob, record.top_token_ids, record.top_logprobs)
print(output.prompt_logprobs[0])               # None: nothing predicts position 0
```

Over the server, `/v1/completions` takes `logprobs` / `prompt_logprobs` directly, and `/v1/chat/completions` follows the OpenAI shape (`logprobs: true` plus `top_logprobs: 5`):

```bash
curl localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-0.6B", "prompt": "The capital of France is",
  "max_tokens": 8, "logprobs": 5, "prompt_logprobs": 5}'
```

Cost, measured with `python benchmarks/serving/bench_observability.py` (A10, Qwen3-0.6B, batch=16, gen=128): `logprobs=5` moves TPOT 4.75 → 5.35 ms (throughput -10.4%) because each step adds a `log_softmax` + `topk` + a device-to-host copy; `prompt_logprobs=5` moves TTFT 23.3 → 32.0 ms and costs -1.5% throughput, since it only touches prefill. Log: [`benchmark_logs/kernels/observability_v0.10.json`](benchmark_logs/kernels/observability_v0.10.json).

### Metrics and Tracing

`rapid-llm serve` exposes a Prometheus endpoint — request counters, in-flight gauges, and the queue-time / TTFT / TPOT histograms on vLLM's bucket grid. The text format is a few lines per metric, so there is no `prometheus_client` dependency to install:

```bash
rapid-llm serve --model-dir my_weight/Qwen3-0.6B &
curl -s localhost:8000/metrics | grep -A2 time_to_first_token
```

```text
# HELP rapid_llm:time_to_first_token_seconds Arrival to first generated token.
# TYPE rapid_llm:time_to_first_token_seconds histogram
rapid_llm:time_to_first_token_seconds_bucket{le="0.5"} 0
rapid_llm:time_to_first_token_seconds_bucket{le="1"} 3
rapid_llm:time_to_first_token_seconds_sum 1.8414203859865665
rapid_llm:time_to_first_token_seconds_count 3
```

Collection is opt-out (`RAPID_LLM_METRICS=0`) and tracing is opt-in: set `RAPID_LLM_OTLP_ENDPOINT=http://localhost:4318` and each request becomes one span carrying its id, prompt and output token counts, and finish reason. Without the endpoint the tracer is a no-op object — nothing is imported, nothing is timed, and the OpenTelemetry SDK stays an optional install. Both together stay inside the 0.5% run-to-run noise of the same benchmark above, because the work is a handful of float additions per request rather than per token.

### Single-Layer Harness

The single-layer harness isolates numerical and dispatch defects before a whole-network run. The harness builds exactly one layer, on one GPU, and can mirror the matching `transformers` layer's weights to compare numerically — no checkpoint download needed, which is the point when the model is 671B and the change is one attention variant:

```bash
# timing + which kernel each op dispatched to, random weights
python -m scripts.verify.layer_harness --model-dir my_weight/Qwen3-0.6B --layer 0

# numerical parity against transformers' own layer, as a gate
python -m scripts.verify.layer_harness --model-dir my_weight/Qwen3-0.6B \
    --layer 3 --weights mirror --tolerance 2e-2

# real weights under a decode-shaped load
python -m scripts.verify.layer_harness --model-dir my_weight/Qwen3-0.6B \
    --layer 3 --weights checkpoint --batch 4 --seq-len 512 --decode-steps 32
```

`--tolerance` turns the comparison into a gate (non-zero exit above it), so the harness works as a pre-flight check in CI as well as by hand.

## Structured Streaming Output

Reasoning and tool-call parsing are declared **per request**, not per deployment. vLLM and SGLang pick one reasoning parser at server start (`--reasoning-parser`), which means one deployment serves one output style; here `reasoning_parser` and `tool_parser` are fields of `ChatCompletionRequest` (validated at the schema layer), so the same server streams R1-style and direct models side by side.

![streaming reasoning parser](images/reasoning.gif)

The GIF is a real Qwen3-1.7B run (`python -m scripts.visualize.gen_reasoning_gif`): the prompt opens the think tag itself, so the splitter is born inside a thinking section (`starts_inside=True`) and has to catch the closing tag mid-stream — every delta lands in its channel and the tags never leak through.

```bash
curl localhost:8000/v1/chat/completions -d '{
  "model": "m", "stream": true,
  "messages": [{"role": "user", "content": "Tokyo weather?"}],
  "reasoning_parser": "deepseek_r1",
  "tool_parser": "deepseek"
}'
```

Two switches, independently composable: `reasoning_parser` routes `<think>…</think>` into `delta.reasoning_content`; `tool_parser` (DeepSeek or Qwen marker families) streams `delta.tool_calls` by call index and flips `finish_reason` to `"tool_calls"`. The parser contract enforces three properties:

- **Streamed output equals one-shot output.** The parsers hold any delta that might complete a tag until it cannot, so an arbitrary chunking of a reply concatenates to what a one-shot parse of the same text says — the parser tests enumerate every two-cut split, and a server-level test asserts the streamed frames merge to the one-shot message on the same request.
- **`finish_reason` is its own frame.** The parser's flush (a tool call cut mid-JSON, a held partial tag) must reach the client before it stops reading, so the terminal frame is an empty delta carrying only the reason — the OpenAI shape.
- **Truncation remains distinguishable.** A call truncated by `max_tokens` reports the fragments it did get, but `finish_reason` stays `"length"` rather than claiming `"tool_calls"`.

Cost, measured with `python benchmarks/serving/bench_parser.py`: reasoning + tool parsing adds ~1.17 µs/token — 0.002–0.005% of decode TPOT, below run-to-run noise.
