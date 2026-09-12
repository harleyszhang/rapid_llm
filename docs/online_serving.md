# 在线推理服务（online batch inference）

`rapid-llm serve` 起一个 OpenAI 兼容的 HTTP 服务，底层是 [连续批处理引擎](./continuous_batching.md)：并发到达的请求被自动合并进同一个 batch，而不是排队串行。

## 快速开始

```bash
pip install -e '.[serve,cuda]'         # 在仓库根目录安装 GPU 服务依赖
rapid-llm serve --model-dir my_weight/Qwen2.5-1.5B-Instruct --port 8000
```

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen2.5-1.5B-Instruct",
       "messages": [{"role": "user", "content": "用一句话解释 GPU 是什么"}],
       "max_tokens": 64}'
```

官方 `openai` 客户端可以直接指过来：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

for chunk in client.chat.completions.create(
    model="Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "写一首关于海的短诗"}],
    max_tokens=128,
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## 端点

CPU 服务使用 `pip install -e '.[serve]'`，启动时加 `--device cpu --max-gpu-num-blocks 2048`。内存预算与能力限制见 [CPU 支持](cpu.md)。

| 端点 | 说明 |
| --- | --- |
| `GET /health` | 存活探针 |
| `GET /v1/models` | 返回 `--served-model-name`（默认取权重目录名） |
| `POST /v1/completions` | 文本补全，prompt 原样送入，不套 chat 模板 |
| `POST /v1/chat/completions` | 对话补全，多轮消息经 tokenizer 的 chat 模板渲染 |

`stream: true` 时返回 `text/event-stream`，逐帧 `data: {...}`，以 `data: [DONE]` 收尾； chat 流的第一帧只带 `role`（与 OpenAI 一致），最后一帧带 `finish_reason`。

支持的采样字段：`max_tokens`、`temperature`、`top_p`、`repetition_penalty`。

**默认值刻意对齐 OpenAI 而不是 rapid_llm 的 CLI**：`temperature` 与 `top_p` 都是 `1.0`、`repetition_penalty` 是 `1.0`。CLI 的 `0.6 / 0.9 / 1.1` 是给交互式聊天调的手感，不该悄悄改变一个按 OpenAI 语义写好的客户端的行为。

**不支持的字段显式报错而不是静默忽略。** `n > 1` 返回 422：客户端要 4 条补全却拿到 1 条，是没有办法自己发现的。

## 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model-dir` | 环境变量 `RAPID_LLM_MODEL_DIR` | 权重目录 |
| `--host` / `--port` | `0.0.0.0` / `8000` | 监听地址 |
| `--served-model-name` | 目录名 | `/v1/models` 里报的名字 |
| `--max-seq-len` | `2048` | 每条请求的上下文窗口；KV 从分页池分配 |
| `--max-num-seqs` | `32` | 同时 decode 的请求数上限（DP 下是**每副本**的上限） |
| `--max-num-batched-tokens` | `8192` | 一次 prefill 分组的 padded token 预算 |
| `--max-gpu-num-blocks` | GPU 自动估算 / CPU 4096 | 指定 KV cache token 行数 |
| `--device` | `cuda` | 执行设备；CPU 使用 `cpu` |
| `--no-cuda-graph` | 关 | 用 eager decode 而不是 replay graph |
| `--no-chat-template` | 关 | base 模型用：消息原样拼接，不套模板 |
| `--tensor-parallel-size` | `1` | 一份权重的 TP 切分数（切权重，装得下大模型） |
| `--data-parallel-size` | `1` | 整模型副本数（每副本一卡起，买吞吐；与 TP 组成 dp×tp 网格） |
| `--load-balancer` | `round_robin` | 请求怎么路由到副本：`round_robin` / `total_requests` / `total_tokens` |
| `--engine-backend` | `thread` | 引擎后端：`thread` 为工作线程内运行（见[线程模型](#线程模型)），`process` 为独立 EngineCore 子进程（ZMQ IPC，需 pyzmq / msgpack，仅服务单副本） |

`--max-num-seqs` 既是显存旋钮也是延迟旋钮：超过某个宽度之后，每 token 的成本不再下降，而单请求延迟还在涨。

## 线程模型

引擎的 `step()` 是阻塞的同步调用，直接在事件循环里跑会让一步计算卡住所有连接。所以 [`AsyncLLMEngine`](../rapid_llm/engine/async_engine.py) 把引擎放在**独立工作线程**上：

```text
    协程 A ──┐                            ┌──> asyncio.Queue A ──> 协程 A
    协程 B ──┼─> SimpleQueue(命令) ──> 工作线程 ─┼──> asyncio.Queue B ──> 协程 B
    协程 C ──┘      (add / abort)      step() 循环  └──> asyncio.Queue C ──> 协程 C
```

- 工作线程**独占**引擎，是唯一碰调度器和 GPU 的执行体——所以两者都不需要加锁；
- 协程从不直接调引擎，只投命令、等增量；
- 回传用 `loop.call_soon_threadsafe`，因为 `asyncio.Queue` 不是线程安全的；
- 空闲时工作线程阻塞在命令队列上，没有流量就不烧 CPU。

每个请求流记住的是**创建它的那个协程所在的事件循环**，不是引擎启动时选定的某一个。这点是被一个真实的死锁逼出来的：早先版本在 `start()` 时绑定一个循环，于是 ASGI 测试客户端（自己在另一个线程里跑一个循环）永远收不到任何数据——不是报错，是挂住。回归测试：`tests/engine/test_async_engine.py::test_the_engine_serves_a_second_event_loop`。

客户端断开时，`generate()` 的 `finally` 会投一条 abort，被放弃的请求**下一步就让出槽位**，而不是继续跑到长度上限。

### 多卡：`--data-parallel-size` / `--tensor-parallel-size`

`--tensor-parallel-size > 1` 时仍是上面这个形状（TP 的 follower 由引擎内部拉起，对服务层透明）。`--data-parallel-size > 1` 时换成 [`AsyncDataParallelEngine`](../rapid_llm/engine/async_data_parallel.py)：每个副本一个进程一条常驻引擎，负载均衡器逐请求选副本，一条**泵线程**把共享结果队列里的消息按 request_id 投回各协程的事件循环——协程仍然只看到同样的 `generate` / `generate_text` 接口，OpenAI 层和所有端点行为不变。

```bash
# 两份整模型副本，按在飞 token 数路由
rapid-llm serve --model-dir my_weight/Qwen2.5-1.5B-Instruct \
    --data-parallel-size 2 --load-balancer total_tokens
```

进程布局与历史实测见[数据并行](./data_parallel.md)。吞吐随工作负载和硬件变化，不保证线性扩展。

### 进程后端：`--engine-backend process`

引擎也可以整体搬进独立子进程（vLLM v1 式的切分）：父进程保留 tokenizer 与流式 detokenizer，子进程独占调度器与执行器，两侧只交换 token id——命令通道 ROUTER ↔ DEALER、输出通道 PUSH ↔ PULL，msgpack 帧加协议版本握手。[`EngineCoreClient`](../rapid_llm/engine/engine_core_client.py) 对外镜像 `AsyncLLMEngine` 的接口，OpenAI 层与所有端点行为不变；子进程崩溃会以 `EngineDeadError` 传播到所有打开的流。该后端只服务单副本（`--data-parallel-size > 1` 会被拒绝）。

## 不经 HTTP 直接用

服务层很薄，引擎自己就能用。同步、step 驱动：

```python
from rapid_llm import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-1.5B-Instruct", max_num_seqs=16
)

a = engine.add_request("解释一下 KV cache", SamplingParams(max_gen_len=128))
b = engine.add_request("写一首俳句", SamplingParams(temperature=0.0, max_gen_len=32))

while engine.has_unfinished_requests():
    for request in engine.step():
        print(f"[{request.request_id}] {request.delta}", end="")
```

异步、流式：

```python
import asyncio
from rapid_llm import AsyncLLMEngine, SamplingParams

async def main():
    async with AsyncLLMEngine.from_pretrained("my_weight/Qwen2.5-1.5B-Instruct") as engine:
        async def ask(prompt):
            async for chunk in engine.generate(prompt, SamplingParams(max_gen_len=64)):
                print(chunk.delta, end="", flush=True)
        await asyncio.gather(ask("你好"), ask("再见"))   # 两个请求共享同一个 batch

asyncio.run(main())
```

离线跑一批 prompt，走的仍是连续批处理调度（先结束的请求立刻让位给排队的）：

```bash
rapid-llm batch --model-dir my_weight/Qwen2.5-1.5B-Instruct \
    --prompts-file prompts.txt --max-num-seqs 16 --show-stats
```

## 性能

单卡 A10、Qwen2.5-1.5B-Instruct、16 个请求每 250 ms 到达一个：吞吐从 93 tok/s 提到 644 tok/s（**×6.9**），平均端到端延迟从 19.1 s 降到 2.3 s（**×8.3**）。完整口径、其他场景与"什么时候没有收益"见 [连续批处理](./continuous_batching.md#实测数据)。

## 当前边界

- **仅文本模型**：多模态 checkpoint 请用 `LLM.generate()`（视觉路径），`serve` 的 DP/TP 都不覆盖它。
- **无鉴权、无限流**：面向内网与本地开发，不要直接暴露到公网。
