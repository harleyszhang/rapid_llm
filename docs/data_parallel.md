# 数据并行（data parallelism）

## 副本与请求路由

数据并行（DP）把请求分给独立模型副本，副本之间不交换前向张量。张量并行（TP）在副本内部切分权重，需要集合通信。两者可以组合，也都支持显式选择 CPU；CPU 副本共享主存带宽，不保证增加副本就提高吞吐。

```python
from rapid_llm import DataParallelController, SamplingParams

if __name__ == "__main__":
    with DataParallelController(
        "my_weight/Qwen2.5-0.5B", device="cpu", data_parallel_size=2,
        max_seq_len=512, max_num_seqs=2, max_gpu_num_blocks=2048,
    ) as engine:
        outputs = engine.generate(["Hello", "Good morning"], SamplingParams(max_gen_len=16))
```

多进程示例应保存为脚本执行，并使用 `__main__` 保护，避免 spawn 子进程重复创建引擎。

因此两者是正交的、可组合的：`dp_size` 份副本，每份 `tp_size` 张卡，构成一个 `dp_size × tp_size` 的 rank 网格（见 `rapid_llm/distributed/parallel_state.py`）。 TP 用延迟换"装得下"，DP 用显存（每卡一份权重）换吞吐。

![data parallel](./images/data_parallel.gif)

上图是 8 个请求经 round-robin 派发到 2 个副本：`GPU0` 拿到偶数号请求、`GPU1` 拿到奇数号，两个副本**同时**解码各自的 batch。两条泳道并排推进——这份并发就是全部的加速来源。

## 分层结构

副本执行、路由策略和进程协调分开实现：

| 角色 | 本仓库实现 | 职责 |
| --- | --- | --- |
| 副本 leader process | `_run_replica_process` + `Scheduler.run_event_loop()` | 读取 typed command、执行连续批处理并发布 typed event |
| TP follower process | `LLM` + `serve_plans()` | 只执行 leader 广播的 `StepPlan`，不读取请求队列 |
| 负载均衡策略 | `RoundRobinPolicy` / `RequestCountPolicy` / `TokenCountPolicy` / `CacheAwarePolicy` | 按请求选择 replica，并按 request id 精确回收负载 |
| 协调器 | `DataParallelController` | 拉起 DP × TP 网格、路由命令、汇聚事件并回收资源 |

**一套 typed 协议，一种执行模型。** 同步 `generate()` 与在线服务都发送 `AddRequest` / `AbortRequest`，副本统一返回 `SchedulerEvents`；启动、失败和停止也分别使用 typed event/command。同步接口只是 controller 上的阻塞 facade，在线接口由 `AsyncLLMEngine` 的 request manager 消费同一事件流，不再有另一套 batch 消息或 DP 专属异步引擎。

- **rank process**（`_run_replica_process`）是网格里的**一个 cell**，不是一个副本：spawn 子进程按 `global_rank` 绑卡，再按副本内位置分化。
  - **leader**（`tp_rank == 0`）构造常驻 `Scheduler` 并直接运行 `run_event_loop()`。请求可以逐条加入，完成的序列下一步释放 slot；空闲时阻塞等待命令，不空转 CPU。
  - **follower**（`tp_rank > 0`）构造 `LLM` 并运行 `serve_plans()`，每次前向由 leader 的 executor 通过控制面广播（见[张量并行](./tensor_parallel.md)）。
- 所有 rank 用 `ReplicaReady` / `ReplicaFailed` 报告启动结果；leader 运行期再发布 `SchedulerEvents` / `SchedulerFailed`，因此 controller 能传播错误而不会永久等待。
- **协调器**（`DataParallelController`）不加载模型权重、KV cache 或 sampler；它只拥有进程、typed channel、路由映射与策略状态。

## 路由策略

`data_parallel.py` 中的策略不持有队列、进程或张量。四种策略如下：

- **`round_robin`**（默认）：0,1,0,1… 轮流发，不管每个请求跑多久。离线批处理的正确默认—— 所有 prompt 一起到，没有哪条特别长。用条带式（0,2,4,… 给副本 0）而不是切连续区间，是为了把长短请求均匀打散，避免一个已排好序的列表把所有长 prompt 都堆给同一个副本。
- **`total_requests`**：发给当前在飞**请求数**最少的副本。所有副本空闲时它退化成轮询（低下标优先），所以能安全地做默认的替身。
- **`total_tokens`**：发给当前在飞**token 数**最少的副本。prefill 的开销与 prompt 长度成正比，长度悬殊时"请求数"是错的计量单位——两条 4k prompt 不等于两条 40 token 的负载。
- **`cache_aware`**：以未命中的 prompt token 数和当前在飞负载共同计分，让共享前缀优先回到已缓存对应 block 的副本。

所有策略使用确定性的低下标 tie-break；计数型策略在冷启动时输出 0,1,2… 序列。

request manager 已经产生 prompt token ids，controller 直接复用这些 ids 做 token 数与 prefix hash，不重复 tokenize。负载 charge 以 request id 记录，完成、拒绝或 abort 时释放准确请求，因此乱序完成不会错误扣除另一请求的负载。

## 与张量并行的关系

CUDA 的 TP 数据面使用 NCCL，CPU 使用 Gloo；控制面使用 Gloo。下文 NCCL 的描述针对 GPU 部署。

网格坐标是纯函数 `grid_coordinates(global_rank, tp_size, dp_size) → (dp_rank, tp_rank)`，按 `global_rank = dp_rank * tp_size + tp_rank` 布局，让一个副本的 TP ranks 连续。 `init_parallel` 只有在 `tp_size > 1` 时才真正 rendezvous 建 NCCL 进程组——纯 DP 的副本之间不共享任何张量，没什么好同步的，NCCL 完全不碰。这也是为什么纯 DP 用普通的 `multiprocessing` 队列而不是 NCCL：worker 从不读另一个 worker 的张量。

**进程数按 cell 算，队列数按副本算。** `tp_size > 1` 时 `init_parallel` rendezvous 的是 `dp_size × tp_size` 个 rank 的世界，所以只 spawn `dp_size` 个进程会**永久挂死**在等待从未启动的 rank 上——一个协调器的任何超时都解释不了的失败。所以协调器为每个 cell 起一个进程，但**请求队列只有 `dp_size` 个**：一条请求发给一个**副本**，而不是发给副本的每个 rank。副本内的 follower 不参与路由，它们跑什么由 leader 的控制面决定（每 step 一次 `StepPlan` 广播，采样出的 token 再从 tp rank 0 广播回去）；只有 leader 回结果，因为协调器每个副本只等一条应答。

这么分层的收益是**路由与同步互不知情**：换 balancer 不碰 TP 的一行代码，改 TP 的广播格式也不碰路由。早期实现里 tp ranks 是**镜像**——每个 rank 一个队列、各自收同一条请求消息——那样 "一条请求"就同时是路由单位和同步单位，两边任何一处不一致都会让某个 rank 少跑一次前向而卡死在集合通信里。

这条网格约束在 CPU 上就能断言，不需要四张卡：测试用假的进程/队列检查 4 个 cell 的 `(global_rank, dp_rank, tp_rank)`，并确认同一副本的两个 rank 拿到同一个 command queue。

## Data Parallelism Attention（DPA）

DPA 改变 DP 轴的含义：每个 DP rank 仍独占请求、Attention 和 KV cache，但 MoE experts 在完整的 `DP × TP` 网格上切分。一次 MoE forward 会先汇聚各 replica 的 token，运行全局专家分片，再把结果规约回请求所属的 replica。这样 KV 容量随 DP rank 数增长，Attention 不为别的 replica 读取 KV，同时专家权重不必在每个 replica 上完整复制。

```python
from rapid_llm import DataParallelController, SamplingParams

with DataParallelController(
    "/path/to/moe-checkpoint",
    data_parallel_size=2,
    tensor_parallel_size=1,
    enable_expert_parallel=True,
    enable_dp_attention=True,
    use_cuda_graph=False,
) as engine:
    outputs = engine.generate(
        ["The capital of France is", "Explain MoE routing"],
        SamplingParams(temperature=0.0, max_gen_len=32),
    )
```

服务入口使用同一组参数：

```bash
rapid-llm serve /path/to/moe-checkpoint \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --enable-dp-attention \
    --no-cuda-graph
```

默认交换后端是 AgRs（ragged all-gather + reduce-scatter），只对真实 token 行运行专家；设置 `RAPID_MOE_A2A_BACKEND=all_to_all` 可切到带 padding 的 a2a 矩形池做 A/B。每个本地 scheduler 独立决定 prefill/decode pass，因此引擎先协调本 step 的最大 forward 数；较早完成或完全空闲的 replica 用 dummy forward 补齐，仍以相同顺序进入 MoE collective。任意请求到达时协调器也会唤醒其他 replica，避免请求倾斜导致 collective 等不到参与者。

当前组合矩阵：

| 组合 | 状态 | 原因 |
| --- | --- | --- |
| DPA + EP | 支持且必需 | experts 才有跨 DP 网格切分与汇聚对象 |
| DPA + TP | 支持 | 每个 replica 内 TP；EP group 扩展到完整网格 |
| DPA + AgRs / a2a | 支持 | AgRs 默认，a2a 用于对照 |
| DPA + CUDA Graph | 暂不支持 | ragged step 几何与跨 replica 锁步尚未纳入捕获契约 |
| DPA + speculative decoding | 暂不支持 | 每 step 的 forward 数和验证 pass 尚未纳入锁步协议 |
| DPA + Sequence Parallelism | 暂不支持 | 两者都会在 MoE 前重分 token 轴 |

回归与真实 checkpoint 门禁：

```bash
pytest tests/distributed/test_dp_attention.py \
    tests/distributed/test_data_parallel.py \
    tests/engine/test_scheduler.py -q
RAPID_LLM_TEST_DPA_DIR=/path/to/moe-checkpoint \
    pytest tests/distributed/test_dp_attention_engine.py -q
RAPID_MOE_A2A_BACKEND=all_to_all \
    RAPID_LLM_TEST_DPA_DIR=/path/to/moe-checkpoint \
    pytest tests/distributed/test_dp_attention_engine.py -q
```

性能脚本同时报告 TPS、TGS、相对单 rank 的 scaling efficiency、同步与通信的端到端开销，以及输出一致率；`balanced`、`idle`、`uneven` 三种场景分别覆盖均衡请求、空闲 replica 和不等长 chunked prefill：

```bash
python benchmarks/parallelism/bench_dp_attention.py \
    --model /path/to/moe-checkpoint \
    --arms local1,ep2,dpa2,dpa2x2 \
    --backends agrs,a2a
```

### DPA 四卡实测（2026-09-30）

Qwen3-30B-A3B-Instruct-2507-FP8，4× NVIDIA H100 80GB HBM3（GPU 间 NV18），CUDA 12.8，Torch `2.7.0a0+7c8ec84dab.nv25.03`，Triton 3.2.0。离线 eager greedy，`max_gen_len=16`，每组 1 次计时；因此数据用于记录功能与通信代价，不代表稳定态容量规划。

`balanced` 固定总 batch=16，是 strong-scaling 口径：

| 配置 | ranks | 延迟 | TPS | TGS | 效率 | sync+comm | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local1 | 1 | 0.799 s | 320.3 | 320.3 | 100.0% | 0.0% | 16/16 |
| ep2 | 2 | 1.530 s | 167.3 | 83.6 | 26.1% | 73.9% | 10/16 |
| dpa2 / AgRs | 2 | 1.306 s | 196.0 | 98.0 | 30.6% | 69.4% | 14/16 |
| dpa2 / a2a | 2 | 1.524 s | 168.0 | 84.0 | 26.2% | 73.8% | 12/16 |
| dpa2x2 / AgRs | 4 | 1.559 s | 164.2 | 41.0 | 12.8% | 87.2% | 12/16 |
| dpa2x2 / a2a | 4 | 1.651 s | 155.1 | 38.8 | 12.1% | 87.9% | 12/16 |

请求倾斜场景确认空闲 replica 与不等长 chunked prefill 不会卡住 collective：

| 配置 | idle TPS / exact | uneven TPS / exact |
| --- | ---: | ---: |
| local1 | 20.42 / 1/1 | 23.10 / 2/2 |
| ep2 | 11.18 / 1/1 | 12.65 / 1/2 |
| dpa2 / AgRs | 12.72 / 1/1 | 18.53 / 1/2 |
| dpa2 / a2a | 11.10 / 1/1 | 16.22 / 1/2 |
| dpa2x2 / AgRs | 10.78 / 1/1 | 15.72 / 1/2 |
| dpa2x2 / a2a | 9.82 / 1/1 | 14.92 / 1/2 |

`exact` 的参考不是 local1 整批输出，而是 local1 按相同 round-robin replica 子 batch 重放后恢复请求顺序；这排除了 batch 宽度变化。FP8/TP/collective 的累加顺序仍可能让近似并列 token 分叉，所以真实 checkpoint 门禁另外检查首个分叉的参考 top-2 gap ≤ 0.5 nat，并要求至少 2/3 生成 token 完全一致；AgRs 与 a2a 均通过。完整环境、输出与原始结果见 [`dp_attention_Qwen3-30B-A3B-Instruct-2507-FP8_20260930_120030.json`](./benchmark_logs/parallel/dp_attention_Qwen3-30B-A3B-Instruct-2507-FP8_20260930_120030.json)。

该 workload 下，DPA 没有带来吞吐扩展：collective、跨 replica 锁步与协调器 IPC 占据 69%–88% 的理想线性吞吐预算。AgRs 在所有 DPA 行均快于带 padding 的 a2a；DPA 的收益在这里是 Attention/KV replica 本地化和 expert 权重跨网格切分，而不是小输出长度下的加速。

## 实测数据

以下是早期 GPU 测量记录，保留用于复现比较，不代表当前版本或 CPU 的扩展效率。

Qwen2.5-1.5B-Instruct，2× A10（23 GB），greedy，`max_gen_len=128`，round-robin。基线是 `data_parallel_size=1` 的协调器（隔离掉副本数以外的变量），另附一行进程内 `LLM` 用来显示协调器的 IPC 开销。

> 每个副本的 `max_num_seqs`（并发上限）都设成它实际收到的条数。这一条必须显式说明：副本里
> 是一个**常驻**引擎，不像一次性的 `LLM` 那样能按手上这批的大小自适应，用服务默认值（32）去
> 对比一个整批一起解码的参考行，量到的是并发度之差而不是并行度之差——实测就是 3320 tok/s
> 对 11376 tok/s。

**weak scaling**（每副本固定 16 条，总量随副本数增长——即"服务吞吐"问题）：

| 配置 | 副本 | batch | 墙钟 | 吞吐 | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM（进程内） | 1 | 16 | 1.07 s | 1907 tok/s | 1.03x |
| DataParallelController | 1 | 16 | 1.10 s | 1857 tok/s | 1.00x |
| DataParallelController | 2 | 32 | 1.10 s | **3716 tok/s** | **2.00x** |

**strong scaling**（固定总量 256 条，切给各副本）：

| 配置 | 副本 | batch | 墙钟 | 吞吐 | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM（进程内） | 1 | 256 | 2.83 s | 11596 tok/s | 1.02x |
| DataParallelController | 1 | 256 | 2.88 s | 11376 tok/s | 1.00x |
| DataParallelController | 2 | 256 | 1.75 s | **18695 tok/s** | **1.64x** |

这组数据中，weak scaling 的吞吐比为 2.00，strong scaling 为 1.64。两种口径回答不同问题，不能互换；复现时应保持每副本并发、prompt 和输出长度一致。

复现：

```bash
python benchmarks/parallelism/bench_data_parallel.py --model my_weight/Qwen2.5-1.5B-Instruct \
    --dp 2 --batch-size 16 --scaling weak
python benchmarks/parallelism/bench_data_parallel.py --model my_weight/Qwen2.5-1.5B-Instruct \
    --dp 2 --batch-size 256 --scaling strong
```

## 正确性检查

上面两次运行里，dp=2 的输出与单卡**逐字节一致**（256/256、32/32 完全相同）。但这需要小心表述，和连续批处理是同一件事：**只有算术完全一致时"文本相同"才是合理预期**。batch 宽度是 GEMM 的 M 维，同一条 prompt 在 batch 32 里和在 batch 16 里 fp16 累加顺序不同，top-2 相差 ~1e-2 的 token 就可能翻转。

所以 DP 的一致性测试比对的是**同构 batch**：不是"dp=2 的 6 条"对"单卡的 6 条"（副本各只跑 3 条，batch 形状就不同），而是让参考 `LLM` 重放**同样的子 batch**——副本 0 的 3 条对单卡的这 3 条。这样 batch 组成完全一致，任何差异都是真正的路由 bug，而不是浮点噪声。见 `tests/distributed/test_data_parallel.py::TestTwoReplicas::test_matches_a_single_gpu_per_replica_batch`。

测试规模：

| 文件 | 数量 | 需要 |
| --- | ---: | --- |
| `tests/distributed/test_parallel_state.py` | 20 | CPU（网格纯函数） |
| `tests/distributed/test_data_parallel_policy.py` | CPU（策略与精确 charge 回收） |
| `tests/distributed/test_data_parallel.py` | CPU 路由/网格/typed IPC + GPU 双副本端到端 |
| `tests/distributed/test_data_parallel_async.py` | CPU（统一 request manager 连接 fake controller） |

这些测试覆盖 grid 启动、typed command 路由、乱序完成的负载回收、DPA wake、子进程静默退出、异步 stream/abort/failure fan-out 和幂等 shutdown。`tests/engine/test_scheduler.py` 另外固定空闲 DPA replica 仍参与全局 forward wave 的行为。

## 当前边界

- **同步 `generate()` 是阻塞 facade，流式服务统一走 `AsyncLLMEngine`。** 两者共享同一个 controller、同一组 typed command/event 和同一批 scheduler processes。`rapid-llm serve --data-parallel-size 2` 启动 `AsyncLLMEngine` request manager，并由它的单一接收循环按 request id 投递增量、完成和错误；客户端断开时发送 `AbortRequest`。load-aware policy 因而能观察跨时间到达的在线请求，批 API 也复用完全相同的路由与负载回收语义。
- **并发上限在建副本时定，不随批大小走。** `max_num_seqs`（默认 32）是常驻引擎的 slot 数，一次发进来 256 条就分批入场。这是服务该有的行为（显存有上限），但离线批处理要吃满卡就得把它开到批的宽度——`DataParallelController(..., max_num_seqs=256)`。上面那张表就是这么测的。
- **文本模型离线批处理。** 多模态的逐请求 processor 输出不走这条路径。
- **每卡一份完整权重。** 这是 DP 的定义，不是限制——放不下单卡就要叠加 TP （`tensor_parallel_size > 1`），两者按网格组合。
- **副本独立 profile 各自的 KV cache。** `tensor_model_parallel_all_reduce_min` 只在副本内的 TP 组里取最小值， DP 副本之间不参与，所以忙卡上的副本可以自持一份更小的 cache。规约张量落在 `torch.cuda.current_device()` 上而不是 `cuda:{tp_rank}`：DP×TP 下进程占的是 `dp_rank * tp_size + tp_rank` 号卡，`CUDA_VISIBLE_DEVICES` 还会再重映射一次，用 tp rank 当设备号会让副本 1 从副本 0 的卡上发起规约。

## 相关文档

- [张量并行](./tensor_parallel.md)：切权重的那一半，可与 DP 组合成网格。
- [连续批处理](./continuous_batching.md)：单副本内请求随到随走的 per-step 调度。
- [量化](./quantization.md)：每卡一份权重太大时的另一个旋钮。
