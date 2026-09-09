# 张量并行（tensor parallelism）

## 权重分片

张量并行（TP）把权重切分到多个 rank，减小每个 rank 的权重占用。行并行层通过 all-reduce 合并局部输出。是否降低延迟取决于矩阵规模、设备计算能力和通信开销。

CPU 部署给 `ContinuousBatchingEngine.from_pretrained` 传入 `device="cpu"`，使用 Gloo 数据面。多进程调用应放在脚本的 `if __name__ == "__main__":` 下。CUDA Graph 和后文的 NCCL 测量仅适用于 GPU，CPU 使用 eager 执行。详见 [CPU 支持](cpu.md)。

两者正交，构成 `dp_size × tp_size` 的 rank 网格，见[数据并行](./data_parallel.md)。

```bash
python -m rapid_llm.cli chat \
    --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --tensor-parallel-size 2
```

```python
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen3-8B", tensor_parallel_size=2
)
```

## Executor

引擎将执行计划交给 `Executor`，由单进程或多进程执行器完成模型前向和采样。

| 角色 | 本仓库 | vLLM |
| --- | --- | --- |
| 一次前向的**数据描述** | `executor/worker.py::ModelInput` | `SchedulerOutput` / `ModelRunnerOutput` |
| 执行一次 plan（本进程） | `UniProcExecutor` | `UniProcExecutor` |
| 执行一次 plan（多进程） | `MultiprocExecutor` | `MultiprocExecutor` |
| 非 driver rank 的全部行为 | `serve_plans` | `WorkerProc.worker_busy_loop` |

接口只有三个成员：`num_slots`、`execute(model_input)`、`shutdown()`。除了返回值，**任何签名里都没有张量**——这是"引擎不感知拓扑"能成立的原因。

- **`UniProcExecutor`**：调本进程的 `ModelWorker`，到此为止。一张卡的默认路径，所以引擎循环里下一个断点就是 kernel 里的断点，`mp.active_children()` 是 0 （`test_tp_engine.py::test_one_gpu_stays_in_one_process`）。单卡不该为多卡的能力付调试成本。
- **`MultiprocExecutor`**：先把 plan 广播给 follower ranks，然后调**同一个** worker 方法。 driver 兼任 rank 0，所以 `tp=2` 只花**两个**进程而不是三个（`test_two_gpus_cost_exactly_one_extra_process`）。

### plan 是数据，所以只有一条代码路径

follower 不持有 scheduler、不持有队列、不持有停止条件；它收 plan，直到收到 `None`。

这是相较**已退场的镜像进程方案**的关键修正。旧方案广播的是 prompt 字符串，每个 rank 各自 **重新推导**该跑哪一批——于是"两边推导一致"变成了一个不变量，而它一旦被破坏（多一次 prefill、少一个 slot），表现不是报错，而是**卡死在 NCCL 里**。现在 plan 本身是唯一真相：布局从同一份数据派生，driver 和 follower 执行的是同一段代码，没有第二份"worker 的前向"要维护同步。

### 死掉的 rank 必须先于集合通信被发现

每个集合通信都假设所有 rank 会到场；一个 rank 死了，其余的**只是等**。静默挂死是多进程执行最坏的失败模式，所以 `execute()` 在提交昂贵的全局事实（broadcast）之前，先查一个廉价的本地事实：进程还活着吗（`ensure_followers_alive`）。同理，`shutdown()` 只在整个组还完整时才广播停止信号——向一个已经死掉的 rank 广播会永久阻塞。

## 控制面与数据面

plan 是 Python 对象，使用 Gloo 控制组广播。GPU 张量集合通信使用 NCCL；CPU 张量集合通信使用 Gloo。`tensor_model_parallel_broadcast_object_list` 在主存传输序列化后的计划。

分层的结果是：**数据面（NCCL，张量）与控制面（gloo，plan）互不知情**，而控制面因为不需要显卡，在 CPU 上就能整套测出来（`test_tp_control_plane.py`：广播语义 + follower 存活检测，7 个测试）。

**rendezvous 端口自选。** 固定 29500 有两个都表现为"挂在 rendezvous"的坑：同机跑两个引擎会撞端口；崩溃残留的 socket 会毁掉下一次运行。所以 `free_port()` 向内核要一个空闲端口， `launch_tensor_parallel(master_port=None)` 默认走它。这个函数放在**生产代码**里、测试 harness 反过来 import 它，是为了让"怎么选端口"只有一处定义。

## 分片维度

| 模块 | 切的维度 | 每步通信 |
| --- | --- | --- |
| `ColumnParallelLinear` | 输出维 | 无（保持切开） |
| `RowParallelLinear` | 输入维 | 一次 `all_reduce` |
| `QKVParallelLinear` | 输出维，**按 q / k / v 分段** | 无 |
| `VocabParallelEmbedding` | vocab 维 | 一次 `all_reduce` |
| `ParallelLMHead` | vocab 维 | 无（logits 留在本地） |

**`QKVParallelLinear` 为什么不是一个 `ColumnParallelLinear`。** q/k/v 三个 GEMM 融成一个 `[q | k | v]`，算术不变但激活只读一次、一次 kernel launch 顶三次——TP 下 decode 走 eager， launch 开销没有 graph replay 帮你藏。但**不能**写成 `ColumnParallelLinear(hidden, q + 2*kv)`： GQA 让 query 头数远多于 kv 头数（Qwen3-8B 是 32 vs 8），两个边界各自独立。对 `q + 2*kv` 做一刀均分是"看不见 q 在哪结束"的：低 rank 会拿到清一色 query 头、高 rank 拿到清一色 kv 头。局部宽度两种切法算出来一样，所以下游**不会报错**——这正是它危险的地方。

**vocab 两端为什么值得单独切。** `[vocab, hidden]` 这两个张量在 151K vocab × 8192 hidden 的 fp16 下各约 4.9 GB，而 decode 步的 `lm_head` GEMM 是 `batch × vocab × hidden`，大词表模型里最大的那次矩阵乘。切 vocab 把两者同时除以 `tp`（实测：`tp=2` 的 embedding 字节数正好是 `tp=1` 的一半，`test_two_ranks_hold_half_the_vocabulary_each`）。对 tied 模型这还是**唯一正确**的选择 ——不切的 head 配上切了的 embedding，就是两个张量都声称自己是同一个（`test_tying_survives_sharding` 断言 tie 关系在分片后仍然成立）。

`ParallelLMHead` 刻意**不 all-gather logits**：sampler 直接吃本地那一段，于是每步的传输量与词表大小**无关**，也从不实体化一个完整的 logits 张量。

## 跨 rank 采样一致性

每个 rank 只有词表的一段，而 top-p / temperature 都需要全局归一化。朴素做法是 all-gather logits——每步搬 `batch × vocab`，正好把上一节省下的东西还回去。

这里用的是**去中心化 log_softmax**：`log_softmax(x)_i = x_i - logsumexp(x)`，而 `logsumexp` 每行只是**一个数**。于是每步只需交换**每行两个标量**：

1. `all_reduce_max` 求全局行最大值（数值稳定用），
2. `all_reduce_sum` 求 `exp(x - max)` 的全局行和。

通信量从 `O(batch × vocab)` 降到 `O(batch)`，且与词表大小无关（`vocab_logsumexp`）。argmax 同理只交换"局部最优值 + 它的全局 id"。

非贪心采样从各 rank 自己的 RNG 抽签，所以最后还要把 rank 0 采出的 id 广播回去（`worker.py::_sync_tp`）——否则各 rank 对"刚生成的 token 是什么"意见不一，后面每一步都在放大这个分歧。这一层的数学在 CPU 上就能整套验证（`test_parallel_sampling.py`，9 个测试，同进程模拟分片）。

## 浮点误差与逐字节比较

分片在**精确算术**下是恒等变换，所以"tp=2 与 tp=1 逐字节相同"看起来是理所当然的断言。它不是。 fp16 归约不满足结合律：row-parallel GEMM 加一次 all-reduce，是把同一批乘积**按另一个顺序**加起来。

这件事上我们没有靠猜。取证脚本在**单卡、tp=1**下，只改变 batch 的组成（因而改变 GEMM 的归约形状）重跑同一条 prompt：

```text
prompt: "The history of the Roman Empire spans many centuries, and"
单独跑           -> "...modern life. From the construction of..."
放进任何一个 batch -> "...modern society. Which of the following..."
                     ^ 两者在第 56 个字符分叉，全程没有张量并行参与
```

这条 prompt 在第 14 个 token 处正好压在一个贪心平局上。**无条件**要求逐字节相等，断言的是 fp16 的性质，而不是分片的正确性。

所以 `test_tp_engine.py` 让测试**自己测出噪声下限**：每条 prompt 答两次——在它的 batch 里，和单独一条。**单卡引擎与自己不一致**的条目就是踩在平局上的条目，其余条目才是恒等式该成立的地方。三层断言：

| 断言 | 范围 |
| --- | --- |
| tp=2 与 tp=1 **逐字节**相同（两种分组都比） | 单卡上 batched == alone 的条目 |
| 共享前缀 ≥ 16 字符 | 踩在平局上的条目——错的 shard offset 会毁掉**第一个** token，不是第十四个 |
| 稳定条目占比 ≥ 2/3 | 防止上面那条强断言被静默架空 |

第三条是关键：没有它，一旦某天大部分 prompt 都变成"不稳定"，最强的那条断言就悄悄退化成空断言而测试依然全绿。实测输出（`pytest -s` 可见）：

```text
batch-shape stable: 7/9; on a tie: [('batch6', 4), ('mixed', 1)]
```

逐字节相等因此覆盖 9 个条目里的 7 个，且这个覆盖率本身被断言着。这是能同时抓住"off-by-one 的 shard offset""mask 漏进了别的 rank 的行"这类**产出看起来合理的数字**的失败的唯一检查——任何 "差值 < eps"的松散比较都不会察觉。

## 集合通信统计

上一节那条"每行两个标量"的主张，到这里之前一直只是 docstring 里的一段论证。问题在于它**没法从 profile 里看出来**：一个 all-gather logits 的采样器会通过 `tests/distributed/` 下**每一个**正确性测试——同样的 token、同样的 logprob——只是每步多搬几个数量级的字节。kernel 名字的火焰图也不会告诉你差别，因为两者都只是"一次 NCCL 调用"。

![tensor parallel](./images/tensor_parallel.gif)

所以每个集合通信都会把自己的 op 与 payload 报给一本账（`rapid_llm/tools/observability/collective_stats.py`）。记账放在 `tools/observability/` 而不是 `distributed/` 里：`distributed/` 负责跑线路，不负责给线路算账，这样这个包本身不带任何上报机器。

```python
from rapid_llm.tools.observability import Collective, CollectiveStats

with CollectiveStats.collect() as stats:
    engine.step()
print(stats.report())
assert stats.tally(Collective.ALL_GATHER).nbytes < 1024   # 关于流量的一句主张
```

五个设计点：

- **窗口式，不是全局的。** 没人开窗时那个 ContextVar 是空 tuple，埋点的代价就是这一个 `if`。测量因此天然是**有作用域的**——问的是"这一步花了多少"，而不是"进程启动以来累计多少"。
- **打开的窗口存在 `ContextVar` 里，不是模块全局变量里。** 于是一个窗口只属于开它的那个线程和 asyncio task。DP 副本是并发推进的（`async_data_parallel.py`），窗口若共享，每个副本的 per-step 测量都会把兄弟副本的流量算进来——得到的是一个看起来完全合理、但恰好错了 DP 倍的数字。
- **窗口可嵌套，事件计入所有打开的窗口。** per-step 窗口套在 whole-run 窗口里，一趟就同时拿到两份数据，调用方不需要做减法。
- **op 与 plane 是枚举，不是字符串。** 打错一个 op 名，字符串写法会安静地开出一行新账，让本该记录的那一行报 0——而"报 0"恰恰是这个模块唯一要回答的问题（这个通信到底有没有发生）。`Collective` 是封闭集合，plane 由 `Collective.plane` 给出：`broadcast_object` 是控制面（pickle 对象走 gloo），其余是数据面（张量走 NCCL）。二者是设计上互相交换的关系——花两个标量的控制流量换掉一次词表规模的 gather——所以按调用点打标签迟早会漂。报告、断言和 GIF 面板都从这一处取 plane。
- **记账点在 world-of-one 早退之后。** 单卡下的 no-op collective 不搬字节，记它就是在量调用点而不是量线路。`tensor_model_parallel_broadcast_object_list` 的字节数需要第二次 pickle，所以只在有窗口打开时才算，并且**在广播之后**算——这样 follower 报的数就是 driver 发出去的数。

实测一次真实的 tp=2 运行（Qwen2.5-1.5B-Instruct，4 条 prompt，24 步，rank 0 视角）：

```text
op                plane      calls       bytes    per call
all_reduce        data        1368     29.2 MB     21.9 KB
broadcast_object  control       24     10.0 KB       428 B
all_gather        data          24       344 B        14 B
broadcast         data          24       344 B        14 B
all_reduce_max    data          24       164 B         7 B
total                         1464     29.2 MB   (data 29.2 MB, control 10.0 KB)
```

两个读法。**层拿走了全部流量**：28 层各两次 row-parallel all-reduce，一步 decode 171 KB，一步 prefill 22.7 MB。**采样没有**：一行一步 12 B（`all_reduce_max` + `all_gather`），而同一行如果 gather logits 是 148.4 KB——**12,661 倍**。控制面全程 10 KB，也就是说"plan 走 gloo"这个选择的代价在总账里是 0.03%。

这本账也让一类新的断言成为可能（`tests/tools/test_collective_stats.py`）：把词表从 4096 放大到 32768，采样流量必须**一个字节都不变**。这是区分"分片采样"和"gather 后采样"的唯一检查。

GIF 由 `scripts/gen_collective_gif.py` 生成，驱动的是真实 tp=2 引擎，每个字节都是量出来的（唯一一条算术而非测量的是"if gathered"那一行，它描述的是另一种实现）。

## 测试规模

| 文件 | 数量 | 需要 |
| --- | ---: | --- |
| `tests/distributed/test_tp_control_plane.py` | 7 | CPU（gloo 控制面 + 存活检测） |
| `tests/distributed/test_parallel_sampling.py` | 9 | CPU（分布式采样数学） |
| `tests/tools/test_collective_stats.py` | 20 | CPU（记账窗口 + 并发隔离 + gloo 双 rank + 词表无关性） |
| `tests/distributed/test_vocab_parallel.py` | 13 | CPU + GPU（vocab 分片；2 个需 4 卡） |
| `tests/distributed/test_qkv_parallel.py` | 17 | CPU（段级切分与权重映射） |
| `tests/distributed/test_tp_engine.py` | 9 | GPU（需 2 卡，端到端） |
| `tests/distributed/test_tp_cuda_graph.py` | 11 | GPU（需 2 卡，TP × CUDA graph × 量化） |

端到端那 9 个测试由**两个 spawn 出的 probe 进程**测量，每个宽度一个：父进程从不 `import` CUDA 也不碰 parallel_state，所以一个崩掉的 rank 不会把一个半初始化的 TP 组泄漏给同一 session 里其余测试。加载 checkpoint 和 rendezvous 是唯一昂贵的部分，所以一个 probe 一次采集全部事实（两种分组的答案、executor 类型、子进程数、embedding 字节数、tie 关系），断言只从 report 里读。

其中一个测试专门跑**在线服务**路径：`AsyncLLMEngine(tensor_parallel_size=2)` 并发处理两条请求。它是唯一从**后台线程**发起集合通信的路径（`step()` 由 worker 线程驱动），实测不失步，且答案与离线 tp=2 逐字节一致。

## TP × CUDA Graph

decode graph 在单卡上录的是一串 kernel；在两卡上还把分片层的 all-reduce 一起录进去。这带来一条额外规则：**每个 rank 每一步必须选同一个 graph**。一个 rank 走 eager 而对端 replay，后果不是答案不同，而是**卡死**——对端在等一个永远不会发出的集合通信。所以启用它的代价不在捕获，而在证明各 rank 结构对称。

三件事让它成立：

1. **通信器先于捕获存在。** NCCL 在首次集合通信时才建设备资源，而捕获区内不能分配。`warmup_collectives()` 在每次 capture 前发一次 all-reduce 把这步挪到捕获之外，并顺手校验规约结果等于 rank 数——同一处不匹配若留到捕获中发现，表现是无消息的挂死。`init_parallel` 另外 `setdefault` 了 `NCCL_GRAPH_MIXING_SUPPORT=1`（prefill 永远 eager、decode 走 graph，两类集合通信共用一个通信器）；1 本就是 NCCL 默认值，这行的作用是把要求写在建组的地方。
2. **网格在上线前对齐。** `ModelRunner.enable_cuda_graph` 捕获完成后，各 rank 用 `tensor_model_parallel_ranks_agree()` 比对一个 `crc32` 网格指纹；不一致就**全体一起丢弃** graph 回落 eager。指纹为 `0`（=什么都没捕获，含 OOM）即使全体一致也判失败，所以这道闸通过就意味着每个 rank 都**有** graph。OOM 分支不能直接 return——对端正走向同一个集合通信，失败必须**报进去**而不是绕开。
3. **replay 判定不读任何本地状态。** `_select()` 只看 `input_ids.shape` 与 `atten_info.max_actual_seq_len`，两者都由 driver 广播的同一份 `ModelInput` 推出，因此各 rank 结构对称，不需要每步协商。

之后还有一道**数值闸**：每个已捕获的 `(batch, bucket)` 都跑一次 graph-vs-eager 的合成 decode 比对，取最大 logit 差，用 min 规约合并——任一 rank 超差则全体退回 eager。容差 `TP_GRAPH_PARITY_ATOL = 1e-2` 给 all-reduce 的求和顺序留了余量，但 H100×2 上 bf16 / fp8 / nvfp4 三种方案实测都是 **0.000e+00**：replay 发的是与捕获完全相同的 kernel 序列，逐位一致。写成 `error <= tol` 而不是 `not error > tol`，是为了让 NaN（graph 读到已释放内存的特征）落到失败一侧。

两个开关：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `RAPID_LLM_TP_CUDA_GRAPH=0` | 未设 | kill-switch，TP 下强制 eager decode，不改代码不重新部署 |
| `RAPID_LLM_TP_GRAPH_CHECK=1` | 未设 | 每步 all-reduce 各 rank 选中的 graph（`None` 也有自己的指纹），不一致就抛异常而不是挂死。默认关闭，因为它给 decode 热路径加了一次集合通信，而 graph 存在的目的正是缩短这条路径 |

实测（Qwen2.5-0.5B、H100×2、`max_seq_len=512` → 8 个 graph）：bf16 / fp8 / nvfp4 三种方案下 graph 引擎与 eager 引擎 32 步 greedy **逐字节一致**（单请求与 3 条 padding 到 batch-4 两种分组都比过）。这里要求逐字节，而 `test_tp_engine.py` 只要求共享前缀，差别是真实的：那边是 1 卡对 2 卡，行并行 GEMM + all-reduce 改变了求和顺序，greedy 平局会翻；这边是 2 卡对同样的 2 卡，只有 launch 来自 Python 还是来自 replay 的区别。

`int4`（AWQ）在 TP 下另有约束：group size 128 必须整除分片后的 `in_features`，0.5B 的 896 两分变 448 就不满足。这是 checkpoint 几何的限制，与 graph 无关，int4 × TP × graph 在更宽的模型上由 `benchmarks/engine/run.py quant` 覆盖。

## 当前边界

- **`vl-chat` 是单卡的。** 视觉路径还跑在一次性批处理的引擎上，`--tensor-parallel-size > 1` 会直接报错退出——旧的"镜像进程假装 TP"正是这一版删掉的东西，不会为了让参数看起来能用而留着。
- **vocab 与两个 head 数都必须能被 `tp` 整除**，否则 `divide()` 在构造时就报错（而不是在某个 kernel 里给出错的形状）。
- **单机。** rendezvous 走 `127.0.0.1`；跨节点需要真正的 `MASTER_ADDR` 与 rank 分配。

## 相关文档

- [数据并行](./data_parallel.md)：切请求流的那一半，可与 TP 组合成网格。
- [连续批处理](./continuous_batching.md)：plan 是怎么被排出来的。
- [在线服务](./online_serving.md)：异步前端与 OpenAI 兼容接口。
