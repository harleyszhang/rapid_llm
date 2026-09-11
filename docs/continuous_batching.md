# 连续批处理（continuous batching）

## 请求生命周期

`LLMEngine.generate()` 在调用的那一刻就把 batch 定死了：所有序列同一步开始，整批一直保持满宽度推进，直到**最长**的那条结束。这带来两个无法在原路径里绕开的问题。

**一是算力浪费。** 一批 8 条请求里 7 条在第 20 个 token 就 EOS 了、1 条要跑到 500，那么后面 480 步里有 7 行在做纯粹的填充计算。序列一旦结束，它占的 batch 行并不会让出来。

**二是无法服务在线请求。** `generate()` 开始 1 毫秒之后到达的请求没有任何办法挤进去，只能等这一整批跑完。对一个服务端来说这等于串行。

连续批处理把"整批一次决定"换成"每步重新决定"：

```text
一次性批处理                          连续批处理
step  A B C D                        step  A B C D   等待队列
 1    ■ ■ ■ ■                         1    ■ ■ ■ ■   E F
 2    ■ ■ ■ ■                         2    ■ ■ ■ ■   E F
 3    ■ □ ■ ■   B 结束但仍占位          3    ■ E ■ ■   F     B 结束，E 立刻补位
 4    ■ □ ■ □                          4    ■ E F ■         C 结束，F 立刻补位
 5    ■ □ ■ □   ← 40% 是填充           5    ■ E F ■   ← 满载
```

## 分层结构

调度、缓存映射和执行分别由以下模块负责：

| 模块 | 职责 | 在哪一侧 |
| --- | --- | --- |
| [`Scheduler`](../rapid_llm/engine/scheduler.py) | 谁 prefill、谁 decode、谁拿哪个槽位 | 纯 host，无张量 |
| [`SlotBatch`](../rapid_llm/executor/slot_batch.py) | 分页 KV 映射与每步 attention 元数据 | host / device |
| [`ContinuousBatchingEngine`](../rapid_llm/engine/continuous_engine.py) | 串起 step 循环、采样、停止判定 | 两侧 |

`Scheduler` 不持有张量，调度测试无需 GPU 或模型权重。模型执行也可以选择 CPU，安装与限制见 [CPU 支持](cpu.md)。

## 每一步做什么

```python
# 对外由 engine.step() 执行；下面只表示阶段关系。
scheduled = scheduler.schedule()
# 分配并写入本步需要的块表。
# 处理 scheduled.prefill 里的 prompt chunks，以及 scheduled.decode。
# 采样后更新请求，释放已结束请求持有的块引用。
```

一个调度步可以同时包含 prefill 和 decode。长 prompt 按 `max_chunk_size` 分块，`max_num_batched_tokens` 限制 padded prefill 的 token 数。执行器根据已有前缀和算子能力选择 prefill、extend 或 decode 路径；不能把续块当作没有历史 KV 的首块处理。

## KV 布局：分页块池

槽位是请求块表的行号，不是一整段固定物理 KV。`b_req_tokens_table[slot, position]` 指向对应的物理 token 行；块池按请求的实际进度分配并维护引用计数。并发同时受 `max_num_seqs`、可用槽位和物理块容量限制。

启用 `enable_prefix_cache` 后，已完成的前缀块按哈希索引。命中请求共享物理块，执行器更新块表，不复制整段 KV。尚未完成的块不能提前供其他请求读取。

`enable_preemption=True` 允许调度器驱逐请求后重计算；已生成 token 会并入恢复上下文。抢占增加计算量，并非免费的内存扩容。launch/harvest pipeline 不能与抢占同时启用，构造时会报错。

`max_gpu_num_blocks` 按 token 行设置物理缓存容量；`prefix_cache_blocks` 则按前缀块计数，不要混用单位。

## 稳态 decode 元数据

`b_req_idx` 与 `b_seq_len` 只在**请求集合发生变化**时才从 host 重建。集合不变时：

```python
self._b_seq_len += 1                                    # 纯设备自增
cur_select_index = table[self._b_req_idx, self._b_seq_len - 1]   # 纯设备 gather
```

稳态下一个 decode 步的元数据只有两个 kernel，零传输。集合变化时才 `torch.tensor(...)` 上传一次——而且刻意每次**新建**张量而不是写回复用的 staging buffer：上一步的张量可能还排在流里等着被 kernel 读，从 host 覆写它会和未执行完的 kernel 竞争。

同样的"仅在集合变化时重建"策略也用在采样参数上： [`BatchedSamplingParams`](../rapid_llm/engine/sampler.py) 把每个请求的 temperature / top-p / repetition_penalty 摊成 `[batch, 1]` 张量，一次采样覆盖整批混合配置，而不是按配置分组多次启动 kernel。

## CUDA Graph：把奇数 batch 补齐

decode graph 是按固定的 `(batch_size, seq_len_bucket)` 网格 capture 的，而连续批处理产生的 batch 大小是负载决定的——batch 7 直接掉回 eager，graph 的收益就没了。 `SlotBatch._pad` 把 batch 补到下一个已 capture 的尺寸，多出来的填充行指向一个 **保留槽位**，其 logits 被丢弃。

填充行的长度取整批的最大长度，这样每一行每步都恰好 +1，上面那条"集合不变就原地自增"的快路径才不会被填充行破坏。保留槽位的 KV 行在初始化时清零：填充行读它不影响任何真实行（没有 kernel 跨 batch 行做归约），但一池 NaN 会污染之后每一次 debug 的读数。

## 一个真实的 bug：按 batch 位置索引槽位

`flash_decoding` 原本这样取一行的 KV 历史：

```python
k_loc = tl.load(b_req_tokens_table + stride_req_to_tokens_b * batch_pid + offs_n_new, ...)
```

即**用 batch 内位置当请求号**。一次性批处理路径永远传 `b_req_idx = arange(batch)`，位置恰好等于槽位号，所以这个假设一直成立、从未暴露。

连续批处理一旦有请求中途结束，后面的请求在 batch 里前移一位，于是每一个都开始读 **邻居的 KV**。症状不是崩溃或乱码，而是：

```text
请求 3 期望： "...we need to identify numbers that are only divisible by 1 and
              themselves. Let's go through the process step by step."
请求 3 实际： "...we need to identify numbers that are only divisible by 1 and
              themselves. Let's go through using electromagnetic induction."
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            这是请求 2 的结尾，逐 token 完全一致
```

修法是把 `b_req_idx` 传进 kernel，用 `cur_req_idx = tl.load(B_Req_Idx + batch_pid)` 转译一次再索引。回归测试见 `tests/kernels/test_flash_decoding.py::test_batch_row_reads_the_slot_named_by_b_req_idx` （把映射反转，逐位置索引必然读错）与 `tests/engine/test_continuous_batching.py::test_survivors_are_byte_identical_when_a_neighbour_leaves` （引擎级，退回旧 kernel 立即失败）。

## 实测数据

以下保留早期版本的测量记录，反映当时的实现与测试环境，不代表当前版本或 CPU 性能。

Qwen2.5-1.5B-Instruct，单卡 A10（23 GB），greedy，`max_gen_len=256`，16 个请求， `max_num_seqs=16`。原始日志：`docs/benchmark_logs/engine/continuous_Qwen2.5-1.5B-Instruct_b16.json`。

| 场景 | 策略 | 墙钟 | 有效吞吐 | 平均延迟 |
| --- | --- | ---: | ---: | ---: |
| offline，整批同时提交 | 一次性 | 2.36 s | 1650 tok/s | 2355 ms |
| offline，整批同时提交 | 连续 | 2.34 s | **1661 tok/s** | **2221 ms** |
| offline，长短混合（4×256 + 12×32） | 一次性（只能给整批一个上限） | 2.34 s | 507 tok/s | 2337 ms |
| offline，长短混合 | 连续（逐请求上限） | 2.18 s | **550 tok/s** | **657 ms** |
| online，每 250 ms 到达一个 | 一次性（只能串行） | 41.6 s | 93 tok/s | 19145 ms |
| online，每 250 ms 到达一个 | 连续 | **6.04 s** | **644 tok/s** | **2309 ms** |

三条结论：

1. **在线到达是主战场：吞吐 ×6.9，平均延迟 ×8.3。** 这就是 online batch inference 与 continuous batching 叠加的收益。
2. **offline 且所有请求长度相近时，两者持平（×1.01）。** 请求同时结束，一次性批处理本来就没浪费，这时连续批处理不该有收益——有的话反而说明测量有问题。
3. **offline 长短混合时，墙钟只快 ×1.07，但平均延迟好 ×3.6，且省下 2688 个没人要的 token（一次性路径 69% 的产出是多余的）。** 墙钟提升有限是因为在 A10 上这个 batch 宽度下 decode 受权重带宽支配，batch 从 16 缩到 4 每步耗时几乎不变；收益体现在短请求早 3.6 倍拿到结果、以及槽位提前释放、承接新请求。

复现：

```bash
python benchmarks/engine/run.py scheduler continuous --model-dir my_weight/Qwen2.5-1.5B-Instruct \
    --scenario all --batch 16 --max-num-seqs 16 --interval 0.25
```

## 正确性检查

"改调度不改数值"这件事需要小心表述，因为**只有算术完全一致时"文本相同"才是合理预期**。 batch 宽度是 GEMM 的 M 维，batch 3 与 batch 4 的 fp16 累加顺序不同，top-2 logits 相差 ~1e-2 的 token 就可能翻转。这是批处理固有的，vLLM 同样如此。所以测试分两档：

- **逐字节比对**只用在两侧 shape 一致的场合：单请求对静态 batch-of-1；以及借 CUDA Graph 填充，让"4 条缩到 3 条"仍在 capture 宽度 4 上执行，从而算术完全一致。
- **其余场合**用 `assert_no_foreign_tail`，直接针对真正要防的失效模式：一个请求吐出**邻居的**文本。这种污染从来不是一个 token 的翻转，而是整句易主。

测试规模：

| 文件 | 数量 | 需要 |
| --- | ---: | --- |
| `tests/engine/test_scheduler.py` | 25 | CPU |
| `tests/entrypoints/test_api_server.py` | 23 | CPU（fake engine） |
| `tests/engine/test_async_engine.py` | 11 + 1 | CPU（stub），1 个需权重 |
| `tests/engine/test_continuous_batching.py` | 15 | GPU + 权重 |
| `tests/engine/test_continuous_perf.py` | 4 | GPU + 权重，`slow` |
| `tests/kernels/test_flash_decoding.py` | +2 | GPU |

## 当前边界

- **仅文本模型。** 视觉 prefill 需要逐请求的 processor 输出，padded prefill 网格放不下；多模态 checkpoint 在构造时就报 `NotImplementedError`。
- **前缀复用与抢占默认关闭。** 分块预填充默认启用，`max_chunk_size=0` 可关闭。
- **功能组合有限制。** 抢占不能搭配 launch/harvest pipeline；FP8 KV 的续块采用能够解码量化缓存的路径。
- **`n > 1` 采样未实现**，HTTP 层显式拒绝而不是静默返回一条。
- **每步一次同步。** 读回采样 token 用于 detokenize 与停止判定。这换来精确的停止语义（EOS 的下一步就离开 batch），也正因为如此才划得来：腾出的槽位立刻给排队请求。

## 相关文档

- [在线推理服务](./online_serving.md)：`rapid-llm serve` 与 OpenAI 兼容端点。
