# 深度优化设计方案（参照 SGLang）

v0.10 之后以性能为目标的专题设计。条目编号 O1–O14，落地时映射回 [ROADMAP.md](../../ROADMAP.md) 的版本条目（F/A/P 系列、地基 0–3、L1–L5）：ROADMAP 决定什么时候做什么，本文给出每件事的具体做法与收益依据。

基准环境：2×A10（24 GB，PCIe 互联、无 NVLink）+ Qwen3-30B-A3B-Instruct-2507-FP8。文中所有收益数字均为该口径下的机制推导估算，最终以 P0 立零点的 on/off 实测为准。

## 1 现状基准：四个结构性瓶颈

**KV 是槽位而非页，prefix 复用靠拷贝。** `executor/slot_batch.py` 中 slot `s` 永久拥有行 `[s*max_seq_len, (s+1)*max_seq_len)`，并发上限等于槽位数；`executor/kv_cache_manager.py` 自述 `One row per token (block_size=1)`，并留有 `TODO: reshape into [blocks, block_size, ...] to support PagedAttention`。因此 `engine/prefix_cache.py` 的命中结果是一串 `(src_slot, start, len)` 拷贝段：每 token 的 KV（fp16、TP=2、48 层）约 96 KB/rank，一个 4K 共享前缀的命中要搬约 200 MB D2D。v0.6 的分页 KV 落地的是行级引用计数，真分页（block_size > 1 + block_table）没有落。

**一步最多三次 forward，metadata 三套。** `engine/continuous_engine.py` 把每步拆成 `PREFILL`（padded 网格、不读 cache）、`EXTEND`（逐 token 行、读 cache）、`DECODE` 三个 pass，各自一套注意力元数据与 kernel。根因是 prefill kernel 不读 cache，续传 chunk 只能走 EXTEND 兜底。

**CPU-GPU 串行，TP 下无 CUDA graph。** `engine/async_engine.py` 的 worker 线程逐条 `step()`：调度 → 发射 kernel → 同步等采样结果回读 → 才调下一步。TP > 1 时 CUDA graph 被显式禁用，48 层 × 2 个 all-reduce 走 NCCL ring over PCIe（batch 小、payload 小，延迟主导），加上数百次 kernel launch 的 Python 派发，batch=1 的 TPOT 里约 30–40% 是非计算开销。

**Profiling 实证（H100，Qwen2.5-0.5B-Instruct，batch=4，max_gen_len=8，greedy，max_seq_len=1024，KV 池 8192 blocks）。**`torch.profiler` 导出 kernel timeline，两边各隔离**一个完整 decode 步**，步内kernel 数相同（脚本对此断言，不相等就报错）：

| 一个 decode 步 | eager | CUDA graph |
|------|-------|------------|
| GPU kernel 数 | 327 | 327 |
| CPU 派发次数 | 326（逐 kernel） | 1 次 `cudaGraphLaunch` + 32 次图外单独派发 |
| GPU 计算时间 | 1038 µs | 1017 µs |
| 步墙钟时间 | 15 724 µs | 1228 µs |
| GPU 占用率 | 7% | 83% |

GPU 计算量两边相同（1038 vs 1017 µs，差 2%；全程 kernel 总数 2647 vs 2654，差 0.3%）——CUDA graph 没减少任何计算。差的是派发：eager 每个 kernel 都要 CPU派发一次，326 次派发把 1 ms 的 GPU 工作摊成 15.7 ms 墙钟，GPU 有 93% 时间在空等下一次派发；graph 用 1 次 `cudaGraphLaunch` 回放捕获区的绝大多数 kernel（本次实测~285 个，即 24 层中的 21 层；剩下 3 层的 kernel 与采样在捕获区外，仍单独派发），同样的 1 ms GPU 工作在 1.2 ms 墙钟内跑完，**步墙钟差 ~13×**。

launch 计数必须同时覆盖两条 CUDA API：torch 算子走 runtime API（`cudaLaunchKernel` / `cudaLaunchKernelExC`），Triton kernel 走 driver API（`cuLaunchKernelEx`，category `cuda_driver`）。只统计 runtime 会漏掉 Triton 的大多数（2647 个 kernel 只数到 535 条 launch），把「每步 326 次派发」误读成 76 次。

复现：`python scripts/gen_cuda_graph_launch_gif.py`（墙钟时间有 run-to-run 波动，本次另一轮为 16 225 µs / 1222 µs，比值同为 ~13×）。

![eager vs graph kernel launch timeline](../images/cuda_graph_launch.gif)

读法：两轨 kernel 数相同（327），时间轴同刻度，可直接对比。上轨 eager 每个蓝色kernel 前都有一次琥珀色 CPU 派发，kernel 之间的暗红色是 GPU 空等；下轨 graph只有 1 条绿色 `cudaGraphLaunch`（另有 32 条琥珀色是捕获区外的单独派发），kernel紧密排列。所以下轨早早跑完、上轨拖满整窗——这就是 15.7 ms vs 1.2 ms。

**A10 上 MoE 是带宽瓶颈，dequant 路径物化中间张量。**sm86 无 fp8 算力，fp8 权重解包成 fp16 再进 GEMM（ROADMAP P3 指出的问题），每步多一轮全量权重读写。

**A10 上 MoE 是带宽瓶颈，dequant 路径物化中间张量。** sm86 无 fp8 算力，fp8 权重解包成 fp16 再进 GEMM（ROADMAP P3 指出的问题），每步多一轮全量权重读写。

## 2 方案全景与依赖

```text
引擎循环层（不改 kernel，收益最直接）
  O2 zero-overhead 循环        O10 tokenize 移出关键路径
调度与内存层
  O1 真分页 + Radix 零拷贝     O6.1 in-batch prefix   O6.2 两级 token budget
  O9 准入滞回 + decode 窗口
通信层
  O3.1 one-shot all-reduce     O3.2 TBO 双批重叠      O11 通信-RMSNorm 融合
kernel 层
  O4 MoE dequant 融合          O7 prefill 桶化 graph   O8 split-kv 自适应
解码算法层
  O5 ngram 投机解码            O6.3 greedy argmax
工程层
  O12 prefill/decode 双 stream 重叠   O13 graph 惰性捕获   O14 fp8 KV 端到端强化
```

依赖关系：O2 / O3.1 / O6.3 / O9 / O10 / O13 无前置，可立即做；O1 是 O5 / O6.2 / O7 的前置；O3.1 解锁 TP + CUDA graph（ROADMAP P8）；O8 / O14 可独立做，收益在 O1 之后放大。

## 3 引擎循环层

### O2 zero-overhead 引擎循环（已落地，默认关）

decode 的输入 token 不再回 CPU：下一步的 embedding 直接查 device 上的采样结果，step 拆成 launch / harvest 两半，隔一步读结果。

**现状。** `ContinuousBatchingEngine.step()` 每步严格串行：调度 → 发射 kernel → 等 GPU 算完 → 采样 token 读回 CPU → detokenize/判停 → 下一轮。两侧互相等待，任一时刻只有一侧在工作。batch=1 decode 的时间线（数字为估算）：

```text
现状，一步 ≈ 7.5ms：
CPU:  [调度+构建 1.5ms][ 同步读回+detok 1ms ][调度...]
GPU:                     [======decode 计算 5ms======]
                                          ↑ CPU 空转等待
```

**提前调度下一步的障碍是 token 回读。** decode 的输入是上一步采样出的 token，`_decode_work` 用 `request.output_token_ids[-1]`（host 上的 Python int）拼输入，再上传回 GPU，一去一回强迫每步同步。sglang 的 overlap 循环（`managers/scheduler.py` 的 `event_loop_overlap`）的前提是：下一步 forward 需要的只是 token id 的数值，而这个数值此刻就在显存里，可以直接把 device 上的采样结果 tensor 交给下一步 embedding 查表。真正需要 token 数值的只有 detokenize 和停止判断，晚一步做没有额外损失——读回来时 GPU 已经在算下一步。

**改法（三处）。**

1. `ModelInput.tokens` 允许传 device Tensor：decode pass 直接引用上一步采样输出 buffer；prefill 的 prompt 本来就在 CPU，不动。
2. step 循环重排：

```python
result_q = deque()
while True:
    plan = scheduler.schedule()            # 纯 CPU；GPU 正算着上一步
    result_q.append(executor.launch(plan)) # 只发射，不同步
    if len(result_q) > 1:                  # 隔一步才读
        last = result_q.popleft()
        tokens = last.sampled().cpu()      # D2H 已被本步 GPU 计算藏住
        advance_requests(tokens)           # detok、停止判断、penalty 计数
```

1. 采样结果落 pinned buffer + event，复用 `executor/overlap.py` 的 `StreamPool`。

CUDA graph 不受影响：replay 的输入本来就是静态 buffer，把 device 采样结果 D2D 拷进输入 buffer，比现在「CPU 读回再上传」少绕一圈。

**代价三条**：停止判定滞后一步（请求 eos 后多占一个槽位一步，sglang 同样如此）；repetition penalty 计数滞后一步（惩罚窗口几十 token，影响可忽略，写进文档）；`max_gen_len` 在 admit 时预扣 1，避免多吐一个 token。

**预期收益**：batch=1 TPOT -20~25%（~7.5ms → ~6ms）；并发下收益更大，CPU 组批时间随 batch 涨、GPU 时间不涨。这条也把 ROADMAP P9 的第一落点从「独立进程 + ZMQ」降级为同进程重叠，成本 10% 拿到 90% 收益，且不再与 F2（单进程 pdb 直达）冲突。

**落地记录（dev-v0.10；dev-v0.12 起在飞深度可配）**：已按本节设计实现，开关 `RAPID_LLM_PIPELINE`（默认关；显式 `1`/`true`/`on` 打开，`from_pretrained(pipeline=True)` 亦可，TP follower 由 driver 经环境变量教会）。循环形态是 `_step_pipelined`：schedule → launch → harvest，在飞深度由 `pipeline_depth` 给出（默认 1，`from_pretrained(..., pipeline_depth=N)` 可调），`len(_inflight) >= pipeline_depth` 或无新工作时才 harvest 最老一步。深度 D 时读到的是 D 步前发射的 token——它的 D2H 拷贝已被中间 D-1 个 forward 藏住，稳态下 `event.synchronize()` 零等待；输出 token 与深度无关，停止处理晚 D 个 token、`pending_tokens` 高 D-1。与设计的差异如下：

- 多吐一个 token 的对策不是 admit 时预扣 `max_gen_len`，而是 harvest 时检查 `request.is_finished`：晚停那步的 pass 照常发射，token 读回后丢弃不追加。占槽代价与设计一致（深度 D 时晚停 D 步、多占槽 D 步），停止语义与同步版逐位一致，admit 逻辑不用动。
- 乐观账目落在 `Request.pending_tokens`：launch +1、harvest -1（丢弃的也减，账目闭合归零）；decode 计划的长度加上它，token 用 `-1` 占位（非法 id，若泄漏到 embedding 会直接报错），真值由 worker 的 `_next_tokens` device 网格 gather 接力。
- 与 recompute preemption 互斥（构造时 ValueError）：领先 `pipeline_depth` 步的账目无法为回滚重算服务，开 pipeline 必须关 `enable_preemption`。
- `StreamPool` 扩了 readback 方向（`_spill` ring，与 upload 的 `_staging` 对称、独立复用）。落地测试抓出并修掉一个真 bug：D2H 的源 tensor 未 `record_stream(copy_stream)`，调用方释放后 block 被 caching allocator 复用，readback 读到的是下一个 pass 覆写的值。upload 路径一直有对称的保护，readback 起步时漏了。
- 契约测试：`tests/engine/test_pipeline_engine.py`（9 例：token 流与同步版一致但晚一步、账目归零、晚停丢弃、互斥拒绝、混合 step）与 `tests/executor/test_overlap.py` 的 readback 三例（含 ring 不覆写）；深度 2 与同步版逐 token 一致、drain 不挂起见 `tests/engine/test_pipeline_depth.py`（GPU + 权重）。

### O10 tokenize 移出关键路径

CLI 与 serve 入口的 tokenize 在主线程串行，几十 K token 的大 prompt 一次几十毫秒直接叠在 TTFT 上。O2 的 launch/harvest 结构给了挂点：tokenize 丢进 harvest 段的线程池，下一个 launch 只用已就绪的请求。预期大 prompt TTFT 减掉几十毫秒；优先级最低。

**落地记录（dev-v0.11）**：已落地，默认关（`async_tokenize=False`）。

- `add_request` 开启且未显式给 `prompt_token_ids` 时，把 encode 丢进惰性创建的线程池（4 worker；tokenizers 的 encode 释放 GIL，可真并行），立即返回空 token 的 Request，engine loop 不再为大 prompt 的几十毫秒 encode 停摆（async serve 下所有在跑请求的 step 都会被拖住）。
- `_collect_tokenized()` 在每步 `step()` 开头（engine 线程，与 `add_request` 同线程故无需锁）收割完成的 encode：请求连同 token 进 scheduler，同一步即可被 admit；`has_unfinished_requests()` 把 tokenizing 计入，worker 不会误判 idle 死等。
- 失败语义对齐同步路径：encode 异常或 scheduler 拒绝（空/超长 prompt）时请求标 `finish_reason="invalid"` + `request.error`，并经 `add_request(on_error=...)` 回调通知调用方。`AsyncLLMEngine._fail_async` 借此把同一个 ValueError 推给对应 stream，serve 语义不变。
- `generate()` 批量路径并行 encode（sum→max）；显式 `prompt_token_ids` 一律走同步注册不进池；`abort` 能取消还在 tokenize 的请求；`shutdown` 关池。
- 契约：`tests/engine/test_async_tokenize.py` 八例（立即返回/次步入队/generate 一致性/显式 token_ids 旁路/空 prompt invalid+on_error/encode 异常/abort 取消/async 前端 ValueError 透传）。

## 4 内存与调度层

### O1 真·分页 KV + Radix 零拷贝共享（工作量最大的一项）

**参照**：sglang `mem_cache/radix_cache.py`（chained-hash 树 + 引用计数 + 只从叶驱逐的 LRU）、`mem_cache/allocator/`（页分配器）、`mem_cache/memory_pool.py`（buffer 直接塑成 `[pages, page_size, kv_heads, head_dim]`）、`layers/attention/triton_backend.py`（decode kernel 直接吃 block_table）。

**设计四步。**

1. **buffer 重塑**：`gpu_kv_buffer` 从 `[max_tokens, 2*kv_heads, head_dim]` 变 `[num_pages, page_size, 2*kv_heads, head_dim]`，page_size=16（与现有 `PREFIX_CACHE_BLOCK_SIZE` 对齐，flashinfer 默认页也是 16）；`b_req_tokens_table` 从恒等映射变成真 block_table `[max_reqs, max_pages]`。
2. **PrefixCache 升级为 RadixCache**：树节点持有 page ids 而非 owner_slot。命中即页引用 +1，零拷贝；`prefix_copies` / `invalidate_slot` / `assign_owner` / `_pending_owners` 整套先拷贝后认领的簿记退场，它们存在的理由就是拷贝有时序（`_promote_pending_owners` 的 docstring 写得很清楚）。
3. **kernel 侧改造（全部工作量的所在）**：`flash_decoding`、`flashattention2_nopad`、`update_kv_buffer` 从行寻址改页寻址，sglang triton_backend 的 decode kernel 就是带 page table 的，形态可直接参照。flashinfer 适配器反而变简单：`BatchPrefillWithPagedKVCacheWrapper` 原生吃页表。
4. **写后读统一（顺带消灭三 forward 分叉）**：sglang 的 extend 语义是先把本 chunk 的 KV 写进 cache，再 attend 完整 `[0, pos+1)`。统一后 PREFILL/EXTEND/DECODE 合成一个 varlen 契约：每行带 `(page_table, q_len, kv_len)`，首 chunk、续传 chunk、decode 都是同一 kernel 的不同 q_len。`PassKind` 三分支、`_prefill_work` 路由、padded 网格全部退役；纯 decode 步继续走 CUDA graph（sglang 同款组合）。

**实例**：10 个请求同时到达，都带同一个 2K system prompt。现状下第一个请求的块只进 `_pending_owners`，`assign_owner` 下一步才认领，于是第 2~10 个请求同一步 admit 时 `copyable_tokens=0`，各自完整 prefill 一遍 2K 前缀（一次昂贵计算付 10 次）；即便命中认领后的 owner，还要按段拷贝 ~200 MB/请求。页化后命中即页引用 +1，第 2~10 个请求直接共享页，TTFT 从 `10 × prefill(2K)` 变 `1 × prefill(2K)`。

**迁移策略**：走 ROADMAP A2（KV 布局可插拔），`PagedLayout` 与现有 `SlotLayout` 并存注册、golden 矩阵双跑，随时回退。

**预期收益**：32K 上下文 + 平均 2K 对话场景，并发容量 ×5–16；多轮对话 TTFT 从「重算 2–4ms + 拷贝 ~1ms」到近零；单步 forward 3 次 → 1 次。

### O6.1 in-batch prefix（同批请求互认）

不依赖 O1 的先行版：admit 时把同一步内块哈希相同的等待请求分组，组内第一个承担计算，其余延一步 admit，下一步命中已认领的 owner，省掉重复 prefill 的算力（拷贝仍在，O1 落地后归零）。预期共享前缀 burst 场景 prefill 算力 -60~90%。

### O6.2 两级 token budget（必须随 O1 同版本落地）

现状 slot 模型下 KV 容量被 `num_slots` 隐式管住，粗但安全。真分页之后容量变成动态共享池，只看 `max_num_batched_tokens`（prefill 预算）不够。

场景走查：32K 上下文模型，KV 池剩 60K 页-token，来了一个 30K prompt，同时 32 个 decode 各占 1K 还在涨。只看 prefill 预算会 admit 这个 30K：它自己分块没问题，但 32 个 decode 每步各长 1 token，几步之内池子见底，触发抢占风暴，刚算完 30K 的请求被踢回 waiting 重算。解法对齐 sglang `ScheduleBatch` 的语义：admit 前把 `rem_total_tokens`（KV 容量账）与 `rem_input_tokens`（prefill 预算账）分开记，模拟完「本请求全程要占多少、现有 running 会涨多少」不够就不放行。页模型缺这道账就是抢占风暴的直接成因。

### O6.3 greedy 直接 argmax

`sampler.py` 现在所有请求统一走 `sample_top_p`：topk(候选池) → softmax → 掩码 → 重归一 → multinomial，五六个小 kernel。greedy 请求一个 argmax 就够，语义上也更直接。采样固定开销 -80%，实现量很小。

**落地记录（dev-v0.10）**：已落地，比设计还多一层。`Sampler._draw` 里 `all_greedy` 时整条 softmax/top_p 流水线跳过，只剩一个 `argmax`（TP 走 `global_argmax`：local max + all-reduce(MAX) + all-gather + amin，每行只过网 2 个值，tie 取最低 id 保持确定性）；混合 batch 才 `torch.where(greedy, ...)` 合流。顺手加了快路径 `all_top_p_one`（`BatchedSamplingParams` 预计算 `all(is_greedy or top_p == 1.0)`），让整批 top_p=1 的随机行直接 `multinomial`，不过 topk/掩码。契约由 `tests/engine/test_sampler.py` 覆盖（greedy argmax、top_p=1 免排序、混合 batch 三形态）。

### O9 准入滞回 + decode 窗口

**滞回**：`kv_cache_manager.can_admit` 的 watermark 是单边阈值，水位在阈值附近抖动时 admit/preempt 来回横跳，每次 preempt 都付一次重算租金。sglang 的 `min_free_slots_delayer` 是对应细节：跌破水位后不立刻恢复准入，等回升到阈值以上一段才放行。搬过来就是 `can_admit` 加恢复带。

**decode 窗口**：burst 到来时立刻打断 decode 去 prefill，所有在跑请求的 TPOT 尖刺。policy：新 prefill 最多攒 N 步再插队，用一点 TTFT 换 decode 平滑。做成 `SchedulerConfig` 开关，benchmark 出 TTFT-TPOT 权衡曲线。预期水位抖动期吞吐掉坑消除，decode TPOT P99 -30~50%。

**落地记录（dev-v0.11）**：两半都落地，默认全关。

- **滞回**：`KVCacheManager` 新增 `hysteresis`（默认 0.05，恢复带块数 `int(blocks * hysteresis)`）。`can_admit` 带惰性锁存：跌破 watermark 后门槛抬到 watermark + 恢复带，直到水位回升过带才复位，水位在单阈值附近抖动时 admit 不再来回翻。需说明的是 `can_admit` 目前无生产调用方（slot 模型下容量由 `num_slots` 隐式管住），滞回现在只是把 v0.12 请求级回收要用的 API 修对，抖动保护要等那时才真实生效。
- **decode 窗口**：`SchedulerConfig.decode_window_steps`（默认 0 = 立即 admit，作为基线；`from_pretrained` 同名参数透传）。`_defer_admission` 只拦本可纯 decode 的步：无 decode 在跑不拦（白等 TTFT），chunked prefill 续传步不拦（打断已付），攒满 N 步必放行。计数器 `_deferred_steps` 在 admit 发生或队清空时归零。
- 契约：`tests/executor/test_kv_cache_manager.py` 三例（单阈值语义、恢复带拒绝、水位振荡防抖）+ `tests/engine/test_scheduler.py` `TestDecodeWindow` 五例（窗口等待/默认立即/无 decode 不等/续传步不等/负值拒绝）。TTFT-TPOT 权衡曲线待基准环境回归后归档。

## 5 通信层

### O3.1 one-shot all-reduce（解锁 TP graph）

TP 组内 one-shot all-reduce：rank 间直接 P2P 写对端 IPC buffer + flag 自举，payload ≤ 阈值时替换 NCCL ring。参照 sglang `distributed/device_communicators/custom_all_reduce.py`（与 `torch_symm_mem.py` 的 multimem 路径）。

实例：现在每步 48 层 × 2 次 ring all-reduce，PCIe 小消息每次 15–25μs，纯延迟主导；one-shot P2P 写 5–8μs，一步省 ~1–1.5ms。它 graph 安全，顺带解锁 ROADMAP P8（TP + CUDA graph 锁步 capture），即 ROADMAP 自己标注的设计空缺。

**预期收益**：TPOT -15~20%；与 O2、TP graph 三项收益有重叠，合计后 CPU + 通信侧开销逼近零，TPOT 逼近 GPU 下限 ~4ms（合计 -40% 左右）。

### O3.2 TBO 双批重叠

decode batch 切两个 micro-batch，micro-A 做 all-reduce 时 micro-B 在算 GEMM。参照 sglang `batch_overlap/`（原 two_batch_overlap）。rapid_llm 已有 `PassKind` / `ModelInput` 抽象，双 pass metadata 现成。做成 batch 超阈值才启用的 policy。PCIe 互联通信占比高时收益真实，NVLink 上气泡更小。

**落地记录（v0.11.5）**：完整 TBO 已落地，`rapid_llm/batch_overlap/` 与 sglang 同布局；开关默认关（`RAPID_LLM_TBO=1` 显式开）。

- **执行器（`operations.py`）**：`YieldOperation` + `StateDict`（键写一次、pop 后才能重写，`clear(expect_keys)` 校验中间量是否按时释放）+ `_StageExecutor` + `execute_overlapped_operations`：op 流按 yield 切 stage，双流按 `delta_stages` 交错推进（lead 领先 N 个 stage，尾部对称收尾）。
- **策略（`operations_strategy.py`）**：`OperationsStrategy.init_new_tbo` 按 layer 类名分派，收的是各层自己的 bound method；dense 流 `[op_attn, yield, op_mlp]` delta 0，EP MoE 流 delta 2（两个 a2a 各带一个 yield）；混合栈（dense 前导层 + MoE）取最宽 lead。
- **切分与 policy（`two_batch_overlap.py`）**：`TboSplitter` 两半等长（奇数 batch 重复末行补齐到 `padded_len`，多余行的 logits 由 `num_rows` 丢弃）+ `TboPolicy`（`min_rows` 阈值 + `capture_eligible` 判定某个 batch 的 graph 是否录交错流）。
- **模型侧**：`models/base.py` 的 `DecoderLayer.forward_attn_stage/forward_mlp_stage` 两段拆分 + 九个 `op_*`（读 `StateDict`、消费即 pop、结果写新键）。
- **数据与结论**：见 `docs/release-v0.11.5.md`。eager 形态负收益的根因是 CPU launch floor（graph 参照臂 6-8 ms 对 eager 27-66 ms）；graph-captured TBO 已实现、replay 与 eager TBO 数值一致，但 dense-PCIe 形状下 interleave 本身净负。
- **启动条件仍然有效**：nsys 基线显示 all-reduce 占步长 >20% 才值得铺开；本版实测该占比只有 ~3-5%，所以 prefill TBO 未做。

### O11 通信-RMSNorm 融合

TP 下 o_proj/mlp 后的 all-reduce + RMSNorm 改成 reduce-scatter → 每 rank 只对持有的 token 段做 norm → all-gather；norm 并行化，且 reduce-scatter 完成的 token 段立刻进 norm 而不用等全量到齐，通信尾部与 norm 重叠。flashinfer 的 fused allreduce+rmsnorm 是同类思路，rapid_llm 已有 `skip_rmsnorm` kernel，融合点现成。

**实例**：batch=1 时 norm 读写是 8KB 级，收益可忽略，这条不适用于 batch=1。batch=32 时 norm 读写省一半，通信尾部可藏。预期 batch≥16 每步 -0.3–0.5ms，batch=1 <5%。它是 O3.1 的增值包，不是独立项。

## 6 kernel 层

### O4 MoE dequant 融合 grouped GEMM

fp8 权重解包进 GEMM mainloop（per-group scale 在 Triton 里就是加载时 `* scale`），消除中间 fp16 权重物化。A10 上 MoE decode 是纯带宽活，省掉的是那次额外的全量权重读。配套：router 提前发射（O2 的 launch/harvest 分离给了挂点）、`moe_align_block_size` 中间 buffer 常驻、`_launch_config` 由 autotune 冻结记录接管（机制 v0.10 已接好，缺一轮 collect）。预期 MoE kernel 时间 -20~25%，TPOT 直接受益。

### O7 prefill 桶化 CUDA graph

O2 只解决 CPU 不等 GPU，没解决 launch 本身。decode 已有 graph（batch 桶 + filler slot 补齐），prefill 没有，而 prefill 恰是 kernel 数最多的路径。机会在于 chunked prefill 的 chunk 大小本来就桶化（`max_chunk_size=512`），PREFILL pass 的形状只有 `(batch, 512)` 有限几种，正是 capture 的合适对象。sglang 最新代码在推 FullCG（连 prefill 也进 graph），方向一致。

设计：PREFILL pass 按 `(batch桶, chunk桶)` 捕获，replay 时 D2D 拷入 token/position。EXTEND pass 先不动——O1 的写后读统一会消灭 EXTEND 整条路径，在它上面做 graph 是白费，O7 排在 O1 之后最省。预期 TTFT 里 CPU 派发占比归零，TTFT -5~10%（大 prompt 场景）。

### O8 decode attention 的 split-kv 自适应

batch=1（单请求聊天）的 decode 有结构性短板：不 split 时一个 seq 的注意力压在少数 SM 上，A10 的 80 个 SM 大部分空闲；长上下文（>4K）时这是 TPOT 大头，8K 上下文的 KV 读 ~750MB/步，与 MoE 的 ~1.8GB 同量级。解法是 split-kv 把 KV 维切开并行算 partial softmax 再合并，split 数按 `(batch, seq_len)` 查表：batch 大时每行自有并行度，split=1 最好（省合并）；batch=1 且 seq 长才开大 split。查表的生成直接挂 autotune 冻结记录（collect → search → persist 三步现成），给 autotune 找到第二个高价值客户。预期长上下文 TPOT -20~30%，与 O14 乘法叠加。

## 7 解码算法层

### O5 ngram 投机解码先行

候选来自 prompt + 已生成文本的 n-gram 查表（纯 Python 先行，热了再下沉）；verify 是一次 varlen pass：draft k 个 token 作为 q_len=k 的行 attend 既有 cache，树形掩码用 varlen kernel 的 mask 位段表达。这正是 O1 统一 varlen 契约的直接受益者——没有 O1，verify 要在 EXTEND 三分叉上再叠树形逻辑。自适应 draft 长度（滑动接受率）对齐 ROADMAP P4 的自有设计；MTP 等 draft 权重类策略仍按 ROADMAP 放 v0.14。sglang 的 ngram（`speculative/`，本机源码已有 `cpp_ngram`）说明这条路在代码/改写负载上性价比高。预期 mean accept 1.5–2.5 → TPS ×1.3–2；验收标准是重复前缀负载 mean accept ≥2。

## 8 工程层

### O12 prefill/decode 双 stream 重叠

prefill（compute-bound）与 decode（memory-bound）无数据依赖（decode 集合本就不含本步刚 prefill 完的请求，`just_prefilled` 已排除），可两条 stream 同时发射。风险是 SM 竞争拖慢 prefill，做成 policy：仅当本步 prefill chunk 总 token < 阈值时并行。

实例：现状一个 512-token chunk 插进来，正在 decode 的 32 个请求这步多等 ~1–2ms，TPOT 尖刺；双 stream 后只被 SM 竞争拖慢 ~20–30%，尖刺 +1.5ms → +0.4ms，代价是 prefill TTFT 变差 5–15%。它是 O9 decode 窗口的另一种解法——窗口是攒着别插，双 stream 是插了别堵门，两个一起 benchmark 按负载选。预期混合负载 decode P99 -30~50%（与 O9 二选一或叠加验证）。

### O13 graph 惰性捕获（ROADMAP P7 的落地设计）

启动只捕获 batch=1 与最大桶两图，其余桶第一遇才捕获（~0.5–1s 一次性代价），空闲时后台补全高频桶；KV 预留按已捕获桶数逐步增长而非一次全扣。

实例：现在 `use_cuda_graph=True` 启动要等全桶捕获（每桶一次 capture + warmup），与 F3「冷启动秒级」目标直接冲突。惰性化后 ~2s 进服务，第一个 batch=16 请求多花 1s，之后零成本。反复起停、跑单测的场景收益最大。预期启动 -80%+（几十秒 → 2–3s），显存预留降到实际用到的桶。

**落地记录（dev-v0.11）**：已落地，默认关（`cuda_graph_lazy=False` 保持全量捕获旧行为），全链同名参数透传（`LLM` / `TextGenerator` / `VisionGenerator` / `ContinuousBatchingEngine.from_pretrained` / `LLMEngine` / `ModelRunner`）。

- **种子对**：`CUDAGraphManager.capture_seed()` 启动只捕两图——最小 batch×最小桶（单个新请求立即上图）+ 最大 batch×最大桶（饱和长上下文 batch 也在图路径上）。网格先经 `max_seq_len` / `max_request_num` clamp 再取端点，退化网格（单形状）自动去重为一图。
- **按需捕获**：`try_replay` miss 且 lazy 时 `_capture_on_miss` 现场捕获：`torch.cuda.synchronize()` 排空流水后，warmup 借用当前 step 的真实 `(b_req_idx, cur_select_index)` 作写目标，warmup 的垃圾 K/V 落在紧随的 replay 要写的同一位置并被覆盖，不污染任何其它请求的行（安全前提：slot 恒等映射，写位置由 `cur_select_index` 决定）。
- **OOM 黑名单**：捕获失败的形状进 `_failed` 永不重试，该形状后续步走 eager；触发 OOM 的那一步本身也降级 eager 正常出 token。
- **KV 预留**：`estimate_capture_workspace(lazy=True)` 只预留 `LAZY_SEED_SHAPES=2` 张图的 workspace；按需捕获从当时空闲显存现拿，失败形状不拖累 KV 池。
- 未做的一项：设计里的「空闲时后台补全高频桶」。on-demand 已覆盖正确性路径，后台补全只省首次 ~0.5–1s 的一次性代价，复杂度不值；启动收益数字待大模型基准回归（本地 0.5B 全量捕获本就秒级，差异不显著）。
- 契约：`tests/executor/test_cuda_graph_manager.py` 十四例 CPU 单测（种子对/按需捕获/黑名单短路/非 lazy 不捕获/prefill 与超长拒绝/退化网格去重）+ `tests/compile/test_cuda_graph.py` 两例 GPU 测试（lazy 输出与 eager 一致、batch-2 首步确实触发按需捕获），无 weights 环境自动 skip。

### O14 fp8 KV 端到端强化

`kv_cache_dtype=fp8` 已存在（uint8 e4m3 存字节），补三件事让它从能跑变为可默认开启：(a) O1 页模型下按字节块共享/拷贝的路径打通；(b) attention kernel 的 dequant（per-tensor scale 起步，head 级可选）；(c) golden 门禁 fp8 KV 单列一行精度指标，不达标不开默认。

实例：KV 每 token 每 rank 96KB（fp16）→ 48KB（fp8），同样的 22GB 池子能装的 token 翻倍，即并发 ×2 或上下文 ×2。另一项收益是 decode attention 属纯带宽活，KV 读量减半即 attention 时间减半，8K 上下文约省 0.7ms/步；与 O8 是乘法关系（一个减总量、一个提效率）。预期 KV 容量 ×2；长上下文 TPOT -10~15%；精度风险由 golden 门禁兜底。

## 9 收益汇总

| 项 | 机制 | 预期收益 | 口径 |
| --- | --- | --- | --- |
| O2 循环重叠 | CPU 串行段藏进 GPU | TPOT -20~25% | batch=1 估算 |
| O3.1 one-shot AR | ring 延迟 → P2P 直写 | TPOT -15~20% | TP=2 PCIe 估算 |
| O3.1 → TP graph | launch 派发归零 | 与上两项合计 TPOT -40% | 合成口径 |
| O1 页 + Radix | 零拷贝共享 + 单 forward | 容量 ×5–16；多轮 TTFT 近零 | 负载相关 |
| O5 ngram | verify 一次付 k token | TPS ×1.3–2 | 代码/改写类负载 |
| O4 dequant 融合 | 省 fp16 权重往返 | MoE kernel -20~25% | A10 带宽瓶颈估算 |
| O7 prefill 桶图 | TTFT 里派发开销归零 | TTFT -5~10% | 大 prompt 场景 |
| O8 split-kv | attention 并行度 | 长上下文 TPOT -20~30% | 8K 上下文 |
| O14 fp8 KV | KV 量减半 | 容量 ×2；长上下文 TPOT -10~15% | 精度门禁前置 |
| O12 双 stream | compute/memory 互补 | decode P99 -30~50% | 混合负载 |
| O13 惰性捕获 | 启动只捕两桶 | 启动 -80%+ | 反复起停场景 |
| O6 / O9 / O10 / O11 | 调度细节 | 各 5–10%，组合有效 | 场景相关 |

组合后的整体预期：batch=1 短上下文 TPOT 7–8ms → ~3.5–4ms；8K 上下文 10–12ms → ~5–6ms；混合负载吞吐 ×1.5–2.5；启动进入秒级。均为机制推导区间，P0 立零点后每项用 on/off 对照重新校准，达不到的项砍掉。

## 10 落地阶段

| 阶段 | 内容 | 出口标准 |
| --- | --- | --- |
| P0 | 立基线：TPOT / TTFT / TPS 三曲线 + nsys 全链路采集一次 | 各项收益的对照基准 |
| P1 | O2（已落地）+ O3.1 + TP graph + O6.3（已落地）+ O9（已落地）+ O10（已落地，默认关）+ O13（已落地，默认关） | TPOT -35%+，启动秒级 |
| P2 | O1 + O6.2（同版本）+ O14 | 容量与 TTFT 对照达标，golden 双布局全绿 |
| P3 | O5 ngram + O7 + O8 | accept ≥2；TTFT / 长上下文 TPOT 达标 |
| P4 | O3.2 TBO + O4 + O12 + O11 | 每项 on/off 对照正收益才保留 |

P1 进度（2026-09）：O2 / O6.3 / O9 / O10 / O13 五项已落地并各有测试与落地记录，其中 O10、O13 与 O2 同为默认关闭的 opt-in 开关；未动的只剩 O3.1 与 TP graph。O3.2 完整 TBO 已在 v0.11.5 落地（`rapid_llm/batch_overlap/`，对齐 sglang 布局），默认关；prefill TBO 按其节内启动条件（all-reduce 占步长 >20%）仍未做，实测占比只有 ~3-5%。

每阶段沿用 ROADMAP 第十一节的铁律：新东西必须有 on/off 对照 benchmark 归档进 `docs/release-vX.Y.Z.md`，golden 双跑；基线用 2×A10 Qwen3-30B-A3B-FP8 的 TPOT / 并发 TPS / prefix 命中 TTFT 三条曲线。

## 11 风险与明确不做

**风险三条**：O1 是唯一的大重构，走 A2 双布局并存迁移、golden 矩阵双跑、随时回退，其余项全部是增量；O12 / O11 的收益依赖 SM 竞争与通信形态，nsys 证伪就砍；O14 的 fp8 KV 精度不达标就只开并发场景，上下文场景退回 fp16，门禁说了算。

**明确不做的**：

- 不抄 sglang 的进程网格（EngineCore / scheduler / detokenizer 多进程全家桶）：单人维护性是前提，O2 已拿到重叠收益。
- 不上 torch.compile：与 F3「冷启动秒级」直接冲突。
- 不引 C++ radix tree：blake2b 链式哈希对 4K 级 prompt 够快，等 prefix cache 真成为 CPU 热点再说。
- EP / DeepEP 不进本期：all_to_all 原语未落地，且 TP MoE 每 token 的全部专家都在本 rank 计算（只切中间维），不存在 expert 负载倾斜问题，2 卡 EP 的通信收益存疑；继续挂 ROADMAP v0.11 通信原语之后。
