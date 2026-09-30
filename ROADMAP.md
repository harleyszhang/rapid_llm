
# 框架差异化亮点(重建:功能 / 性能 / 架构 / 自动化)

标注「已有」「待建」,每条附"为什么 vLLM/SGLang 不会做"。

> **多硬件平台原则**:rapid_llm 面向多硬件平台——NVIDIA 全系 GPU(A10/A100 sm80/sm86,H100/H200 sm90,B200 sm100+),并预留 AMD ROCm / CPU 路径(架构维度 A9)。核心论据:kernel 层**纯 Triton**(Triton 前端本身跨 CUDA/ROCm/CPU 后端),没有平台绑定的 C++/CUTLASS 编译。NVIDIA 卡内部再按 SM 版本细分:地基 2 的 `capability` 字段(SM 窗口 + device 过滤)让不同 SM 自动选不同 kernel 实现(如 fp8 tensor core 仅 sm89+,DeepGEMM 仅 sm90+),无 fp8 的卡自动回退到 Triton dequant 路径。开发与验证在 A10(sm86,PCIe 互联)上进行——它是"最低配"场景,能在 A10 上跑通的设计在 H/B 系列及 ROCm/CPU 上必然可用。

## 功能维度

| # | 亮点 | 状态 | 他们为何不做 |
| --- | --- | --- | --- |
| F1 | **任意模型任意单层的独立运行 harness**:单层跑 forward、对比 HF、测延迟/显存 | 已有(v0.10) | 他们没有这个抽象;大模型验证靠整模型跑 + 8 卡 |
| F2 | 默认单卡路径全程单进程,`pdb` 可直达 kernel 调用点 | 已有 | v1 全量多进程隔离(pdb 进不了 EngineCore);这里多进程只在 TP/DP>1 时启用(地基 0 的 UniProc/Multiproc 双实现),单卡默认仍是单进程 |
| F3 | 冷启动秒级(无 CUDA C++ 编译、无 torch.compile、graph 捕获可关) | 已有 | 他们为长驻服务优化,启动 30s–2min 不在乎 |
| F4 | 后端缺失自动回退原生,永不硬失败 | 待建 | 他们缺库常直接报错退出 |
| F5 | **bf16 权重与 KV**:参数与 cache dtype 脱离 fp16 硬编码(现散在 config/base/moe/attention 多处),由 checkpoint dtype 驱动 | 已有(v0.11) | 他们早已全面支持;fp16-only 是我们刻意保持的最小精度面,补 bf16 需连带各量化 method 的 supported_dtypes 与 kernel cast 策略 |
| F6 | **logprobs / prompt_logprobs**:采样时同步产出 top-k 对数概率,覆盖 prompt + 生成段 | 已有(v0.10) | vLLM 需要独立的 `PromptLogprobsWorker` 二次前向;这里一次 forward 拿全 |
| F7 | **结构化输出 / 约束生成(guided decoding)**:JSON Schema / 正则 / CFG / choice 约束,grammar bitmask 作用于采样 logits | 待建 | 对标 vLLM `StructuredOutputManager` + xgrammar(旧 API 即 guided_json/guided_regex/guided_choice);自有:Triton bitmask kernel 复用现有 sampler |
| F8 | **Reasoning parser / Tool parser**:推理段(`think` 标签拆分)与工具调用(函数调用 JSON)在协议层流式拆分,OpenAI 兼容 `reasoning_content` / `tool_calls` 字段 | 已有(v0.11) | 对标 vLLM `reasoning/`(DeepSeekR1/Qwen3 等 30+ parser)与 `tool_parsers/`(ToolParser ABC);自有:parser 与增量 detokenizer 协同,流式解析不等完整段,且解析器按请求声明(vLLM/SGLang 均为服务级单选) |

## 性能维度(主动技术创新,非防守论点)

借鉴 TileRT(tile 级运行时,把 compute/IO/comm 动态重叠,追极低 TPOT)、TensorRT-LLM(overlap scheduler)、Flux/TokenWeave/DualPipe(comm-compute 重叠),但每条转成自有设计。每条给目标指标。

| # | 亮点 | 状态 | 借鉴 / 自有设计 | 目标指标 |
| --- | --- | --- | --- | --- |
| P1 | **decode 层 megakernel**:整层 decode 路径(RMSNorm→QKV→attn→o_proj→MLP)融进极少数持久化 Triton kernel,消除几十次 launch + Python 派发 | 待建 | 借 TileRT tile 运行时;**纯 Triton 可融,vLLM 的 C++/CUTLASS kernel 融不动** | batch=1 TPOT 下降 |
| P2 | **TP 通信-计算 overlap**(详见第六节) | 待建 | 借 Flux/TokenWeave;PCIe 互联 GPU(如 A10)通信占比高,NVLink GPU(如 H100/B200)气泡更小但仍有收益 | TP=2 decode 隐藏大部分 all-reduce |
| P3 | **MoE dequant 融合 grouped GEMM + weight-stationary tiling** | 待建 | sm86(如 A10)无 fp8 算力→MoE 卡带宽;sm90+(H100)有 fp8 但仍有 dequant 收益;dequant 融进 epilogue 消除中间张量往返 | MoE decode 带宽利用率提升 |
| P4 | **投机解码全家桶**:MTP / ngram(prompt lookup) / DFlash / DSpark,统一 draft-verify 接口,按模型形态自动选策略 | 待建(最后开发) | 借 SGLang `SpeculativeAlgorithm`(NGRAM/DFLASH/DSPARK/EAGLE)、vLLM `ngram_proposer`;自有:统一 verify 接口 + 在线接受率自适应调 draft 长度 | decode 吞吐随接受长度近线性提升 |
| P5 | **overlap 调度进 CUDA graph**:多 stream 的 计算+通信+拷贝 整体 capture | 待建 | 借 TRT-LLM overlap scheduler;replay 零 CPU 派发 | 多卡 decode CPU 侧 gap 归零 |
| P6 | **自动调优 tile 配置落盘复用**(见工具 autotune 模块) | 待建 | 针对真实 shape 分布,自动生成而非手工 JSON | 高频 shape 命中最优 tile |
| P7 | CUDA graph 惰性捕获:首遇 (batch, bucket) 组合再 capture,省启动时间与预留显存 | 待建 | 中途 capture 有运行中 OOM(KV profiler 的 workspace 按全网格预扣)与首步尾延迟风险,vLLM 同样是启动时全量 capture | 启动时间与显存预留双降 |
| P8 | **DP/TP 与 CUDA Graph 同时生效**:TP all-reduce 可被 capture,DP 各副本独立 graph replay | 已落地 | vLLM 显式禁用 TP+CUDAGraph(见 `gpu_model_runner`);此处捕获后用跨 rank 指纹 + 数值闸确认结构对称再上线 | H100×2 实测 graph 与 eager 逐字节同答,logit 差 0.000e+00 |
| P9 | **引擎级异步调度**(CPU-GPU overlap):调度器独立进程,ZMQ 收请求,最多 N 个 batch 同时在流水线上 | 待建 | 借 vLLM `EngineCoreProc`+ZMQ、SGLang `zmq_to_scheduler`;自有:N-batch 流水线 + 双缓冲 slot | decode 步 CPU 侧等待归零,GPU 利用率逼近 100% |
| P10 | **DP 负载均衡策略族**:round-robin / 最小请求数 / 最小 token 数 / cache-aware(prefix 命中感知路由) | 待建 | 借 SGLang `LoadBalanceMethod` 四策略;自有:cache-aware 用各副本 prefix cache 命中估计打分,共享前缀请求聚到同 rank | DP 副本间负载倾斜 <10%,命中率同步提升 |

## 架构设计维度

| # | 亮点 | 状态 | 差异点 |
| --- | --- | --- | --- |
| A1 | **稀疏后端注册表 + 保底行**:外部后端只注册擅长的 (scheme, arch) 格,其余自动落原生 | 待建 | 避免 N×M 类爆炸,这是单人能维护多后端的前提 |
| A2 | **KV 布局可插拔**:token-level / paged / 未来 DiffKV 并存,重构可增量迁移随时回退 | 待建 | 把高风险大重构变成低风险增量,他们靠人力硬切 |
| A3 | 调度策略 policy 化:one-shot / continuous / chunked-prefill 共用一个 step 循环 | 待建 | 顺带消灭现有"两套生成循环"债务 |
| A4 | 模型定义薄:一个模型 = 一个类体十几行 + 一行注册 | 已有 | 他们的模型文件动辄上千行 |
| A5 | 每个 Triton kernel 旁并排 PyTorch 参考实现,作为语义定义者 | 部分已有 | 他们参考实现散在测试里 |
| A6 | **算子一等公民分发**:ABC 签名 + 声明式清单 + 确定性 dispatch,从现有 `registry.py` 雏形(availability+priority)升级到完整链路 | 待建(雏形已有) | 对标 sglang `KernelSpec`+selector;自有:实测排序自动选最快,sglang 甩给用户手选 |
| A7 | **运行时可观测性内置**:metrics/tracing 是一等 API,非离线工具;每个 step 产出 per-request 延迟、KV 占用、后端选择、overlap 气泡 | 已有(v0.10) | vLLM 的 metrics 面向运维仪表盘;这里面向开发者 debug,粒度到算子级 |
| A8 | **前沿注意力可插拔**:MLA/DSA/SWA/HCA 作为 `attention.*` 逻辑算子的不同实现,共享 paged KV 接口 | 待建 | vLLM 的 MLA 是独立类(`MLAAttention`);这里走统一 dispatch,新增变体只注册不写新类 |
| A10 | **多进程隔离引擎**(地基 0,对齐 vLLM/SGLang 进程模型):EngineCore(调度)与 Worker(GPU 执行)分离,调度决策只算一次、广播 SchedulerOutput;单卡默认仍单进程 | 待建 | vLLM `EngineCoreProc`+`MultiprocExecutor`/SGLang scheduler 进程网格是多年踩坑后的定论;当前 TP"镜像进程"/DP 一次性批处理是最大架构债(详见地基 0) |
| A11 | **并行 module 补齐**:`QKVParallelLinear`(q_proj+kv_proj 合体一次 GEMM,按 head 对齐切) / `VocabParallelEmbedding` / `ParallelLMHead` 按 vocab 维切分,采样走"去中心化 log_softmax"(只规约 logsumexp 标量 + gather 局部 top-k,logits 永不物化全量) | 待建 | vLLM `QKVParallelLinear`/`VocabParallelEmbedding`/`ParallelLMHead` 参照;当前 q/kv 两次 GEMM、embed/lm_head 全量复制是 v0.7.0 遗留决策(按小 vocab 估 0.3GB/rank 不值),Qwen3 151K vocab 下 embed+lm_head ≈4.9GB/rank、decode lm_head GEMM 是算力大头,必须切(详见第四节) |

## 工具维度(按功能模块划分)

统一放在 `tools/`(或 `<pkg>.tools`)命名空间,每模块一组子命令。这套工具既是差异化,也是单人维护多后端框架的质量前提。

**模块 A · 性能分析 (perf)**

| 工具 | 做什么 | 独特性 |
| --- | --- | --- |
| perf.profile | 一次 forward → 逐算子延迟 + roofline 归因(compute/memory-bound)+ achieved vs 峰值算力/带宽 | vLLM 要挂 torch profiler 且输出面向专家;这里一条命令出人可读归因 |
| perf.timeline | 多 stream 时间线,comm/compute 重叠可视化 | 直接服务第六节 overlap 验证 |
| perf.watchdog | benchmark 入库,劣化超阈值 CI 报警 | 保证"更快"不悄悄退步 |

**模块 B · 精度 (accuracy)**

| 工具 | 做什么 | 独特性 |
| --- | --- | --- |
| acc.golden | 多模型×多路径逐 token 门禁,GPU runner 强制跑,禁止静默 skip | 修掉"skip 了还显绿"的假安全 |
| acc.align | 外部后端 vs 原生 max-abs-diff 阈值门禁 | 使测试量为 N+M 而非 N×M |
| acc.divergence | **精度断层定位器**:整模型 vs HF 逐层对比,自动定位第一个超阈的层/算子 | 新模型接入 debug 神器,vLLM 无 |

**模块 C · 可视化 (viz)**

| 工具 | 做什么 |
| --- | --- |
| viz.structure | 模型结构树/图:每层算子、shape、dtype、参数量、选中后端 |
| viz.memory | 显存去向:静态分区(weight/KV/activation/graph)+ 随 step 的时间线曲线 |
| viz.flow | 请求执行流程:scheduler 决策、slot 分配、抢占、prefill/decode 切换 |
| viz.schedule | 层内 DAG + stream 分配 + overlap 时间线 |

**模块 D · 后端 explain (explain)**

每算子选了谁、候选有谁、为何被排除(库缺失/架构不匹配/优先级低)。vLLM backend 选择不透明是社区高频吐槽点。

**模块 E · shape 采集与自动调优 (autotune)**

| 阶段 | 做什么 |
| --- | --- |
| collect | 跑真实负载,导出所有 GEMM/attn/MoE shape(含出现频次) |
| search | 对高频 shape 搜 tile / num_warps / num_stages |
| persist | 按 (gpu, op, shape, dtype) 落盘 JSON 到缓存目录 + 可选入仓,启动命中即用、未命中回退启发式并后台补搜 |

覆盖对象:`fused_moe`(替换 `_launch_config`)、`flashattention2_nopad`(启用被注释的 144 组配置为搜索空间)、量化 GEMM。

**模块 F · 运行时可观测性 (observe)**

| 工具 | 做什么 | 独特性 |
| --- | --- | --- |
| observe.metrics | 每 step 产出 per-request 的 TTFT/TPOT/KV 占用/采样配置/选中后端;导出 Prometheus 或 JSON | vLLM 的 metrics 面向 SRE;这里面向开发者,粒度到单请求单算子 |
| observe.trace | 调用链 trace:一次 generate 的 scheduler 决策→slot 分配→forward→sampler→detokenize,每段附耗时;对接 OpenTelemetry span 或自研 JSON timeline | 直接服务 debug,不是 profiling 的事后分析 |
| observe.overlap | 实时 overlap 时间线:compute/comm/copy 三条 stream 的 event 水位 + 气泡标注,直接驱动第六节验证 | perf.timeline 的在线版,服务运行时调参而非离线分析 |

设计原则:metrics/tracing 是**一等 API**,不是离线工具的附属品。`Engine.step()` 返回的 `RequestOutput` 内嵌 `ObservabilitySpan`,调用方可零成本获取每步的 per-request 延迟分解。单元测试:每个 metric 有 fixture 驱动的 assertion(不是"能跑",是"数值在预期区间")。

真正没人做的:perf.profile 的人可读 roofline 归因、acc.divergence 精度断层定位、viz.*全套、explain、autotune 自动生成、observe.* 运行时可观测 —— 加上功能维度的 F1(单层 harness)、架构维度的 A1/A2/A6/A7。面试讲"我怎么用自动化保证单人维护多后端框架的质量"是很强的叙事。

# 三、四个地基设计方案

后面所有功能都挂在这四个地基上,顺序不能反。地基 0 是并行的一切前提——进程模型不对,TP/DP 都是纸面。

## 地基 0:多进程隔离引擎(对齐 vLLM/SGLang)

> 结论先行:当前 TP/DP 的实现是"每进程跑一个完整引擎",而 vLLM/SGLang 的共识是"调度算一次、执行进程隔离"。这个差距不补齐,P8/P9/P10、在线服务×TP、DP 常驻副本全都落不了地。

### v0.7.0 现状问题清单(代码证据)

**架构债(4 条)**:

| # | 问题 | 代码证据 | vLLM/SGLang 的做法 |
| --- | --- | --- | --- |
| 1 | **TP 是"镜像进程"不是引擎**:每个 rank 跑一个完整 `TextGenerator`(含 tokenizer/sampler/停止判断),rank 0 把 prompt tokens `dist.broadcast` 给 mirror worker 陪跑,输出丢弃 | `cli.py:_tp_mirror_worker` 的 broadcast 循环 | 调度只在 leader 算一次,广播的是**结构化 SchedulerOutput**,worker 只执行 forward(vLLM `WorkerProc`、SGLang leader scheduler) |
| 2 | **旧 TP 与持续批处理互斥**:调度/停止/detokenize 在每个 rank 独立执行,靠"相同输入→相同决策"隐式锁步;抢占/异步停止等 rank-local 决策一旦引入就 desync | 旧连续批处理入口直接拒绝 tp>1 | 调度决策单点,天然一致 |
| 3 | **旧 DP worker 是一次性批处理**:副本闲在两次 `generate()` 之间;请求不能中途加入;KV 每次 `free_all()` 全量重置 | 旧 `data_parallel.py` worker 调 `LLM.generate()` | 副本改为常驻 scheduler process，请求随到随入 |
| 4 | **DP×TP 组合死锁** | `data_parallel.py:99` `init_parallel(global_rank=dp_rank*tp_size, tp_size, dp_size)` 声明 dp×tp 的 NCCL world,但只 spawn 了 dp 个进程 | spawn 完整 dp×tp 进程网格(SGLang)或组内嵌套 spawn TP worker(vLLM) |

**bug(3 条)**:

| # | bug | 位置 | 修法 |
| --- | --- | --- | --- |
| 5 | `all_reduce_min` 用 `_TP_RANK` 当 CUDA device index;dp>1 时非 leader 副本的 TP rank 0 会算到别人的卡上 | `parallel_state.py:245` `cuda:{_TP_RANK}` | `torch.cuda.current_device()` |
| 6 | 旧 DP 路由用字符数 `len(prompt)` 当 token 数，least-loaded 的参数与语义不一致 | 旧 controller 路由与独立 policy 模块 | 路由层复用 token ids，policy 按 request id 精确记账 |
| 7 | TP 采样 RNG 不同步(已修,保留监控) | `tensor_model_parallel_broadcast` 采样后广播 | — |

### 目标进程模型

```
单卡(默认,F2 不破):  Frontend ──同进程── EngineCore( Scheduler + Worker 同体 )
TP:                    Frontend ── EngineCore(leader) ── broadcast SchedulerOutput(gloo) ── Worker 进程 × tp
DP:                    Frontend ── Router(P10) ── EngineCore 进程 × dp(每个内嵌自己的 TP 组)
```

**设计决策**:

1. **Executor 接口先行**(对标 vLLM `v1/executor/`):`Executor` ABC 只暴露 `init()/forward()/shutdown()`;两个实现——`UniProcExecutor`(EngineCore 直接持有 ModelRunner,单进程,pdb 直达 kernel 保住 F2)与 `MultiprocExecutor`(spawn tp 个 WorkerProc,各持 ModelRunner+KV cache)。调度器只见 Executor,不知进程拓扑——单卡/多卡同一套调度代码。
2. **调度只算一次**:leader 序列化 `SchedulerOutput`(prefill/decode 列表 + slot + block 号),经 gloo(CPU 组)broadcast 给 TP worker——控制面小张量;数据面 all_reduce 仍走 NCCL。镜像进程模式里"各 rank 独立跑 Scheduler"的整段隐式锁步代码退场。
3. **DP 常驻循环**:DP worker 从一次性 `LLM` 换成常驻 EngineCore 循环(请求队列随到随入,持续批处理永续),负载/缓存水位上报给路由层(P10 的输入)。
4. **进程网格一次 spawn**:dp×tp 个 Worker 进程统一拉起,每个进程 `init_parallel(global_rank=dp*tp+tp_rank,...)`——修复 bug 4 的死锁。

### 验收

- TP=2 下连续批处理 `Scheduler` 可用，golden 全绿;
- DP×TP(2×2)能起能推理(当前是死锁);
- `AsyncLLMEngine`(HTTP 服务路径)+ TP=2 可跑;
- 单卡默认路径仍单进程(冒烟:pdb 断点直达 Triton kernel 调用点);
- 每个阶段有单元测试:mock 进程网格,断言 SchedulerOutput 广播内容一致、RNG 同步。

### 落地顺序

1. 先修 bug 4/5/6(小改,不动架构)→ **v0.8.0**;
2. Executor 抽象 + UniProc/Multiproc 双实现 + SchedulerOutput 广播 + TP 接持续批处理 + DP worker 换常驻循环 → **v0.8.0**;
3. 与第八节合流:EngineCore 拆独立进程 + ZMQ + N-batch 流水线(地基 0 是第八节的前置,第八节是地基 0 的异步化终态)→ **v0.12**。

## 地基 1:真分页 KV + 请求级动态管理(解锁 chunked prefill + prefix caching)

现状问题:`block_size` 恒为 1,按 token 行分配 + refcount,`alloc_contiguous_kvcache` 慢路径每 decode 步 3 次 D2H 同步,每请求预占 `max_seq_len` 行;`free_all` 每次 `generate()` 全量重置,无请求级回收;缺显存不足时的抢占。

设计要点:

- `block_size=16`(中小显存卡如 A10 上 16 比 32 更省碎片;大显存卡如 H100 可选 32),`block_table[req_id] → [block_ids]`
- **通过 A2(KV 布局可插拔)并存迁移**:新 `PagedLayout` 与现有 `TokenLayout` 同时注册,env 切换。golden 矩阵在两种布局下都必须过,通过后再删旧的。这是把"高风险重写"降级为"可回退增量"的关键手法。
- 分配路径纯 GPU,消除 `nonzero` + `.item()` 同步
- block hash(prompt token 前缀哈希)+ refcount → prefix caching 的挂点
- attention kernel 同步改造:`flash_decoding` 与 `flash_attention2_no_pad` 从行索引寻址改块寻址
- **请求级动态管理**:请求结束即回收 block(取代 `free_all` 全量重置);block 池按 watermark 做准入,不足时进入抢占(见第七节 KV 内存管理)

验收:golden 全绿;并发容量提升(不再预占 max_seq_len);decode 步无 D2H 同步;请求结束后 block 立即可复用。

## 地基 2:算子作为一等公民 —— 一个算子 / 一个签名 / N 份实现 / 一份清单 / 一条确定性分发

参考 sglang `python/sglang/kernels`(spec/registry/selector/fused_op)的成熟设计,并补上它刻意留白的一环:**如何自动调到性能最佳的实现**。

> **现状(v0.9.0)**:五根支柱的**机制**已按三层落地——机制层 `kernels/dispatcher/`(ABC 签名 `interfaces.py`、声明式 KernelSpec、确定性 dispatch + 逐条拒绝理由的 `explain` + 调用 trace),注册行在九个算子域组 `kernels/ops/<group>/__init__.py`(native 行与外部后端的行同处一地),接入层 `kernels/backend/<lib>/` 一库一包(INSTALL 元数据 + available() 检测 + adapter)。v0.8 的雏形 `kernels/backends/registry.py` 与整个平铺 `backends/` 目录已随本迁移删除;它唯一未被吸收的能力(per-op 环境变量)已泛化为对每个注册 op 都生效的 `op_backend_env()`。
>
> **仍缺的是数据而非机制**:`golden.verified` 与 `perf_key` 两个字段 dispatch 已在过滤/排序时读取,但对齐门禁工具与冻结的实测记录分别要到 M3.1 / M3.2 才产出;`layout` 目前只做"不满足即排除",尚不会自动插入转换。

### 五根支柱

**① 一个逻辑算子**:收敛成固定清单,只定义"算什么",id 用 `<group>.<name>`。v0.9.0 共 12 个契约,其中 9 个 op 已有 native 实现行:
`attention.prefill / attention.decode / linear / moe / rmsnorm / rope / kv_write / elementwise.swiglu / elementwise.swiglu_split`。
`elementwise` 是开放命名空间,它的两个成员 `elementwise.swiglu`(packed)与 `elementwise.swiglu_split`(两半分开)各占一行,差别只在 arity——注册表共 11 个 op id、21 行(native 13 行 + 外部 8 行)。
剩下 5 个契约**刻意只声明、不注册 native 行**——契约先定住,实现随对应里程碑到位:

| 契约 | 为何没有 native 行 |
| --- | --- |
| `sample` | 采样归 `engine/sampler.py`,它跑在 TP 切分后的 vocab 分片上并带 repetition penalty;再写一份等于让采样有两处可分歧 |
| `comm.dispatch` / `comm.combine` | EP 组已随 v0.11.5 落地(`--enable-expert-parallel`,TP 组兼作 EP 组):`modules/ep_dispatch.py` 的等分 padding `all_to_all` dispatcher 承担 dispatch/combine,本地 permute 那半仍由 `fused_moe` 的 `moe_align_block_size` 承担;DeepEP 低延迟后端需 sm90+,A10(sm86)走纯 `torch.distributed` 路径 |
| `attention.mla_decode` | 树内尚无 MLA 模型;flashmla 行 + 单层 harness(`models/mla_single_layer.py`)已入库,契约仅有外部行 |
| `elementwise` | 开放命名空间的根本身,只有成员注册行 |

这份清单由 `tests/ops/test_native_specs.py` 的 `OPS_WITHOUT_NATIVE_ROW` 精确断言——"某个契约没有实现"可以,但不能悄悄没有。
现有 `flashattention2_nopad.py`、`flashdecoding.py` 命名混乱的根因就是缺这层——它们其实是 `attention.prefill` / `attention.decode` 两个逻辑算子的不同实现(legacy v1/v2 学习型 kernel 已移入 `benchmarks/kernels/`,随本迁移一并删除)。

**② 一个签名(ABC,吃语义张量,不含 layout)**:所有实现共享同一 `forward()` 签名与语义,`forward_native`(纯 PyTorch/参考实现)是**正确性基准**,必须存在。这一模式项目已有雏形——`models/quantization/methods/base.py` 的 `LinearQuantMethod.apply(layer, x)`,推广到全部算子即可。layout 差异(DeepGEMM 要转置权重、FlashInfer 要自己的 KV 布局)不进签名,进元数据。
**形参名也是契约的一部分**:因为支柱③ 刻意不写适配层,dispatch 把 kernel 函数本身交给调用方,所以"形参顺序对得上但名字不对"会无声通过。`tests/ops/test_native_specs.py::TestTargetsMatchTheirContract` 逐名比对 target 与 ABC——正是它抓出 `update_kv_buffer(K_Values, ...)` 与 `skip_rmsnorm(X, ...)` 两处漂移。

**③ N 份实现(不新增目录层,只多一张声明清单)**:三个正交的位置回答三个问题——

```
kernels/
  ops/            # 谁来算 + 注册行:九个算子域组,模型层按名字直调组内 native 实现
    attention/  gemm/  moe/  layernorm/  rope/  kvcache/  activation/  sampling/  embeddings/
    # 每组 __init__.py 持该算子的全部注册行: native 行与外部后端的行同处一地
  dispatcher/     # 怎么选: spec.py + registry.py + dispatch.py + interfaces.py,torch-free
    autotune/     # 实测记录的冻结与复用(地基 3)
  backend/        # 能算什么: 一库一包,INSTALL 元数据 + available() 检测 + adapter
    flashinfer/  deepgemm/  flashmla/  deepep/   # adapter 按 ABC 契约签名写
    availability.py # 真 import 检测 + 安装配方(BackendInstall),survey() 打印
```

**刻意不做的事**:不建 `impls/` 这类中间目录,也不写转发适配器。KernelSpec 的 `target` 是 `"module:attr"` 字符串,直接指向真实 kernel 函数,所以 `modules/attention.py` 里读到的仍是 `flash_attention2_no_pad` / `flash_decoding` 这种一眼可辨的算子名,而不是某个包装层。代价是 kernel 的公开签名必须干净——例如 `flash_attention2_no_pad` 原先要求调用方自己乘 `log2(e)`,这个 kernel 私有约定已下沉到 wrapper 内部,契约统一成 plain `1/sqrt(d)`。

迁移已落地:实现本体随算子域进 `ops/<group>/`,native 行与外部后端的行同写在各组 `__init__.py`,`benchmarks/kernels/` 下两个 legacy attention 文件已删。

**④ 一份元数据清单(声明式,集中一处,torch-free)**:借 sglang `KernelSpec`——注册时只存 `target="module:attr"` 字符串,**惰性 import**,注册全程不加载 torch/kernel(保证冷启动秒级、CPU 机器可 `import`)。每份实现声明:

| 字段 | 作用 | 借鉴 |
| --- | --- | --- |
| `available()` | import 检测 | sglang 惰性 load |
| `capability`(device + SM 窗口,OR 语义) | 硬过滤,如 DeepGEMM `>=sm90` | sglang `CapabilityRequirement` |
| `dtypes` / `scheme` | 支持的精度/量化方案 | 合并现有量化 method |

| `shape`(hard 约束 + prefer 偏好) | 过滤 + 排序 | 本项目新增 |
| `layout`(输入/输出布局要求) | dispatcher 决定转换或排除 | 防抽象泄漏 |

| `golden`(verified + max_abs_diff) | 未过对齐门禁不进默认分发 | 挂钩 acc.align |

| `perf_key`(gpu, op, shape_bucket, dtype) | 指向冻结的实测记录 | 本项目新增,见支柱⑤ |

**⑤ 一条确定性分发规则(sglang 的确定性 + 实测排序)**:

```
select(op, key=(arch, dtype, shape_bucket)) -> impl:
  1. 过滤:available ∧ capability匹配 ∧ dtype支持 ∧ shape.hard满足 ∧ layout可获得 ∧ golden.verified
  2. 排序:显式 backend= 覆盖  >  该 key 的冻结实测最优  >  shape.prefer/priority
  3. 取 top;native 保底行保证非空
  4. 结果按 key 缓存(lru_cache)+ 记录决策链供 explain
```

**确定性来源**:第 2 步的"实测最优"来自 autotune/profiling **预先冻结**的记录(存盘,见地基 3),不是运行时现测——同一 key 永远选同一实现,benchmark/golden/bug 全部可复现。这正是对 sglang selector(多后端可用时甩给用户显式指定)的超越:**rapid_llm 用实测记录自动选最快,选完仍确定**。

### 两个必须做对的地方

1. **不造两个注册表**:项目已有 `models/quantization/methods/` 按 scheme 注册。若再建一个按 (arch,shape) 的 kernel 注册表,`linear` 会被两套机制管辖。正解:**scheme 只是 dispatch key 的一维**,量化 method 就是 `linear` 在某 dtype 下的一份实现,统一进同一张清单。重构时就合并。
2. **layout 转换必须显式、可缓存、可被 explain 看到**:实现要求的 layout 与输入不符时,dispatcher 要么插入声明过的转换(权重转置只做一次并缓存),要么排除该实现,绝不允许实现内部偷偷假设 layout。

### 配套设施(直接复用 sglang 思路)

- **强制后端开关**:对标 sglang `SGLANG_FORCE_FUSED_OP_BACKEND`,二分数值 bug 时把整模型钉到 native。两级粒度,越窄越优先——`backend=` 参数 > per-op `RAPID_LLM_<OP>_BACKEND`(如 `RAPID_LLM_ATTENTION_DECODE_BACKEND`) > 全局 `RAPID_LLM_FORCE_BACKEND`。per-op 才是实际需要的粒度:一台机器可能想让 attention 走 flashinfer 而 linear 留在 native Triton GEMM。
- **调用 trace**(对标 sglang `enable_fused_op_trace`):记录每次调用的 (op, backend, shape/dtype),**直接产出 ops-collector(地基 3 的 collect 阶段)要的真实 shape 清单**。
- 外部后端各自占一个 `kernels/backend/<backend>/` 包(注册行在 `ops/<group>/__init__.py`,包内是 adapter + INSTALL 元数据),永不进核心依赖。**检测用真 import 而非 `find_spec`**:这些库是编译扩展或 JIT,"目录在"与"这张卡上能加载"是两个问题,dispatch 只关心后者(`backend/availability.py`)。
- **安装方式不是一句话**:四个后端里只有 flashinfer 是 wheel(`rapid-llm[flashinfer]`),DeepGEMM / FlashMLA / DeepEP 是带 submodule 的源码编译(DeepEP 还要先装 NVSHMEM)。所以后三者**不给 extra**——给一个装不了东西的 extra 比不给更误导——安装配方作为数据写在各自包的 `INSTALL` 里,由 `survey()` 打出来。

### 落地顺序

先只做 `linear` 一个算子打通"注册→过滤→排序→explain"全链路(纯 Python,不需 GPU 就能验证 dispatch 与测试骨架),再 attention → MoE → 通信。

## 地基 3:自动调优与配置持久化(T1+T2)

这是你 comment 6 的正式方案:

| 阶段 | 做什么 |
| --- | --- |
| 采集 | T2 shape 采集器跑一遍真实负载,导出 shape 清单(含出现频次) |
| 搜索 | warm-up 或离线对高频 shape 搜索 tile/num_warps/num_stages |
| 落盘 | 按 `(gpu_name, op, shape_key, dtype)` 存 JSON 到用户缓存目录 + 可选提交进仓库 |
| 复用 | 启动时按 key 命中即用,未命中回退启发式并可后台补搜 |
| 门禁 | T4 看门狗对比历史最优,劣化报警 |

覆盖对象:`fused_moe`(`_launch_config` 替换)、`flashattention2_nopad`(把注释掉的 144 组配置启用为搜索空间)、量化 GEMM(`_launch_config`)。

**当前覆盖度(2026-09 实测校正)**:「量化 GEMM」这一项目前只做到五分之一——dense 路径的五个量化 kernel 里只有 `w4a16_matmul` 会查 `ConfigStore`,fp8 W8A8、fp8/int8 W8A16、NVFP4 都是无条件算 launch config,搜到的配置无消费者。`benchmarks/kernels/bench_quant_gemm.py --tune` 对这四个如实报「no consumer」而不写死没人读的条目;要补齐,得先给它们加上和 `w4a16.py` 同样的 `get_best_config` 查询点。详见 `docs/quantization.md` 的「w4a16 中的第二个 tile 启发式缺陷」节。

# 四、并行与服务能力路线

依赖关系(箭头 = 前置):

```
多进程引擎(地基 0)─┬─→ TP 持续批处理 ──→ 在线服务×TP(AsyncLLMEngine)
                    └─→ DP 常驻副本 ──→ P10 cache-aware 路由
真分页 KV ──┬─→ chunked prefill ──→ PD 分离
            └─→ prefix caching
通信原语补全 ─┬─→ EP ──→ 专家负载均衡(EPLB 族)
              ├─→ DCP
              └─→ CP/PCP
```

当前通信层只有 `all_reduce`(SUM) 和 `tensor_model_parallel_all_reduce_min`,**缺 all_gather / reduce_scatter / all_to_all / P2P send-recv**——这是 EP/DCP/CP 的共同前置。

| 能力 | 前置 | 2×GPU 可实测? | 优先级 | 说明 |
| --- | --- | --- | --- | --- |
| **TP 持续批处理** | 地基 0 Executor 抽象 | 可(TP=2) | **高** | 当前 TP 只有 CLI 镜像进程+一次性批处理,服务路径完全不支持 TP(见地基 0 问题 1/2) |
| chunked prefill | 分页 KV + token budget 调度 | 可 | **高** | varlen attention 你已有,主要改调度和 KV 部分写 |
| prefix caching | 分页 KV + block hash | 可 | **高** | 多轮对话/共享 system prompt 收益直观 |
| EP | all_to_all | 可(EP=2,已验证) | 中高 | **已有(v0.11.5,提前于 v0.13)**:`--enable-expert-parallel` 让 TP 组兼作 EP 组,routed experts 按整专家切分(intermediate 不再 TP 切),等分 padding `all_to_all` dispatch/combine;DeepSeek-V2/V3/V4 与 qwen3_moe 共享 `SparseMoeBlock` 即自动可用。首版限非量化 + eager |
| DCP | all_gather + LSE 校正 | 可(DCP=2) | 中 | 长上下文 decode 扩容;MLA 场景价值最大 |
| CP / PCP | P2P ring 或 zigzag 切分 + LSE 合并 | 可 | 中 | 降 prefill 延迟 |
| **PD 分离** | chunked prefill + KV 传输 connector | **可(1P1D)** | 中 | 2 卡就能演示,不需要更多卡 |
| 专家负载均衡 | EP + 专家负载统计 | 可 | 低 | 冗余专家 + 重平衡;需先确认 EPLB 规格 |
| **DP 负载均衡策略族** | DP 调度器 + 负载/缓存水位上报 | 可 | **高** | SGLang 四策略之上加 cache-aware 路由(见下方小节) |
| **DP attention** | 通信原语补全 + MLA | 可 | 中 | attention 按请求维 DP、免 TP all-reduce,MoE 走 EP;MLA latent KV 小、DP 复制代价低(见第九节) |

关键判断:**全部能力在 2×GPU(如 A10)上都至少能做到"机制正确 + 小规模可测"**,且设计不绑定特定 GPU 型号——在 H100/B200 上同样可用,只是收益规模随互联带宽和算力而变。chunked prefill / prefix caching 的收益在任何 GPU 上都是真实的。

## DP 负载均衡策略(P10)

SGLang `DataParallelController.LoadBalanceMethod` 有四种:round-robin / follow-bootstrap-room / 最小总请求数 / 最小总 token 数。rapid_llm 在此之上加自有的 **cache-aware 路由**:请求 prefix hash 先查各副本的 prefix cache 命中估计,路由到"命中块数最多"的副本——共享 system prompt 的多轮对话天然聚到同 rank,命中率与负载均衡双收。实现依赖两个前置:地基 1 的 block hash;各副本负载/缓存水位经 observe.metrics 上报给路由层(第十节 Router 是它的多机推广)。

## 并行 module 补齐:QKVParallelLinear / VocabParallelEmbedding / ParallelLMHead + 分布式采样(A11)

**现状(v0.7.0)**:`models/base.py` 注释明确"vocab 张量保持复制"——embed 是全量 `nn.Embedding`,lm_head 是全量 `nn.Parameter`,`forward` 末尾 `F.linear(hidden, lm_head_weight)` 输出全量 logits,采样吃完整 logits。该决策按小 vocab 估(省 ~0.3GB/rank、代价每步 2 个 collective);对 Qwen3(151K vocab×8192 hidden)不成立:**embed+lm_head ≈4.9GB/rank(fp16),TP=4 每 rank 省 3.7GB;decode 每步 lm_head GEMM = batch×vocab×hidden 是大 vocab 模型 decode 算力大头,切分直接 /tp**。tied embedding 模型(Qwen3 系)切 lm_head 必须同切 embed——**tie 一致性是切 embedding 的真正理由**,顺带省显存。同节的另一个缺口:`attention.py` 里 q_proj、kv_proj 是**两个独立 ColumnParallelLinear**,decode 每步 attention 投影是两次 GEMM,缺 vLLM 式 `QKVParallelLinear` 的合体切分。

**设计**:类名对齐 vLLM `model_executor/layers/vocab_parallel_embedding.py`;weight 布局独立于 quant(复用 quant_method 组合,`_check_shard_alignment` 已保证 scale 网格不被切坏)。

- `QKVParallelLinear`:q_proj + kv_proj 合为一个 `qkv_proj` 参数(输出 `[q_total + 2*kv_total]`),一次 GEMM 替代两次;q 段按 `num_q_heads/tp`、kv 段按 `num_kv_heads/tp` **各自对齐**(GQA 两段 head 边界独立),kv 段是 k+v 连续存放,local 输出 = `2*(num_kv_heads/tp)*head_dim`,k/v 边界不因切分破坏;
- `VocabParallelEmbedding`:weight `[local_vocab, hidden]`,forward = `F.embedding` + `tensor_model_parallel_all_reduce`(查表结果 hidden 维规约);
- `ParallelLMHead`:weight `[local_vocab, hidden]`,forward 输出**局部 logits** `[*, local_vocab]`,不 gather——采样层直接消费局部 logits。

**自有设计(对比 vLLM 的关键差异)**

1. **去中心化 log_softmax(标量规约,非 gather)**:`log_softmax(x)_i = x_i - logsumexp(x)`。每 rank 只 all_reduce 一个标量 `logsumexp(local_logits)`(SUM),局部 logits 减之即得**数学等价的完整分布 logprobs**——带宽 O(1);vLLM 需 gather 全量/局部 logits,O(batch×vocab)。推理主路径 logits 永不物化。
2. **局部 top-k + 小张量 gather**:采样时每 rank 局部 top-k → all_gather 各 rank 的 `(token_id, logprob)` 对(O(k×tp),k=请求的 logprobs 数)→ 全局 argmax/multinomial。**gather 量由 API 的 logprobs 参数决定,与 vocab 无关**。
3. **tie 共享局部切片**:tie_word_embeddings 时 embed 与 lm_head 是**同一局部参数**;`weights.py` 的 tied 机制从"全量别名"升级为"局部切片别名"。加载路径零新代码:`_SHARD_DIM` 表用 incoming-tensor 表述切分(现有架构),embed/lm_head 各加一条 vocab 维规则,覆盖校验/量化 scale 全复用。
4. **collective 记账可视化**:切分后每步固定 +2 collective(embed 侧 all_reduce、采样侧标量规约+top-k gather);viz.structure 把 collective 作为一等节点显示(local/full shape + collective 位置 + 每步计数);CUDA Graph(P8)capture 时与主 all-reduce 同组锁步。
5. **q|kv 两段 fused(非 vLLM 的 q|k|v 三段)**:rapid_llm 本就 fused k/v(`weights.py _FUSED_KV`),QKV 合体后仍是两段——`_FUSED_KV` 映射直接复用(HF k_proj/v_proj → qkv_proj 的 kv 段),`_SHARD_DIM` 从"整参一条规则"升级为**段级规则**(q/kv 段各自 head 边界),"每参数恰好覆盖一次"校验不变,不引入 vLLM 的 weight_loader 多切片机制。dense MLP 的 gate/up 同族(MergedColumnParallelLinear,MoE gate_up 已 stacked)列低优先。

**验收**

- TP=2 采样 logprobs 与 TP=1 逐元素一致(机器精度内);
- TP=2 下 fused qkv 单次 GEMM 与 q/kv 两次 GEMM 逐元素一致;GQA 下 q/kv 段 head 边界独立对齐;
- Qwen3 TP=2 加载后每 rank embed+lm_head 显存减半;tied 模型 `embed.weight is lm_head.weight`;
- decode 步 lm_head 耗时 /tp(时间测量或 Nsight 佐证);
- 单测:TP=1/2 随机权重等价性、非整除 vocab 报错、量化 scale 切分对齐、局部 top-k gather 正确性;viz 输出每个并行 module 的 local/full shape。

落点:**v0.8.0**(与地基 0 同属并行重构;分布式采样与地基 0 的"采样结果随广播走"协同)。

# 五、DeepSeek 单层 harness(你 comment 3 的正式方案)

这是我认为**投入产出比最高的一个新增工具**,它一次解决四件事。

设计:

- 输入:HF config + 层索引 + 可选真实权重(或随机初始化)
- 只实例化一层,跑 forward,支持 prefill/decode 两种输入形态
- 输出:输出张量 vs HF 同层的 max-abs-diff、逐算子延迟、峰值显存、选中的后端
- 同一 harness 可跑 vLLM 的对应单层做横向对比

价值:

| 用途 | 说明 |
| --- | --- |
| MLA 正确性验证 | 不需要 671B 权重,随机权重也能验证结构与数值路径 |
| DSA / Lightning Indexer 验证 | MQA logits + TopK 稀疏选择的机制正确性 |
| 性能对比 | 单层 fp8 ≈ 11.5GB,单张 A10(24GB)可跑,vs vLLM 同层可比;H100/B200 更充裕 |
| 品化为 F1 亮点 | 任意模型任意层的通用调试工具,这是 vLLM 没有的 |

前置:必须先拆薄 `models/base.py` 的 `Attention`(当前硬编码 fused-KV + RoPE + RMSNorm,MLA 接不进来)。这和地基 2 是同一次改造。

补充建议:用 **DeepSeek-V2-Lite(16B,MLA 完整)** 跑整模型验证端到端正确性,再用单层 harness 打 V3/V4。两条路互补——一个验证全链路,一个验证前沿架构。

# 、计算-通信 overlap 设计

不同 GPU 互联拓扑决定 overlap 的收益量级:**PCIe 互联的 GPU(如 A10/A100 PCIe 版)之间无 NVLink**,TP 的 all-reduce 走 PCIe、通信占比高,"把通信藏进计算"是真实大头收益;**NVLink 互联的 GPU(如 H100/H200/B200)**带宽充裕、通信占比低,但仍有调度气泡可藏,且 L4 tile-signaling 在 NVLink 上收益全开。以下设计面向两种拓扑,由 `capability` 字段自适应。

## 三条 ping-pong 流水轴

o rlap 不止"TP 通信藏进计算"一件事。把所有重叠拆成三条正交的 ping-pong 轴,每条轴独立度量、独立验证,组合后逼近零气泡:

| 轴 | 流水两端 | 粒度 | 目标 | 对应层级 |
| --- | --- | --- | --- | --- |
| **A. Host-Device** | CPU 调度 ↔ GPU 执行 | **batch 级** | CPU 在 GPU 跑 batch i 时即调度 batch i+1,CPU 侧零等待 | P9(第八节) |
|**B. Memo-Compute** | HBM 读写 ↔ tensor core 计算 | **tile 级** | 算完一块 tile 即可发射下一块的 load/store,访存与计算流水 | L4 |
| **C. Compute-Comm** | 计算 kernel ↔ 通信 kernel | **batch/micro-batch 级** | A 块做 all-reduce 时 B 块的 GEMM 已在算,通信藏进计算 | L2/L3/L5 |

三条轴由易到难,分五级落地:

| 级 | 轴 | 技术 | 借鉴 | 你的设计 | PCIe GPU | NVLink GPU |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | C | 跨 stream 算子重叠(shared/routed expert 并行;下层权重 H2D 预取) | — | `torch.cuda.Stream`+event,零 kernel 改动,先验证调度抽象 | 稳定收益 | 稳定收益 |
| L2 | C | TP 粗粒度 ping-pong:token 切两半,A 做 all-reduce 时 compute 算 B | TokenWeave | 做成 batch 超阈值才启用的 policy | 就有效 | 收益较小 |
| L3 | C | TP 细粒度分解:GEMM 输出切 chunk,算完一块即发 reduce-scatter | Google 分解 | chunk 流水,气泡更小 | 可 | 可 |
| L4 | B | **tile-signaling 原语**(核心创新):Triton GEMM 每算完一个 tile 置 flag,消费者自旋消费,访存与计算细到单 tile | Flux(靠 CUTLASS C++) | 用 `tl.atomic`+共享 flag buffer 在**纯 Triton**里实现 tile 级信号量,可读可教学 | 机制可验证;收益有限 | 收益全开 |
| L5 | C | MoE all-to-all 重叠:token 切 micro-batch,dispatch i+1 与 expert 算 i 重叠 | DualPipe / SGLang batch_overlap | **已有(v0.11.5)**:EP dispatch/combine 各拆成 a/b 两段,`operations_strategy` 把两 micro-batch 的 a2a 与对侧计算交错(decode delta_stages=2) | 可(EP=2,已验证) | 可(EP=2) |

> **轴 A(Host-Device)** 单独成节(第八节),因为它改的是引擎架构而非 kernel,层级最高。

## DP/TP 与 CUDA Graph 同时生效(P8,已落地)

难点从来不是捕获,而是证明各 rank 结构对称:captured graph 里的 all-reduce 要求每个 rank 每一步选同一个 graph,一个 rank 走 eager 而对端 replay,后果是**挂死**而非答案不同。vLLM 的做法是直接禁用。

**两层各自的落法:**

1. **TP + CUDA Graph**:捕获前 `warmup_collectives()` 把 NCCL 通信器的惰性初始化挪出捕获区;捕获后各 rank 用 `tensor_model_parallel_ranks_agree()` 比对 `crc32` 网格指纹(指纹 `0`=什么都没捕获,即使全体一致也判失败),再各自跑一遍 graph-vs-eager 的 logit 比对并以 min 规约合并——任一 rank 超差则全体退回 eager。replay 判定只读广播来的 `ModelInput`,故不需要每步协商。详见 `docs/tensor_parallel.md` 的「TP × CUDA Graph」节。
2. **DP + CUDA Graph**:DP 副本互不通信,各自 capture 各自 replay,随常驻引擎循环一同生效。

**实测**(H100×2,Qwen2.5-0.5B,`tests/distributed/test_tp_cuda_graph.py`):bf16 / fp8 / nvfp4 三方案各捕获 8 个 graph,graph 与 eager 的最坏 logit 差 **0.000e+00**,32 步 greedy 逐字节同答。`RAPID_LLM_TP_CUDA_GRAPH=0` 是 kill-switch,`RAPID_LLM_TP_GRAPH_CHECK=1` 每步校验两 rank 选中同一 graph。

## 支撑抽象与验证

**overlap 调度器** —— 每层声明 compute/comm/copy 的 DAG,调度器分 stream、插 event、选重叠策略,整体进 CUDA graph(P5+P8),并由 explain 打印每层用了哪种重叠、气泡在哪。这是 TileRT 运行时的"可读+可解释缩小版",既是架构亮点也是调试利器。

**验证**:Nsight timeline 看三条轴是否真重叠 + overlap on/off 对照 + 纳入 perf.watchdog 和 observe.overlap(模块 F)。L4 边界诚实标注:PCIe GPU 收益不如 NVLink GPU,但"纯 Triton 实现 Flux 式细粒度融合"本身是招牌叙事,不依赖特定 GPU 大数字。单元测试:每个 overlap policy 有 on/off 对照的延迟断言(不是"能跑",是"气泡减少 X%")。

# 七、KV 内存管理与传输

把三件相关的事收拢到一个抽象下:动态管理、分层存储、PD 分离——它们本质都是"KV block 在不同存储层间搜运"。

**三层能力**:

| 能力 | 现状 | 归属 | 说明 |
| --- | --- | --- | --- |
| 分配粒度(block table) | ROADMAP 已有 | 地基 1 | block_size>1,不再预占 max_seq_len |
| 请求级 alloc/free + watermark 准入 | 缺 | 地基 1 补充 | 取代 `free_all` 全量重置 |
| 抢占(recompute / swap-out) | 缺 | v0.9 调度 | 缺块时把低优先序列踢回 waiting |

**统一的 `KVTransfer` 抽象**(新增,架构核心):GPU↔CPU↔磁盘↔远端节点 的 block 搬运用同一套接口。它有两个 consumer:

- **分层存储**(新增,挂 A2 KV 布局可插拔):CPU 层 / 磁盘层作为 KV 的 tier provider,和 paged layout 同一注册机制。在 2×GPU(如 2×A10=48GB 或 2×H100=160GB)上,它的正当理由不是"扩 KV 容量"(中等规模模型 KV 不是瓶颈),而是 **prefix cache 溢出到 CPU/磁盘**——多轮对话/共享 system prompt 场景把冷前缀换出、热的留 GPU,提命中率。对齐 vLLM 的 kv_offload 子系统(CPU 主层 + 磁盘二级层,次级层不直访 GPU、经主层中转)。
- **PD 分离**(已有 v0.12):远端节点作为一个 tier;prefill→decode 的 KV 传输和 offload 的 GPU↔CPU 搬运是同一件事的不同拓扑。对齐 TileRT 的 KVConnector 模式。

关键决策:`KVTransfer` 必须先于 PD 建好,因为分层存储和 PD 共用它。落点:请求级回收 + watermark 在 **v0.7**;抢占在 **v0.9**;`KVTransfer` 抽象 + 分层存储 + PD 在 **v0.12**(共用传输层)。

# 八、引擎级异步调度架构(P9)

> 对应第六节 ping-pong **轴 A(Host-Device)**。这是改动最大的一条:把调度器从主线程拆出去,让 CPU 和 GPU 各跑各的。**前置是地基 0**:没有 Executor 抽象和进程边界,本节的架构图落不了地;反过来本节是地基 0 的异步化终态——EngineCore 从"同进程可选拆"升级为"独立进程 + ZMQ + N-batch 流水线"。

## 现状与动机

当前 `AsyncLLMEngine` 是单 worker 线程 + `queue.SimpleQueue`,同步调用 `engine.step()`——每步 CPU 等 GPU 返回后才调度下一步,CPU-GPU 串行,气泡明显。vLLM v1 的 `EngineCoreProc` 和 SGLang 的 `zmq_to_scheduler` 已证明:调度器拆成独立进程、ZMQ 通信,可让 CPU 调度与 GPU 执行重叠。

## 架构设计

```
┌───────────────┐     ZMQ (req in)     ┌─────────────────┐     SchedulerOutput queue   ┌──────────────┐
│  Frontend     │ ──────────────────▶  │  Scheduler      │ ─────────────────────────▶  │  Executor    │
│  (API/CLI)    │ ◀──────────────────  │  (独立进程)       │ ◀───────────────────────── │  (GPU worker)│
└───────────────┘     ZMQ (output)     └─────────────────┘     ResultEvent queue       └──────────────┘
```

**四个组件**:

| 组件 | 职责 | 借鉴 | 自有设计 |
| --- | --- | --- | --- |
| **Frontend** | 收 HTTP/CLI 请求,序列化后发给 Scheduler | vLLM `AsyncLLM` | 复用现有 `AsyncLLMEngine` 的 asyncio 层,只换 IPC 后端 |
| **Scheduler** | 独立进程;跑 `Scheduler.schedule()`,产出 `SchedulerOutput`;不碰 GPU | vLLM `EngineCoreProc`;SGLang `zmq_to_scheduler` | **N-batch 流水线**:允许多个 `SchedulerOutput` 同时在 queue 里未被执行,不是严格 1:1 的 request-response |
| **Executor** | GPU worker;消费 `SchedulerOutput`,跑 `model_runner.forward()`,采样,回传 result | vLLM `gpu_worker` | **双缓冲 slot**:batch i 在 GPU 跑时,batch i+1 的 input tensor 已在另一块预分配 buffer 里 ready |
| **Output Router** | 按 request_id 把 result 路由回对应 Frontend stream | vLLM output queue | 复用现有 `_RequestStream` 的 `call_soon_threadsafe` 机制 |

## N-batch 流水线(核心创新)

vLLM/SGLang 的调度器与执行器是严格 1:1 的 request-response——调度一步、执行一步、回传一步。rapid_llm 的原创设计是**流水线化**:

| 时刻 | CPU(Scheduler) | GPU(Executor) |
| --- | --- | --- |
| t0 | 调度 batch 0 | idle |
| t1 | 调度 batch 1 | 执行 batch 0 |
| t2 | 调度 batch 2 | 执行 batch 1 |
| ... | ... | ... |

最多 N 个 batch 同时在流水线上飞(N 由 `max_num_batched_tokens` 和 slot 双缓冲数决定)。这要求:

1. **双缓冲 KV slot**:batch i 写的 KV 行和 batch i+1 读的 KV 行不能是同一组——否则 decode 步的 `update_kv_index` 会覆写正在被 forward 读的行。
2. **无锁 result 回传**:Executor 产出 result 后通过 ZMQ push 回 Scheduler/Frontend,不阻塞下一个 batch 的调度。
3. **反压(backpressure)**:当流水线满(N 个 batch 在飞)时,Scheduler 阻塞在 push 上,自然反压 Frontend。这比 vLLM 的显式 `can_schedule()` 检查更简洁。

## ZMQ 通信细节

- **请求通道**:Frontend → Scheduler,`ZMQ_PUSH` / `ZMQ_PULL`,序列化 `(request_id, prompt_token_ids, sampling_params)`。
- **调度输出通道**:Scheduler → Executor,`ZMQ_PUSH` / `ZMQ_PULL`,序列化 `SchedulerOutput`(prefill/decode 列表 + chunk_lens)。
- **结果通道**:Executor → Frontend(经 Scheduler 或直连),`ZMQ_PUB` / `ZMQ_SUB` 按 request_id 路由,或 `ZMQ_PUSH` / `ZMQ_PULL` + Frontend 侧 demux。
- ZMQ socket 的 `recv`/`send` 释放 GIL,天然可与 asyncio 事件循环共存(vLLM 已验证)。

## 落地顺序

1. 先把现有 `AsyncLLMEngine` 的 `queue.SimpleQueue` 换成 ZMQ,Scheduler 仍在同进程内但独立线程——验证 IPC 正确性。
2. 再拆 Scheduler 到独立进程——验证 N-batch 流水线和双缓冲。
3. 最后接 `observe.overlap`(模块 F)实时监控 CPU/GPU 两条流水线的气泡。

单元测试:mock GPU 执行时间,断言 N=2 时 CPU 等待时间 < N=1 时的 50%;mock Scheduler 延迟,断言 GPU 空闲率下降。

# 九、前沿注意力变体(A8)

> 对标 vLLM `MLAAttention` + `SlidingWindowMLASpec`。设计原则:所有注意力变体作为 `attention.*` 逻辑算子的不同实现,走地基 2 的统一 dispatch,新增变体只注册不写新类。

## 四种注意力变体

| 变体 | 全称 | 代表模型 | 核心机制 | vLLM 对标 |
| --- | --- | --- | --- | --- |
| **MLA** | Multi-head Latent Attention | DeepSeek-V2/V3/V4 | 低秩压缩 KV:存 latent `c_kv` 而非完整 K/V;decode 时 on-the-fly 上采成 K/V;KV cache 只占 `L_kv` 维而非 `N × P` | `MLAAttention` + `MLAAttentionSpec` |
| **DSA** | Dynamic Sparse Attention | DeepSeek-V3/V4 | Top-K 稀疏选择:latent KV 经 `MLA` 粗筛后,只取 top-K 相关行做细粒度 attention;Lightning Indexer 加速 top-K | `MLAAttentionSpec(model_version=4)` + `sparse_swa.py` |
| **SWA** | Sliding Window Attention | Mistral / Qwen3 部分 | 固定窗口 W:只 attend 最近 W 个 token;与 full attention 混用(底层 SWA + 顶层 full) | `SlidingWindowMLASpec` |
| **HCA** | Hybrid Chunk Attention | InfLLM / 长上下文变体 | 分块层次化:近距 chunk 做 full attention,远距 chunk 经 chunk-level 聚合后做 sparse attention | vLLM 暂无独立类;自有设计 |

## Attention 后端矩阵

地基 2 的 dispatch 在 attention 上的具体展开——后端 × 变体的二维矩阵,每个组合一份实现,新增后端/变体只加表项不写新类:

| 后端 | prefill | decode | 变体覆盖 | 前置条件 |
| --- | --- | --- | --- | --- |
| Triton FA2(自有,现 `flash_attention2_nopad` / `flash_decoding`) | ✓ | ✓ | GQA / SWA | 无,永远存在的保底行 |
| Triton MLA(自有) | ✓ chunked | ✓ MQA 路径 | MLA | 无 |
| FlashAttention-3(external) | ✓ | ✓ | GQA / SWA | sm90+;import 失败自动回退 Triton |
| FlashInfer(external) | ✓ | ✓ | GQA / MLA | 库存在 + capability 过滤 |
| DSA indexer(自有) | — | ✓ | DSA | MLA 先行 |

对标 vLLM `AttentionBackend` 枚举(FLASH_ATTN / FLASH_ATTN_3 / FLASHINFER / TRITON / MLA 等,按 platform 选默认);差异:vLLM 每平台各写一份适配层,这里 native Triton 是全局保底行,external 后端只在 capability 命中时替换对应表项。

## 统一接口设计

所有注意力变体共享 `attention.prefill` / `attention.decode` 两个逻辑算子的 ABC 签名(见地基 2),差异只在实现层:

```python
# kernels/ops/attention.py
class AttentionOp(ABC):
    @abstractmethod
    def forward(self, q, k, v, metadata: AttentionMetadata, layer_idx: int) -> Tensor: ...

# kernels/ops/attention/ 下的实现本体,由该组 __init__.py 的注册行指名
class FlashAttn2Prefill(AttentionOp): ...    # 现 flash_attention2_nopad
class FlashDecoding(AttentionOp): ...        # 现 flash_decoding
class MLADecode(AttentionOp): ...            # MLA decode 路径
class MLAPrefill(AttentionOp): ...           # MLA chunked prefill
class SlidingWindowDecode(AttentionOp): ...  # SWA decode
class HybridChunkPrefill(AttentionOp): ...    # HCA prefill
```

Dispatcher 按 `(arch, model_type, attention_variant, shape_bucket, dtype)` 自动选实现——MLA 模型走 `MLADecode`,Mistral 走 `SlidingWindowDecode`,传统 GQA 走 `FlashDecoding`。

## MLA 实现要点

对标 vLLM `MLAAttention` 的两条路径(compute-friendly vs data-movement-friendly):

- **compute-friendly**(prefill):latent `c_kv` 上采成完整 K/V 后做 MHA,headdim = P + R,访存多但算力利用率高。
- **data-movement-friendly**(decode):直接用 latent `c_kv` 做 MQA,headdim = L_kv + R,访存少。decode 场景访存是瓶颈,走这条。
- **chunked prefill**:当 `Skv` 过大时,把 context 按 workspace `W` 分块,逐块上采+attention+merge LSE,避免 `k_nope = (kv_c @ W_UK).view(Skv, N, P)` OOM。

KV cache 格式:MLA 存的是 latent `c_kv`(`[Skv, L_kv]`)而非完整 K/V(`[Skv, N, P]`),所以 `PagedAttention` 的 cache 写入路径需要分叉——这正好是地基 2 的 layout 元数据要解决的:MLA 实现声明 `layout="latent_kv"`,dispatcher 在 cache 写入时选择不同的 `kv_write` 实现。

## DP attention(MLA 的并行组合技)

SGLang `enable_dp_attention` 已验证的 DeepSeek 风格并行:attention 部分按请求维 DP——每个 rank 持有自己请求的**完整 KV**(不按 head 切分),attention 全程零 TP all-reduce;MoE 部分跨 rank 聚合走 EP。层间需要 all_gather(进 MoE)/ 分发(回 attention),正好是第四节"通信原语补全"的直接 consumer。

对 MLA 格外划算:latent KV 只有 `L_kv` 维(如 512,对比完整 K/V 的 `N × 2 × P`),DP 各存一份的代价小;换来 attention 免通信、prefill 不做 TP 切分。rapid_llm 落点:并行策略进 dispatch key(`(arch, model_type, attention_variant, shape_bucket, dtype)` 加一维 `parallel="tp|dp"`),与 A8 统一接口是同一次改造;依赖本节 MLA 先行。

## DSA 实现要点

DSA 是在 MLA 基础上加稀疏选择:decode 时不扫全部 `Skv` 行,而是用 latent `c_kv` 做粗粒度打分(MQA over latent),取 top-K 行做细粒度 attention。Lightning Indexer 是 vLLM 对这个 top-K 选择的优化(预排序 + block-level 索引)。rapid_llm 的实现路径:

1. 先做 MLA 正确性(DeepSeek-V2-Lite 端到端验证)。
2. 再加 DSA 的 top-K 选择逻辑(单层 harness 验证)。
3. top-K 选择用 Triton kernel 实现,复用 `tl.argsort` + `tl.gather`。

## SWA + HCA 实现要点

- **SWA**:`flash_decoding` 的 grid 加一个 `window_size` 参数,start 取 `max(0, seq_len - window)`。与 full attention 混用时,模型层声明 per-layer 的 attention type,dispatcher 逐层选不同实现。
- **HCA**:近距 chunk 走 full prefill kernel,远距 chunk 经 chunk-level 聚合(取每个 chunk 的代表向量)后做 sparse attention。这是自有设计,需要在 ABC 签名里加 `chunk_size` 元数据。

# 十、服务化与运维

可观测(A7 / 工具模块 F)是引擎内视角;本节补服务外视角——rapid_llm 要能当生产服务跑,不只是研究框架。

| 能力 | 做什么 | 借鉴 | 自有设计 |
| --- | --- | --- | --- |
| **Prometheus metrics** | `/metrics` 端点暴露 TTFT / TPOT / 队列深度 / KV 占用 / 后端选中率 / overlap 气泡 | vLLM metrics(SRE 导向的标准计数器);SGLang prometheus 输出 | 复用 observe.metrics 的 per-step 数据流;metric 命名与 vLLM 对齐(社区 Grafana 面板直接可用),额外暴露 rapid_llm 特有的后端/算子维度 |
| **请求追踪** | 请求全链路 span:enqueue → 调度决策 → forward → 采样 → detokenize → 返回 | OpenTelemetry;vLLM request-level tracing | observe.trace 直接导出 OTLP;trace_id 贯穿 ZMQ 进程边界(第八节跨进程透传) |
| **Router** | 多实例入口:负载均衡 / 健康检查 / 故障转移 / prefix 感知路由 | SGLang router(Rust 实现);vLLM 生产栈 router(k8s gateway) | **路由决策吃 metrics 反馈**:Router 从各实例 /metrics 拉负载与 cache 水位,把第四节的 cache-aware 路由从引擎内 DP 推广到集群级;纯 Python 单进程实现(不引 Rust 依赖),定位教学级生产化 |

落点:Prometheus + 追踪在 **v0.10**(observe.metrics/trace 同版本,顺手开 /metrics 端点);Router 在 **v0.12**(多实例 PD/DP 就绪后才有意义)。

# 十一、版本计划

条目沿用前文的标号:F / A / P 系列见"框架差异化亮点"一节,地基 0-3 见第三节,L1-L5 与 ping-pong 轴 A/B/C 见第六节。

每个版本都有一项固定交付物:该版本支持的全部模型跑一遍性能(TTFT / TPOT / TPS)与精度(golden 逐 token + 数据集分数)benchmark,与上一版同配置结果并列,归档为 `docs/release-vX.Y.Z.md`,原始数据落 `docs/benchmark_logs/*.json`。这部分基线不在各版本重复,各版本的 **benchmark** 只写该版本**新增的对照维度**——新做的东西必须给出"比什么快了/退了多少"的对照,否则不算做完。

执行路径是工具而非人工:采集入口统一挂到 Makefile 的 `bench-*` 目标(`benchmarks/bench_*.py` + `examples/benchmark.py`),`perf.watchdog` 与上一版最优比对并标出劣化项,`acc.golden` 出精度门禁、`acc.divergence` 定位精度断层,最后由一个 release-bench agent 按固定模板汇总成发版文档。人只审两处:被标红的劣化项,和结论段的因果解释。工具链自身在 v0.4(watchdog 入库)与 v0.5(autotune collect)成型,所以 v0.4 那份报告的作用是立零点,没有上一版可比。

当前状态:v0.4 - v0.11.0 已发版(发版文档与原始数据如上所述);v0.11.5 起未动。

## v0.4 可信基线(已发版)

- **fix**
  - TP 采样 RNG 不同步
  - AWQ/GPTQ:修通,或撤下对外的支持声明
- **test**
  - acc.golden 升为强制门禁:上 GPU runner、禁止静默 skip、覆盖面扩到 continuous / 量化 / VL / DP
  - perf.watchdog:benchmark 入库,劣化超阈值报警
- **benchmark**
  - 立零点:全部在库模型跑 `bench_e2e` 与 `bench_hf_baseline`,同 prompt 同口径,作为后续版本的对照基准
  - 量化路径(fp8 / W8A16 / W4A16 / SmoothQuant)单列一张表,性能与精度成对给出
- **验收**
  - 无卡时 CI 明确报"未验证",不再判绿
  - TP=2 采样不 diverge

## v0.5 自动化调优(已发版)

- **feat**
  - autotune 三阶段(collect / search / persist)接入 `fused_moe`、`flashattention2_nopad`、量化 GEMM
  - P1 megakernel 雏形:先融 RMSNorm + QKV
- **refactor**
  - w4a16 用 `tl.dot` 重写
- **benchmark**
  - 调优命中与未命中同 shape 对照,按算子分列,说明搜到的配置比启发式好在哪
  - w4a16 重写前后、megakernel 融合前后:TPOT + kernel launch 次数
- **验收**
  - 高频 shape 的调优配置落盘,启动命中即用
  - w4a16 给出重写前后的对比数字

## v0.6 分页 KV(已发版)

- **feat**
  - 地基 1 落地:真分页 KV,block_size > 1 且不再按 max_seq_len 预占
  - 请求级 alloc/free + watermark 准入,取代 `free_all` 全量重置
  - viz.flow:调度决策、slot 分配、抢占、prefill/decode 切换
  - viz.structure 先出文本树,viz.memory 先出静态预算表
- **refactor**
  - KV 布局做成可插拔:新旧两种布局并存,并给出迁移路径
- **benchmark**
  - 新旧 KV 布局同配置对照:TPS、可容纳并发数、峰值显存
  - watermark 准入下的显存水位曲线与请求拒绝率
- **验收**
  - 两种布局的 golden 都绿
  - decode 路径无 D2H 同步
  - 请求结束即回收其 block,不等全批结束
  - 能导出结构树与显存预算表

## v0.7 调度能力(已发版)

- **feat**
  - chunked prefill + prefix caching
  - 抢占:缺块时把低优先序列踢回 waiting,recompute 与 swap-out 两条路径
  - P5 overlap 进 CUDA graph
  - MTP 接入
- **refactor**
  - 调度 policy 化,两个引擎循环合并为一份
- **benchmark**
  - chunked prefill on/off 在长短 prompt 混合负载下的 TTFT-TPOT 权衡曲线
  - prefix caching:命中率随共享前缀比例的变化,以及命中带来的 TTFT 降幅
  - 抢占触发时的 P99 尾延迟,recompute 与 swap-out 分别计
- **验收**
  - 长 prompt 期间 decode 不停顿
  - 共享前缀命中率可观测
  - 缺块时能抢占而非拒绝请求

## v0.8.0 多进程隔离引擎 + 并行修复与 module 补齐(地基 0,已发版)

- **fix**(并行 bug 修复包:问题 4/5/6,小改不动架构)
  - DP×TP 死锁:改为一次 spawn 出 dp×tp 进程网格
  - `tensor_model_parallel_all_reduce_min` 取错 device index:`_TP_RANK` 换成 `torch.cuda.current_device()`
  - DP 路由的 token 数估算:从字符数改为真实 token 数,balancer 命名对齐 SGLang 语义
- **refactor**(引擎重构)
  - Executor 抽象:UniProc / Multiproc 两份实现,对标 vLLM `v1/executor/`
  - SchedulerOutput 经 gloo 控制面广播,镜像进程模式退场
  - DP worker 从一次性 `generate()` 换成常驻引擎循环
  - 进程网格一次 spawn(dp×tp),不再分层拉起
- **feat**
  - TP 接入连续批处理 `Scheduler`
  - `AsyncLLMEngine` 支持 TP
  - 并行 module 补齐(A11):`QKVParallelLinear`(q 与 kv 两段 fused 成一次 GEMM,按 head 边界切)、`VocabParallelEmbedding` / `ParallelLMHead` 按 vocab 切分、去中心化 log_softmax 分布式采样(标量 logsumexp 规约 + 局部 top-k gather,详见第四节)
- **benchmark**
  - TP=1/2 扩展效率,并给出 all-reduce 在单 step 中的占比
  - 单进程 executor 与多进程 executor 的单 step 主机侧开销差(单卡不该因重构变慢)
  - DP×TP 2×2 聚合吞吐与副本间负载倾斜度
- **验收**
  - bug 修复:TP=2 采样不 diverge;DP×TP 2×2 能起能推理,不再死锁
  - 引擎:TP=2 下 continuous golden 全绿;在线服务 × TP=2 可跑;单卡默认仍走单进程(pdb 能断点),mock 进程网格断言各 rank 收到的 SchedulerOutput 一致
  - module:TP=2 的采样 logprob 与 TP=1 逐元素一致;embed + lm_head 显存减半

## v0.9 多后端 + overlap 骨架(进行中)

- **feat(已完成)**
  - 地基 2:实际落地形态超出本版"雏形"目标,直接按三层建满
    - `kernels/ops/` 按算子域分组(gemm / attention / moe / layernorm / rope / activation / sampling / kvcache / embeddings / quantization),每组 `__init__.py` 持有该算子全部注册行,共 11 个算子 21 行
    - `kernels/dispatcher/` 是 torch-free 机制层:KernelSpec 六维声明(available / capability / dtypes+schemes / shape 硬约束+偏好 / layout / golden)+ 注册表 + dispatch 四步(filter → rank → cache → report)。explain 打印逐条拒绝理由与落选者排名,`RAPID_LLM_KERNEL_TRACE=1` 输出 JSON 决策线
    - `kernels/backend/` 一库一包(flashinfer / deepgemm / flashmla / deepep)+ available() 真 import 检测:缺库是排名事件,不是崩溃
    - golden gate:未验证(verified=False)的行默认不参与 dispatch,只有显式 `backend=`(参数或 `RAPID_LLM_*_BACKEND` 环境变量)可越过;flashinfer 的 attention / rmsnorm / rope / sample 行已带 max-abs-diff 记录
    - 默认全 native:外部行 priority=UNMEASURED(-1) 排在 native(0) 之下,翻盘等 v0.10 冻结实测数据接线
  - A9 Platform 抽象:设备检测 + 能力声明(CapabilityRequirement),dispatch 按 capability 过滤(deepgemm / flashmla 的 sm90+ 窗口在 A10 上被拒),可 mock 测试
  - prefix caching 支持 DP:负载均衡按前缀亲和路由,各副本的 cache 合成一个池
- **refactor(已完成)**
  - attention 接口拆薄:`PagedAttention` 下沉到 modules/(KV 写入 + prefill/decode 分派),`models/base.py` 的 Attention 只管投影与 RoPE;dispatch 在构造期一次决策,热路径是普通属性调用,MLA 才接得进来
- **feat(剩余)**
  - overlap 调度器抽象 + L1 跨 stream 重叠
- **顺带提前入库**(归属后继版本,在此记账)
  - v0.11 的 MLA 算子侧:`MinimalMlaLayer` 单层 harness + flashmla 后端行(golden 未验证,默认不 dispatch)
  - v0.13 的 FlashInfer attention 后端行(prefill + decode 两行,golden 已验证)
- **benchmark**
  - (已完成)同 shape 下 native 与 flashinfer 逐一对照(`bench_flashinfer` 等已入库),静态 priority 顺序与实测顺序的出入,即是 v0.10 换成实测排序的依据
  - (已完成)L1 跨 stream 重叠 on/off 的端到端差值,附 timeline 佐证
- **验收**
  - (已达成)一条命令切后端,并能解释为何选它
  - (已达成)缺库时自动回退到 native
  - L1 重叠有 timeline 作佐证
  - (已达成)Platform 可 mock 测试

## v0.10.0 可观测性 + 算子分发(已发版)

- **feat**
  - 地基 2 收尾:原计划"从雏形升级",但声明式 KernelSpec 清单、确定性 dispatch、registry 雏形与 `gen_backend_registry_gif.py` 的退场已随 v0.9 提前完成,本版只补两件
    - 冻结实测排序接线:autotune store 经 `set_perf_provider` 接进 dispatch 的 rank 步,排序依据从静态 priority 换成预先冻结的实测记录——同一 key 永远选同一实现,外部后端实测更快时才翻盘
    - ABC 签名:评估给全部实现统一 `forward()` 语义(现以 target 字符串 + 各 ops 组注释钉死的调用契约替代);签名统一是"kernel 函数本体直接交给调用方、不必写转发适配器"的前提
  - F6 logprobs / prompt_logprobs
  - A7 运行时可观测性(observe.metrics + observe.trace)+ `/metrics` Prometheus 端点 + OTLP 追踪导出
- **test**
  - F1 单层 harness:`rapid_llm/tools/harness/` + `scripts/layer_harness.py`——meta 骨架只给指定层材料化存储,权重可来自 checkpoint(只读该层的 key)、transformers 同层镜像或随机;prefill 与 decode 两形态各自对 HF 同层比 max-abs-diff,同时给逐模块 CUDA event 计时、峰值显存与 dispatch 决策(MLA 侧 `MinimalMlaLayer` 继续作 benchmark 载体)
- **benchmark**
  - (已达成)dispatch 开销:`benchmarks/kernels/bench_dispatch.py`——A10 上首次决策 761 ms(一次性,全在后端检测的 import 上)、换 key 后的 filter+rank 27 us、命中缓存 15 us(其中真正的缓存查找 0.5 us,余下是每次重取的平台快照);调用点在构造期决策一次并存成属性,每步 forward 连这次查找都不做。冻结实测排序相对 v0.9 静态 priority 的端到端差值在噪声内(见下),因为这张 GPU 上实测赢家与静态顺序的首选一致——排序换成实测的意义是"外部后端真快时才翻盘",不是无条件提速
  - (已达成)logprobs / prompt_logprobs 开启后的额外开销:`benchmarks/bench_observability.py`——Qwen3-0.6B batch16 gen128,TPOT 4.75 -> 5.35 ms、吞吐 -10.4%(logprobs=5);prompt_logprobs=5 只压在 prefill 上,TTFT 23 -> 32 ms 而吞吐 -1.5%;两个都开 -12.7%。默认关闭,按请求 opt-in
  - (已达成)metrics 与 trace 全开时的性能损耗:同上脚本,两者都低于基线自身的 0.5% 抖动,无需降采样
- **验收**
  - (已达成)logprobs 与 HF 对齐:`tests/golden/test_logprob_parity.py` 对 transformers teacher-forced 的 log_softmax 逐位置比对,一次性与 chunked prefill 两条采集路径各一遍
  - (已达成)每 step 给出 per-request 的延迟分解(queue / TTFT / TPOT 三段)
  - (已达成)dispatcher 按 shape / dtype 选实现,并能 explain 决策链(v0.9 已达成,此处作回归项)
  - (已达成)Prometheus 能抓到标准 metric

## v0.11.0 前沿架构 + 结构化输出(已发版)

- **feat**
  - (已达成)MLA:DeepSeek-V2-Lite 端到端(TP=2)——kv_lora_rank 512 + rope 64 的 latent cache 全链路(投影/吸收/分页池/decode kernel);偏差:V3/V4 单层未做,V2 架构已覆盖 MLA 全部核心路径,V3/V4 差异集中在 router 与权重组织,随下轮
  - (偏差:F7 结构化输出明确移出本轮,见发版文档范围裁剪;grammar bitmask 接口位置已预留)
  - (已达成)F8 reasoning parser / tool parser:`engine/reasoning.py`(think 标签流式状态机,后缀窗口处理跨 delta 标签)+ `engine/tool_parser.py`(DeepSeek/Qwen 双族标记集,JSON 字符级扫描器),protocol 层补 `reasoning_content`/`tool_calls`,解析器按**请求**声明
  - (偏差:SWA / HCA / DSA 明确移出本轮,相关接口预留)
  - (已达成)通信原语补全:all_gather / reduce_scatter / all_to_all / P2P send-recv,gloo 后端双进程数值正确性入测
  - (已达成)TP 引擎释放对称化:MultiprocExecutor.shutdown 补销毁 rank-0 侧进程组——拥有 followers 即拥有 group;同进程先 TP=2 后 TP=1 的切换挂死(bench_mla 实测踩到)的问题由此根除,golden/探针的手动规避序列随之删除,主进程回归测试反向验证过(撤销修复即失败)
- **test**
  - (已达成)新增算子 / module / 模型的精度与性能测试:MLA 模块单测 + DeepSeek-V2-Lite TP=2 golden(5 用例,drift 预算门禁)+ 通信原语双进程测试;全量回归 1359 passed
  - (已达成)**acc.divergence**:整模型对 HF 逐层对比不变,定位从"第一个超阈层"升为"逐层 diff + 扰动注入 + 预算式门禁"——BOS 热点排查证明单层超阈可能只是 ULP 算术假热点,预算式门禁比硬阈值更诚实
- **benchmark**
  - (已达成)MLA 模型首份报告:`benchmarks/bench_mla.py`——KV 几何从各自 config.json 解析(V2-Lite latent 576 vs 未压缩 5120 vs Qwen3-1.7B GQA 2048 elements/token/层),同负载实测 TTFT/TPOT/峰值显存/KV 池容量;TP 下 latent 复制(每 rank 全量)的口径在报告里标明
  - (已达成,部分)reasoning parser 开启后的开销:`benchmarks/bench_parser.py`——reasoning +0.07 us/token、reasoning+tools +1.17 us/token(相对本版实测 decode TPOT 21.7–63.0 ms 占比 0.002%–0.005%,噪声内);bitmask 侧随 F7 移出
  - (偏差:SWA 对照随 SWA 移出)
- **验收**
  - (已达成)HF 单层 max-abs-diff 达阈值:golden TP=2 逐 token + parity probe 校准(prompt drift mean 0.046/max 0.298,decode 步 mean 0.221/max 1.967,greedy match 86%)
  - (偏差:JSON Schema 100% 合法随 F7 移出)
  - (已达成)reasoning_content / tool_calls 与 vLLM 对齐:双族标记集对齐 vLLM deepseek/qwen parser,流式分块==一次性解析的公理在 parser 层与 server 层各验证一遍
  - (偏差:SWA 混用随 SWA 移出)

## v0.11.5 计算通信重叠

- **feat**
  - B 轴(Memory-Compute)与 C 轴(Compute-Comm)落地:L2 粗粒度 ping-pong + L3 GEMM 输出分解 + L4 tile-signaling 原语(A 轴归 P9,在 v0.12)
  - **EP + L5(提前于 v0.12/v0.13)**:EP 组状态(`--enable-expert-parallel`,TP 组兼作 EP 组)+ `SparseMoeBlock` 整专家切分与 `expert_map` 权重加载 + `modules/ep_dispatch.py` 等分 padding `all_to_all` dispatcher;L5 把 MoE forward 拆成 sglang 式 op 流(gate/dispatch_a/shared/dispatch_b/experts/combine_a/combine_b),`TwoBatchOverlap` 改由 `operations_strategy` 驱动,两 micro-batch 的 a2a 与对侧计算交错
  - **refactor**:`executor/tbo.py` 与 `executor/comm_overlap.py` 迁进独立包 `batch_overlap/`(对齐 sglang `srt/batch_overlap/`),新增 `operations_strategy.py`;`executor/overlap.py` 只留 host↔device 轴(L1)
  - P8 DP + CUDA Graph 先行:DP 副本互不通信,各自 capture 各自 replay,无锁步约束
  - 完整支持 deepseekv2 v3 v4 模型，跑多层 layers 可以跑通，性能提速，精度对齐 transformersvllm（确保覆盖到deepseekv2/v3/v4 所有算子和 module）
- **docs**
  - Nsight 对照报告:确认三条轴是否真重叠
- **benchmark**
  - L2 / L3 / L4 逐级叠加的收益分解,PCIe 与 NVLink 两种拓扑分列(L4 在 PCIe 上收益有限,如实标注)
  - DP + CUDA Graph 的 decode TPOT 与 capture 耗时、显存增量
  - EP=2 与 TP MoE 同卡数对照 + L5 op 流 on/off 的 decode TPOT(收益分解待 profiling;等分 padding 的浪费上界为每层 ep_size-1 行,decode 小 batch 可接受)
- **验收**
  - 每个 overlap policy 有 on/off 对照数据,断言气泡减少幅度而非只断言"能跑"
  - DP + CUDA Graph 下 decode 无退化
  - EP=2 权重按 `expert_map` 分区加载正确、forward 与全专家参照数值等价(bf16 容差);EP+TBO op 流与非 TBO EP 路径输出一致(集合通信顺序不交叉错配);端到端 DeepSeek-V2-Lite TP2+EP2(含 TBO)与 TP2 greedy 在 tie-gap 容差内一致

## v0.12.0 异步调度 + KV 传输

- **feat**
  - P9 引擎级异步调度(地基 0 的异步化终态,即 A 轴):EngineCore 独立进程 + ZMQ + N-batch 流水线
  - P10 DP 负载均衡策略族:round-robin / 最少请求数 / 最少 token 数 / cache-aware
  - DP attention:MLA + EP 组合
  - Router 雏形:metrics 反馈路由 + 健康检查
  - 分层存储:CPU 层与磁盘层作为 KV 的 tier provider,目标是 prefix cache 溢出后仍命中,而非扩 KV 容量
  - PD 分离 1P1D
  - CP / PCP
  - L5 MoE all-to-all 重叠
  - 专家负载均衡
- **refactor**
  - `KVTransfer` 统一抽象:`GPU ↔ CPU`(主层)、`CPU ↔ 磁盘`(二级层,经主层中转)、`GPU ↔ 远端 GPU`(PD 拓扑)三条路径收进同一套接口。本版内它要先于分层存储与 PD 落地,那两者都建在它上面
- **benchmark**
  - 1P1D 与同卡数单实例对照:TTFT、TPS、KV 传输在 TTFT 中的占比
  - N-batch 流水线 N=1/2 的 CPU 等待时长与 GPU 空闲比
  - 四种 DP 路由策略在倾斜负载下的副本利用率,cache-aware 单独看命中率增益
  - prefix 溢出到 CPU / 磁盘后的命中率与换回延迟
- **验收**
  - 1P1D 端到端可演示
  - N=2 时 CPU 等待 < N=1 的 50%
  - DP 副本负载倾斜 < 10%
  - 前缀溢出到 CPU 后命中率不降

## v0.13.0 前沿注意力 + TP Graph

- **feat**
  - P8 TP + CUDA Graph:所有 rank 锁步 capture 与 replay,保证 collective 调用顺序一致
  - DSA indexer:top-K 选择下沉为 Triton kernel(`tl.argsort` + `tl.gather`)
  - HCA:近距 chunk 走 full prefill,远距 chunk 经 chunk 级聚合后做 sparse attention
  - attention external 后端接入:FA3(sm90+)/ FlashInfer(后者已随 v0.9 提前入库)
  - EP(EP=2)
  - DCP(DCP=2)
  - perf.timeline + viz.schedule
- **benchmark**
  - TP + CUDA Graph on/off 的 decode TPOT,补上 v0.8 起一直缺的这块
  - external attention 后端与原生 Triton 实现在多档 seq_len / batch 上的对照
  - EP=2 与 TP MoE 同卡数对照;DCP=2 在长上下文 decode 上的扩展效率
- **验收**
  - TP + CUDA Graph 下 all-reduce 不挂死
  - DSA top-K 选择正确
  - external 后端的 capability 命中与回退两条路径均可测
  - EP 可跑并有数据

## v0.14.0 投机解码全家桶(P4)

- **feat**
  - ngram(prompt lookup)先行:不需 draft 模型,用最简路径验证 verify 链路
  - MTP
  - DFlash:借 target lm_head 的无 head draft
  - DSpark
  - 在线接受率自适应调 draft 长度
- **refactor**
  - 统一 draft-verify 接口,按模型形态自动选策略
- **benchmark**
  - 四种 draft 方式的 mean accept length 与端到端加速比,verify 开销单独计,并按负载类型(重复前缀 / 代码 / 开放问答)分列
  - 接受率自适应开启前后的对照,含 draft 长度的实际分布
- **验收**
  - 重复前缀负载下 ngram 接受长度 ≥ 2
  - MTP mean accept 有报告
  - verify 正确性 golden 全绿

## v0.15.0 收敛

- **chore**
  - 公开 API 稳定下来
- **test**
  - 全量回归:golden、精度门禁、服务端 API 三套一次跑绿
- **docs**
  - 文档站
- **benchmark**
  - 全矩阵收口:模型 × 并行配置 × 后端 三维覆盖,并与 vLLM / SGLang 在同机同配置下对照
  - 汇总 v0.4 到 v1.0 的性能演进曲线,逐版标注收益来自哪项改动
- **验收**
  - 公开 API 语义稳定
