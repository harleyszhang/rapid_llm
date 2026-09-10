# Release v0.11.0 — MLA 端到端 + 流式 reasoning/tool 解析

**Date:** 2026-09-02 **Branch:** `dev-v0.11` **Theme:** DeepSeek-V2-Lite 的 latent KV 从 config 到 decode kernel 跑通并纳入 golden 门禁；思考内容与工具调用变为**按请求声明**的协议层能力

## Summary

v0.11 交付两条独立主线：KV 显存占用最小的注意力架构 MLA 端到端跑通；思考过程与工具调用两类输出从 content 通道流式分离。

**MLA 端到端（commit b43013c）：** `DeepseekV2Model`（MLA + DeepSeek-MoE）注册进 registry，KV cache 行从「每头两列」泛化为 `(dim,)` 元组——MLA 每 token 只缓存一条 576 维 latent（512 lora + 64 rope），TP 下按 vLLM 的口径在每 rank 复制而非切分。精度验证不依赖人工比对：`tests/golden/test_deepseek_v2_tp2.py` 在 2×A10 上对 transformers 逐 token 比对，门禁不是硬阈值而是从 parity probe 实测校准的 drift 预算——**BOS 排查证明单层 max-abs 超阈可能只是 ULP 算术假热点**（BOS 的 MoE 输出范数是普通 token 的千倍，同相对误差落在 [1024,2048) 桶恰好差 1 个 bf16 ULP=8.0），预算式门禁比「首个超阈层即失败」的硬阈值更贴合实测误差分布。

**acc.divergence（同 commit）：**不止定位第一个超阈层，还做逐层 diff、扰动注入自证、预算式门禁。CLI 挂 `rapid-llm acc divergence`，`tools/accuracy/divergence.py` 568 行。

**F8 流式解析（本版收尾 commit）：** `engine/reasoning.py` + `engine/tool_parser.py` 把 think 标签与 DeepSeek/Qwen 双族 tool 标记从增量 detokenizer 文本里流式拆出。与 vLLM 的差异在两处：解析器**按请求声明**（vLLM/SGLang 都是服务启动时单选一个，一个部署只能服务一种模型），以及流式分块与一次性解析的等价性是**不变式**——parser 层穷举全部 two-cut 切分，server 层再验证流式帧拼接等于一次性 message。

**地基（散在多个 commit）：** bf16 从 checkpoint dtype 驱动而非散落硬编码（commit 63e8616 等）；完整 YaRN（beta_fast/beta_slow + mscale）；通信原语补全 all_gather / reduce_scatter / all_to_all / P2P send-recv 并统一去掉 `_tp` 后缀（commit 38e044a），gloo 后端双进程数值正确性入测。

## Feature

### MLA：DeepSeek-V2-Lite 端到端（commit b43013c）

```bash
# 双卡 TP=2 起服务
rapid-llm serve --model-dir my_weight/DeepSeek-V2-Lite --tensor-parallel-size 2

# golden 门禁（2×A10，需 checkpoint 在 my_weight/DeepSeek-V2-Lite）
pytest tests/golden/test_deepseek_v2_tp2.py -q        # 5 passed
```

三个取舍：

- **KV 行泛化为 `(dim,)` 元组，不改 KVCacheManager 公共路径。** MLA 每 token 一条 latent 行、GQA 每 token 每头一对 K/V，都被同一个「每层一个 (tokens, dim) 池」表达；MLA 的写入路径独立，失败可回退——这是计划里写明的侵入性约束。
- **TP 下 latent 复制，不切分。** latent 是 single-KV-head，切了就没有 rank 能独立算注意力；每 rank 持全量 latent 与 vLLM 口径一致。代价是 KV 池容量不随 rank 数翻倍，`bench_mla` 报告中每张涉及容量的表都标注了这条口径。
- **门禁是预算，不是阈值。** 实测校准（parity probe）：prompt 位置 drift mean 0.046 / max 0.298，decode 步 mean 0.221 / max 1.967，greedy match rate 86%（0.1 以内的 near-tie 计为 match）。预算取约 2× 分离度：mean ≤ 0.12/0.5、max ≤ 0.7/4.0、match ≥ 75%。router 是 fp32 upcast GEMM + fp32 softmax，两侧（lite 与 transformers 5.x）同构——「HF 是 bf16 router」是 4.x 时代的结论，5.x 已不成立。

### acc.divergence（commit b43013c）

```bash
# 对 HF 逐层 diff，指认第一个发散层
rapid-llm acc divergence --model-dir my_weight/DeepSeek-V2-Lite \
    --prompt "Explain KV caches." --reference transformers

# 扰动注入自证：给指定层注入已知扰动，工具必须指认回同一层
rapid-llm acc divergence --inject layer=7,scale=1e-2 ...
```

### F8 reasoning / tool parser（收尾 commit）

```bash
curl localhost:8000/v1/chat/completions -d '{
  "model": "deepseek-v2-lite", "stream": true,
  "messages": [{"role": "user", "content": "Tokyo weather?"}],
  "reasoning_parser": "deepseek_r1",
  "tool_parser": "deepseek"
}'
```

请求带 `"reasoning_parser": "deepseek_r1"`，流式帧里 `delta.reasoning_content` 先输出、`delta.content` 在 think 块闭合后接续；带 `"tool_parser": "deepseek"`（或 `"qwen"`），`delta.tool_calls` 按调用 index 流式合并，`finish_reason` 变为 `"tool_calls"`。两个开关彼此独立、可组合，未知值在 schema 层 422。

设计取舍三条：

- **解析器按请求声明，不是服务级配置。** vLLM 的 `--reasoning-parser` / `--tool-call-parser` 是启动期单选，一个部署只能服务一种输出形态；这里的 `ChatCompletionRequest` 把它变成请求字段，同一部署可混服务 R1 式与直出式模型。代价是每个请求多一次 Literal 校验——微基准见下。
- **流式 == 一次性是穷举验证的不变式。** `ReasoningSplitter` 的后缀窗口（hold 可能补全标签的最长后缀）与 `_JsonCallScanner` 的字符级扫描（字符串感知括号计数、键序无关、支持 args-first）共同保证：任意切分的 feed+finish 拼接等于 parse 整段。parser 层对每个 fixture 穷举全部 two-cut 切分，server 层再断言流式帧三通道拼接等于一次性 message 的三通道。
- **finish_reason 独立成帧，截断时如实报 length。** 流式 chat 的终止原因不再附加在最后内容帧上，而是 OpenAI 官方形态的空 delta 帧——因为 parser 的 flush（截断的 tool call、held 的部分标签）必须在它之前发完，客户端在 finish_reason 处停止读取。截断的调用（max_tokens 切在 JSON 半中间）仍然上报已到的碎片，但 `finish_reason` 保持 `"length"` 而非 `"tool_calls"`。

### 地基修正

- **bf16 由 checkpoint dtype 驱动**（commit 63e8616 等）：`moe.py` gate_weight 等散落的 fp16 硬编码清到 `config.dtype` 一处。
- **YaRN 完整实现**：beta_fast/beta_slow 分段 + mscale 缩放，名称与实现对齐。
- **通信原语补全 + `_tp` 后缀统一去掉**（commit 38e044a）：all_gather / reduce_scatter / all_to_all / P2P send-recv 齐备，gloo 后端双进程数值正确性挂在 `tests/distributed/`（tp_harness 起真进程组）。
- **TP 引擎释放对称化**：`MultiprocExecutor.shutdown` 此前只 reap follower 进程，rank-0 侧的进程组留在一个活进程里——下一个同进程引擎会读到没人要求的 TP 尺寸（`bench_mla` 的 TP=2→TP=1 切换挂死就是这么来的，golden 测试和探针脚本一直在手动规避）。现在拥有 followers 即拥有 group，两半一起销毁；DP cell leader（空 followers）不动 coordinator 的 grid。回归测试在主进程起真 TP=2 引擎再断言 world of one，golden 的手动规避序列随之删除——它现在守卫的就是这个修复。

## Benchmark

全部为 2×A10（23 GB，PCIe 互联）实测，greedy，batch=8，gen=128。

### MLA 首份报告（`benchmarks/bench_mla.py`）

数据口径先说明：V2-Lite 是 16B MoE（激活 2.4B），没有「同尺寸」dense 模型可对照，所以 KV 列从各自 config.json 解析、延迟列跑同一份负载，两组数字并列呈现，不宣称可比。TP=2 decode 走 eager（NCCL 集合通信不进 graph 捕获），TP=1 走 CUDA graph——执行路径差异如实标注。

| | DeepSeek-V2-Lite TP=2（eager decode） | Qwen3-1.7B TP=1（CUDA graph） |
| --- | --- | --- |
| TTFT（batch=8 prefill） | 64.8 ms | 22.1 ms |
| TPOT | 63.01 ms（p50 62.35） | 21.72 ms（p50 21.70） |
| 吞吐 | 126.9 tok/s | 368.3 tok/s |
| 权重显存 | 29.32 GiB（两卡合计） | 3.78 GiB |
| KV 池容量 | 161137 tokens | 145363 tokens |
| rank-0 进程峰值显存 | 19.41 GiB | 19.33 GiB |

两个引擎都在同一张卡上把 KV 池用到 profiling 预留的 90% 上限（V2-Lite 池 4.79 GiB、Qwen3 池 15.55 GiB，权重较小者池更大），所以容量数字本身不可比；可比的是密度：**每 GiB KV 显存换 33.6k tokens（V2-Lite latent） vs 9.3k（Qwen3 GQA）**，3.6 倍差距就是 config 解析出的 30.4 vs 112.0 KiB/token 在实测池上的直接验证。TP=2 的 eager decode 是惩罚项不是收益项，如实标注。

KV 几何（每 token 每层 elements，从 config.json 解析）：V2-Lite MLA latent **576**（512 lora + 64 rope），同一架构不压缩需 **5120**（16 头 × (128 nope + 64 rope + 128 v)），Qwen3-1.7B GQA **2048**（2 × 8 heads × 128）。latent 口径下 TP=2 的池容量即全模型容量（latent 复制，不切分）。

日志：[`docs/benchmark_logs/kernels/mla_v0.11.json`](benchmark_logs/kernels/mla_v0.11.json)。

### F8 parser 开销（`benchmarks/bench_parser.py`）

| 配置 | 每 token 解析成本 | 相对基线增量 |
| ------ | ----------------- | ------------- |
| off（裸循环） | 0.04 µs | — |
| reasoning | 0.11 µs | +0.07 µs |
| reasoning + tools | 1.21 µs | +1.17 µs |

语料是 think 块 + 正文 + tool 调用段的混合流，按 detokenizer 尺度（~4 字符）切增量，标签跨块边界是常态而非特例。增量 1.17 µs 相对本版实测 decode TPOT（21.7–63.0 ms，见上表）占比 **0.002%–0.005%**——远低于噪声下限，符合「纯 Python 字符处理不该被感知」的预期；如实报数，不四舍五入成「零开销」。

日志：[`docs/benchmark_logs/kernels/parser_v0.11.json`](benchmark_logs/kernels/parser_v0.11.json)。

## 测试结果

```text
1359 passed, 82 skipped, 11 xfailed in 101s        全量 pytest（serve extra 安装后）
160 passed, 2 skipped in 140s                       tests/distributed/（2×A10，RAPID_LLM_TEST_MODEL_DIR=Qwen3-1.7B；
                                                   含 shutdown 释放回归测试，撤销修复即失败，已反向验证）
5 passed in 24s                                    tests/golden/test_deepseek_v2_tp2.py（2×A10 + V2-Lite）
25 passed                                          tests/engine/test_reasoning.py（含 two-cut 穷举不变式）
22 passed                                          tests/engine/test_tool_parser.py（含 two-cut 穷举不变式）
41 passed                                          tests/entrypoints/test_api_server.py（含流式==一次性 server 级不变式）
```

golden 的 UNVERIFIED 语义与上一版一致：checkpoint 缺失时 xfail 并写明原因，`RAPID_LLM_GOLDEN_STRICT=1` 时 fail——没有静默变绿。`test_deepseek_v2_tp2.py` 不带共享 `weights` 标记：那个标记绑定 Qwen2.5-0.5B 的共享 fixture，checkpoint 由文件内 fixture 管理，UNVERIFIED 守卫同样由它承担。

## 文件清单（相对 v0.10.0，只列主干）

| 操作 | 路径 |
| ------ | ------ |
| 新建 | `rapid_llm/models/deepseek_v2.py`、`rapid_llm/kernels/ops/attention/mla.py`（MLA prefill/decode 算子与参考实现） |
| 新建 | `rapid_llm/tools/accuracy/`（acc.divergence：PrefillCache/Checker/报告） |
| 新建 | `rapid_llm/engine/reasoning.py`、`rapid_llm/engine/tool_parser.py`（F8 双 parser） |
| 修改 | `rapid_llm/entrypoints/{protocol,api_server}.py`（reasoning_content / tool_calls / 请求级开关 / 独立 finish 帧） |
| 修改 | `rapid_llm/executor/{kv_cache_manager,model_runner}.py`（KV 行泛化为 `(dim,)` 元组） |
| 修改 | `rapid_llm/modules/{moe,mlp}.py`、`rapid_llm/models/config.py`（bf16 dtype 驱动、V2 配置面） |
| 新建 | `benchmarks/bench_mla.py`、`benchmarks/bench_parser.py` |
| 新建 | `tests/golden/test_deepseek_v2_tp2.py`、`tests/engine/test_{reasoning,tool_parser}.py`、`tests/tools/test_divergence.py` |
| 新建 | `scripts/dsv2_tp2_parity_probe.py`、`scripts/dsv2_layer_diff.py`（校准与排查探针）、`scripts/gen_reasoning_gif.py`（README gif，真实运行渲染） |
| 修改 | `rapid_llm/executor/executor.py`（shutdown 对称销毁 rank-0 group）、`tests/distributed/test_tp_engine.py`（对应回归测试） |
| 新建 | `docs/benchmark_logs/kernels/{mla,parser}_v0.11.json`、`docs/images/reasoning.gif` |

## Upgrade

```bash
git checkout dev-v0.11 && uv pip install -e .

# MLA 端到端（2 卡）
rapid-llm serve --model-dir my_weight/DeepSeek-V2-Lite --tensor-parallel-size 2

# 请求级解析开关（两个开关独立、可组合）
curl localhost:8000/v1/chat/completions -d '{
  "model": "m", "stream": true, "messages": [...],
  "reasoning_parser": "deepseek_r1", "tool_parser": "deepseek"
}'

# 复现本版 benchmark
python benchmarks/bench_mla.py --json docs/benchmark_logs/kernels/mla_v0.11.json
python benchmarks/bench_parser.py --json docs/benchmark_logs/kernels/parser_v0.11.json

# golden 门禁
pytest tests/golden/test_deepseek_v2_tp2.py -q
```

## 相关文档

- [ROADMAP](../ROADMAP.md)：v0.11 章的 feat / test / benchmark / 验收逐条状态与偏差记录
- [docs/release-v0.10.0.md](release-v0.10.0.md)：logprobs 与可观测性的上一版交付
- [docs/tensor_parallel.md](tensor_parallel.md)：TP 面的既有说明，latent 复制口径见本文 Benchmark 节
