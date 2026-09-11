# 精度评估

精度有两条独立验证路径，本文是两条的单一事实来源：

- **`benchmarks/eval/`**：多数据集对比基准，sglang `benchmark/` 的布局——每个数据集一个目录，`bench_rapid_vllm.py`（rapid_llm 引擎侧，CUDA graph 默认开启，即优化模式）和 `bench_hf.py`（transformers baseline 侧）跑**同一套提示词与判分**，rapid 与 HF 准确率之差只能来自引擎；
- **`tests/evals/`**：GSM8K 回归套件（`make test-eval`），按 `configs/*.yaml` 声明的阈值判定通过/失败，外加 golden token parity 这类逐 token 证据，是 CI 语义的"门"；基准只出数不判定。

## benchmarks/eval/：多数据集对比基准

```
benchmarks/eval/
├── _common.py            # 共享层：checkpoint 解析、HF 装载/生成、rapid 与 HF 的续写似然打分、证据落盘
├── gsm8k/                # 5-shot 数学链，最后一个整数精确匹配
│   ├── bench_rapid_vllm.py   # 委托 tests.evals.gsm8k.evaluate_gsm8k，与回归套件永不漂移
│   ├── bench_hf.py           # HF baseline，同一 build_prompts/score
│   ├── bench_vllm.py         # vLLM 对照（需另装 vllm 的环境）
│   ├── common.py → 覆盖在 tests/evals/gsm8k.py 里
│   └── README.md
├── boolq/                # 5-shot 是非阅读理解，True/False 精确匹配
│   ├── bench_rapid_vllm.py / bench_hf.py / common.py / README.md
└── hellaswag/            # 20-shot 四选一续写，按选项对数似然打分（两种判分口径）
    ├── bench_rapid_vllm.py / bench_hf.py / common.py / README.md
```

每个数据集的口径（few-shot 格式、解码预算、判分规则、与 sglang 的差异及理由）定义在各自的 `common.py` / README 里，rapid 与 HF 共享——这是"分差只能来自引擎"的结构保证。

### 实测环境与共同口径（2026-09-10）

NVIDIA A10（23 GB）× 1（GPU1），torch 2.13.0+cu129 / triton 3.7.1 / transformers 5.15 / Python 3.13（rapid_llm venv，rapid 与 HF 同一环境）。greedy；rapid 侧 CUDA graph 开启（引擎默认 = 优化模式，`--eager` 仅调试用）；HF 侧 `torch_dtype="auto"`，与引擎同 dtype——对比是 engine-vs-engine，不是 fp16-vs-bf16。题数一律 200（各验证集前 N 条，固定前缀）。checkpoint：Qwen2.5-0.5B 取 HF hub 缓存快照，其余取 `/data/shared/llm_weights`。

每行数字都有对应 JSON：`docs/benchmark_logs/accuracy/accuracy_{gsm8k,boolq,hellaswag}_{rapid,hf}_20260910_*.json`（26 个文件）。

### GSM8K（5-shot，≤256 token 数学推理链）

| 模型 | chat | rapid_llm | HF | Δ（rapid−HF） | 吞吐 rapid / HF（tok/s） |
| --- | :---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 否 | 33.50% | 37.00% | −3.5pp | 3040 / 1184 |
| Qwen3-0.6B | 否 | 38.50% | 39.50% | −1.0pp | 2352 / 458 |
| Qwen2.5-1.5B-Instruct | 是 | **64.50%** | 56.00% | +8.5pp | 816 / 499 |
| Qwen3-1.7B | 否 | 61.00% | 61.00% | 0 | 1047 / 347 |

rapid 与 HF 无效率全部为 0。吞吐对比有个必须说明的口径差：HF `generate` 没有停止串，模型不吐 EOS 就跑满 256 token（0.5B 侧 200 题共生成 51200 token，rapid 侧靠停止串只生成 30667），延迟口径对 HF 不公平，所以表里用每秒 token 数——该口径下引擎全面领先 1.6×~5.1×。

四组准确率 rapid 赢一平一输二，分差与生成长度的关系见「跨数据集结论」：±8.5pp 不是"谁对谁错"，是贪心长链对 bf16 kernel 数值差的敏感度。

### BoolQ（5-shot，≤6 token 是非题）

| 模型 | rapid_llm | 无效 | HF | 无效 | Δ | q/s rapid / HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 47.50% | 0.5% | 48.50% | 0.5% | −1.0pp | 54.4 / 29.1 |
| Qwen3-0.6B | 61.00% | 0% | 61.00% | 0% | 0 | 36.5 / 15.6 |
| Qwen2.5-1.5B-Instruct | **69.50%** | 0% | 66.00% | 1.5% | +3.5pp | 19.8 / 12.5 |
| Qwen3-1.7B | 68.00% | 0% | 68.00% | 0% | 0 | 16.9 / 10.1 |

BoolQ 每题只生成约 6 个 token 且带 `\n` 停止，rapid 与 HF 生成 token 数几乎一致，q/s 是公平口径。1.5B 与 1.7B 两组里 rapid/HF 各有一组逐题一致，1.5B 的 3.5pp 全部来自 5 道题翻面——生成长度短，分歧量级与 HellaSwag 的打分类接近。

**口径教训：BoolQ 不对 instruct 模型开 chat 模板。** 初版曾对 1.5B-Instruct rapid 与 HF 都包 chat 模板：rapid 59.50%（无效 2%），HF 48.00%（无效 **26.5%**）。26.5% 不是引擎 bug：chat 模板把 few-shot 示例包进 user turn，与模型的微调格式相冲，True/False 首 token 的 logit 差被压到边缘，两引擎的 bf16 数值差足以让选择翻面（HF fp32 单条与批量仲裁结果一致，翻面是真实输出而非 padding 伪影）。sglang 的原生口径本就不带 chat，按它对齐后无效归零、rapid 与 HF 回到 1.5pp 内。GSM8K 上 chat 模板是实打实的 +8.5pp 增益（回归套件一节有对照表），但那是把裸指令包成模型认识的格式；BoolQ 的 few-shot 格式自己就是提示工程，再包 chat 属于双重包装，两头不讨好。

### HellaSwag（20-shot，四选一续写似然打分）

每个选项单独 tokenize 后拼接（lm-eval-harness 的做法），rapid 与 HF 对每个选项打分的 token 序列逐位相同。两种判分都报：`acc` 是选项总 log-prob 取 argmax（lm-eval 口径），`acc_len` 是平均 log-prob 取 argmax（sglang `sgl.select` 默认的 token_length_normalized 口径）。

| 模型 | rapid acc / acc_len | HF acc / acc_len | Δacc | 延迟 rapid / HF |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 44.00% / 48.00% | 44.00% / 48.00% | 0 | 21.1s / 18.2s |
| Qwen3-0.6B | 44.50% / 46.50% | 45.00% / 48.00% | −0.5pp | 26.7s / 31.0s |
| Qwen2.5-1.5B-Instruct | **52.00% / 58.00%** | 52.00% / 58.00% | 0 | 44.0s / 46.8s |
| Qwen3-1.7B | 46.50% / 51.00% | 45.50% / 51.00% | +1.0pp | 50.3s / 58.4s |

8 组判分（4 模型 × 2 口径）里 5 组 rapid 与 HF 完全一致，其余差 1~2 题。两引擎的 per-token logprob 有真实的 bf16 级数值差（Qwen3-0.6B 冒烟实测：|Δ| 均值 0.039、最大 0.21，8 题里 7 题选项选择一致），但 argmax 判分对此不敏感。

性能上 rapid 与 HF 接近（0.86×~1.16×）：这类负载没有 decode，每题就是 4 次大 prefill，引擎的优势项（连续批处理、CUDA graph 重放 decode 步）用不上，prompt logprobs 的 top-k 物化还多一块开销。0.5B/0.6B 在 batch 16 下触发过一次显存分配器重试警告（4.3 GB 的 logits 网格，非致命），1.5B/1.7B 用 batch 8——README 里 `--batch-size` 的默认值就是这么定的。

### 跨数据集结论

三个数据集把"每题的生成/打分链条长度"拉开成三档，rapid 与 HF 分差随之单调放大：

| 数据集 | 每题链条 | rapid 与 HF 分差 | rapid 吞吐优势 |
| --- | --- | --- | --- |
| HellaSwag | 0 token（纯 prefill 打分） | 0 ~ 1pp | 无（0.86×~1.16×） |
| BoolQ | ≤6 token | 0 ~ 3.5pp | 1.6×~2.3×（q/s） |
| GSM8K | ≤256 token | 0 ~ 8.5pp | 1.6×~5.1×（tok/s） |

机制：greedy 长链是路径依赖的——某一步 top-1 与 top-2 的 logprob 差小于两引擎的 bf16 数值差时选择翻面，此后整条链走向不同。单步分歧概率很小，链条越长累积越多；似然打分把整条链压缩成一个 argmax，对数值差几乎免疫。所以 GSM8K 上的 ±8.5pp 不构成对任一引擎的裁决，判引擎对错要靠逐 token 证据（回归套件的 golden token parity、HellaSwag 的 per-token logprob 对照）。性能结论同样清晰：decode 占比越高引擎优势越大——这正是连续批处理加 CUDA graph 的作用面，也是对比基准坚持用优化模式跑的原因。

### 跑法

```bash
# rapid_llm 侧（CUDA graph 默认开启 = 优化模式；--eager 关闭仅用于调试）
python -m benchmarks.eval.gsm8k.bench_rapid_vllm --model-dir Qwen2.5-1.5B-Instruct --chat-template
python -m benchmarks.eval.boolq.bench_rapid_vllm --model-dir Qwen2.5-0.5B
python -m benchmarks.eval.hellaswag.bench_rapid_vllm --model-dir Qwen3-1.7B --batch-size 8

# HF baseline 侧（参数同构）
python -m benchmarks.eval.gsm8k.bench_hf --model-dir Qwen2.5-1.5B-Instruct --chat-template
python -m benchmarks.eval.boolq.bench_hf --model-dir Qwen2.5-0.5B
python -m benchmarks.eval.hellaswag.bench_hf --model-dir Qwen3-1.7B --batch-size 8
```

`--model-dir` 接受路径或名字（依次在当前路径、`RAPID_LLM_MODELZOO`（默认 `/data/shared/llm_weights`）、`my_weight/` 下查找）。数据集首次运行自动下载到 `~/.cache/rapid_llm/evals/`（GSM8K 4.7 MB、BoolQ 4 MB、HellaSwag 12 MB）。每次运行把完整结果追加到 `docs/benchmark_logs/accuracy/accuracy_{dataset}_{engine}_<时间戳>.json`。各数据集口径细节与 sglang 的逐条差异见各自目录的 README。

## tests/evals/：GSM8K 回归套件

对齐 vLLM `tests/evals/` 的口径：train split 前 5 条作示例，统一 `Question: ...\nAnswer:` 格式；greedy，关 `repetition_penalty` 与 `stop_on_repeat`；截断到下一个 `Question`；取 completion 最后一个整数与参考答案 `####` 后的数字精确匹配。`benchmarks/eval/gsm8k/bench_rapid_vllm.py` 的引擎侧直接调 `tests.evals.gsm8k.evaluate_gsm8k`，所以回归套件与对比基准的引擎侧数字永不漂移。

套件的设计与扩展方式见 [`tests/evals/README.md`](../tests/evals/README.md)。

### 回归基线（历史实测）

torch 2.13.0+cu129 / triton 3.7.1 / Python 3.13，fp16 权重，CUDA graph 开启：

| 模型 | 题数 | few-shot | chat | 准确率 | 无效率 | 耗时 (s) |
| --- | ---: | ---: | :---: | ---: | ---: | ---: |
| Qwen2.5-0.5B（base） | 200 | 5 | 否 | **35.00%** | 0.00% | 12.6 |
| Qwen2.5-0.5B（base） | 1319（全量） | 5 | 否 | **35.94%** | 0.00% | 82.1 |
| Qwen2.5-1.5B-Instruct | 200 | 5 | 是 | **63.00%** | 0.00% | 43.1 |
| Qwen2.5-1.5B-Instruct | 1319（全量） | 5 | 是 | **63.76%** | 0.00% | 277.3 |

200 题子集与全量的差都在 1 个点以内（35.00 vs 35.94、63.00 vs 63.76），200 题足以当日常回归信号，全量留给需要参考数值的场合。无效率全程为 0：每条 completion 都能解析出数字——这一列是判断"模型答错了"还是"评测根本没看到答案"的分界。

同一协议今天的对比矩阵（上一节）里 0.5B 引擎侧跑出 33.50%：与 35.00% 的 1.5pp 差在 200 题子集 ±3.4pp 的统计噪声内（历史数字测于 `my_weight` 副本，今天用 HF hub 缓存快照）。

### 复测记录：v0.9 kernels 三层重构（2026-08）

kernels 目录重组为 ops/dispatcher/backend 三层后，在 torch 2.11.0+cu129 / triton 3.6.0 / Python 3.12 环境下复测：

- **Qwen2.5-1.5B-Instruct / 200 题：61.50%，无效率 0，通过**（阈值 0.63±0.05）；同一环境在重构前的 main 分支上复测得到**完全相同的 61.50%**——重构对生成结果零影响。与历史 63.00% 的 1.5 点差来自环境，在 200 题子集 ±3.4 点的统计噪声内。
- 同一分支上 **Qwen3-0.6B（bf16）与 Qwen3-0.6B-FP8 的 golden token parity 全部通过**：eager 与 CUDA graph 重放、与各自入库基线字节级一致（4 种 batch 布局 × repetition penalty 全组合），这是比分数更强的逐 token 证据。
- e2e 性能复测（10 个 checkpoint × eager/CUDA graph）见 `docs/benchmark_models.md`「模型 e2e benchmark 汇总」：graph 加速从 0.5B 的 5.3x 收敛到 14B 的 1.01x，launch-bound 到 compute-bound 的过渡与规模相符。

### 复现

```bash
# pytest：按阈值判定通过/失败
make test-eval                              # 默认 models-small.txt（0.5B / 200 题，约 20 s）
make test-eval EVAL_CONFIGS=models-all.txt  # 全部配置

# 独立脚本：只出数，不判定
python -m tests.evals.gsm8k --model-dir Qwen2.5-1.5B-Instruct \
    --num-questions 1319 --batch-size 16 --max-gen-len 256 --chat-template

# 追加结果到 JSON lines
python -m tests.evals.gsm8k --model-dir Qwen2.5-0.5B --save-results eval_runs.jsonl
```

首次运行把 GSM8K 下载到 `~/.cache/rapid_llm/evals/gsm8k/`。`RAPID_LLM_EVAL_DATA_DIR` 改缓存目录，`RAPID_LLM_EVAL_BASE_URL` 换下载源。

greedy 解码是确定性的：同一 checkpoint 重复跑得到逐字节相同的输出，准确率可精确复现（0.5B/200 题的 35.00% 在独立脚本与 pytest 两条路径下取到同一个值）。

### 敏感性验证

只有一个准确率数字说明不了它测的是什么。下面几组都在 Qwen2.5-0.5B / 200 题上跑，用来确认口径本身没有引入偏差。

| 变量 | 设置 | 准确率 | 无效率 | 说明 |
| --- | --- | ---: | ---: | --- |
| `max_gen_len` | 256 | 35.00% | 0.00% | 基线 |
| `max_gen_len` | 512 | 35.00% | 0.00% | **完全一致** —— 256 步没有截断任何一条推理链 |
| few-shot | 0 | 31.50% | 0.00% | base 模型没有示例也能按格式作答，但掉 3.5 个点 |
| few-shot | 5 | 35.00% | 0.00% | 基线 |
| few-shot | 8 | 33.50% | 0.00% | 再加示例不再有增益 |

`max_gen_len` 从 256 加到 512 准确率一字不差，配合 0% 的无效率，说明 256 的解码预算对 GSM8K 足够；这条是选 256 作默认值的依据。

chat 模板对 instruct 模型的影响单列（Qwen2.5-1.5B-Instruct / 200 题）：

| chat 模板 | 准确率 | 无效率 |
| :---: | ---: | ---: |
| 关 | 54.50% | 0.00% |
| 开 | **63.00%** | 0.00% |

差 8.5 个点。instruct 模型被微调的格式是模板而不是裸文本，所以配置里 `chat_template: true` 对它们不是可选项——关掉测到的是"模型在陌生格式下的表现"。base 模型没有对应的微调格式，必须保持关闭。这与 BoolQ 的口径教训（`benchmarks/eval` 一节）合起来读：chat 模板该开在模型被微调过的格式上，不该叠在 few-shot 提示工程上。

### 阈值与回归判定

每个配置声明一条实测基线，测试断言 `准确率 ≥ accuracy_threshold - tolerance`：

| 配置 | 模型 | 题数 | 阈值 | 容差 |
| --- | --- | ---: | ---: | ---: |
| `Qwen2.5-0.5B.yaml` | Qwen2.5-0.5B | 200 | 0.35 | 0.05 |
| `Qwen2.5-0.5B-full.yaml` | Qwen2.5-0.5B | 1319 | 0.36 | 0.03 |
| `Qwen2.5-1.5B-Instruct.yaml` | Qwen2.5-1.5B-Instruct | 200 | 0.63 | 0.05 |

用下界而不是相等：greedy 是确定性的，但 kernel 改动即便数值上没问题，也可能让若干道临界题翻面。容差吸收这部分抖动，超出就是真的回归。题数越多抖动越小，所以全量配置的容差收到 0.03。

另有 `max_invalid_rate` 单独断言无效率上限。两种失败模式要分开看：准确率低但无效率也低 = 模型算错了；无效率高 = 评测没拿到答案，此时准确率不含任何关于模型的信息。

### 未覆盖的模型

`my_weight/` 下其余 checkpoint 没有纳入回归套件：

- **Qwen3-0.6B**：`my_weight` 目录里只有 `config.json`，没有 `*.safetensors`（`/data/shared/llm_weights` 的完整副本已在对比基准里跑过）；
- **llava-1.5-7b-hf / Qwen3-VL-4B-Instruct**：多模态，GSM8K 是纯文本任务，需要另配视觉基准；
- **Qwen3-30B-A3B-Instruct-2507-FP8**：A10 的 23 GB 装不下。

配置里点名的 checkpoint 不存在时用例自己 skip 并说明原因，所以 `models-all.txt` 在任何机器上都能直接跑。补齐权重后新增一个 YAML、把文件名写进 `models-all.txt` 即可，无需改测试代码。

## HotpotQA（examples/eval_accuracy.py）

HellaSwag 已由 `benchmarks/eval/hellaswag/` 承接（rapid 与 HF、固定口径、证据落盘）。[`examples/eval_accuracy.py`](../examples/eval_accuracy.py) 仍服务于 HotpotQA：出 exact match / F1。数据集类型从文件名识别（`hotpot*`），原始文件需自备——仓库不打包该数据集，也没有提交阈值，所以它是**只出数的脚本**，不参与回归判定。

```bash
python examples/eval_accuracy.py \
    --dataset /path_to/hotpot_dev_distractor_v1.json \
    --model /path_to/Llama-3.2-3B-Instruct \
    --batch 10 --max-gen-len 1900
```

与 GSM8K 口径的两点差异：解码用采样（temperature 0.7 / top_p 0.8）而非 greedy，同一 checkpoint 重复跑不保证逐字节相同；`--max-gen-len` 默认 1900，给推理链留长度，评分只取答案。

运行前提：`examples/evaluator/datasets.py` 依赖 `sentence_transformers`（不在 rapid_llm 自身的 requirement 里），需先 `pip install sentence_transformers`。该导入按数据集按需发生，没装时 `--help` 仍可用，报错只在真正跑数据集时出现。
