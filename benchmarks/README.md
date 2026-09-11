# benchmarks/：性能与精度基准

`benchmarks/` 是 rapid_llm 的基准场景层：每个脚本回答一个问题、跑一组对照，把数字连环境一起写进 JSON。测量本身的实现（指标口径、引擎驱动、日志、GPU 清理）在 [rapid_llm/benchmark/](../rapid_llm/benchmark/)——放在被测的包里，与引擎同版本演进；这里只放 import 它的场景脚本，同一份实现不写两遍。

结果按子域归档在 [docs/benchmark_logs/](../docs/benchmark_logs/)；面向读者的结论整理在 [docs/benchmark_models.md](../docs/benchmark_models.md)（性能）、[docs/eval_models.md](../docs/eval_models.md)（精度）与 [docs/optimization_features.md](../docs/optimization_features.md)（开关收益）。

## 设计原理

**驱动循环只有一处。** 所有被测系统（rapid_llm 的各个入口、DP 协调器、HF、vLLM、多模态）适配成 [backends.py](../rapid_llm/benchmark/backends.py) 里的一个 `Backend`，`measure_rows` 是唯一的驱动循环。这不是洁癖：早先 offline runner 自带三份功能相同的驱动拷贝，对 warm-up、报告轮数、"greedy 是什么意思"各持一词，同一模型的数字因此不可比。库内自下而上分层为 `datasets/`（数据行）→ `workloads`（prompt/采样预设）→ `metrics`（TTFT/TPOT/TPS）→ `backends`（每引擎一个实现）→ `utils`（GPU 清理、显存足迹、JSON 日志、表格）。

**一个目录一类问题。** kernel 层回答"一个 kernel 多快"，engine 层回答"引擎开关每步省多少"，eval 回答"答得对不对"；并行、重叠、服务路径、单模型各有自己的目录（见下表）。对照实验只保留一个变量轴，切换开关、量化或并行档位，数字之差才能归因到那个轴。

**先验证，再计时。** 内核层任何时间读数前先过 `microbench.verify` 的数值正确性门；引擎与套件层的 `--verify` 断言贪心输出与基线逐 token 一致——把输出移动了的加速是失败，不是收益。计时纪律也统一在 [microbench.py](kernels/microbench.py)：先 warm、刷 L2、设备同步计时。精度层把判分口径钉在数据集目录的 `common.py`，两个引擎组共用同一份 `build_prompts`/`score`，差异只剩引擎。

**结果自带环境。** [write_json_log](../rapid_llm/benchmark/utils.py) 给每条 JSON 自动盖章 `environment()`：GPU 型号与卡数、SM 数、互联拓扑、驱动与库版本、host CPU/内存与推理模式。报告欠读者的环境必须留在文件里，收集一次，脚本忘不掉。

**判读要噪声底。** 对照组两次跑出的比值就是纯噪声的标定（例二），单次结果不许概括成通用加速比——[docs/README.md](../docs/README.md) 对文档提交的要求同样适用于这里的结果。

## 目录布局

| 目录 | 回答的问题 | 入口 | 结果落到 |
|---|---|---|---|
| [kernels/](kernels/) | 单个 kernel 多快、值不值得用 | `python benchmarks/kernels/bench_*.py` | `docs/benchmark_logs/kernels/` |
| [engine/](engine/) | 引擎开关（CUDA graph、prefix cache、chunked prefill、量化、KV offload）每步省多少 | `python benchmarks/engine/run.py <命令>` | `docs/benchmark_logs/engine/`；pipeline/kv-transfer 落 `.../kv_transfer/` |
| [suites/](suites/) | 整套 checkpoint 横评：vs HF、eager vs graph、modelzoo、QK-Norm A/B | `python benchmarks/suites/run.py <命令>` | `docs/benchmark_logs/`（qk_norm 在其子目录） |
| [eval/](eval/) | 答得对不对：GSM8K / BoolQ / HellaSwag，双引擎同口径 | `python -m benchmarks.eval.<数据集>.bench_*` | `docs/benchmark_logs/accuracy/` |
| [overlap/](overlap/) | 计算/通信重叠：L1 拷贝流、L2 双 batch、SBO、EP+TBO | `python -m benchmarks.overlap.policies` 等 | `docs/benchmark_logs/overlap/` |
| [parallelism/](parallelism/) | TP / DP / EP 对比，batch 与上下文长度扫描 | `python benchmarks/parallelism/bench_*.py` | `docs/benchmark_logs/parallel/` |
| [models/](models/) | 单个模型深挖：MLA 的 KV 经济、DeepSeek-V4 前向对比 | `python benchmarks/models/bench_*.py` | `docs/benchmark_logs/models/` |
| [serving/](serving/) | 服务路径的每 token CPU 开销（观测面、parser） | `python benchmarks/serving/bench_*.py` | `docs/benchmark_logs/serving/` |

各脚本 docstring 记录了完整口径与命令，`--help` 同步输出。overlap 的 L4（tile 级信号）与 TBO 成本模型不经过引擎，放在 `kernels/` 与其余微基准同住。

## 工作流程

新增一个 benchmark 的路径：

1. **定位**：问题属于哪一类（上表）。单 kernel 进 `kernels/`，引擎开关进 `engine/`，跨模型横评进 `suites/`，精度进 `eval/`。
2. **写脚本**：import [rapid_llm.benchmark](../rapid_llm/benchmark/__init__.py) 的公共 API——`PROMPTS`/`GREEDY_PARAMS`（负载预设）、`make_backend`/`build_arm`（被测系统）、`run_requests`/`pctl`（计时与分位）、`require_gpus`/`free_gpu`/`gpu_tag`（GPU 卫生）、`write_json_log`/`environment`（落盘）。不重写驱动循环，不重定义 JSON 形状。
3. **声明对照**：一组配置在同一驱动循环下互比（开关、量化、并行档位或引擎），差异能归结到你要测的那一个轴。
4. **过验证门**：引擎/套件层用 `--verify`，内核层用 `microbench.verify`。
5. **落盘与成文**：JSON 进 `docs/benchmark_logs/<子域>/`，然后把环境、口径、原始数据路径和限制写进对应文档（性能→`benchmark_models.md`，精度→`eval_models.md`，开关收益→`optimization_features.md`）；专项实验（如 QK-Norm）另起子目录 README。

## 使用方案

准备：GPU 基准需要能跑 CUDA 的 torch 构建（项目 `.venv`；套件脚本跑前预检 CUDA，构建与驱动不匹配会拦下并提示换 `PYTHON=`）。权重按约定查找：套件在 `my_weight/<name>` 下取 checkpoint；engine 脚本用 `--model-dir` 显式给路径（惯例也是 `my_weight/` 下的 checkpoint）；eval 依次查当前路径、`RAPID_LLM_MODELZOO`（默认 `/data/shared/llm_weights`）、`my_weight/`。

**套件（suites）**：

```bash
python benchmarks/suites/run.py compare        # rapid_llm vs HF 全模型批大小矩阵
python benchmarks/suites/run.py e2e --out /tmp/e2e   # eager vs CUDA graph 单批
python benchmarks/suites/run.py models         # modelzoo 套件（默认路径见 --help）
python benchmarks/suites/run.py qk-norm run fused docs/benchmark_logs/qk_norm --scope single
python benchmarks/suites/run.py qk-norm summarize    # 从归档 JSON 重算比值
```

默认解释器 `.venv/bin/python`，用 `--python` 或 `PYTHON=` 更换；`--dry-run` 只打印将执行的命令。

**引擎（engine）**：

```bash
python benchmarks/engine/run.py scheduler matrix --model-dir <ckpt> --graph --prefix-cache
python benchmarks/engine/run.py scheduler continuous --model-dir <ckpt>
python benchmarks/engine/run.py optimizations --model-dir <ckpt> --workload long \
    --features chunked_prefill pipeline --verify
python benchmarks/engine/run.py quant --model-dir <ckpt> --schemes fp16 fp8 int4 nvfp4
python benchmarks/engine/run.py pipeline --model-dir my_weight/Qwen3-0.6B
python benchmarks/engine/run.py kv-transfer --model-dir my_weight/Qwen3-0.6B
python benchmarks/engine/run.py cpu            # 不带 --model-dir 时用本地 tiny LLaMA
```

**精度（eval）**：

```bash
python -m benchmarks.eval.gsm8k.bench_rapid_vllm --model-dir Qwen2.5-0.5B --num-questions 200
python -m benchmarks.eval.gsm8k.bench_hf --model-dir Qwen2.5-0.5B --num-questions 200
```

每个数据集目录自带 README（[gsm8k](eval/gsm8k/README.md)、[boolq](eval/boolq/README.md)、[hellaswag](eval/hellaswag/README.md)），口径与命令以它们为准。

**内核 / 重叠 / 并行 / 单模型 / 服务**：

```bash
python benchmarks/kernels/bench_fused_moe.py --json out.json
python -m benchmarks.overlap.policies --policy sbo --graph
python benchmarks/parallelism/bench_expert_parallel.py --model <moe-ckpt>
python benchmarks/models/bench_mla.py --batch 8 --max-gen-len 128
python benchmarks/serving/bench_observability.py --model-dir <ckpt>
```

常用环境变量：`CUDA_VISIBLE_DEVICES` 选卡；`RAPID_LLM_AUTOTUNE=0` 关 autotune 走启发式 tile（跨机对照时必开，见 [microbench.py](kernels/microbench.py) 的 `_RELEVANT_ENV`）；`RAPID_LLM_MODELZOO` 指模型库。

## 结果实例解析

每条结果的 JSON 形状统一：`{"config": ..., "results": ...}`。`config` 是命令行参数加时间戳，再加自动收集的 `environment`（机器与软件栈的完整事实：GPU 型号与卡数、SM、拓扑、驱动、torch/triton 版本等）；`results` 是这次运行的全部表格。打开文件先看 `config.environment`，确认数字来自哪台机器，再看 `results`；行与行由同一驱动循环产生，可以直接互比。

两个真实例子的读法。

**例一：读一张 engine 表——CUDA graph 的收益为什么随模型增大衰减。** [benchmark_models.md 的 e2e 表](../docs/benchmark_models.md)（A10 单卡、batch 8、greedy）里取三行：

| 模型 | TPOT eager | TPOT graph | graph 加速 |
|---|---:|---:|---:|
| Qwen3-0.6B-FP8 | 25.94 ms | 4.49 ms | 5.8x |
| Qwen2.5-1.5B | 20.47 ms | 8.54 ms | 2.4x |
| Qwen3-8B | 38.66 ms | 37.33 ms | 1.04x |

三行放在一起读出的规律：0.6B 的 decode 步只有几毫秒，kernel launch 开销占大头，graph 重放把 launch 拿掉换来 5.8x；8B 算术时间主导，launch 占比小，加速收敛到 1.04x。同一张表里 Qwen3-0.6B 的 FP8 与 bf16 TPOT 几乎一样（4.49 vs 4.61 ms），说同一件事：小模型 decode 是 launch-bound，权重减半的收益显不出来——FP8 的收益要到大模型才兑现。

**例二：读一组 A/B——先定噪声底，再认收益。** [QK-Norm 融合的专题 README](../docs/benchmark_logs/qk_norm/README.md)带了一个对照组（Qwen2.5-0.5B，`use_qk_norm=False`，两次运行代码完全相同）。对照组比值（fused ÷ baseline，<1 更快）在 eager 是 1.004–1.016、graph 是 0.995–1.006，即噪声约 ±1.6% / ±0.6%；受影响模型 batch=1 的 TPOT 比 0.956、0.957 超出噪声，收益成立；batch=32 的 0.995 在噪声内，不认。没有这层标定，0.956 与 0.995 都只是"小于 1 的数字"，无从判断哪个是真的。

同一份 README 还演示了另一个纪律：TP2 那组对照自己波动 +27.6%，作者标"不可判读"，不强行给结论。读本目录的结果时，"不可判读"是结论，不是缺数据。

## 相关文档

- 性能结果与复现命令：[docs/benchmark_models.md](../docs/benchmark_models.md)
- 精度结果：[docs/eval_models.md](../docs/eval_models.md)
- 优化特性对照：[docs/optimization_features.md](../docs/optimization_features.md)
- QK-Norm A/B 专题：[docs/benchmark_logs/qk_norm/README.md](../docs/benchmark_logs/qk_norm/README.md)
- 各数据集口径：[eval/gsm8k](eval/gsm8k/README.md)、[eval/boolq](eval/boolq/README.md)、[eval/hellaswag](eval/hellaswag/README.md)
