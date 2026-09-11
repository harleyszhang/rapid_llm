"""The ``rapid-llm`` command line: chat, vl-chat, serve, batch and acc.divergence.

Each subcommand is a :class:`CliCommand` that owns its parser section and its
wiring, so a new command never touches ``main``; knobs shared by the engine
commands live once in :class:`TextEngineOptions`, while the accuracy tool
talks to :mod:`rapid_llm.tools.accuracy` directly — no engine involved.

Usage:
    rapid-llm chat --help
    rapid-llm acc.divergence --model-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from .engine import ContinuousBatchingEngine, SamplingParams, VisionGenerator
from .engine.dp_load_balancer import LOAD_BALANCERS
from .engine.scheduler import DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS
from .modules.quantization import RUNTIME_SCHEMES
from .tools.accuracy import (
    DEFAULT_PROMPT,
    DEFAULT_REL_THRESHOLD,
    find_first_divergent_layer,
)
from .utils.env_compat import getenv
from .utils.prompt_templates import ChatPrompter, PrompterResolver

# ---------------------------------------------------------------------------
# 第一层:声明式 CLI 参数表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliOption:
    """一条 CLI 参数声明;``register`` 将其翻译成一次 ``add_argument`` 调用。"""

    flag: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def register(self, sub: argparse.ArgumentParser) -> None:
        sub.add_argument(self.flag, **self.kwargs)


# 所有子命令共享的参数。repetition_penalty 默认 1.1 而非 1.0:小参数 base
# 模型在 fp16 argmax 平局(~0.02 logit gap)下极易滑入重复死循环,默认
# 开启轻量惩罚是更安全的出厂行为,传 1.0 可显式关闭。
COMMON_OPTIONS: tuple[CliOption, ...] = (
    CliOption("--model-dir", {"help": "Checkpoint directory"}),
    CliOption("--max-seq-len", {"type": int, "default": 2048}),
    CliOption("--max-gpu-num-blocks", {"type": int, "default": None}),
    CliOption("--device", {"default": "cuda"}),
    CliOption("--temperature", {"type": float, "default": 0.6}),
    CliOption("--top-p", {"type": float, "default": 0.9}),
    CliOption("--max-gen-len", {"type": int, "default": None}),
    CliOption(
        "--repetition-penalty",
        {
            "type": float,
            "default": 1.1,
            "help": "Penalise logits of already-generated tokens (1.0 disables it; "
            "1.1 is the default because small base models easily loop on repeats)",
        },
    ),
    CliOption(
        "--quantization",
        {
            "choices": sorted(RUNTIME_SCHEMES),
            "default": None,
            "help": "Runtime weight quantisation for fp16 checkpoints "
            "(fp8 checkpoints are detected automatically from config.json)",
        },
    ),
    CliOption(
        "--kv-cache-dtype",
        {
            "choices": ["auto", "fp8", "fp8_e4m3"],
            "default": "auto",
            "help": "KV-cache element type: 'auto' keeps fp16; the fp8 spellings "
            "store e4m3 bytes, halving the cache footprint (vLLM-style)",
        },
    ),
    CliOption(
        "--tensor-parallel-size",
        {
            "type": int,
            "default": 1,
            "help": "Number of GPUs for tensor parallelism (splits weights across cards)",
        },
    ),
    CliOption(
        "--enable-expert-parallel",
        {
            "action": "store_true",
            "help": "Split MoE experts whole-across-ranks over the tensor-parallel "
            "group instead of splitting each expert's intermediate dim (vLLM "
            "--enable-expert-parallel semantics)",
        },
    ),
)


def _cuda_graph_option(*, default: bool) -> CliOption:
    """The one CUDA-graph flag every command that talks to an engine takes.

    ``argparse.BooleanOptionalAction`` registers both spellings — positive
    ``--cuda-graph`` and negative ``--no-cuda-graph`` — against a single
    boolean dest, so no command needs a second flag with the opposite sense
    (the ``use or not no_`` double negative ``from_args`` used to carry). Only
    the default differs by command, and it lives here in the declaration,
    where a flag's semantics belong: throughput commands (``batch``,
    ``serve``) capture by default, the REPLs (``chat``, ``vl-chat``) run
    eager — one turn in flight never amortises capture latency.
    """
    return CliOption(
        "--cuda-graph",
        {
            "action": argparse.BooleanOptionalAction,
            "default": default,
            "help": "Capture decode CUDA graphs (default: %(default)s)",
        },
    )


# ---------------------------------------------------------------------------
# 第二层:引擎构造参数(配置对象 + Builder)
# ---------------------------------------------------------------------------


def _model_dir_from(args: argparse.Namespace) -> str:
    """Resolve ``--model-dir`` (falling back to ``RAPID_LLM_MODEL_DIR``); exits if neither.

    Public because the engine commands and the accuracy tool answer this
    question identically — the fallback rule must not drift between them.
    """
    model_dir = args.model_dir or getenv("RAPID_LLM_MODEL_DIR")
    if not model_dir:
        raise SystemExit(
            "Model directory not provided. Pass --model-dir <path> or set RAPID_LLM_MODEL_DIR."
        )
    if not Path(model_dir).is_dir():
        raise SystemExit(f"model directory {model_dir!r} does not exist")
    return model_dir


@dataclass(frozen=True)
class BaseOptions:
    """Construction options every engine build shares, text or vision.

    Exactly the fields :class:`~rapid_llm.engine.generator.VisionGenerator`
    takes, ``use_cuda_graph`` included. Multimodal models used to be refused
    at the capture boundary; a decode step with no vision payload is the
    same graph text models replay (the vision tokens are ordinary KV-cache
    rows by then), so vl-chat now parses the same switch the text commands
    take.
    """

    model_dir: str
    max_seq_len: int = 2048
    max_gpu_num_blocks: int | None = None
    device: str = "cuda"
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    use_cuda_graph: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BaseOptions:
        return cls(**cls._collect(args))

    @staticmethod
    def _collect(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "model_dir": _model_dir_from(args),
            "max_seq_len": args.max_seq_len,
            "max_gpu_num_blocks": args.max_gpu_num_blocks,
            "device": args.device,
            "quantization": getattr(args, "quantization", None),
            "kv_cache_dtype": getattr(args, "kv_cache_dtype", "auto"),
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
            "enable_expert_parallel": getattr(args, "enable_expert_parallel", False),
            # ``cuda_graph`` is the dest every --cuda-graph flag writes (text
            # commands and vl-chat alike); the fallback fires only if a command
            # forgot to register the flag, and eager is the safe default.
            "use_cuda_graph": getattr(args, "cuda_graph", False),
        }

    def build_vision_generator(self) -> VisionGenerator:
        return VisionGenerator(
            checkpoints_dir=self.model_dir,
            max_seq_len=self.max_seq_len,
            max_gpu_num_blocks=self.max_gpu_num_blocks,
            device=self.device,
            use_cuda_graph=self.use_cuda_graph,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            kv_cache_dtype=self.kv_cache_dtype,
        )


@dataclass(frozen=True)
class TextEngineOptions(BaseOptions):
    """Base options plus the switches only the text engine has.

    One factory serves ``chat``, ``batch`` and ``serve`` because what separates
    them is scheduling, not construction — and tensor parallelism arrives with
    the factory: ``from_pretrained`` spawns the follower ranks and hands back
    an engine whose executor drives them, so a command only picks its
    concurrency.
    """

    def build_engine(
        self,
        *,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
    ) -> ContinuousBatchingEngine:
        return ContinuousBatchingEngine.from_pretrained(
            self.model_dir,
            max_seq_len=self.max_seq_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_gpu_num_blocks=self.max_gpu_num_blocks,
            device=self.device,
            use_cuda_graph=self.use_cuda_graph,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            kv_cache_dtype=self.kv_cache_dtype,
        )


# ---------------------------------------------------------------------------
# 第三层:Prompter 选择策略
# ---------------------------------------------------------------------------

# PrompterResolver 住在 utils/prompt_templates.py:base-vs-instruct 的判定必须
# 全链路唯一(serve 的 /v1/chat/completions 与 chat/batch 共用同一套规则),
# 放在 CLI 里会让 serve 长出第二套模板策略——一个带 chat_template 的 base
# checkpoint 在 chat 里被原样直传、在 serve 里却被套上模板,正是曾经的真实分歧。


# ---------------------------------------------------------------------------
# 第四层:子命令(Command + Template Method)
# ---------------------------------------------------------------------------


class CliCommand(ABC):
    """CLI 子命令基类。

    ``register`` 是模板方法:建子 parser → 注册公共参数 → 注册命令特有
    参数(``add_arguments`` 钩子)→ 绑定 handler。子类只补充差异部分。
    公共参数集由 ``common_options`` 决定:引擎命令沿用全量
    ``COMMON_OPTIONS``(采样参数在内),不走引擎的工具命令覆写为定位
    模型所需的子集——给 ``acc.divergence`` 挂上 ``--temperature`` 只会
    误导。
    """

    name: ClassVar[str]
    help: ClassVar[str]
    common_options: ClassVar[tuple[CliOption, ...]] = COMMON_OPTIONS

    def register(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(self.name, help=self.help)
        for opt in self.common_options:
            opt.register(sub)
        self.add_arguments(sub)
        sub.set_defaults(handler=self)
        return sub

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:  # noqa: B027
        """注册命令特有参数;默认无。"""

    @staticmethod
    def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_gen_len=args.max_gen_len,
            repetition_penalty=args.repetition_penalty,
        )

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """执行子命令,返回进程退出码。"""


class ChatCommand(CliCommand):
    """``chat``:交互式多轮文本对话(REPL 循环,逐 token 流式输出)。

    每轮重渲染全部历史(instruct 走 tokenizer 的 chat template,base 原文
    拼接),回复结束后写回历史;``/clear`` 清空会话但不重载权重。

    单卡多卡走同一条路径:引擎自己决定 rank,本类只管一次对话一个请求。
    """

    name = "chat"
    help = "Interactive multi-turn text chat"

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        # A REPL pays capture latency up front for a single stream of turns, so
        # graphs are opt-in here; throughput commands default the other way.
        _cuda_graph_option(default=False).register(sub)

    def run(self, args: argparse.Namespace) -> int:
        opts = TextEngineOptions.from_args(args)
        # A REPL has one turn in flight at a time, so a single slot is the whole
        # of the concurrency and the KV cache is not split eight ways for nobody.
        engine = opts.build_engine(max_num_seqs=1)
        prompter = PrompterResolver.build(opts.model_dir, engine.tokenizer)
        params = self.build_sampling_params(args)
        # The conversation across turns: user and assistant messages in order.
        # Every turn re-renders it whole, so a REPL session is one growing
        # context the model stays inside of.
        history: list[dict[str, str]] = []

        self._print_banner(opts.model_dir, prompter, params)
        try:
            while True:
                try:
                    user_input = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C 退出
                    print()
                    return 0
                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    return 0
                if user_input == "/clear":
                    history.clear()
                    print("(conversation cleared)\n")
                    continue
                self._stream_reply(engine, prompter, params, user_input, history)
        finally:
            # Under tensor parallelism the follower ranks are waiting for the
            # next plan; without this they would outlive the session.
            engine.shutdown()

    @staticmethod
    def _print_banner(
        model_dir: str, prompter: ChatPrompter | None, params: SamplingParams
    ) -> None:
        print(f"Loaded {Path(model_dir).name}. Type 'exit' to quit, '/clear' to start over.")
        if prompter is None:
            print("(no chat template: prompts are sent verbatim)")
        if params.is_greedy and params.repetition_penalty == 1.0:
            # greedy + base 模型是经典的重复循环组合:fp16 argmax 平局翻转
            # (~0.02 logit gap) 之后循环无法自我纠正,提前给用户提个醒
            print(
                "(greedy decoding without a repetition penalty can loop on repeats; "
                "try --temperature 0.6 or --repetition-penalty 1.1)"
            )
        print()

    @staticmethod
    def _stream_reply(
        engine: ContinuousBatchingEngine,
        prompter: ChatPrompter | None,
        params: SamplingParams,
        user_input: str,
        history: list[dict[str, str]],
    ) -> None:
        """一轮对话:渲染全部历史 → 流式打印 → 回写助手消息。

        多轮的关键在"渲染全部历史":instruct 模型走 :meth:`ChatPrompter.apply`
        (与 ``/v1/chat/completions`` 同一条模板路径),base 模型原文拼接各轮
        ——和服务端 ``_render_chat`` 一个策略,两个入口不会漂移。只发当前
        这一句,模型每轮都是失忆的,这正是"继续"换来一段全新开场白的根因。
        """
        history.append({"role": "user", "content": user_input})
        prompt = (
            prompter.apply(history)
            if prompter is not None
            else "\n".join(turn["content"] for turn in history)
        )

        try:
            request = engine.add_request(prompt, params)
        except ValueError as exc:  # empty, or longer than the context window
            # The turn never ran; leaving it in the history would poison every
            # later prompt with a message the model never answered.
            history.pop()
            print(f"[{exc}]", file=sys.stderr)
            print(
                "[conversation no longer fits the context window; /clear starts a fresh one]\n",
                file=sys.stderr,
            )
            return

        # One request is in flight, so whoever a step advanced is this one; the
        # loop reads it rather than assuming, since a step that scheduled nothing
        # would otherwise reprint the previous delta.
        while engine.has_unfinished_requests():
            for advanced in engine.step():
                print(advanced.delta, end="", flush=True)
        # A reply cut short by "repeat" or "length" is still what the model
        # said; it enters the history either way so the next turn stays coherent.
        history.append({"role": "assistant", "content": request.text})
        if request.finish_reason == "repeat":
            print(
                "\n[stopped early: degenerate repetition detected; try a higher "
                "--repetition-penalty or --temperature]",
                file=sys.stderr,
            )
        print("\n")


class VlChatCommand(CliCommand):
    """``vl-chat``:单轮图像条件对话(LLaVA / Qwen3-VL)。"""

    name = "vl-chat"
    help = "Single-turn image-conditioned chat"

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        # Same REPL trade-off as ``chat``: one turn in flight never amortises
        # capture latency, so graphs are opt-in here.
        _cuda_graph_option(default=False).register(sub)
        CliOption(
            "--image",
            {"nargs": "+", "required": True, "help": "One or more image paths"},
        ).register(sub)
        CliOption(
            "--prompt",
            {"help": "Prompt text; must contain '<image>' for LLaVA, plain text for Qwen3-VL"},
        ).register(sub)

    def run(self, args: argparse.Namespace) -> int:
        opts = BaseOptions.from_args(args)

        if opts.tensor_parallel_size > 1 or opts.enable_expert_parallel:
            # Text commands shard through the engine's executor; the vision path
            # still runs one replica, and the scheme that used to fake it here --
            # a mirror process re-deriving the batch from a broadcast prompt --
            # is exactly what this release removed.
            raise SystemExit(
                "vl-chat is single-GPU: --tensor-parallel-size > 1 and "
                "--enable-expert-parallel need the "
                "continuous-batching engine, which does not host vision models yet"
            )

        generator = opts.build_vision_generator()
        params = self.build_sampling_params(args)

        images = [Image.open(p).convert("RGB") for p in args.image]
        # LLaVA 吃裸的 "USER: <image> ... ASSISTANT:" 字符串;Qwen3-VL 只要
        # 一条纯 user 消息,由 VisionGenerator 自己套 chat template
        default_prompt = (
            "Describe this image."
            if generator.is_qwen3_vl
            else "USER: <image>\nDescribe this image. ASSISTANT:"
        )
        for delta in generator.stream(args.prompt or default_prompt, images, params):
            print(delta, end="", flush=True)
        print()
        return 0


class ServeCommand(CliCommand):
    """``serve``:启动 OpenAI 兼容的 HTTP 服务(连续批处理引擎)。"""

    name = "serve"
    help = "Start an OpenAI-compatible API server"

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        for option in (
            CliOption("--host", {"default": "0.0.0.0"}),
            CliOption("--port", {"type": int, "default": 8000}),
            CliOption(
                "--served-model-name",
                {"help": "Name reported by /v1/models (defaults to the directory name)"},
            ),
            CliOption(
                "--max-num-seqs",
                {
                    "type": int,
                    "default": DEFAULT_MAX_NUM_SEQS,
                    "help": "How many requests may decode concurrently",
                },
            ),
            CliOption(
                "--max-num-batched-tokens",
                {
                    "type": int,
                    "default": DEFAULT_MAX_NUM_BATCHED_TOKENS,
                    "help": "Padded token budget for one prefill group",
                },
            ),
            _cuda_graph_option(default=True),
            CliOption(
                "--data-parallel-size",
                {
                    "type": int,
                    "default": 1,
                    "help": "Whole-model replicas for throughput (one GPU each; "
                    "combines with --tensor-parallel-size into a dp x tp grid)",
                },
            ),
            CliOption(
                "--load-balancer",
                {
                    "choices": list(LOAD_BALANCERS),
                    "default": "round_robin",
                    "help": "How requests are routed between data-parallel replicas",
                },
            ),
            CliOption(
                "--no-chat-template",
                {
                    "action": "store_true",
                    "help": "Send /v1/chat/completions messages verbatim (base models)",
                },
            ),
        ):
            option.register(sub)

    def run(self, args: argparse.Namespace) -> int:
        from .entrypoints.api_server import ServerConfig, run_server

        opts = TextEngineOptions.from_args(args)
        # Sampling flags do not apply here: every HTTP request carries its own.
        config = ServerConfig(
            model_dir=opts.model_dir,
            served_model_name=args.served_model_name,
            max_seq_len=opts.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_gpu_num_blocks=opts.max_gpu_num_blocks,
            device=opts.device,
            use_cuda_graph=opts.use_cuda_graph,
            quantization=opts.quantization,
            tensor_parallel_size=opts.tensor_parallel_size,
            enable_expert_parallel=opts.enable_expert_parallel,
            kv_cache_dtype=opts.kv_cache_dtype,
            data_parallel_size=args.data_parallel_size,
            load_balancer=args.load_balancer,
            # None = auto: the same base-vs-instruct detection chat/batch apply,
            # so a Base checkpoint is served verbatim instead of templated just
            # because its tokenizer happens to ship a template.
            chat_template=False if args.no_chat_template else None,
        )
        print(f"Serving {config.model_name} on http://{args.host}:{args.port}")
        run_server(config, host=args.host, port=args.port)
        return 0


class BatchCommand(CliCommand):
    """``batch``:把一批 prompt 交给连续批处理引擎离线跑完。

    与 ``chat`` 的区别不在于接口而在于调度:这里每个 prompt 是一个独立请求,
    先结束的请求立即释放槽位给后面排队的 prompt,而不是拖到整批里最长的那一条。
    """

    name = "batch"
    help = "Run a prompt file through the continuous-batching engine"

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        for option in (
            CliOption(
                "--prompts-file",
                {"help": "Text file with one prompt per line; omit to use a built-in demo set"},
            ),
            CliOption("--max-num-seqs", {"type": int, "default": DEFAULT_MAX_NUM_SEQS}),
            _cuda_graph_option(default=True),
            CliOption(
                "--no-chat-template",
                {"action": "store_true", "help": "Send prompts verbatim (base models)"},
            ),
            CliOption(
                "--show-stats",
                {"action": "store_true", "help": "Print throughput and per-request timings"},
            ),
        ):
            option.register(sub)

    def run(self, args: argparse.Namespace) -> int:
        import time

        opts = TextEngineOptions.from_args(args)
        prompts = self._load_prompts(args.prompts_file)

        engine = opts.build_engine(max_num_seqs=args.max_num_seqs)
        prompter = PrompterResolver.build(
            opts.model_dir, engine.tokenizer, use_template=not args.no_chat_template
        )
        if prompter is not None:
            prompts = [prompter.insert_prompt(p) for p in prompts]

        params = self.build_sampling_params(args)
        started = time.perf_counter()
        requests = [engine.add_request(prompt, params) for prompt in prompts]
        try:
            while engine.has_unfinished_requests():
                engine.step()
        finally:
            elapsed = time.perf_counter() - started
            engine.shutdown()  # releases the tensor-parallel followers, if any

        for index, request in enumerate(requests):
            print(f"--- [{index}] {request.finish_reason} ---")
            print(request.text.strip())
            print()

        if args.show_stats:
            self._print_stats(requests, elapsed, started)
        return 0

    @staticmethod
    def _load_prompts(path: str | None) -> list[str]:
        if path is None:
            return [
                "What is the capital of France?",
                "Write a haiku about the sea.",
                "List three prime numbers.",
                "Explain what a GPU is in one sentence.",
            ]
        lines = [line.strip() for line in Path(path).read_text().splitlines()]
        prompts = [line for line in lines if line]
        if not prompts:
            raise SystemExit(f"{path!r} contains no prompts")
        return prompts

    @staticmethod
    def _print_stats(requests: list, elapsed: float, started: float) -> None:
        generated = sum(len(r.output_token_ids) for r in requests)
        ttfts = [
            (r.first_token_time - started) * 1000
            for r in requests
            if r.first_token_time is not None
        ]
        print(f"{len(requests)} requests, {generated} tokens in {elapsed:.2f}s")
        print(f"throughput {generated / elapsed:7.1f} tok/s")
        if ttfts:
            print(f"TTFT mean {sum(ttfts) / len(ttfts):7.1f} ms | max {max(ttfts):7.1f} ms")


class AccuracyCommand(CliCommand):
    """``acc.divergence``:整模型对 ``transformers`` 参考实现的逐层精度对比。

    同一 checkpoint 装两个实现、同一 prompt 喂两侧,逐层比 decoder layer
    输出,报第一个超出噪声带的层及该层内 attention/MLP 的二次定位;
    ``--json`` 给工具链,默认人读表格给终端。不构造引擎,公共参数只留
    定位模型所需的三条。退出码即结论(0 全层通过,1 有散度),可直接当
    CI 精度门禁用。
    """

    name = "acc.divergence"
    help = "Locate the first decoder layer that diverges from the transformers reference"
    common_options = (
        CliOption("--model-dir", {"help": "Checkpoint directory"}),
        CliOption("--max-seq-len", {"type": int, "default": 2048}),
        CliOption("--device", {"default": "cuda"}),
    )

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        CliOption(
            "--prompt",
            {"default": DEFAULT_PROMPT, "help": "Prompt to compare the two models on"},
        ).register(sub)
        CliOption(
            "--rel-threshold",
            {
                "type": float,
                "default": DEFAULT_REL_THRESHOLD,
                "help": "Diff-to-reference ratio past which a layer counts as diverged",
            },
        ).register(sub)
        CliOption("--json", {"action": "store_true", "help": "Emit the report as JSON"}).register(
            sub
        )

    def run(self, args: argparse.Namespace) -> int:
        report = find_first_divergent_layer(
            _model_dir_from(args),
            prompt=args.prompt,
            rel_threshold=args.rel_threshold,
            device=args.device,
            max_seq_len=args.max_seq_len,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render())
        return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# 装配层:命令注册表 + 入口
# ---------------------------------------------------------------------------

COMMANDS: tuple[CliCommand, ...] = (
    ChatCommand(),
    VlChatCommand(),
    ServeCommand(),
    BatchCommand(),
    AccuracyCommand(),
)
"""已注册子命令;新增命令 = 实现一个 :class:`CliCommand` 子类并加入此表。"""


def build_parser() -> argparse.ArgumentParser:
    # description 只取模块 docstring 首行;完整架构说明留在文档注释里
    parser = argparse.ArgumentParser(
        prog="rapid-llm", description=__doc__.splitlines()[0] if __doc__ else None
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Silence a noisy but harmless warning from torch._utils.
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
    args = build_parser().parse_args(argv)
    return args.handler.run(args)


if __name__ == "__main__":
    sys.exit(main())
