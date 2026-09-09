"""Measure CPU model forward latency; omit --model-dir for a local tiny LLaMA."""

import argparse
import json
import statistics
import tempfile
import time

import torch

from rapid_llm.executor.model_runner import ModelRunner


def measure(call, repeats: int) -> float:
    for _ in range(3):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir")
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)


@torch.inference_mode()
def run(args: argparse.Namespace) -> int:
    if min(args.prompt_length, args.repeats, args.threads) < 1:
        raise SystemExit("length, repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="rapid-cpu-bench-") as temporary:
        path = args.model_dir
        if path is None:
            from transformers import LlamaConfig, LlamaForCausalLM

            model = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=64,
                    hidden_size=128,
                    intermediate_size=256,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=args.prompt_length + 2,
                )
            ).to(torch.bfloat16)
            model.save_pretrained(temporary)
            del model
            path = temporary
        length = args.prompt_length
        runner = ModelRunner.build(path, length + 2, device="cpu", max_gpu_num_blocks=length + 32)
        ids = torch.ones((1, length), dtype=torch.long)
        positions = torch.arange(length)[None]
        runner.prefill_alloc_kv_cache(
            length, torch.tensor([length], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
        )
        prefill = measure(lambda: runner.model(ids, positions, runner.atten_info), args.repeats)
        runner.decode_alloc_kv_cache(1)
        token = torch.ones((1, 1), dtype=torch.long)
        position = torch.tensor([[length]])
        decode = measure(lambda: runner.model(token, position, runner.atten_info), args.repeats)
        print(
            json.dumps(
                {
                    "benchmark": "cpu_model_forward",
                    "model": args.model_dir or "tiny_random_llama",
                    "torch": torch.__version__,
                    "threads": args.threads,
                    "prompt_length": length,
                    "batch_size": 1,
                    "repeats": args.repeats,
                    "prefill_median_ms": prefill,
                    "decode_median_ms": decode,
                    "decode_tokens_per_second": 1000 / decode,
                    "scope": "model forward only; excludes loading, scheduling, sampling and tokenization",
                },
                indent=2,
            )
        )
    return 0
