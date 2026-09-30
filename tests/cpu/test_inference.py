"""CPU inference checks using local random checkpoints; no downloads or CUDA."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from rapid_llm import (
    LLM,
    DataParallelController,
    SamplingParams,
    Scheduler,
    SchedulerConfig,
)
from rapid_llm.executor.model_runner import ModelRunner
from rapid_llm.kernels import flash_attention2_chunked, flash_attention2_no_pad, skip_rmsnorm


@pytest.fixture
def checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(42)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
            )
        )
        .to(torch.bfloat16)
        .eval()
    )
    model.save_pretrained(tmp_path)
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
    ).save_pretrained(tmp_path)
    return tmp_path, model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_residual_norm(dtype):
    torch.manual_seed(4)
    x = torch.randn(3, 14).to(dtype)[:, ::2]
    residual = torch.randn_like(x)
    original = residual.clone()
    weight = torch.randn(7).to(dtype)
    values = x.float() + original.float()
    expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)).to(
        dtype
    ) * weight
    output, updated = skip_rmsnorm(weight=weight, residual=residual, x=x)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(updated, values.to(dtype))
    assert updated.data_ptr() == residual.data_ptr()


def test_chunked_attention_matches_full_prefix():
    torch.manual_seed(2)
    q, k, v = torch.randn(3, 7, 4, 8).unbind(0)
    k, v = k[:, :2], v[:, :2]
    starts = torch.tensor([0])
    lengths = torch.tensor([7])
    expected = flash_attention2_no_pad(q, k, v, 0.3, starts, lengths, 7)
    actual = flash_attention2_chunked(
        q[3:], k, v, 0.3, starts, starts, torch.tensor([3]), lengths, 4
    )
    torch.testing.assert_close(actual, expected[3:])


def test_cpu_dispatch_and_mla_pages():
    from rapid_llm.kernels import dispatch
    from rapid_llm.platform import PlatformInfo

    selected = dispatch(
        "attention.mla_decode",
        dtype=torch.float32,
        layout=frozenset({"kv:mla_latent"}),
        platform_info=PlatformInfo(),
    )
    assert selected.spec.backend == "cpu"
    torch.manual_seed(6)
    q = torch.randn(1, 2, 80)
    cache = torch.randn(4, 2, 80)
    table = torch.tensor([[3, 1]])
    latent = torch.cat((cache[3], cache[1]))[:3]
    expected = (q[0] @ latent.T * 0.2).softmax(-1) @ latent[:, :16]
    actual = selected.load()(q, cache, table, torch.tensor([3]), max_seq_len=4, sm_scale=0.2)
    torch.testing.assert_close(actual[0], expected)


@torch.inference_mode()
def test_prefill_decode_logits(checkpoint, monkeypatch):
    path, reference = checkpoint
    expected_prefill = reference(torch.tensor([[1, 4, 6]])).logits.float()
    expected_decode = reference(torch.tensor([[1, 4, 6, 8]])).logits[:, -1:].float()
    runner = ModelRunner.build(str(path), 16, max_gpu_num_blocks=64, device="cpu")
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: pytest.fail("CPU entered CUDA")
    )
    runner.enable_cuda_graph()
    assert not runner.uses_cuda_graph
    ids = torch.tensor([[1, 4, 6]])
    runner.prefill_alloc_kv_cache(
        3, torch.tensor([3], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    )
    actual = runner.model(ids, torch.arange(3)[None], runner.atten_info)
    torch.testing.assert_close(actual.float(), expected_prefill, atol=0.008, rtol=0.03)
    runner.decode_alloc_kv_cache(1)
    actual = runner.model(torch.tensor([[8]]), torch.tensor([[3]]), runner.atten_info)
    torch.testing.assert_close(actual.float(), expected_decode, atol=0.008, rtol=0.03)


@pytest.mark.parametrize("quantization", [None, "int8", "int4", "fp8", "smoothquant"])
def test_offline_and_continuous_generation(checkpoint, quantization, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setenv("RAPID_LLM_TBO", "1")
    llm = LLM(
        str(path), device="cpu", max_seq_len=16, max_gpu_num_blocks=128, quantization=quantization
    )
    llm.stop_token_ids = set()
    params = SamplingParams(temperature=0, max_gen_len=3, stop_on_repeat=False)
    output = llm.generate(["hello world"], params)
    assert output[0].outputs[0].finish_reason == "length"
    engine = Scheduler(
        llm, SchedulerConfig(max_seq_len=16, max_num_seqs=2, max_num_batched_tokens=4)
    )
    results = engine.generate(["hello world", "hello"], params)
    assert len(results) == 2
    engine.shutdown()


@pytest.mark.parametrize("dp,tp,pipeline", [(2, 1, False), (1, 2, False), (1, 2, True)])
def test_parallel_generation(checkpoint, dp, tp, pipeline, monkeypatch):
    from rapid_llm.executor.worker import PIPELINE_ENV

    path, _ = checkpoint
    monkeypatch.setenv(PIPELINE_ENV, "1")
    monkeypatch.setenv("RAPID_LLM_COMM_OVERLAP", "1")
    monkeypatch.setenv("RAPID_LLM_TBO", "1")
    with DataParallelController(
        str(path),
        device="cpu",
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        pipeline=pipeline,
        max_num_seqs=2,
        max_seq_len=16,
        max_gpu_num_blocks=128,
    ) as engine:
        results = engine.generate(
            ["hello world", "hello"], SamplingParams(temperature=0, max_gen_len=2)
        )
        assert len(results) == 2
        assert all(result.outputs for result in results)


@pytest.mark.parametrize("kv_dtype", ["auto", "fp8"])
def test_chunk_prefix_preemption_combination(checkpoint, kv_dtype):
    path, _ = checkpoint
    engine = Scheduler.from_pretrained(
        str(path),
        device="cpu",
        max_seq_len=32,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        max_chunk_size=2,
        max_gpu_num_blocks=128,
        enable_prefix_cache=True,
        enable_preemption=True,
        kv_cache_dtype=kv_dtype,
    )
    params = SamplingParams(temperature=0, max_gen_len=3, stop_on_repeat=False)

    def generate():
        requests = [
            engine.add_request("hello world " * 10, params),
            engine.add_request("hello", params),
        ]
        for _ in range(100):
            if not engine.has_unfinished_requests():
                return requests
            engine.step()
        pytest.fail("CPU generation did not finish")

    try:
        first, second = generate(), generate()
        assert second[0].num_cached_tokens >= 16
        assert [r.output_token_ids for r in first] == [r.output_token_ids for r in second]
    finally:
        engine.shutdown()


def test_cpu_launch_harvest_pipeline(checkpoint, monkeypatch):
    import os

    from rapid_llm.executor.worker import PIPELINE_ENV

    monkeypatch.delenv(PIPELINE_ENV, raising=False)
    path, _ = checkpoint
    engine = Scheduler.from_pretrained(
        str(path),
        device="cpu",
        max_seq_len=16,
        max_num_seqs=2,
        max_gpu_num_blocks=128,
        pipeline=True,
        async_tokenize=True,
    )
    try:
        results = engine.generate(
            ["hello world", "hello"], SamplingParams(temperature=0, max_gen_len=3)
        )
        assert len(results) == 2
        assert PIPELINE_ENV not in os.environ
    finally:
        engine.shutdown()
