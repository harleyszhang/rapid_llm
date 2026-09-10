"""Quick smoke: vLLM over DeepSeek-V3-4layers on 2x A10 (SM86), TP-2.

The parity gate needs a vLLM arm for V3; this checks the trimmed BF16
checkpoint loads and generates on Ampere before we wire it into the
benchmark scripts.
"""

from vllm import LLM, SamplingParams

CKPT = "/data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16"
# Same regroup the V3 golden gate uses: the trimmed config keeps 8 experts
# over 8 groups, which degenerates the grouped noaux_tc router; n_group=2 /
# topk_group=1 / num_experts_per_tok=2 restores grouped semantics.
OVERRIDES = {"n_group": 2, "topk_group": 1, "num_experts_per_tok": 2}

llm = LLM(
    model=CKPT,
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    hf_overrides=OVERRIDES,
)
sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)

outs = llm.generate(
    ["Explain what a GPU tensor core is and why it matters for deep learning."],
    sp,
)
gen = outs[0].outputs[0]
print("text:", repr(gen.text))
print("token_ids:", list(gen.token_ids))
top = sorted(gen.logprobs[0].items(), key=lambda kv: -kv[1].logprob)[:5]
print("step0 top5:", [(int(i), round(l.logprob, 4)) for i, l in top])
print("SMOKE OK")
