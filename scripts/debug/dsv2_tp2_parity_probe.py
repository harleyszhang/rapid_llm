"""One-shot probe: DeepSeek-V2-Lite TP=2 greedy vs a teacher-forced HF reference.

Development scaffold for the golden gate: prints the per-step agreement
statistics that decide the committed thresholds (how many tokens match the
reference argmax outright, how many sit in a tie, how far the reported
logprobs drift). Not part of the test suite.

Usage:
    .venv/bin/python scripts/dsv2_tp2_parity_probe.py
"""

from __future__ import annotations

import gc

import torch

_MODEL = "my_weight/DeepSeek-V2-Lite"
_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea.",
    "List three prime numbers.",
    "Explain what a GPU is in one sentence.",
]
_MAX_GEN = 32


def _lite_run() -> list[dict]:
    """Greedy completions with per-step logprobs from a two-rank engine."""
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.engine.sampler import SamplingParams

    engine = ContinuousBatchingEngine.from_pretrained(
        model=_MODEL,
        device="cuda:0",
        max_seq_len=1024,
        max_gpu_num_blocks=8192,
        max_num_seqs=4,
        use_cuda_graph=False,
        tensor_parallel_size=2,
    )
    try:
        params = SamplingParams(
            temperature=0.0, max_gen_len=_MAX_GEN, logprobs=2, prompt_logprobs=2
        )
        tokenizer = engine.tokenizer
        runs = []
        for prompt, output in zip(_PROMPTS, engine.generate(_PROMPTS, params), strict=True):
            records = output.outputs[0].logprobs or []
            runs.append(
                {
                    "prompt_ids": tokenizer.encode(prompt, add_special_tokens=True),
                    "tokens": [r.token_id for r in records],
                    "logprobs": [r.logprob for r in records],
                    # Prompt-side records too: teacher-forced positions carry no
                    # sampling feedback, so their drift is the tight budget the
                    # golden gate can hold this model to.
                    "prompt": [
                        None if r is None else r.logprob for r in output.prompt_logprobs or []
                    ],
                }
            )
        return runs
    finally:
        # shutdown reaps the followers and tears down the rank-0 half of their
        # group with them, so the transformers load below sees a plain process.
        engine.shutdown()
        del engine  # drop the weight reference before the HF reference loads
        gc.collect()
        torch.cuda.empty_cache()


def _reference(runs: list[dict]) -> list[dict]:
    """HF teacher-forced log-softmax rows over each prompt + generation."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        _MODEL,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        # The checkpoint's auto_map points at DeepSeek's remote-code class,
        # whose weight names transformers 5.x fails to auto-convert; the
        # built-in DeepseekV2ForCausalLM shares their naming and loads cleanly.
        trust_remote_code=False,
    ).eval()
    try:
        out = []
        with torch.no_grad():
            for run in runs:
                ids = torch.tensor([run["prompt_ids"] + run["tokens"]], device="cuda:0")
                logits = model(ids).logits.float().cpu()
                out.append(torch.log_softmax(logits, dim=-1)[0])
        return out
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    runs = _lite_run()
    refs = _reference(runs)

    ties = matches = steps = 0
    drifts: list[float] = []
    prompt_drifts: list[float] = []
    reported_drifts: list[float] = []
    for run, ref in zip(runs, refs, strict=True):
        for i, reported in enumerate(run["prompt"]):
            if i == 0 or reported is None:
                continue
            token = run["prompt_ids"][i]
            prompt_drifts.append(abs(reported - float(ref[i - 1, token])))
        base = len(run["prompt_ids"])
        for step, token in enumerate(run["tokens"]):
            row = ref[base - 1 + step]
            reported_drifts.append(abs(run["logprobs"][step] - float(row[token])))
            top2 = row.topk(2)
            gap = float(top2.values[0] - top2.values[1])
            mine = float(row[token])
            best = float(top2.values[0])
            drifts.append(best - mine)
            steps += 1
            if token == int(top2.indices[0]):
                matches += 1
            elif gap < 0.1:
                ties += 1
                print(f"  tie at step {step}: gap {gap:.4f}, lite took {mine:.4f}")
            else:
                print(
                    f"  MISMATCH step {step}: ref top {int(top2.indices[0])} at "
                    f"{best:.4f} (gap {gap:.4f}), lite took {token} at {mine:.4f}"
                )

    drifts.sort()
    print(f"\nsteps={steps} matches={matches} ties={ties}")
    print(
        f"drift: max={drifts[-1]:.4f} p95={drifts[int(len(drifts) * 0.95)]:.4f} "
        f"mean={sum(drifts) / len(drifts):.4f}"
    )
    prompt_drifts.sort()
    print(
        f"prompt drift ({len(prompt_drifts)} positions): "
        f"max={prompt_drifts[-1]:.4f} p95={prompt_drifts[int(len(prompt_drifts) * 0.95)]:.4f} "
        f"mean={sum(prompt_drifts) / len(prompt_drifts):.4f}"
    )
    reported_drifts.sort()
    print(
        f"reported drift ({len(reported_drifts)} steps): "
        f"max={reported_drifts[-1]:.4f} p95={reported_drifts[int(len(reported_drifts) * 0.95)]:.4f} "
        f"mean={sum(reported_drifts) / len(reported_drifts):.4f}"
    )
    for run in runs:
        print("text sample:", run["tokens"][:8])


if __name__ == "__main__":
    main()
