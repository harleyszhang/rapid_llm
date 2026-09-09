"""Bisect: capture/replay one decoder layer (EP on) vs eager.

Trace the output magnitude across repeated replays: linear growth means an
in-place accumulate (index_add_ / all-reduce) re-applies to its own result on
every replay; a constant-but-wrong value means a stale buffer read.
"""
import sys

import torch
import torch.distributed as dist


def payload(rank: int) -> dict:
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.executor.attention_metadata import AttentionMetadata
    from rapid_llm.kernels import skip_rmsnorm

    eng = ContinuousBatchingEngine.from_pretrained(
        model="my_weight/DeepSeek-V2-Lite",
        device=f"cuda:{rank}",
        max_seq_len=256,
        max_gpu_num_blocks=8192,
        max_num_seqs=8,
        use_cuda_graph=False,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
    )
    try:
        runner = eng.engine.model_runner
        model = runner.model
        B, L = 4, 32
        ai = AttentionMetadata()
        ai.kv_buffer = runner.kv_cache_manager.gpu_kv_buffer
        ai.b_req_tokens_table = runner.b_req_tokens_table
        ai.cur_select_index = torch.zeros(B, dtype=torch.int32, device="cuda")
        ai.b_seq_len = torch.full((B,), L, device="cuda")
        ai.b_req_idx = torch.arange(B, device="cuda")
        ai.max_actual_seq_len = L
        ai.is_prefill = False

        ids = torch.randint(0, 1000, (B, 1), device="cuda")
        pos = torch.full((B, 1), L - 1, device="cuda")
        emb = model.get_input_embeddings(ids)
        pos_emb = model.rotary_emb(emb, pos)
        layer = model.layers[0]

        def run(x):
            h, r = layer(x, ai, 0, pos_emb, None)
            return skip_rmsnorm(h, r, model.norm_weight, model.config.rms_norm_eps)

        def flat(t):
            return tuple(x.clone() for x in t) if isinstance(t, (tuple, list)) else t.clone()

        def maxdiff(a, b):
            if isinstance(a, (tuple, list)):
                return max((x - y).abs().max().item() for x, y in zip(a, b, strict=True))
            return (a - b).abs().max().item()

        def mean(t):
            if isinstance(t, (tuple, list)):
                return sum(x.abs().mean().item() for x in t) / len(t)
            return t.abs().mean().item()

        with torch.no_grad():
            ref = flat(run(emb.clone()))
            torch.cuda.synchronize()
            for _ in range(3):
                run(emb.clone())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = run(emb)
            # 5 replays with values traced: linear growth => an in-place
            # accumulate re-applies to its own output on each replay;
            # constant-but-wrong => stale buffer reads.
            seq = []
            for i in range(5):
                try:
                    if dist.is_initialized():
                        dist.barrier()
                except Exception:
                    pass
                g.replay()
                torch.cuda.synchronize()
                cur = flat(out)
                seq.append({"i": i, "mean": round(mean(cur), 6),
                            "vs_eager": round(maxdiff(cur, ref), 6)})
        return {"eager_mean": round(mean(ref), 6), "seq": seq}
    finally:
        eng.shutdown()


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from tests.distributed.tp_harness import run_on_tp_ranks
    print(run_on_tp_ranks(payload, tp_size=2))
