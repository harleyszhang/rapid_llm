---
name: add-speculative-decoding
description: Extend rapid_llm's speculative decoding beyond the shipped O5 n-gram proposer — the proposer interface and registry, single-pass verification, sampled-mode rejection sampling, KV budget and rollback, and the four high-risk coupling points (KV rollback, batch-state divergence, CUDA Graph, sampler RNG), with vLLM's MRV2 module as the reference design. Losslessness is the hard bar. Use when adding a draft-model proposer (EAGLE/MTP/draft LM), when enabling speculation under sampled decoding, when acceptance accounting or KV rollback misbehaves, or when reviewing a speculative-decoding change.
---

# Adding Speculative Decoding

rapid_llm already ships O5: an n-gram proposer with greedy single-pass
verification. This skill extends that base — new proposers (draft model,
EAGLE, MTP) and sampled-mode verification. vLLM's MRV2 module
(`vllm/v1/worker/gpu/spec_decode/`) is the reference design throughout.
**Losslessness is the hard bar** — a faster engine that changes the output
is a broken engine.

## The existing base (extend, never parallel)

| Responsibility | Module | State |
| --- | --- | --- |
| Propose | `rapid_llm/engine/ngram_proposer.py` | `NgramProposer(max_ngram_size=5, max_draft=6)`, token-ids-only input |
| Verify | `continuous_engine._speculate_verify` + `executor/worker.py` full-logits path | one forward per step, greedy argmax per position, bonus = argmax at the first mismatch |
| KV budget | `scheduler.reserve_speculative(request, draft_rows)` | `fit = min(draft_rows, limit - seq_len)`; an exhausted pool truncates drafts, never evicts |
| Switch | env `LITE_LLAMA_SPECULATE` | off by default; decode only, prefill chunks never participate |
| Tests | `tests/engine/test_ngram_proposer.py`, `test_continuous_engine.py` | |

If a responsibility does not map to this table (a second engine family, a
different sampler home), stop and ask — do not create a parallel module.
Two consequences of the current design:

- Verification is **greedy-only**: accept while `draft[j] ==
  argmax(logits[j])`, bonus is the argmax at the first mismatch. Lossless
  for greedy decoding only. Turning speculation on under `temperature > 0`
  requires the rejection sampler below — do not just flip the flag.
- `NgramProposer` is standalone, not behind an interface. A second proposer
  must not fork the engine flow — build the interface first.

## Proposer interface and registry

MRV2's shape, mapped onto rapid_llm (sketch, not final signatures):

```python
class Proposer(Protocol):
    def propose(self, token_ids: list[int], context: DraftContext) -> list[int]: ...
    graph_safe: bool           # capture-safe inside decode graphs
    needs_hidden_states: bool  # eagle/mtp need them; ngram does not
```

- **Registry by name, factory with hard errors.** Config selects the
  proposer by name; an unsupported name raises immediately. vLLM's
  `init_speculator()` (`worker/gpu/spec_decode/__init__.py`) throws
  `NotImplementedError` for methods the MRV2 stack does not support (ngram,
  medusa stay on the MRV1 stack) — an explicit error beats a silent
  fallback.
- **CUDA-graph honesty lives in the interface.** MRV2 puts
  `init_cudagraph_manager()` / `capture()` on the speculator base class, so
  every method opts in explicitly. rapid_llm's precedent is the KernelSpec
  `graph_safe` field: a proposer that assembles per-step state on the host
  is `graph_safe=False` and the runner refuses to capture with it, the same
  way `add-model`'s `supports_cuda_graph=False` works.
- **Model-based proposers share a base.** MRV2's `DraftModelSpeculator`
  holds what every draft-model method needs: weight loading, draft KV
  management, greedy draft sampling, the TP argmax-reduction path (comm
  cost O(vocab) -> O(2 x tp_size)), and the draft-logits cache for
  probabilistic drafting. Per-method code is only the architecture-specific
  forward. Copy that split — the second proposer must not re-implement
  sampling and KV plumbing. `needs_hidden_states=True` maps to MRV2's
  `use_aux_hidden_state_outputs`: the runner must emit aux layers, which
  changes the target model's output contract.

Stage 0 — interface alignment, still no code: one interface document per
new draft source: type (draft model / EAGLE / MTP / n-gram), inputs (token
ids only, or hidden states / aux layers too), outputs (does it produce
draft probabilities? sampled verification needs them), call cadence, batch
shapes, error semantics. The n-gram proposer and a future EAGLE proposer
already differ on inputs; stage 0 exists to expose that before the
interface hardens.

## Verification and sampling

1. **Single batched verification — keep.** The k drafts enter one forward
   (length k+1 per request), never k serial passes. Candidate positions use
   a chain mask (tree candidates are out of scope). Requests in a batch may
   carry different k — state the padding or varlen policy.
2. **Rejection sampling for sampled mode.** Greedy argmax compare is the
   degenerate case (`p_draft` one-hot). The sampled path compares a uniform
   against `p_target / p_draft` position by position; at the first
   rejection, sample from the corrected distribution
   `(p_target - p_draft).clamp(min=0).normalize()`. Port the algorithm from
   `vllm/v1/sample/rejection_sampler.py` (PyTorch, full LogitsProcessor
   stack). Reach for a Triton port (MRV2's GPU-native version) only when a
   profile says so, per `kernel-microbenchmark`.
3. **The bonus token comes from the regular sampler,** not from inside the
   rejection sampler. vLLM passes it in as an argument because the bonus
   must honor top_p/top_k/penalties, which the verification path does not
   support. Dropping the bonus throws away one free token per
   fully-accepted step.
4. **Separate RNG streams for draft and target.** MRV2 salts the draft
   noise (`_DRAFT_NOISE_SALT = 1 << 30`) so probabilistic drafting cannot
   perturb the target's sampling stream — that separation is what makes
   same-seed reproducibility achievable at all. rapid_llm's sampler keeps
   per-request generators; the draft stream gets its own derived generator,
   never a shared one.

## The four high-risk coupling points

Each needs a design note and dedicated tests:

1. **KV rollback.** `reserve_speculative` budgets k draft rows for the
   verify stretch; after verification accepts `a <= k` drafts, token
   accounting, KV rows and the position cursor roll back `k - a` together,
   as one consistent operation. Cover accept-all / accept-some /
   accept-none — and the truncate-not-evict path, where a full pool
   silently shrinks the draft count.
2. **Batch-state divergence.** Requests accept different counts in the same
   step. MRV2's answer is persistent per-request state tables with per-step
   inputs built by gather, so assembling one step's inputs never mutates
   the canonical state; its zero-bubble async mode goes further —
   scheduling optimistically as if all drafts will be accepted and
   correcting on the next step. Copy the persistent-table pattern; treat
   optimistic scheduling as a later optimization with its own correctness
   tests.
3. **CUDA Graph.** Variable speculation length vs fixed graph shapes:
   enumerate (k+1)-shaped decode graphs at capture time, and capture
   draft-side graphs after the target's so both share one batch-shape
   enumeration — that ordering is what MRV2 uses in `capture_model()`. A
   proposer that cannot be capture-safe sets `graph_safe=False` and runs
   eager while the rest captures. Conclusion plus measured data, not vibes.
4. **Sampler RNG.** Rejection-sampling uniforms and draft noise are
   state-managed per request and separated by salt (above); replaying the
   same seed must reproduce the same token stream.

## Acceptance bars

1. **Losslessness, per proposer**: greedy — speculation on vs off is
   token-identical; sampled — same-seed token-identical (preferred), or a
   distribution check: >= 100k tokens at the same temperature, chi-square
   p > 0.05, method and raw data reported.
2. **Acceptance rate is a committed measurement, not a guess.** vLLM keeps
   `tests/v1/e2e/spec_decode/acceptance_rates/` — per-method tests that
   assert acceptance on fixed workloads. Add the rapid_llm equivalent: a
   high-acceptance load (code continuation) shows positive
   TPOT/throughput gain; a low-acceptance load (high-temperature open
   generation) degrades <= 5%, and the adaptive cutoff (disable speculation
   below an acceptance threshold) triggers correctly. Report acceptance
   rate and average accepted tokens per step, on/off and across regimes.
3. The four coupling points are test-covered, including 0% and 100%
   acceptance edges.
4. Cross-matrix, cell by cell (each cell = greedy-identical + no perf
   regression): speculation x {CUDA Graph, TP, quantization, comm-compute
   overlap, chunked prefill, prefix caching}; negative interactions
   recorded with the cause.
5. Soak: >= 30 minutes at high concurrency — no state corruption, memory
   curve flat.
6. Library isolation: every proposer has a mock; unit tests never need the
   real draft model; an upstream interface change fails a test immediately.
7. Extensibility demo: adding a proposer (a mock is fine) touches only its
   file plus one registration.

## Deliverables

1. One interface document per draft source.
2. Design note: the three-way separation, the integration points, the four
   coupling points.
3. Proposer support matrix: method x interface version x status, with the
   explicit unsupported list.
4. Test and benchmark report per `model-benchmark-and-report`, plus
   speculation metrics: acceptance rate and average accepted tokens per
   step, on/off and across acceptance regimes.
5. README: principle, switches and tuning knobs, with a recorded on/off
   speed comparison.
6. Git: new branch — interface+core / each proposer / runner integration /
   cross-validation and docs as separate commits, each green alone.

## Cross-references

- `model-benchmark-and-report` — evidence rules for the speedup claims
- `kernel-microbenchmark` — when the rejection sampler or draft forward
  shows up in profiles and earns a Triton port
- `write-test` — test placement; `locate-numeric-divergence` — when the
  on/off token streams disagree
- `add-model` — `supports_cuda_graph=False` precedent for capture opt-out;
  draft-model proposers reuse its weight-loading machinery
- vLLM reference (local checkout `open_source/vllm`):
  `vllm/v1/worker/gpu/spec_decode/` — MRV2 speculator hierarchy and factory;
  `vllm/v1/sample/rejection_sampler.py` — the PyTorch rejection-sampling
  algorithm; `tests/v1/e2e/spec_decode/acceptance_rates/` — the
  acceptance-rate test pattern
