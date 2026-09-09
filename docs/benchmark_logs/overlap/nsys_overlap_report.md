# nsys overlap evidence — TP=2 decode, Qwen2.5-1.5B, batch 16, eager

Two payloads identical except the overlap switches
(`RAPID_LLM_OVERLAP`/`RAPID_LLM_TBO`/`RAPID_LLM_COMM_OVERLAP`);
traced with `nsys profile --trace=cuda`, kernels exported via
`nsys stats -r cuda_gpu_trace --format csv`, then aggregated per GPU
over every NCCL kernel (both warmup and steady passes — the overlap
behaviour is the same in both). Compute side excludes memcpys: a copy
engine overlaps a reduction for free and is not the claim under test.

| trace | gpu | NCCL kernels | NCCL total | hidden under compute | exposed (serial) |
| --- | --- | --- | --- | --- | --- |
| overlap off | 0 | 6778 | 1150.66 ms | 0.00 ms (0.0%) | 1150.66 ms (100.0%) |
| overlap off | 1 | 6778 | 213.77 ms | 0.00 ms (0.0%) | 213.77 ms (100.0%) |
| overlap on | 0 | 12934 | 2103.34 ms | 206.12 ms (9.8%) | 1897.22 ms (90.2%) |
| overlap on | 1 | 12934 | 1464.44 ms | 126.04 ms (8.6%) | 1338.41 ms (91.4%) |

GPU NVIDIA A10 (0): reduction time hidden under compute goes 0.0% -> 9.8%.
GPU NVIDIA A10 (1): reduction time hidden under compute goes 0.0% -> 8.6%.

Interconnect: PCIe (2x A10, no NVLink hardware); the fractions are
about this machine and say nothing about NVLink topologies.
