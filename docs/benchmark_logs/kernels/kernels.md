# Triton 内核 benchmark 测试

各算子的 benchmark 结果图按算子分类存放于 [`../../images/kernels_benchamrk/`](../../images/kernels_benchamrk/)，目录结构如下（每个算子一个子目录，含 `.png` 图与对应 `.csv` 原始数据）：

```text
kernels_benchamrk/
├── softmax/          # softmax
├── matmul/           # linear / matmul（fp16、fp8）
├── rmsnorm/          # rmsnorm、skip_rmsnorm 融合算子
├── layernorm/        # layernorm
├── mlp_silu/         # MLP + SiLU
├── attention/        # flashattention / flashattention_v2_no_pad / flashdecoding
├── token_embedding/  # token embedding
└── misc/             # 其他汇总结果（result.png、results.html）
```

## softmax

softmax benchmark test result：

![softmax](../../images/kernels_benchamrk/softmax/softmax-performance.png)

单程 native kernel 对比：

![softmax-native](../../images/kernels_benchamrk/softmax/softmax-native-performance.png)

## linear（matmul）

linear(matmul) benchmark test result：

![matmul-fp16](../../images/kernels_benchamrk/matmul/matmul-performance-fp16.png)

fp8 精度：

![matmul-fp8](../../images/kernels_benchamrk/matmul/matmul-performance-fp8.png)

## rmsnorm

rmsnorm benchmark test result：

![rms-norm](../../images/kernels_benchamrk/rmsnorm/rms-norm-forward.png)

残差 skip 与 rmsnorm 融合后的 `skip_rmsnorm` 算子：

![skip-rmsnorm](../../images/kernels_benchamrk/rmsnorm/skip_rmsnorm_benchmark.png)

## layernorm

layernorm benchmark test result：

![layer-norm-forward](../../images/kernels_benchamrk/layernorm/layer-norm-forward.png)

## mlp_silu

MLP_Silu test result：

![MLP_Silu](../../images/kernels_benchamrk/mlp_silu/mlp-silu-performance_ret.png)

## flashattention

flashattention benchmark test result：

![flashattention benchmark test](../../images/kernels_benchamrk/attention/fused-attention-batch8-head64-d64-fwd-causal=False.png) ![flashattention benchmark test](../../images/kernels_benchamrk/attention/fused-attention-batch4-head32-d64-fwd-causal=False.png)

## flashattention_v2_no_pad

flashattention_v2_no_pad benchmark test result：

![flashattention_v2_no_pad](../../images/kernels_benchamrk/attention/flashattention_nopad_benchamrk.png) ![flashattention_v2_no_pad](../../images/kernels_benchamrk/attention/flashattentionv2_nopad_benchamrk.png)

## flashdecoding

flashdecoding benchmark test result：

![flashdecoding](../../images/kernels_benchamrk/attention/flashdecoding_benchamrk.png)

## token_embedding

token embedding benchmark test result：

![token_embedding](../../images/kernels_benchamrk/token_embedding/token_embedding_benchmark.png)
