"""Parallelism state: the DP x TP rank grid and the collectives its axes need.

Re-exports :mod:`rapid_llm.distributed.parallel_state` so that rank queries
(``get_tensor_model_parallel_rank``) and group setup (``init_parallel``) come from
one import. The collective names follow vLLM's
``vllm.distributed.parallel_state`` spelling (``tensor_model_parallel_all_reduce``
and friends) so both codebases read the same at the call site.

The DP axis has two meanings, and both live here: independent replicas (nothing
to say across the axis) and :mod:`rapid_llm.distributed.dp_attention`, where the
replicas hold one model between them -- attention per rank, experts pooled
across them, which is how sglang and vLLM deploy DeepSeek-V3.

Usage:
    from rapid_llm.distributed import get_tensor_model_parallel_rank, init_parallel
"""

from .dp_attention import (
    DPMetadata,
    coordinate_tokens_across_dp,
    current_dp_metadata,
    dp_attention_region,
    dp_combine,
    dp_dispatch,
    dp_gather,
    dp_scatter,
)
from .parallel_state import (
    all_gatherv,
    all_to_all,
    data_parallel_all_gather,
    data_parallel_all_reduce,
    destroy_parallel,
    destroy_tensor_parallel,
    divide,
    dp_attention_enabled,
    expert_parallel_enabled,
    get_data_parallel_cpu_group,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_ep_group,
    get_ep_rank,
    get_ep_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_size,
    grid_coordinates,
    init_parallel,
    init_tensor_parallel,
    recv,
    reduce_scatter,
    reduce_scatterv,
    send,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_max,
    tensor_model_parallel_all_reduce_min,
    tensor_model_parallel_broadcast,
    tensor_model_parallel_broadcast_object_list,
    tensor_model_parallel_ranks_agree,
    warmup_collectives,
)
from .sequence_parallel import (
    SequenceParallelPass,
    SequenceParallelRegion,
    sequence_parallel_enabled,
    sequence_parallel_region,
    sp_active,
)

__all__ = [
    # Data-parallel attention: the step geometry the DP ranks agree on, and the
    # gather/scatter that pools their tokens for the expert stage
    "DPMetadata",
    # Sequence parallelism: the pass that marks the boundaries, and the region
    # they are boundaries of
    "SequenceParallelPass",
    "SequenceParallelRegion",
    "all_gatherv",
    "all_to_all",
    "coordinate_tokens_across_dp",
    "current_dp_metadata",
    "data_parallel_all_gather",
    "data_parallel_all_reduce",
    "destroy_parallel",
    "destroy_tensor_parallel",
    "divide",
    "dp_attention_enabled",
    "dp_attention_region",
    "dp_combine",
    "dp_dispatch",
    "dp_gather",
    "dp_scatter",
    "expert_parallel_enabled",
    "get_data_parallel_cpu_group",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_ep_group",
    "get_ep_rank",
    "get_ep_world_size",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "get_world_size",
    "grid_coordinates",
    "init_parallel",
    "init_tensor_parallel",
    "recv",
    "reduce_scatter",
    "reduce_scatterv",
    "send",
    "sequence_parallel_enabled",
    "sequence_parallel_region",
    "sp_active",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_all_reduce_max",
    "tensor_model_parallel_all_reduce_min",
    "tensor_model_parallel_broadcast",
    "tensor_model_parallel_broadcast_object_list",
    "tensor_model_parallel_ranks_agree",
    "warmup_collectives",
]
