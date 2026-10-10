"""Pure rank geometry for prefill and decode context parallelism.

The launch world remains a ``DP x TP`` grid.  Inside each replica the TP axis is
factored as ``PCP x attention-TP``; DCP is a consecutive subgroup of an
attention-TP group.  Keeping this arithmetic free of ``torch.distributed`` makes
configuration validation deterministic and unit-testable before rendezvous.
"""

from __future__ import annotations

from dataclasses import dataclass

PCP_STRATEGIES = frozenset({"zigzag", "interleave"})
DCP_COMM_BACKENDS = frozenset({"ag_rs", "a2a"})


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    """Validated parallel widths and this rank's coordinates."""

    global_rank: int
    tp_size: int
    dp_size: int
    pcp_size: int = 1
    dcp_size: int = 1
    pcp_strategy: str = "zigzag"
    dcp_comm_backend: str = "ag_rs"

    def __post_init__(self) -> None:
        for name, value in (
            ("tp_size", self.tp_size),
            ("dp_size", self.dp_size),
            ("prefill_context_parallel_size", self.pcp_size),
            ("decode_context_parallel_size", self.dcp_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        world_size = self.tp_size * self.dp_size
        if not 0 <= self.global_rank < world_size:
            raise ValueError(
                f"global_rank {self.global_rank} is outside a {self.dp_size}x{self.tp_size} "
                f"grid of {world_size} ranks"
            )
        if self.tp_size % self.pcp_size:
            raise ValueError(
                f"tensor_parallel_size={self.tp_size} must be divisible by "
                f"prefill_context_parallel_size={self.pcp_size}"
            )
        if self.attention_tp_size % self.dcp_size:
            raise ValueError(
                f"attention tensor-parallel size {self.attention_tp_size} must be divisible by "
                f"decode_context_parallel_size={self.dcp_size}"
            )
        if self.pcp_strategy not in PCP_STRATEGIES:
            choices = ", ".join(sorted(PCP_STRATEGIES))
            raise ValueError(
                f"unknown prefill context-parallel strategy {self.pcp_strategy!r}; "
                f"choose from {choices}"
            )
        if self.dcp_comm_backend == "fi_a2a":
            raise ValueError(
                "dcp_comm_backend='fi_a2a' requires FlashInfer MNNVL support, which "
                "rapid_llm does not provide; use 'ag_rs' or 'a2a'"
            )
        if self.dcp_comm_backend not in DCP_COMM_BACKENDS:
            choices = ", ".join(sorted(DCP_COMM_BACKENDS))
            raise ValueError(
                f"unknown DCP communication backend {self.dcp_comm_backend!r}; "
                f"choose from {choices}"
            )

    @property
    def world_size(self) -> int:
        return self.tp_size * self.dp_size

    @property
    def dp_rank(self) -> int:
        return self.global_rank // self.tp_size

    @property
    def tp_rank(self) -> int:
        return self.global_rank % self.tp_size

    @property
    def attention_tp_size(self) -> int:
        return self.tp_size // self.pcp_size

    @property
    def pcp_rank(self) -> int:
        return self.tp_rank // self.attention_tp_size

    @property
    def attention_tp_rank(self) -> int:
        return self.tp_rank % self.attention_tp_size

    @property
    def dcp_rank(self) -> int:
        return self.attention_tp_rank % self.dcp_size

    @property
    def dcp_group_index(self) -> int:
        return self.attention_tp_rank // self.dcp_size

    def replica_members(self, dp_rank: int | None = None) -> list[int]:
        replica = self.dp_rank if dp_rank is None else dp_rank
        start = replica * self.tp_size
        return list(range(start, start + self.tp_size))

    def attention_tp_groups(self) -> list[list[int]]:
        groups: list[list[int]] = []
        width = self.attention_tp_size
        for replica in range(self.dp_size):
            base = replica * self.tp_size
            for pcp_rank in range(self.pcp_size):
                start = base + pcp_rank * width
                groups.append(list(range(start, start + width)))
        return groups

    def pcp_groups(self) -> list[list[int]]:
        groups: list[list[int]] = []
        width = self.attention_tp_size
        for replica in range(self.dp_size):
            base = replica * self.tp_size
            for attention_rank in range(width):
                groups.append(
                    [base + pcp_rank * width + attention_rank for pcp_rank in range(self.pcp_size)]
                )
        return groups

    def dcp_groups(self) -> list[list[int]]:
        groups: list[list[int]] = []
        for attention_group in self.attention_tp_groups():
            for start in range(0, len(attention_group), self.dcp_size):
                groups.append(attention_group[start : start + self.dcp_size])
        return groups


def resolve_topology(
    global_rank: int = 0,
    tp_size: int = 1,
    dp_size: int = 1,
    *,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_strategy: str = "zigzag",
    dcp_comm_backend: str = "ag_rs",
) -> ParallelTopology:
    """Build and validate the parallel topology without creating process groups."""

    return ParallelTopology(
        global_rank=global_rank,
        tp_size=tp_size,
        dp_size=dp_size,
        pcp_size=prefill_context_parallel_size,
        dcp_size=decode_context_parallel_size,
        pcp_strategy=prefill_context_parallel_strategy,
        dcp_comm_backend=dcp_comm_backend,
    )
