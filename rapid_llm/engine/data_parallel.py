"""Data-parallel scheduler process controller and routing policies.

The controller owns the DP x TP process grid, typed command/event channels and
request-to-replica routing. Each replica leader runs ``Scheduler.run_event_loop``;
TP followers only execute plans broadcast by that leader.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import queue
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch.multiprocessing as mp

from ..utils.logger import get_logger
from .outputs import CompletionOutput, RequestOutput
from .prefix_cache import PREFIX_CACHE_BLOCK_SIZE, iter_block_hashes
from .sampler import PositionLogprobs, SamplingParams
from .scheduler import DEFAULT_MAX_CHUNK_SIZE, DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS
from .scheduler_ipc import (
    PROTOCOL_VERSION,
    AbortRequest,
    AddRequest,
    AddRequestBatch,
    ReplicaFailed,
    ReplicaReady,
    SchedulerEvents,
    SchedulerFailed,
    SchedulerReady,
    ShutdownScheduler,
    UtilityRequest,
    WakeScheduler,
)

logger = get_logger(__name__)

LOAD_BALANCE_POLICIES = ("round_robin", "total_requests", "total_tokens", "cache_aware")
STARTUP_TIMEOUT_S = 900.0
_LIVENESS_POLL_S = 0.25
_JOIN_GRACE_S = 30.0
_TERMINATE_GRACE_S = 10.0


class _LoadPolicy(ABC):
    """Select replicas and account for each request until its exact completion."""

    needs_token_ids = False

    def __init__(self, dp_size: int) -> None:
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        self.dp_size = dp_size
        self._anonymous_ids = itertools.count()

    def _request_id(self, request_id: str | None) -> str:
        return request_id or f"policy-{next(self._anonymous_ids)}"

    @abstractmethod
    def select(
        self,
        estimated_tokens: int = 0,
        token_ids: Sequence[int] | None = None,
        *,
        request_id: str | None = None,
    ) -> int:
        """Choose a replica and record this request's charge."""

    def release(self, request_id: str) -> None:  # noqa: B027
        """Release the exact request charge; stateless policies do nothing."""


class RoundRobinPolicy(_LoadPolicy):
    """Choose replicas in index order without tracking load."""

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._counter = 0

    def select(
        self,
        estimated_tokens: int = 0,
        token_ids: Sequence[int] | None = None,
        *,
        request_id: str | None = None,
    ) -> int:
        replica = self._counter % self.dp_size
        self._counter += 1
        return replica


class _CountingPolicy(_LoadPolicy):
    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._load = [0] * dp_size
        self._charges: dict[str, tuple[int, int]] = {}

    @abstractmethod
    def _weight(self, estimated_tokens: int) -> int:
        pass

    def select(
        self,
        estimated_tokens: int = 0,
        token_ids: Sequence[int] | None = None,
        *,
        request_id: str | None = None,
    ) -> int:
        request_id = self._request_id(request_id)
        replica = min(range(self.dp_size), key=self._load.__getitem__)
        charge = self._weight(estimated_tokens)
        self._load[replica] += charge
        self._charges[request_id] = (replica, charge)
        return replica

    def release(self, request_id: str) -> None:
        charge = self._charges.pop(request_id, None)
        if charge is None:
            return
        replica, amount = charge
        self._load[replica] = max(0, self._load[replica] - amount)

    @property
    def load(self) -> tuple[int, ...]:
        return tuple(self._load)


class RequestCountPolicy(_CountingPolicy):
    """Choose the replica with the fewest in-flight requests."""

    def _weight(self, estimated_tokens: int) -> int:
        return 1


class TokenCountPolicy(_CountingPolicy):
    """Choose the replica with the fewest estimated in-flight prompt tokens."""

    def _weight(self, estimated_tokens: int) -> int:
        return max(1, estimated_tokens)


class CacheAwarePolicy(_LoadPolicy):
    """Balance uncached prefill cost while preserving prefix affinity."""

    needs_token_ids = True
    DEFAULT_INDEX_CAPACITY = 65536

    def __init__(
        self,
        dp_size: int,
        *,
        block_size: int = PREFIX_CACHE_BLOCK_SIZE,
        hash_seed: int = 0,
        index_capacity: int = DEFAULT_INDEX_CAPACITY,
    ) -> None:
        super().__init__(dp_size)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if index_capacity < 1:
            raise ValueError(f"index_capacity must be >= 1, got {index_capacity}")
        self.block_size = block_size
        self.hash_seed = hash_seed
        self.index_capacity = index_capacity
        self._resident: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(dp_size)]
        self._load = [0] * dp_size
        self._charges: dict[str, tuple[int, int]] = {}

    def select(
        self,
        estimated_tokens: int = 0,
        token_ids: Sequence[int] | None = None,
        *,
        request_id: str | None = None,
    ) -> int:
        request_id = self._request_id(request_id)
        hashes = (
            list(iter_block_hashes(token_ids, self.block_size, self.hash_seed)) if token_ids else []
        )
        prompt_tokens = len(token_ids) if token_ids else max(0, estimated_tokens)
        costs = []
        cached_by_replica = []
        for replica in range(self.dp_size):
            cached = self._cached_blocks(replica, hashes)
            cached_by_replica.append(cached)
            costs.append(self._load[replica] + self._uncached(prompt_tokens, cached))
        replica = min(range(self.dp_size), key=costs.__getitem__)
        charge = self._uncached(prompt_tokens, cached_by_replica[replica])
        self._load[replica] += charge
        self._charges[request_id] = (replica, charge)
        self._remember(replica, hashes)
        return replica

    def release(self, request_id: str) -> None:
        charge = self._charges.pop(request_id, None)
        if charge is None:
            return
        replica, amount = charge
        self._load[replica] = max(0, self._load[replica] - amount)

    @property
    def load(self) -> tuple[int, ...]:
        return tuple(self._load)

    def cached_tokens(self, token_ids: Sequence[int], replica: int) -> int:
        hashes = list(iter_block_hashes(token_ids, self.block_size, self.hash_seed))
        return self._cached_blocks(replica, hashes) * self.block_size

    def resident_blocks(self, replica: int) -> int:
        return len(self._resident[replica])

    def _uncached(self, prompt_tokens: int, cached_blocks: int) -> int:
        return max(1, prompt_tokens - cached_blocks * self.block_size)

    def _cached_blocks(self, replica: int, hashes: Sequence[int]) -> int:
        resident = self._resident[replica]
        cached = 0
        for block_hash in hashes:
            if block_hash not in resident:
                break
            cached += 1
        return cached

    def _remember(self, replica: int, hashes: Sequence[int]) -> None:
        resident = self._resident[replica]
        for block_hash in hashes:
            resident[block_hash] = None
            resident.move_to_end(block_hash)
        while len(resident) > self.index_capacity:
            resident.popitem(last=False)


def create_load_policy(name: str, dp_size: int) -> _LoadPolicy:
    policies = {
        "round_robin": RoundRobinPolicy,
        "total_requests": RequestCountPolicy,
        "total_tokens": TokenCountPolicy,
        "cache_aware": CacheAwarePolicy,
    }
    try:
        policy = policies[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown load_balancer {name!r}; choose from {LOAD_BALANCE_POLICIES}"
        ) from exc
    return policy(dp_size)


def _run_replica_process(
    global_rank: int,
    dp_rank: int,
    tp_rank: int,
    dp_size: int,
    tp_size: int,
    model: str,
    scheduler_kwargs: dict[str, Any],
    enable_dp_attention: bool,
    command_queue,
    event_queue,
) -> None:
    """Run one DP x TP rank; only replica leaders own schedulers and IPC."""
    import torch

    from .. import __version__
    from ..distributed.parallel_state import init_parallel
    from ..executor.executor import serve_plans
    from .llm import LLM
    from .scheduler import Scheduler

    scheduler = None
    try:
        options = dict(scheduler_kwargs)
        on_cpu = options.pop("device", "cuda") == "cpu"
        if not on_cpu:
            torch.cuda.set_device(global_rank)
        expert_parallel = options.get("enable_expert_parallel", False)
        init_parallel(
            global_rank=global_rank,
            tp_size=tp_size,
            dp_size=dp_size,
            enable_expert_parallel=expert_parallel,
            enable_dp_attention=enable_dp_attention,
            backend="gloo" if on_cpu else "nccl",
        )
        device = "cpu" if on_cpu else f"cuda:{global_rank}"
        if tp_rank == 0:
            scheduler = Scheduler.from_pretrained(model, device=device, **options)
            ready = SchedulerReady(
                protocol_version=PROTOCOL_VERSION,
                engine_version=__version__,
                max_model_len=scheduler.config.max_seq_len,
                num_gpu_blocks=scheduler.num_kv_blocks,
                max_num_seqs=scheduler.planner.max_num_seqs,
            )
        else:
            model_keys = {
                "tokenizer",
                "max_seq_len",
                "max_gpu_num_blocks",
                "use_cuda_graph",
                "quantization",
                "tensor_parallel_size",
                "enable_expert_parallel",
                "kv_cache_dtype",
                "cuda_graph_lazy",
                "hf_overrides",
            }
            follower = LLM(
                model=model,
                device=device,
                **{key: value for key, value in options.items() if key in model_keys},
            )
            ready = None
        event_queue.put(ReplicaReady(global_rank, ready))
    except BaseException:
        event_queue.put(ReplicaFailed(global_rank, traceback.format_exc()))
        return

    if scheduler is None:
        try:
            serve_plans(
                follower, scheduler_kwargs["max_num_seqs"], pipeline=options.get("pipeline")
            )
        except BaseException:
            event_queue.put(ReplicaFailed(global_rank, traceback.format_exc()))
            raise
        return

    try:
        scheduler.run_event_loop(
            command_queue,
            event_queue,
            parent_pid=os.getppid(),
            enable_dp_attention=enable_dp_attention,
        )
    except BaseException:
        event_queue.put(ReplicaFailed(global_rank, traceback.format_exc()))
        raise
    finally:
        scheduler.shutdown()


@dataclass(frozen=True, slots=True)
class _Route:
    replica: int


class DataParallelController:
    """Own scheduler processes, typed IPC, routing state and blocking generation."""

    def __init__(
        self,
        model: str,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        load_balancer: str = "round_robin",
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill: bool = True,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        enable_prefix_cache: bool = False,
        prefix_cache_blocks: int | None = None,
        enable_preemption: bool = False,
        enable_dp_attention: bool = False,
        *,
        startup_timeout_s: float = STARTUP_TIMEOUT_S,
        process_context: Any = None,
        process_target: Any = _run_replica_process,
        **engine_kwargs: Any,
    ) -> None:
        import torch

        if data_parallel_size < 1:
            raise ValueError(f"data_parallel_size must be >= 1, got {data_parallel_size}")
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
        if "device" in engine_kwargs and engine_kwargs["device"] != "cpu":
            raise ValueError(
                "CUDA device is derived from the replica's rank; use device='cpu' for CPU workers"
            )
        if load_balancer not in LOAD_BALANCE_POLICIES:
            raise ValueError(
                f"unknown load_balancer {load_balancer!r}; choose from {LOAD_BALANCE_POLICIES}"
            )
        if enable_dp_attention:
            if data_parallel_size <= 1:
                raise ValueError("DP attention requires data_parallel_size > 1")
            if not engine_kwargs.get("enable_expert_parallel", False):
                raise ValueError("DP attention requires enable_expert_parallel=True")
            if engine_kwargs.get("use_cuda_graph", True):
                raise ValueError(
                    "DP attention does not support CUDA Graph yet; set use_cuda_graph=False"
                )
            if os.environ.get("LITE_LLAMA_SPECULATE", "0").strip().lower() in {
                "1",
                "true",
                "on",
            }:
                raise ValueError("DP attention does not support speculative decoding yet")

        self.model = model
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_dp_attention = enable_dp_attention
        self.world_size = data_parallel_size * tensor_parallel_size
        visible = torch.cuda.device_count()
        if engine_kwargs.get("device") != "cpu" and self.world_size > visible:
            raise ValueError(
                f"data_parallel_size={data_parallel_size} x "
                f"tensor_parallel_size={tensor_parallel_size} needs {self.world_size} GPUs, "
                f"but only {visible} are visible"
            )

        self._policy = create_load_policy(load_balancer, data_parallel_size)
        self._select = self._policy.select
        self._routes: dict[str, _Route] = {}
        self._route_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._tokenizer = None
        self._request_ids = itertools.count()
        self._closed = False
        self._failure: RuntimeError | None = None

        scheduler_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_chunked_prefill": enable_chunked_prefill,
            "max_chunk_size": max_chunk_size,
            "enable_prefix_cache": enable_prefix_cache,
            "prefix_cache_blocks": prefix_cache_blocks,
            "enable_preemption": enable_preemption,
            **engine_kwargs,
        }
        if scheduler_kwargs.get("use_cuda_graph") is None:
            scheduler_kwargs.pop("use_cuda_graph", None)
        self._scheduler_kwargs = scheduler_kwargs

        context = process_context or mp.get_context("spawn")
        self._command_queues = [context.Queue() for _ in range(data_parallel_size)]
        self._event_queue = context.Queue()
        self._workers = [
            context.Process(
                target=process_target,
                args=(
                    global_rank,
                    global_rank // tensor_parallel_size,
                    global_rank % tensor_parallel_size,
                    data_parallel_size,
                    tensor_parallel_size,
                    model,
                    scheduler_kwargs,
                    enable_dp_attention,
                    self._command_queues[global_rank // tensor_parallel_size],
                    self._event_queue,
                ),
                daemon=False,
            )
            for global_rank in range(self.world_size)
        ]
        try:
            for worker in self._workers:
                worker.start()
            self._ready = self._await_ready(startup_timeout_s)
        except BaseException:
            self.shutdown()
            raise

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> DataParallelController:
        return cls(model, **kwargs)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from .llm_engine import LLMEngine

            self._tokenizer = LLMEngine._load_tokenizer(self.model)
        return self._tokenizer

    @property
    def scheduler_info(self) -> SchedulerReady:
        return self._ready

    def send(self, command) -> None:
        """Route one typed command to its replica or broadcast it to the grid."""
        if self._closed:
            raise RuntimeError("DataParallelController has been shut down")
        if self._failure is not None:
            raise self._failure
        if isinstance(command, AddRequestBatch):
            request_ids = [request.request_id for request in command.requests]
            with self._route_lock:
                if len(set(request_ids)) != len(request_ids):
                    raise ValueError("request ids in a batch must be unique")
                duplicate = next(
                    (request_id for request_id in request_ids if request_id in self._routes), None
                )
                if duplicate is not None:
                    raise ValueError(f"request id {duplicate!r} is already active")
                grouped: list[list[AddRequest]] = [[] for _ in range(self.data_parallel_size)]
                for request in command.requests:
                    token_ids = request.prompt_token_ids
                    replica = self._select(len(token_ids), token_ids, request_id=request.request_id)
                    self._routes[request.request_id] = _Route(replica)
                    grouped[replica].append(request)
            try:
                for replica, requests in enumerate(grouped):
                    if requests:
                        self._send_replica(replica, AddRequestBatch(tuple(requests)))
            except BaseException:
                for request_id in request_ids:
                    self._release(request_id)
                raise
            return
        if isinstance(command, AddRequest):
            token_ids = command.prompt_token_ids
            with self._route_lock:
                if command.request_id in self._routes:
                    raise ValueError(f"request id {command.request_id!r} is already active")
                replica = self._select(len(token_ids), token_ids, request_id=command.request_id)
                self._routes[command.request_id] = _Route(replica)
            try:
                self._send_replica(replica, command)
            except BaseException:
                self._release(command.request_id)
                raise
            return
        if isinstance(command, AbortRequest):
            with self._route_lock:
                route = self._routes.get(command.request_id)
            if route is not None:
                self._send_replica(route.replica, command)
                self._release(command.request_id)
            return
        if isinstance(command, UtilityRequest | WakeScheduler):
            for command_queue in self._command_queues:
                command_queue.put(command)
            return
        if isinstance(command, ShutdownScheduler):
            for command_queue in self._command_queues:
                command_queue.put(command)
            return
        raise TypeError(f"unsupported scheduler command: {type(command).__name__}")

    def receive(self, timeout: float | None = None):
        """Receive one typed event while detecting silent worker death."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = _LIVENESS_POLL_S
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                wait = min(wait, remaining)
            try:
                event = self._event_queue.get(timeout=wait)
            except queue.Empty:
                dead = [worker for worker in self._workers if not worker.is_alive()]
                if dead and not self._closed:
                    detail = (
                        f"data-parallel rank(s) {[worker.pid for worker in dead]} exited "
                        f"with codes {[worker.exitcode for worker in dead]}"
                    )
                    self._failure = RuntimeError(detail)
                    raise self._failure from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise
                continue

            if isinstance(event, ReplicaFailed):
                self._failure = RuntimeError(
                    f"data-parallel rank {event.global_rank} failed:\n{event.message}"
                )
                return SchedulerFailed(str(self._failure))
            if isinstance(event, SchedulerEvents):
                for output in event.outputs:
                    if output.finished or output.error is not None:
                        self._release(output.request_id)
            return event

    def is_alive(self) -> bool:
        return (
            self._failure is None
            and not self._closed
            and all(worker.is_alive() for worker in self._workers)
        )

    @property
    def exitcode(self):
        return tuple(worker.exitcode for worker in self._workers)

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Blocking facade over the same typed request/event path used by serving."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompts:
            return []
        params = sampling_params or SamplingParams()
        token_ids = [
            list(self.tokenizer.encode(prompt, add_special_tokens=True)) for prompt in prompts
        ]
        request_ids = [f"batch-{next(self._request_ids)}" for _ in prompts]
        generated: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
        reasons: dict[str, str | None] = {}
        output_logprobs: dict[str, list[PositionLogprobs]] = {
            request_id: [] for request_id in request_ids
        }
        prompt_logprobs: dict[str, list[PositionLogprobs | None] | None] = {}

        with self._sync_lock:
            try:
                arrival_time = time.monotonic()
                self.send(
                    AddRequestBatch(
                        tuple(
                            AddRequest(
                                request_id,
                                tuple(ids),
                                params,
                                arrival_time=arrival_time,
                                stream=params.logprobs is not None,
                            )
                            for request_id, ids in zip(request_ids, token_ids, strict=True)
                        )
                    )
                )
                pending = set(request_ids)
                while pending:
                    event = self.receive()
                    if isinstance(event, SchedulerFailed):
                        raise RuntimeError(event.message)
                    if not isinstance(event, SchedulerEvents):
                        continue
                    for output in event.outputs:
                        if output.request_id not in generated:
                            continue
                        generated[output.request_id].extend(output.new_token_ids)
                        if output.delta_logprobs is not None:
                            output_logprobs[output.request_id].append(output.delta_logprobs)
                        if output.prompt_logprobs is not None:
                            prompt_logprobs[output.request_id] = list(output.prompt_logprobs)
                        if output.error is not None:
                            raise ValueError(output.error)
                        if output.finished:
                            reasons[output.request_id] = output.finish_reason
                            pending.discard(output.request_id)
            except BaseException:
                for request_id in request_ids:
                    with contextlib.suppress(Exception):
                        self.send(AbortRequest(request_id))
                raise

        outputs = []
        for prompt, request_id in zip(prompts, request_ids, strict=True):
            records = output_logprobs[request_id] or None
            outputs.append(
                RequestOutput(
                    prompt=prompt,
                    outputs=[
                        CompletionOutput(
                            0,
                            self.tokenizer.decode(generated[request_id], skip_special_tokens=True),
                            reasons.get(request_id),
                            records,
                        )
                    ],
                    prompt_logprobs=prompt_logprobs.get(request_id),
                )
            )
        return outputs

    def _send_replica(self, replica: int, command) -> None:
        self._command_queues[replica].put(command)
        if self.enable_dp_attention:
            for peer, command_queue in enumerate(self._command_queues):
                if peer != replica:
                    command_queue.put(WakeScheduler())

    def _release(self, request_id: str) -> None:
        with self._route_lock:
            route = self._routes.pop(request_id, None)
            if route is not None:
                self._policy.release(request_id)

    def _await_ready(self, timeout_s: float) -> SchedulerReady:
        pending = set(range(self.world_size))
        leader_ready: SchedulerReady | None = None
        deadline = time.monotonic() + timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"data-parallel ranks did not report ready within {timeout_s:.0f}s"
                )
            try:
                event = self.receive(timeout=min(_LIVENESS_POLL_S, remaining))
            except queue.Empty:
                continue
            if isinstance(event, SchedulerFailed):
                raise RuntimeError(event.message)
            if not isinstance(event, ReplicaReady):
                continue
            pending.discard(event.global_rank)
            if event.scheduler is not None:
                if event.scheduler.protocol_version != PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"scheduler protocol mismatch on rank {event.global_rank}: "
                        f"child v{event.scheduler.protocol_version}, parent v{PROTOCOL_VERSION}"
                    )
                leader_ready = leader_ready or event.scheduler
        if leader_ready is None:
            raise RuntimeError("data-parallel grid started without a scheduler leader")
        logger.info(
            "data parallel ready: %d replicas x TP %d = %d ranks (%s)",
            self.data_parallel_size,
            self.tensor_parallel_size,
            self.world_size,
            type(self._policy).__name__,
        )
        return leader_ready

    def shutdown(self) -> None:
        """Stop admission, drain schedulers, and reclaim every rank. Idempotent."""
        if self._closed:
            return
        with contextlib.suppress(Exception):
            self.send(ShutdownScheduler())
        self._closed = True
        for worker in self._workers:
            worker.join(timeout=_JOIN_GRACE_S)
            if worker.is_alive():
                logger.warning("data-parallel rank %s did not stop; terminating", worker.pid)
                worker.terminate()
                worker.join(timeout=_TERMINATE_GRACE_S)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=_TERMINATE_GRACE_S)
        for channel in (*self._command_queues, self._event_queue):
            with contextlib.suppress(Exception):
                channel.close()
                channel.join_thread()

    def __enter__(self) -> DataParallelController:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
