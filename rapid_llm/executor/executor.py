"""Executor: the one interface an engine has for running a model pass.

:class:`Executor` is the abstract seam — ``execute`` a :class:`ModelInput`
in, sampled tokens out; :class:`UniProcExecutor` runs the worker in-process
while :class:`MultiprocExecutor` forwards plans to TP follower ranks.

Usage:
    executor = UniProcExecutor(engine, max_num_seqs, max_seq_len)
    tokens, logprobs = executor.execute(model_input)
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import socket
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..distributed.parallel_state import tensor_model_parallel_broadcast_object_list
from ..utils.logger import get_logger
from .worker import ModelInput, ModelWorker, PassLogprobs

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..engine.llm_engine import LLMEngine

_log = get_logger(__name__)

#: How long :meth:`MultiprocExecutor.shutdown` waits for a follower to notice the
#: stop signal before killing it (a follower's remaining work is one pass).
SHUTDOWN_TIMEOUT_S = 30.0

#: How long a teardown waits for ``destroy_process_group`` before giving up. Eager
#: groups destroy in ms; the wait is only for the graph-captured case, where NCCL's
#: abort can sit in a futex forever.
DESTROY_DEADLINE_S = 15.0


def _destroy_with_deadline(destroy: Callable[[], None], abandon: Callable[[], None]) -> None:
    """Tear the group down, but never park the caller on a wedged abort.

    ``destroy_process_group`` runs on a daemon thread with a deadline: eager engines
    destroy promptly; a communicator whose collectives were captured into a CUDA graph
    can block the abort indefinitely (a PyTorch/NCCL interaction nothing clears from
    outside). On the deadline the grid is abandoned: parallel-state globals reset to a
    world of one, and the wedged communicator dies with the process.
    """
    import threading

    teardown = threading.Thread(target=destroy, daemon=True)
    teardown.start()
    teardown.join(DESTROY_DEADLINE_S)
    if teardown.is_alive():
        _log.warning(
            "group teardown did not finish within %.0fs (a graph-captured "
            "NCCL communicator can wedge its abort); abandoning the group — "
            "it dies with this process",
            DESTROY_DEADLINE_S,
        )
        abandon()


class Executor(ABC):
    """Runs model passes on behalf of an engine.

    Implementations own the model, the KV cache and the processes holding them. The
    engine's contract is narrow: ask the slot count, submit a plan, shut down.
    """

    @property
    @abstractmethod
    def num_slots(self) -> int:
        """How many cache slots plans may address, i.e. the concurrency ceiling."""

    @property
    def num_kv_blocks(self) -> int:
        """Cache blocks the scheduler may hand out, or ``0`` if the executor cannot say.

        ``0`` leaves the scheduler to size its pool from the slot geometry (what a fake
        executor in a test wants); a real one reports its profiled cache.
        """
        return 0

    @abstractmethod
    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        """Run one pass: its sampled token ids, one per sampled row, and any
        logprob records the plan asked for (``None`` when none did)."""

    def execute_verify(
        self, model_input: ModelInput
    ) -> tuple[torch.Tensor, PassLogprobs | None, torch.Tensor | None]:
        """O5 speculative decoding: like execute but also returns full logits.

        Default falls back to execute with ``None`` logits. Subclasses override
        to actually return the logits tensor for per-position verification.
        """
        tokens, records = self.execute(model_input)
        return tokens, records, None

    @abstractmethod
    def shutdown(self) -> None:
        """Release whatever the executor owns beyond this object's lifetime."""

    def readback_async(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Stage sampled tokens for the host without waiting for their pass.

        Default is the blocking degradation (``.cpu()``, no event), keeping the
        launch/harvest contract for executors with no copy stream (fakes, CPU workers).
        Real executors forward to their worker's pool.
        """
        return tokens.cpu(), None

    def release_readback(  # noqa: B027
        self, host: torch.Tensor
    ) -> None:
        """Return a staged token buffer once the host has read it.

        Default is a no-op, matching :meth:`readback_async`'s blocking
        degradation: a ``.cpu()`` result owns its own storage, so there is no
        ring buffer to give back. Real executors forward to their worker's pool.
        """

    def timeline_summary(self) -> str:
        """Region table of the streams this executor ran on, for overlap diagnostics.

        Empty unless ``RAPID_LLM_OVERLAP_TIMELINE`` is on, so callers print it
        unconditionally; an executor owning no streams keeps this default.
        """
        return ""


class UniProcExecutor(Executor):
    """One process, one model, no message passing.

    Args:
        engine: Built :class:`~rapid_llm.engine.llm_engine.LLMEngine`; the executor
            takes its KV cache over.
        max_num_seqs: Concurrency ceiling.
        max_seq_len: Context bound.
        pipeline: Worker feeds decode inputs back on the device (O2); ``None`` defers
            to :data:`~rapid_llm.executor.worker.PIPELINE_ENV`.
    """

    def __init__(
        self,
        engine: LLMEngine,
        max_num_seqs: int,
        max_seq_len: int,
        *,
        pipeline: bool | None = None,
    ) -> None:
        self._worker = ModelWorker(engine, max_num_seqs, max_seq_len, pipeline=pipeline)

    @property
    def num_slots(self) -> int:
        return self._worker.num_slots

    @property
    def num_kv_blocks(self) -> int:
        return self._worker.num_kv_blocks

    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        return self._worker.execute(model_input)

    def execute_verify(
        self, model_input: ModelInput
    ) -> tuple[torch.Tensor, PassLogprobs | None, torch.Tensor | None]:
        return self._worker.execute_verify(model_input)

    def readback_async(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        return self._worker.readback(tokens)

    def release_readback(self, host: torch.Tensor) -> None:
        self._worker.release_readback(host)

    def timeline_summary(self) -> str:
        return self._worker.timeline.summary()

    def shutdown(self) -> None:
        """Nothing to tear down: the caller still owns the engine it passed in."""


class MultiprocExecutor(Executor):
    """Tensor parallelism: this rank plans, every rank runs.

    The driver is rank 0 *and* a worker, so TP size two costs two processes, not three.
    Each :meth:`execute` publishes the plan on the CPU group then does its share of the
    forward; the model's collectives and the sampler line the ranks up.

    Args:
        engine: This rank's engine, holding its weight shard.
        max_num_seqs: Concurrency ceiling.
        max_seq_len: Context bound.
        followers: Processes running ranks 1.. (from :func:`launch_tensor_parallel`);
            empty when someone else owns them (CLI, DP controller), so shutdown only
            sends the stop signal.
        pipeline: Worker feeds decode inputs back on the device (O2); ``None`` defers
            to :data:`~rapid_llm.executor.worker.PIPELINE_ENV` (how followers learn
            the driver's choice).
    """

    def __init__(
        self,
        engine: LLMEngine,
        max_num_seqs: int,
        max_seq_len: int,
        followers: Sequence[mp.process.BaseProcess] = (),
        *,
        pipeline: bool | None = None,
    ) -> None:
        self._worker = ModelWorker(engine, max_num_seqs, max_seq_len, pipeline=pipeline)
        self._followers = tuple(followers)
        self._live = True

    @property
    def num_slots(self) -> int:
        return self._worker.num_slots

    @property
    def num_kv_blocks(self) -> int:
        return self._worker.num_kv_blocks

    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        ensure_followers_alive(self._followers)
        tensor_model_parallel_broadcast_object_list(model_input)
        return self._worker.execute(model_input)

    def execute_verify(
        self, model_input: ModelInput
    ) -> tuple[torch.Tensor, PassLogprobs | None, torch.Tensor | None]:
        ensure_followers_alive(self._followers)
        tensor_model_parallel_broadcast_object_list(model_input)
        return self._worker.execute_verify(model_input)

    def readback_async(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        # Rank 0's copy is the only one that matters: followers discard their tokens
        # as they discard their sampled results.
        return self._worker.readback(tokens)

    def release_readback(self, host: torch.Tensor) -> None:
        self._worker.release_readback(host)

    def timeline_summary(self) -> str:
        """Only this rank's regions; the followers trace their own streams."""
        return self._worker.timeline.summary()

    def shutdown(self) -> None:
        """Tell the followers to leave their loop, then reap them.

        Idempotent, and conditional on the group being whole: broadcasting the stop
        signal to a dead rank would block forever, so a crashed follower is joined
        without being asked. An empty follower tuple means someone else owns the
        processes. Owning the followers means owning the rank-0 half of their group,
        so a non-empty tuple tears the group down too (left standing, it re-shards the
        next engine this process builds).
        """
        if not self._live:
            return
        self._live = False
        if all(process.is_alive() for process in self._followers):
            tensor_model_parallel_broadcast_object_list(None)
        if self._followers:
            # Our half of the group goes down BEFORE the reap: a follower's teardown
            # only completes when every rank destroys with it, so a follower whose
            # rank 0 is parked in ``join`` is stuck in its destructor. Invisible to an
            # eager run, but a graph-captured communicator has device-side state only
            # the group-wide destroy releases. The barrier lines ranks up at the destroy
            # itself, since ``ncclCommAbort`` is collective in some NCCL versions.
            from ..distributed.parallel_state import (
                abandon_parallel,
                destroy_parallel,
                tensor_model_parallel_barrier,
            )

            # Release this rank's captured graphs first — the same discipline the
            # follower teardown applies to its own: a captured region holds NCCL
            # kernels registered with the communicator, and the destroy below
            # would block on them (the deadline then abandons the group, wedging
            # any engine this process tries to build next).
            self._worker._runner.release_cuda_graph()
            tensor_model_parallel_barrier()
            _destroy_with_deadline(destroy_parallel, abandon_parallel)
        for process in self._followers:
            process.join(timeout=SHUTDOWN_TIMEOUT_S)
            if process.is_alive():
                _log.warning("tp follower pid %s did not exit; terminating", process.pid)
                process.terminate()


def ensure_followers_alive(followers: Sequence[mp.process.BaseProcess]) -> None:
    """Raise if any follower has exited, before a collective can hang on it.

    Every collective assumes all ranks arrive; when one dies the rest wait, and a
    silent hang is multi-process execution's worst failure — so the driver checks the
    cheap local fact (process alive) before the expensive global one. Ranks start at 1
    (rank 0 is the checker).
    """
    for rank, process in enumerate(followers, start=1):
        if not process.is_alive():
            raise RuntimeError(
                f"tensor-parallel rank {rank} (pid {process.pid}) exited with code "
                f"{process.exitcode}; see its traceback above"
            )


def free_port() -> int:
    """A port the OS says is free, so a rendezvous never inherits a stale one.

    A fixed default (29500) makes two engines collide and a crashed run's lingering
    socket break the next — both surface as a hang at rendezvous, not an error.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_tensor_parallel(
    tp_size: int,
    engine_kwargs: dict[str, Any],
    max_num_seqs: int,
    master_port: int | None = None,
    enable_expert_parallel: bool = False,
    device: str = "cuda",
    pipeline: bool | None = None,
) -> tuple[mp.process.BaseProcess, ...]:
    """Start ranks 1..``tp_size``-1 and join this process as rank 0.

    Blocks until the group has rendezvoused, so on return the caller may build its
    engine and the shard widths are already right (layers read the TP size from
    :mod:`rapid_llm.distributed.parallel_state`, not an argument).

    Args:
        tp_size: Ranks in the group, including this one.
        engine_kwargs: Constructor arguments every rank builds its engine from,
            minus ``device``, which is the rank's own GPU. Must be picklable.
        max_num_seqs: Concurrency ceiling, so followers size their scratch to
            match.
        master_port: Rendezvous port; rank 0 listens. Defaults to a free one.
        enable_expert_parallel: MoE expert split mode every rank's group state
            must agree on before its engine builds (see
            :func:`~rapid_llm.distributed.parallel_state.init_parallel`).

    Returns:
        The follower processes, in rank order.
    """
    from ..distributed.parallel_state import init_tensor_parallel

    master_port = free_port() if master_port is None else master_port
    context = mp.get_context("spawn")
    followers = [
        context.Process(
            target=run_follower,
            args=(
                rank,
                tp_size,
                engine_kwargs,
                max_num_seqs,
                master_port,
                enable_expert_parallel,
                device,
                pipeline,
            ),
            name=f"rapid-llm-tp{rank}",
            daemon=True,
        )
        for rank in range(1, tp_size)
    ]
    for process in followers:
        process.start()
    init_tensor_parallel(
        rank=0,
        world_size=tp_size,
        backend="gloo" if torch.device(device).type == "cpu" else "nccl",
        master_port=master_port,
        enable_expert_parallel=enable_expert_parallel,
    )
    return tuple(followers)


def serve_plans(engine: LLMEngine, max_num_seqs: int, *, pipeline: bool | None = None) -> None:
    """Run broadcast plans until the driver sends ``None``. The whole of a follower.

    A follower holds no scheduler, queue or stop criteria, and discards the tokens it
    samples (rank 0 sampled the same ones and detokenises them). What keeps ranks in
    step is identical code over an identical plan. Separate from :func:`run_follower`
    because who *starts* a follower varies (this module for a lone replica, the DP
    controller for a grid cell) while what it *does* must not.

    Args:
        engine: This rank's engine, holding its weight shard.
        max_num_seqs: Concurrency ceiling, so the scratch matches the driver's
            (``max_seq_len`` comes from the engine: it only sizes local scratch).
    """
    worker = ModelWorker(engine, max_num_seqs, engine.max_seq_len, pipeline=pipeline)
    while (plan := tensor_model_parallel_broadcast_object_list()) is not None:
        # Records are discarded as the tokens are: every rank computed identical ones
        # and rank 0 reports them.
        worker.execute(plan)


def run_follower(
    rank: int,
    tp_size: int,
    engine_kwargs: dict[str, Any],
    max_num_seqs: int,
    master_port: int,
    enable_expert_parallel: bool = False,
    device: str = "cuda",
    pipeline: bool | None = None,
) -> None:
    """Body of a non-driver tensor-parallel rank: rendezvous, build, serve plans.

    Module-level so ``spawn`` can pickle it by name.
    """
    from ..distributed.parallel_state import (
        abandon_parallel,
        destroy_parallel,
        init_tensor_parallel,
    )
    from ..engine.llm_engine import LLMEngine

    on_cpu = torch.device(device).type == "cpu"
    if not on_cpu:
        torch.cuda.set_device(rank)
    device = "cpu" if on_cpu else f"cuda:{rank}"
    init_tensor_parallel(
        rank=rank,
        world_size=tp_size,
        backend="gloo" if on_cpu else "nccl",
        master_port=master_port,
        enable_expert_parallel=enable_expert_parallel,
    )
    engine = None
    try:
        engine = LLMEngine(
            device=device,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=enable_expert_parallel,
            **engine_kwargs,
        )
        _log.info("tp rank %d ready on %s", rank, device)
        serve_plans(engine, max_num_seqs, pipeline=pipeline)
    except BaseException:
        # A follower that dies here would otherwise vanish silently: the spawn
        # parent only reports an exception that propagates through
        # ``Process._bootstrap``, so a crash whose cleanup below raises a
        # second error (or that trips a native teardown path) prints nothing
        # and the driver only sees a closed connection. Announce the failure
        # on this rank's own stderr before the teardown gets a chance.
        traceback.print_exc()
        raise
    finally:
        # Release the captured graphs before the group: they hold NCCL kernels
        # registered with the communicator, and destroy_parallel would block on
        # them (this rank would then be terminated, hanging rank 0 in turn).
        if engine is not None:
            engine.model_runner.release_cuda_graph()
        # Meet the driver's shutdown barrier before this process exits: the
        # driver lines ranks up at the destroy itself (executor.shutdown), and
        # a follower that exits first closes the gloo pair under that barrier
        # — the driver reads it as a crashed peer. A driver already gone turns
        # the barrier into an error of its own; the destroy still runs below.
        from ..distributed.parallel_state import tensor_model_parallel_barrier

        with contextlib.suppress(Exception):
            tensor_model_parallel_barrier()
        # Same deadline the driver takes (``MultiprocExecutor.shutdown``): a
        # communicator whose collectives a graph captured can wedge its abort
        # forever, and a follower parked there gets SIGTERM'd by the driver's
        # join timeout — the terminate is what surfaces as an interpreter
        # abort. Abandoning keeps both sides' exits clean.
        _destroy_with_deadline(destroy_parallel, abandon_parallel)
