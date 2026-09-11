"""Run a payload on a real TP process group so sharded layers can be tested.

``run_on_tp_ranks`` spawns one process per TP rank, initialises the
grid, and returns the payload's results — the bridge between CPU unit
tests and full TP engine tests.

Usage:
    results = run_on_tp_ranks(payload, tp_size=2)
"""

from __future__ import annotations

import logging
import os
import queue as queue_module
import socket
import subprocess
import time
import traceback
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from rapid_llm.distributed import parallel_state as ps

_log = logging.getLogger(__name__)


def _free_port() -> int:
    """Return a local rendezvous port without importing the model executor."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def needs_gpus(count: int):
    """Mark a test as needing ``count`` real devices, one per rank.

    Two marks in one: the skip for machines that lack the devices, and the
    ``gpu`` marker for the tier system. The tier must exclude the test even
    where the devices *exist* -- a CPU-tier run on a GPU box must not
    rendezvous NCCL just because it could, which is how a busy GPU once
    failed 23 unrelated tests through the state a failed rendezvous left
    behind.
    """
    skip = pytest.mark.skipif(
        torch.cuda.device_count() < count,
        reason=f"needs {count} CUDA devices, found {torch.cuda.device_count()}",
    )

    def decorate(func):
        return pytest.mark.gpu(skip(func))

    return decorate


def _worker(
    payload: Callable[[int], Any],
    rank: int,
    tp_size: int,
    dp_size: int,
    port: int,
    backend: str,
    results: mp.Queue,
    acks: mp.Queue,
    enable_expert_parallel: bool = False,
    enable_dp_attention: bool = False,
) -> None:
    """One rank: take a device, join the grid, run the payload, report answer or traceback.

    The device is claimed *before* ``init_parallel`` because nccl binds the calling
    thread's current device at rendezvous; leaving every rank on device 0 is the classic
    way to get a hang instead of a result. On gloo there is no device to claim. The
    traceback travels as text rather than being raised: an exception in a child process
    is invisible to pytest, and a rank that dies silently leaves its peers blocked in a
    collective with nothing to explain why.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)  # not setdefault: an inherited value is stale
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        ps.init_parallel(
            global_rank=rank,
            tp_size=tp_size,
            dp_size=dp_size,
            master_port=port,
            backend=backend,
            enable_expert_parallel=enable_expert_parallel,
            enable_dp_attention=enable_dp_attention,
        )
        results.put((rank, payload(rank), None))
    except BaseException:  # reported to the parent, which re-raises it verbatim
        results.put((rank, None, traceback.format_exc()))
    finally:
        ps.destroy_parallel()
    # torch tensors ride the queue as shared-memory fds the parent picks up
    # while *unpickling* its get(); a daemon worker that returns here can die
    # before that rendezvous, leaving the parent with ConnectionResetError.
    # Park until the parent signals every result has been drained.
    acks.get()


#: Slack above the pre-grid memory level the settle fence tolerates while it
#: waits: driver and sibling-process noise, orders below a dead rank's claim.
_SETTLE_MARGIN_MIB = 512

#: How long the settle fence may wait for a slow driver to finish reclaiming.
_SETTLE_TIMEOUT = 30.0


def _device_memory_used() -> dict[int, int] | None:
    """MiB in use per device, straight from nvidia-smi; ``None`` if it cannot answer.

    A subprocess and not ``torch.cuda``: querying through torch would build a
    CUDA context in the parent, whose memory would then pollute both the
    baseline and every payload that assumes the parent holds no device.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        _log.debug("nvidia-smi cannot answer (%s); skipping the settle fence", error)
        return None
    used: dict[int, int] = {}
    for line in out.strip().splitlines():
        index, mib = line.split(",")
        used[int(index.strip())] = int(mib.strip())
    return used


def _reap_workers(workers: list[mp.Process]) -> None:
    """Join every worker, escalating polite join -> SIGTERM -> SIGKILL.

    A worker stuck in a driver call does not process SIGTERM (the signal is
    handled at the next Python bytecode, not inside the ioctl), so a terminate
    that times out with no escalation leaves the process alive and holding its
    whole device claim -- the leak that poisons the next grid. SIGKILL cannot
    be caught: the process dies at the kernel's next chance and the driver
    starts releasing its memory.
    """
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=10)
        if worker.is_alive():
            _log.warning(
                "worker pid %s survived SIGKILL; its device memory stays claimed", worker.pid
            )


def _wait_for_devices_to_settle(
    baseline: dict[int, int], *, timeout: float = _SETTLE_TIMEOUT
) -> None:
    """Block until every device is back near the level seen before the grid.

    The release of a killed worker's memory is the driver's job, and it is
    asynchronous: a grid launched inside the teardown window asks for a fresh
    checkpoint on a device still holding the dead rank's gigabytes, and dies
    of OOM where nothing in the traceback looks like its cause. Bounded and
    best-effort -- the numbers also move for reasons that are not ours on a
    shared machine, so a straggler is warned about, never fatal.
    """
    deadline = time.monotonic() + timeout
    while True:
        used = _device_memory_used()
        if used is None:
            return
        stragglers = {
            index: (used.get(index, 0), floor)
            for index, floor in baseline.items()
            if used.get(index, 0) > floor + _SETTLE_MARGIN_MIB
        }
        if not stragglers:
            return
        if time.monotonic() >= deadline:
            _log.warning(
                "devices still above their pre-grid memory after %.0fs (used, pre-grid, MiB): %s",
                timeout,
                stragglers,
            )
            return
        time.sleep(0.5)


def _attempt(
    payload: Callable[[int], Any],
    tp_size: int,
    dp_size: int,
    timeout: float,
    backend: str,
    enable_expert_parallel: bool,
    enable_dp_attention: bool,
) -> list[Any]:
    """One grid on one freshly picked rendezvous port.

    The whole of :func:`run_on_tp_ranks` except the retry, so a lost port can
    be answered with a fresh one cleanly: the ``finally`` reaps every worker --
    up to SIGKILL -- and waits out the driver's memory release before this
    returns, and every other failure propagates untouched.
    """
    world_size = tp_size * dp_size
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        raise RuntimeError(f"{world_size} ranks need {world_size} devices, one each")

    # The memory levels this grid starts from; the fence at the bottom waits
    # for the devices to come back down to them.
    baseline = _device_memory_used() if backend == "nccl" else None

    context = mp.get_context("spawn")
    results: mp.Queue = context.Queue()
    acks: mp.Queue = context.Queue()  # workers park on it until results are drained
    port = _free_port()
    workers = [
        context.Process(
            target=_worker,
            args=(
                payload,
                rank,
                tp_size,
                dp_size,
                port,
                backend,
                results,
                acks,
                enable_expert_parallel,
                enable_dp_attention,
            ),
            daemon=True,
        )
        for rank in range(world_size)
    ]
    collected: dict[int, Any] = {}
    for worker in workers:
        worker.start()
    try:
        deadline = time.monotonic() + timeout
        while len(collected) < world_size:
            left = deadline - time.monotonic()
            if left <= 0:
                missing = sorted(set(range(world_size)) - set(collected))
                raise TimeoutError(f"ranks {missing} did not report within {timeout}s")
            try:
                rank, value, error = results.get(timeout=left)
            except queue_module.Empty:
                continue
            if error is not None:
                raise AssertionError(f"rank {rank} failed:\n{error}")
            collected[rank] = value
    finally:
        # Release the workers parked on the ack queue *before* joining them;
        # the fds behind tensor results stay valid only while their sender
        # lives, so every get() must have finished unwrapping first.
        for _ in range(world_size):
            acks.put(True)
        _reap_workers(workers)
        if baseline is not None:
            _wait_for_devices_to_settle(baseline)
    return [collected[rank] for rank in range(world_size)]


#: Fresh ports one grid may cost. The port is picked by binding, released, then
#: re-bound by the workers after their spawns — a window as long as a handful of
#: interpreter startups, and a machine busy enough hands the port to someone else
#: inside it. The loser is rank 0's ``TCPStore``: ``EADDRINUSE`` before a single
#: collective ran, so the grid is retried on a fresh port instead of failing the
#: claim it guards.
_RENDEZVOUS_ATTEMPTS = 3


def run_on_tp_ranks(
    payload: Callable[[int], Any],
    tp_size: int,
    *,
    dp_size: int = 1,
    timeout: float = 300.0,
    backend: str = "nccl",
    enable_expert_parallel: bool = False,
    enable_dp_attention: bool = False,
) -> list[Any]:
    """Run ``payload(rank)`` on every rank of a ``dp_size x tp_size`` grid.

    Args:
        payload: Module-level function (spawn has to pickle it) called once per rank with
            its global rank; must return plain Python data.
        tp_size: Ranks per replica.
        dp_size: Number of replicas.
        timeout: Seconds to wait for all ranks. Exceeding it means a collective
            mismatch — a rank calling a collective its peers do not is a deadlock, not a
            wrong answer, so it has to fail as a bounded test rather than hang the suite.
        backend: ``"nccl"`` for the data plane (one device per rank, see
            :func:`needs_gpus`) or ``"gloo"`` for a device-free grid — enough to
            exercise the control plane, which is what
            :func:`~rapid_llm.distributed.parallel_state.broadcast_object` and the
            executor's plan hand-off live on.
        enable_expert_parallel: Set the EP group state before the payload runs, so
            expert-parallel code paths see
            :func:`~rapid_llm.distributed.parallel_state.get_ep_group`.
        enable_dp_attention: Make the DP axis a shard of one model rather than a
            set of independent replicas: builds the DP group and widens the EP
            group to the whole grid, which is what the DP-attention tests need.

    Returns:
        One result per global rank, in rank order.

    A grid whose rendezvous port another process won — rank 0's ``TCPStore``
    failing with ``EADDRINUSE`` before any collective — is retried on a fresh
    port (up to ``_RENDEZVOUS_ATTEMPTS`` times). Every other failure
    propagates on its first occurrence.

    Raises:
        AssertionError: If any rank raised; the message carries that rank's traceback.
        TimeoutError: If some rank never reported.
    """
    attempt = 0
    while True:
        try:
            return _attempt(
                payload,
                tp_size,
                dp_size,
                timeout,
                backend,
                enable_expert_parallel,
                enable_dp_attention,
            )
        except AssertionError as error:
            attempt += 1
            if "EADDRINUSE" not in str(error) or attempt >= _RENDEZVOUS_ATTEMPTS:
                raise
            _log.warning(
                "rendezvous port lost to another process (attempt %d/%d); retrying on a fresh one",
                attempt,
                _RENDEZVOUS_ATTEMPTS,
            )
