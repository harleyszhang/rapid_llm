"""The harness's own contract: a lost rendezvous port costs a retry, nothing more.

:func:`~tests.distributed.tp_harness.run_on_tp_ranks` picks a port by binding
and releasing it, then its workers re-bind it after their spawns — a window a
busy machine can win, and the loser is rank 0's ``TCPStore`` failing with
``EADDRINUSE`` before a single collective ran. The wrapper answers only that
loss, only with a fresh port, and only so many times: every other failure is
a real one, and a retry around it would hide the deadlock it reports. Pinned
here because the semantics are the harness's; ``_attempt`` is swapped, never
spawned — the port mechanics themselves are exercised by every test that uses
the harness for real.

Usage:
    pytest tests/distributed/test_tp_harness.py
"""

from __future__ import annotations

import pytest

from tests.distributed import tp_harness

#: The shape rank 0's failure travels in: the ``AssertionError`` wrapper the
#: parent raises, carrying the child's traceback text (port verbatim from a
#: real loss; the number is noise).
_EADDRINUSE = (
    "rank 0 failed:\n"
    "torch.distributed.DistNetworkError: The server socket has failed to listen "
    "on any local network address. port: 53637, useIpv6: false, code: -98, "
    "name: EADDRINUSE, message: address already in use"
)


def _scripted_attempt(monkeypatch, failures: list[str | None]) -> list[int]:
    """Swap ``_attempt`` for one scripted outcome per call; return the call log.

    Outcome ``None`` means success; a string is raised as the ``AssertionError``
    every rank failure arrives as. The log records how many attempts ran, which
    is the whole of what the wrapper decides.
    """
    calls: list[int] = []

    def scripted(*args, **kwargs):
        calls.append(1)
        failure = failures[len(calls) - 1]
        if failure is not None:
            raise AssertionError(failure)
        return [{"rank": 0}, {"rank": 1}]

    monkeypatch.setattr(tp_harness, "_attempt", scripted)
    return calls


def test_a_lost_port_costs_exactly_one_fresh_attempt(monkeypatch):
    calls = _scripted_attempt(monkeypatch, [_EADDRINUSE, None])
    results = tp_harness.run_on_tp_ranks(lambda rank: rank, tp_size=2, backend="gloo", timeout=5)
    assert results == [{"rank": 0}, {"rank": 1}]
    assert len(calls) == 2, "the loss must be answered by a fresh attempt"


def test_a_real_failure_propagates_without_a_retry(monkeypatch):
    calls = _scripted_attempt(monkeypatch, ["rank 1 failed:\nValueError: boom"])
    with pytest.raises(AssertionError, match="boom"):
        tp_harness.run_on_tp_ranks(lambda rank: rank, tp_size=2, backend="gloo", timeout=5)
    assert len(calls) == 1, "a retry around a real failure would hide it"


def test_a_permanently_lost_port_fails_after_the_attempt_budget(monkeypatch):
    calls = _scripted_attempt(monkeypatch, [_EADDRINUSE] * tp_harness._RENDEZVOUS_ATTEMPTS)
    with pytest.raises(AssertionError, match="EADDRINUSE"):
        tp_harness.run_on_tp_ranks(lambda rank: rank, tp_size=2, backend="gloo", timeout=5)
    assert len(calls) == tp_harness._RENDEZVOUS_ATTEMPTS, "the budget must be finite"
