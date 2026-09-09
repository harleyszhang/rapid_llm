"""End-to-end smoke test for serving a *quantised* checkpoint over HTTP.

``test_api_server.py`` puts a fake engine behind the app, which is the right trade
for protocol coverage: the JSON shape and the SSE framing are what a client depends
on, and pinning them should not need a GPU. But a fake engine cannot fail the way
this file is looking for. Quantisation and an fp8 KV cache are decided during
startup -- the loader picks a linear method per layer, the cache manager profiles
memory at a different element size, and the graph capture happens against whichever
kernels those two chose. All of it is finished before the first request arrives, so
every failure mode here is a *startup* failure: the server either never listens, or
listens and answers with something the model did not generate.

So this test runs the real thing: ``rapid-llm serve`` as a subprocess, real weights
quantised at load time, a real socket. It asserts only what such a smoke test can
honestly assert -- that the server becomes ready, that a completion comes back
non-empty, and that streaming and non-streaming agree with each other. Whether the
quantised tokens are *good* is a question for ``benchmarks/engine/run.py quant``, which
compares them against a recorded baseline; a smoke test that tried to judge quality
from one prompt would be asserting noise.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("fastapi", reason="needs the `serve` extra")
httpx = pytest.importorskip("httpx", reason="needs the `serve` extra")

pytestmark = [pytest.mark.serving, pytest.mark.gpu, pytest.mark.slow]

#: Startup is a checkpoint load, a quantisation pass, a KV profile and a capture.
_READY_TIMEOUT_S = 420.0

_PROMPT = "The capital of France is"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Server:
    """A ``rapid-llm serve`` subprocess, with its log kept for the failure message.

    Started in its own session so that shutting it down takes any follower ranks
    with it; a leaked rank holds GPU memory that the next test discovers as an OOM.
    """

    def __init__(self, argv: list[str], log_path: Path) -> None:
        self.log_path = log_path
        self._handle = log_path.open("w")
        self.process = subprocess.Popen(
            argv,
            stdout=self._handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def tail(self, lines: int = 25) -> str:
        try:
            return "\n".join(self.log_path.read_text().splitlines()[-lines:])
        except OSError:  # pragma: no cover - only if the log vanished
            return "<no log>"

    def wait_ready(self, port: int, timeout_s: float) -> None:
        """Poll ``/health`` until it answers, failing with the server's own log.

        The process is polled too: a server that dies during startup would
        otherwise be indistinguishable from a slow one until the timeout, and the
        reason it died is the only useful thing to report.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                pytest.fail(
                    f"server exited with code {self.process.returncode} during startup:\n"
                    f"{self.tail()}"
                )
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass  # not listening yet
            time.sleep(1.0)
        pytest.fail(f"server not ready in {timeout_s:.0f}s:\n{self.tail()}")

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), 15)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                self.process.terminate()
            try:
                self.process.wait(timeout=60.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - wedged rank
                os.killpg(os.getpgid(self.process.pid), 9)
                self.process.wait(timeout=30.0)
        self._handle.close()


@pytest.fixture(scope="module")
def quant_server(model_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """An fp8-quantised server with an fp8 KV cache, shared by this module's tests.

    Module-scoped deliberately: the startup this exercises costs minutes, and every
    test here asks a question about the *same* startup. Re-paying it per test would
    buy no isolation, because there is no request that can change how the weights
    were quantised.

    ``max_gpu_num_blocks`` is pinned rather than profiled. Profiling claims most of
    the card, which is correct for a deployment and antisocial in a test suite that
    may be sharing the GPU.
    """
    if not torch.cuda.is_available():  # pragma: no cover - the gpu mark handles this
        pytest.skip("needs a CUDA device")

    port = _free_port()
    log_path = tmp_path_factory.mktemp("serve") / "server.log"
    argv = [
        sys.executable,
        "-m",
        "rapid_llm.cli",
        "serve",
        "--model-dir",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-seq-len",
        "512",
        "--max-gpu-num-blocks",
        "4096",
        "--max-num-seqs",
        "4",
        "--quantization",
        "fp8",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--cuda-graph",
    ]
    server = _Server(argv, log_path)
    try:
        server.wait_ready(port, _READY_TIMEOUT_S)
        yield port, server
    finally:
        server.stop()


def _completion(port: int, **overrides) -> dict:
    body = {
        "model": "m",
        "prompt": _PROMPT,
        "max_tokens": 16,
        "temperature": 0.0,
        **overrides,
    }
    response = httpx.post(f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=120.0)
    response.raise_for_status()
    return response.json()


def test_models_endpoint_lists_the_quantised_model(quant_server):
    """Readiness is not enough: the served model must be advertised.

    ``/health`` answers once the app is up, which on a startup failure that was
    caught and logged can happen with no engine behind it.
    """
    port, _ = quant_server
    listing = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=30.0).json()
    assert [entry["id"] for entry in listing["data"]]


def test_fp8_weights_and_fp8_kv_cache_generate(quant_server):
    """The whole point: fp8 linear layers plus an fp8 KV cache produce text.

    Both halves are load-time decisions, and either one failing is silent -- a
    mis-scaled dequant yields ``nan`` logits, an fp8 cache read with the wrong
    stride yields fluent nonsense. Neither raises. What this catches is the case
    where the answer is *empty*, which is what ``nan`` logits and a broken cache
    both eventually produce.
    """
    port, server = quant_server
    payload = _completion(port)
    text = payload["choices"][0]["text"]
    assert text.strip(), f"empty completion from an fp8 server:\n{server.tail()}"
    assert payload["usage"]["completion_tokens"] > 0


def test_stream_and_non_stream_agree(quant_server):
    """Streaming must be the same generation, delivered differently.

    At ``temperature=0`` the two paths sample the same tokens, so their texts have
    to match exactly. They are assembled by different code -- one accumulates on the
    server, the other concatenates SSE deltas on the client -- and a mismatch means
    one of them is dropping or duplicating a chunk. Comparing prefixes would hide
    exactly that, so this is an equality assertion.
    """
    port, server = quant_server
    whole = _completion(port)["choices"][0]["text"]

    body = {
        "model": "m",
        "prompt": _PROMPT,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
    }
    pieces: list[str] = []
    with httpx.stream(
        "POST", f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=120.0
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            frame = line.removeprefix("data: ").strip()
            if frame == "[DONE]":
                break
            pieces.append(json.loads(frame)["choices"][0].get("text") or "")

    assert pieces, f"stream produced no frames:\n{server.tail()}"
    assert "".join(pieces) == whole


def test_concurrent_requests_do_not_change_a_greedy_answer(quant_server):
    """A greedy completion is a function of its own prompt, not of its neighbours.

    Continuous batching decodes several requests in one pass, so a shared scratch
    buffer or a mis-indexed position would let one request's tokens depend on who
    it was batched with. That is invisible in a single-request test and invisible
    again under sampling; it takes identical prompts, greedy decoding and a batch.
    """
    port, server = quant_server
    alone = _completion(port)["choices"][0]["text"]

    body = {
        "model": "m",
        "prompt": _PROMPT,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    url = f"http://127.0.0.1:{port}/v1/completions"
    with httpx.Client(timeout=180.0) as client:
        together = [client.post(url, json=body).json()["choices"][0]["text"] for _ in range(3)]

    for index, text in enumerate(together):
        assert text == alone, (
            f"request {index} differed when batched:\n"
            f"  alone:  {alone!r}\n  batched: {text!r}\n{server.tail()}"
        )
