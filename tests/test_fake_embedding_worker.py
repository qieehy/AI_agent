from __future__ import annotations

import asyncio
import sys

import pytest

from rag.embedding_worker_protocol import (
    PROTOCOL_VERSION,
    EmbeddingRequest,
    EmbeddingResult,
    WorkerFailure,
    WorkerReady,
    decode_worker_response,
    encode_message,
)

REQUEST_ID_1 = "11111111-1111-4111-8111-111111111111"
REQUEST_ID_2 = "22222222-2222-4222-8222-222222222222"


async def _start_worker(mode: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.support.fake_embedding_worker",
        "--mode",
        mode,
        "--generation",
        "7",
        "--model-version",
        "fake-v1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _stop_worker(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
    await asyncio.wait_for(process.wait(), timeout=3)


async def _read_stdout(process: asyncio.subprocess.Process) -> bytes:
    assert process.stdout is not None
    return await asyncio.wait_for(process.stdout.readline(), timeout=3)


async def _read_stderr(process: asyncio.subprocess.Process) -> bytes:
    assert process.stderr is not None
    line = await asyncio.wait_for(process.stderr.readline(), timeout=3)
    return line.rstrip(b"\r\n")


async def _send_request(
    process: asyncio.subprocess.Process,
    *,
    request_id: str,
    texts: tuple[str, ...],
) -> None:
    assert process.stdin is not None
    process.stdin.write(encode_message(EmbeddingRequest(
        protocol_version=PROTOCOL_VERSION,
        type="embed",
        generation=7,
        request_id=request_id,
        texts=texts,
    )))
    await process.stdin.drain()


@pytest.mark.anyio
async def test_fake_worker_is_persistent_and_correlates_responses() -> None:
    process = await _start_worker("success")
    try:
        ready = decode_worker_response(await _read_stdout(process))
        assert isinstance(ready, WorkerReady)
        assert ready.generation == 7
        assert ready.model_version == "fake-v1"
        assert ready.dimension == 2

        await _send_request(process, request_id=REQUEST_ID_1, texts=("first",))
        first = decode_worker_response(await _read_stdout(process))
        assert isinstance(first, EmbeddingResult)
        assert first.request_id == REQUEST_ID_1
        assert first.vectors == ((1.0, 1.0),)

        await _send_request(
            process,
            request_id=REQUEST_ID_2,
            texts=("second", "third"),
        )
        second = decode_worker_response(await _read_stdout(process))
        assert isinstance(second, EmbeddingResult)
        assert second.request_id == REQUEST_ID_2
        assert second.vectors == ((1.0, 1.0), (2.0, 1.0))

        assert process.returncode is None
        assert process.stdin is not None
        process.stdin.close()
        await process.stdin.wait_closed()
        assert await asyncio.wait_for(process.wait(), timeout=3) == 0
    finally:
        await _stop_worker(process)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mode", "marker"),
    [
        ("startup_hang", b"STARTUP_HANG"),
        ("crash_before_ready", b"CRASH_BEFORE_READY"),
    ],
)
async def test_fake_worker_startup_failure_modes_are_observable(
    mode: str,
    marker: bytes,
) -> None:
    process = await _start_worker(mode)
    try:
        assert await _read_stderr(process) == marker
        if mode == "startup_hang":
            assert process.returncode is None
        else:
            assert await asyncio.wait_for(process.wait(), timeout=3) == 21
    finally:
        await _stop_worker(process)


@pytest.mark.anyio
async def test_fake_worker_inference_hang_has_a_deterministic_barrier() -> None:
    process = await _start_worker("inference_hang")
    try:
        assert isinstance(decode_worker_response(await _read_stdout(process)), WorkerReady)
        await _send_request(process, request_id=REQUEST_ID_1, texts=("block",))

        assert await _read_stderr(process) == b"INFERENCE_HANG"
        assert process.returncode is None
    finally:
        await _stop_worker(process)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode",
    [
        "worker_error",
        "wrong_request_id",
        "wrong_generation",
        "malformed_json",
        "crash_during_inference",
    ],
)
async def test_fake_worker_inference_failure_modes_reach_the_parent(mode: str) -> None:
    process = await _start_worker(mode)
    try:
        assert isinstance(decode_worker_response(await _read_stdout(process)), WorkerReady)
        await _send_request(process, request_id=REQUEST_ID_1, texts=("query",))

        if mode == "crash_during_inference":
            assert await _read_stderr(process) == b"CRASH_DURING_INFERENCE"
            assert await asyncio.wait_for(process.wait(), timeout=3) == 22
        else:
            raw_response = await _read_stdout(process)
            if mode == "malformed_json":
                assert raw_response == b"not-json\n"
                return

            response = decode_worker_response(raw_response)
            if mode == "worker_error":
                assert isinstance(response, WorkerFailure)
                assert response.error_type == "model_inference_failed"
            elif mode == "wrong_request_id":
                assert isinstance(response, EmbeddingResult)
                assert response.request_id != REQUEST_ID_1
            else:
                assert mode == "wrong_generation"
                assert isinstance(response, EmbeddingResult)
                assert response.generation == 8
    finally:
        await _stop_worker(process)
