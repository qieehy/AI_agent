from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from errors import (
    EmbeddingWorkerError,
    EmbeddingWorkerProtocolError,
    EmbeddingWorkerTimeoutError,
)
from observability import logger
from rag.process_embeddings import ProcessEmbeddingClient, WorkerState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker_log_records() -> Iterator[list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []

    def capture_worker_event(message: Any) -> None:
        records.append(dict(message.record))

    handler_id = logger.add(
        capture_worker_event,
        level="DEBUG",
        filter=lambda record: (
            record["extra"].get("component") == "process_embedding_client"
        ),
    )
    try:
        yield records
    finally:
        logger.remove(handler_id)


def _event_extras(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record["extra"] for record in records]


def _worker_environment() -> dict[str, str]:
    environment = {"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    for name in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _make_client(mode: str, **overrides) -> ProcessEmbeddingClient:
    options = {
        "command": (
            sys.executable,
            "-m",
            "tests.support.fake_embedding_worker",
            "--mode",
            mode,
            "--model-version",
            "fake-v1",
        ),
        "cwd": PROJECT_ROOT,
        "environment": _worker_environment(),
        "expected_model_version": "fake-v1",
        "startup_timeout_s": 3.0,
        "inference_timeout_s": 3.0,
        "lock_timeout_s": 1.0,
        "shutdown_timeout_s": 3.0,
    }
    options.update(overrides)
    return ProcessEmbeddingClient(**options)


async def _wait_for_stderr_marker(
    client: ProcessEmbeddingClient,
    marker: str,
) -> None:
    async def wait_until_observed() -> None:
        while marker not in client.stderr_tail:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_observed(), timeout=3)


@pytest.mark.anyio
async def test_client_reuses_one_worker_and_closes_cleanly() -> None:
    client = _make_client("success")

    await client.start()
    first_pid = client.worker_pid
    assert first_pid is not None
    assert client.state == WorkerState.READY

    assert await client.embed_batch(["first"]) == [[1.0, 1.0]]
    assert await client.embed_batch(["second", "third"]) == [
        [1.0, 1.0],
        [2.0, 1.0],
    ]
    assert client.worker_pid == first_pid
    assert client.generation == 1

    await client.aclose()
    assert client.state == WorkerState.CLOSED
    assert client.worker_pid is None
    await client.aclose()


@pytest.mark.anyio
async def test_successful_lifecycle_emits_safe_structured_events(
    worker_log_records: list[dict[str, Any]],
) -> None:
    client = _make_client("success")
    sensitive_text = "customer-secret-must-not-be-logged"

    await client.embed_batch([sensitive_text])
    await client.aclose()

    extras = _event_extras(worker_log_records)
    events = [extra["event"] for extra in extras]
    assert events == [
        "embedding_worker_starting",
        "embedding_worker_ready",
        "embedding_inference_started",
        "embedding_inference_completed",
        "embedding_worker_closing",
        "embedding_worker_closed",
    ]

    ready = extras[1]
    assert ready["operation"] == "startup"
    assert ready["outcome"] == "success"
    assert ready["generation"] == 1
    assert ready["dimension"] == 2
    assert isinstance(ready["worker_pid"], int)
    assert ready["duration_ms"] >= 0

    completed = extras[3]
    assert completed["operation"] == "inference"
    assert completed["outcome"] == "success"
    assert completed["batch_size"] == 1
    assert completed["dimension"] == 2
    assert completed["duration_ms"] >= 0

    assert all(
        sensitive_text not in record["message"]
        and sensitive_text not in repr(record["extra"])
        for record in worker_log_records
    )


@pytest.mark.anyio
async def test_timeout_and_recovery_events_show_generation_change_without_stderr(
    worker_log_records: list[dict[str, Any]],
) -> None:
    client = _make_client(
        "inference_hang_once",
        inference_timeout_s=0.5,
    )
    try:
        with pytest.raises(EmbeddingWorkerTimeoutError):
            await client.embed_batch(["first"])

        assert await client.embed_batch(["recovered"]) == [[1.0, 1.0]]
    finally:
        await client.aclose()

    extras = _event_extras(worker_log_records)
    timeout_event = next(
        extra
        for extra in extras
        if extra["event"] == "embedding_inference_timed_out"
    )
    assert timeout_event["outcome"] == "timeout"
    assert timeout_event["generation"] == 1
    assert timeout_event["timeout_s"] == 0.5
    assert timeout_event["forced_stop"] is True
    assert timeout_event["duration_ms"] >= 0

    starting_events = [
        extra
        for extra in extras
        if extra["event"] == "embedding_worker_starting"
    ]
    assert [extra["generation"] for extra in starting_events] == [1, 2]
    assert [extra["restart"] for extra in starting_events] == [False, True]
    assert "INFERENCE_HANG" not in repr(extras)
    assert client.state == WorkerState.CLOSED
    assert client.worker_pid is None


@pytest.mark.anyio
async def test_client_startup_timeout_kills_worker() -> None:
    client = _make_client("startup_hang", startup_timeout_s=0.5)

    with pytest.raises(EmbeddingWorkerTimeoutError) as exc_info:
        await client.start()

    assert exc_info.value.context == {"phase": "startup", "timeout_s": 0.5}
    assert client.state == WorkerState.NEW
    assert client.worker_pid is None
    await client.aclose()


@pytest.mark.anyio
async def test_client_inference_timeout_kills_worker() -> None:
    client = _make_client("inference_hang", inference_timeout_s=0.5)

    with pytest.raises(EmbeddingWorkerTimeoutError) as exc_info:
        await client.embed_batch(["query"])

    assert exc_info.value.context == {"phase": "inference", "timeout_s": 0.5}
    assert client.state == WorkerState.NEW
    assert client.worker_pid is None
    assert "INFERENCE_HANG" in client.stderr_tail
    await client.aclose()


@pytest.mark.anyio
async def test_client_rejects_wrong_ready_model_version_and_kills_worker() -> None:
    client = _make_client("success", expected_model_version="different-model")

    with pytest.raises(EmbeddingWorkerProtocolError, match="model_version"):
        await client.start()

    assert client.state == WorkerState.NEW
    assert client.worker_pid is None
    await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mode", "expected_exit_code"),
    [
        ("crash_before_ready", 21),
        ("crash_during_inference", 22),
    ],
)
async def test_client_translates_worker_crashes_and_cleans_up(
    mode: str,
    expected_exit_code: int,
) -> None:
    client = _make_client(mode)
    try:
        with pytest.raises(EmbeddingWorkerError) as exc_info:
            await client.embed_batch(["query"])

        assert exc_info.value.context["exit_code"] == expected_exit_code
        assert client.state == WorkerState.NEW
        assert client.worker_pid is None
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode",
    ["wrong_request_id", "wrong_generation", "malformed_json"],
)
async def test_client_rejects_uncorrelated_or_malformed_response(mode: str) -> None:
    client = _make_client(mode)
    try:
        with pytest.raises(EmbeddingWorkerProtocolError):
            await client.embed_batch(["query"])
        assert client.state == WorkerState.NEW
        assert client.worker_pid is None
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_structured_worker_failure_does_not_destroy_healthy_process() -> None:
    client = _make_client("worker_error")
    try:
        with pytest.raises(EmbeddingWorkerError) as exc_info:
            await client.embed_batch(["query"])

        assert exc_info.value.context["error_type"] == "model_inference_failed"
        assert client.state == WorkerState.READY
        assert client.worker_pid is not None
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_client_rejects_calls_after_close() -> None:
    client = _make_client("success")
    await client.aclose()

    with pytest.raises(EmbeddingWorkerError, match="closed"):
        await client.start()
    with pytest.raises(EmbeddingWorkerError, match="closed"):
        await client.embed_batch(["query"])


@pytest.mark.parametrize(
    "override",
    [
        {"command": ()},
        {"expected_model_version": ""},
        {"startup_timeout_s": 0},
        {"startup_timeout_s": float("inf")},
        {"inference_timeout_s": float("nan")},
        {"inference_timeout_s": -1},
        {"lock_timeout_s": True},
        {"shutdown_timeout_s": 0},
        {"max_request_bytes": 0},
        {"max_response_bytes": 0},
        {"max_stderr_chars": 0},
    ],
)
def test_client_configuration_is_validated(override) -> None:
    with pytest.raises(ValueError):
        _make_client("success", **override)


@pytest.mark.anyio
async def test_lock_timeout_does_not_kill_the_current_worker_and_next_generation_recovers() -> None:
    client = _make_client(
        "inference_hang_once",
        inference_timeout_s=1.0,
        lock_timeout_s=0.1,
    )
    first = asyncio.create_task(client.embed_batch(["first"]))
    try:
        await _wait_for_stderr_marker(client, "INFERENCE_HANG")
        first_pid = client.worker_pid
        assert first_pid is not None

        with pytest.raises(EmbeddingWorkerTimeoutError) as exc_info:
            await client.embed_batch(["second"])

        assert exc_info.value.context == {"phase": "lock", "timeout_s": 0.1}
        assert client.worker_pid == first_pid

        with pytest.raises(EmbeddingWorkerTimeoutError) as first_error:
            await first
        assert first_error.value.context["phase"] == "inference"
        assert client.worker_pid is None
        assert client.state == WorkerState.NEW

        assert await client.embed_batch(["recovered"]) == [[1.0, 1.0]]
        assert client.generation == 2
        assert client.worker_pid is not None
    finally:
        if not first.done():
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        await client.aclose()


@pytest.mark.anyio
async def test_cancellation_kills_worker_before_propagating_and_then_recovers() -> None:
    client = _make_client(
        "inference_hang_once",
        inference_timeout_s=10.0,
    )
    task = asyncio.create_task(client.embed_batch(["cancel me"]))
    try:
        await _wait_for_stderr_marker(client, "INFERENCE_HANG")
        assert client.worker_pid is not None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client.worker_pid is None
        assert client.state == WorkerState.NEW
        assert await client.embed_batch(["recovered"]) == [[1.0, 1.0]]
        assert client.generation == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_startup_cancellation_kills_worker_before_propagating_and_then_recovers() -> None:
    client = _make_client(
        "startup_hang_once",
        startup_timeout_s=10.0,
    )
    task = asyncio.create_task(client.start())
    try:
        await _wait_for_stderr_marker(client, "STARTUP_HANG")
        assert client.worker_pid is not None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client.worker_pid is None
        assert client.state == WorkerState.NEW
        await client.start()
        assert client.state == WorkerState.READY
        assert client.generation == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_protocol_failure_restarts_with_a_new_generation() -> None:
    client = _make_client("wrong_generation_once")
    try:
        with pytest.raises(EmbeddingWorkerProtocolError, match="generation"):
            await client.embed_batch(["first"])

        assert client.worker_pid is None
        assert client.generation == 1
        assert await client.embed_batch(["second"]) == [[1.0, 1.0]]
        assert client.generation == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_concurrent_close_interrupts_active_inference_and_rejects_new_calls() -> None:
    client = _make_client(
        "inference_hang",
        inference_timeout_s=10.0,
        shutdown_timeout_s=0.5,
    )
    active = asyncio.create_task(client.embed_batch(["block close"]))
    await _wait_for_stderr_marker(client, "INFERENCE_HANG")

    close_task = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)
    with pytest.raises(EmbeddingWorkerError, match="closed"):
        await client.embed_batch(["must be rejected"])

    with pytest.raises(EmbeddingWorkerError):
        await active
    await close_task
    assert client.state == WorkerState.CLOSED
    assert client.worker_pid is None
