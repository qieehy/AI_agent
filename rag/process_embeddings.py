from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, NoReturn
from uuid import uuid4

from pydantic import ValidationError

from errors import (
    EmbeddingWorkerError,
    EmbeddingWorkerProtocolError,
    EmbeddingWorkerTimeoutError,
)
from observability import logger
from rag.embedding_worker_protocol import (
    MAX_PROTOCOL_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    EmbeddingRequest,
    EmbeddingResult,
    WorkerFailure,
    WorkerReady,
    WorkerResponse,
    decode_worker_response,
    encode_message,
)


class WorkerState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    CLOSED = "closed"


class ProcessEmbeddingClient:
    """Bounded JSONL client for one persistent embedding subprocess.

    This component owns the subprocess and its standard streams. Calls are serialized
    because one worker processes one embedding batch at a time. The production
    composition root owns one client and closes it when the CLI lifecycle ends.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        expected_model_version: str,
        startup_timeout_s: float,
        inference_timeout_s: float,
        lock_timeout_s: float,
        shutdown_timeout_s: float,
        max_request_bytes: int = 256 * 1024,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_stderr_chars: int = 4096,
    ) -> None:
        normalized_command = tuple(command)
        if not normalized_command or any(not isinstance(part, str) or not part for part in normalized_command):
            raise ValueError("command must contain non-empty string arguments")
        resolved_cwd = cwd.resolve()
        if not resolved_cwd.is_dir():
            raise ValueError("cwd must resolve to an existing directory")
        if not expected_model_version.strip():
            raise ValueError("expected_model_version must be a non-empty string")
        for name, value in (
            ("startup_timeout_s", startup_timeout_s),
            ("inference_timeout_s", inference_timeout_s),
            ("lock_timeout_s", lock_timeout_s),
            ("shutdown_timeout_s", shutdown_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be greater than 0")
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_response_bytes", max_response_bytes),
            ("max_stderr_chars", max_stderr_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_request_bytes > MAX_PROTOCOL_MESSAGE_BYTES:
            raise ValueError("max_request_bytes exceeds the protocol hard limit")
        if max_response_bytes > MAX_PROTOCOL_MESSAGE_BYTES:
            raise ValueError("max_response_bytes exceeds the protocol hard limit")

        self._command = normalized_command
        self._cwd = resolved_cwd
        self._environment = dict(environment)
        self._expected_model_version = expected_model_version
        self._startup_timeout_s = float(startup_timeout_s)
        self._inference_timeout_s = float(inference_timeout_s)
        self._lock_timeout_s = float(lock_timeout_s)
        self._shutdown_timeout_s = float(shutdown_timeout_s)
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._max_stderr_chars = max_stderr_chars

        self._operation_lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""
        self._generation = 0
        self._dimension: int | None = None
        self._state = WorkerState.NEW
        self._closed = False
        self._close_requested = False

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.returncode is None else None

    @property
    def stderr_tail(self) -> str:
        return self._stderr_tail

    async def start(self) -> None:
        acquired = await self._acquire_operation_lock(operation="startup")
        try:
            self._ensure_open()
            await self._ensure_started_locked()
        finally:
            if acquired:
                self._operation_lock.release()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        acquired = await self._acquire_operation_lock(operation="inference")
        try:
            self._ensure_open()
            await self._ensure_started_locked()
            request = self._build_request(texts)
            self._state = WorkerState.BUSY
            inference_started_at = perf_counter()
            self._emit_event(
                "DEBUG",
                event="embedding_inference_started",
                operation="inference",
                outcome="started",
                batch_size=len(request.texts),
            )
            try:
                response = await asyncio.wait_for(
                    self._exchange_locked(request),
                    timeout=self._inference_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                worker_pid = self.worker_pid
                await self._stop_process_locked(force=True)
                self._state = WorkerState.NEW
                self._emit_event(
                    "WARNING",
                    event="embedding_inference_timed_out",
                    operation="inference",
                    outcome="timeout",
                    duration_ms=self._elapsed_ms(inference_started_at),
                    timeout_s=self._inference_timeout_s,
                    batch_size=len(request.texts),
                    worker_pid=worker_pid,
                    forced_stop=True,
                )
                raise EmbeddingWorkerTimeoutError(
                    "inference",
                    self._inference_timeout_s,
                ) from exc
            except asyncio.CancelledError:
                worker_pid = self.worker_pid
                await self._run_cleanup_shielded(
                    asyncio.create_task(self._stop_process_locked(force=True))
                )
                self._state = WorkerState.NEW
                self._emit_event(
                    "INFO",
                    event="embedding_inference_canceled",
                    operation="inference",
                    outcome="canceled",
                    duration_ms=self._elapsed_ms(inference_started_at),
                    batch_size=len(request.texts),
                    worker_pid=worker_pid,
                    forced_stop=True,
                )
                raise
            except EmbeddingWorkerError as exc:
                worker_pid = self.worker_pid
                await self._stop_process_locked(force=True)
                self._state = WorkerState.NEW
                self._emit_event(
                    "ERROR",
                    event="embedding_inference_failed",
                    operation="inference",
                    outcome="failed",
                    duration_ms=self._elapsed_ms(inference_started_at),
                    batch_size=len(request.texts),
                    worker_pid=worker_pid,
                    forced_stop=True,
                    **self._safe_error_fields(exc),
                )
                raise

            if isinstance(response, WorkerFailure):
                self._state = WorkerState.READY
                self._emit_event(
                    "WARNING",
                    event="embedding_inference_failed",
                    operation="inference",
                    outcome="worker_failure",
                    duration_ms=self._elapsed_ms(inference_started_at),
                    batch_size=len(request.texts),
                    worker_error_type=response.error_type,
                    forced_stop=False,
                )
                raise EmbeddingWorkerError(
                    "embedding worker reported an inference failure",
                    context={
                        "error_type": response.error_type,
                        "generation": response.generation,
                    },
                )
            if not isinstance(response, EmbeddingResult):
                await self._fail_protocol_locked("worker returned a non-result response")
            if response.generation != self._generation:
                await self._fail_protocol_locked("worker response generation does not match")
            if response.request_id != request.request_id:
                await self._fail_protocol_locked("worker response request_id does not match")
            if len(response.vectors) != len(request.texts):
                await self._fail_protocol_locked("worker response batch size does not match")
            if self._dimension is None or any(
                len(vector) != self._dimension for vector in response.vectors
            ):
                await self._fail_protocol_locked("worker response dimension does not match READY")

            self._state = WorkerState.READY
            self._emit_event(
                "INFO",
                event="embedding_inference_completed",
                operation="inference",
                outcome="success",
                duration_ms=self._elapsed_ms(inference_started_at),
                batch_size=len(request.texts),
                dimension=self._dimension,
            )
            return [list(vector) for vector in response.vectors]
        finally:
            if acquired:
                self._operation_lock.release()

    async def aclose(self) -> None:
        if self._state == WorkerState.CLOSED:
            self._emit_event(
                "DEBUG",
                event="embedding_worker_close_skipped",
                operation="shutdown",
                outcome="already_closed",
            )
            return
        shutdown_started_at = perf_counter()
        self._close_requested = True
        self._emit_event(
            "INFO",
            event="embedding_worker_closing",
            operation="shutdown",
            outcome="started",
        )
        acquired = False
        try:
            try:
                acquired = await self._acquire_operation_lock(
                    operation="shutdown",
                    timeout_s=self._shutdown_timeout_s
                )
            except EmbeddingWorkerTimeoutError:
                await self._run_cleanup_shielded(
                    asyncio.create_task(self._interrupt_current_process_for_close())
                )
                try:
                    acquired = await self._acquire_operation_lock(
                        operation="shutdown",
                        timeout_s=self._shutdown_timeout_s
                    )
                except EmbeddingWorkerTimeoutError as exc:
                    raise EmbeddingWorkerTimeoutError(
                        "shutdown",
                        self._shutdown_timeout_s,
                    ) from exc

            if self._state == WorkerState.CLOSED:
                return
            self._closed = True
            self._state = WorkerState.STOPPING
            await self._stop_process_locked(force=False)
            self._state = WorkerState.CLOSED
            self._emit_event(
                "INFO",
                event="embedding_worker_closed",
                operation="shutdown",
                outcome="success",
                duration_ms=self._elapsed_ms(shutdown_started_at),
            )
        except EmbeddingWorkerTimeoutError as exc:
            self._emit_event(
                "ERROR",
                event="embedding_worker_close_timed_out",
                operation="shutdown",
                outcome="timeout",
                duration_ms=self._elapsed_ms(shutdown_started_at),
                timeout_s=exc.context.get("timeout_s"),
                timeout_phase=exc.context.get("phase"),
            )
            raise
        except asyncio.CancelledError:
            self._state = WorkerState.STOPPING
            await self._run_cleanup_shielded(
                asyncio.create_task(self._interrupt_current_process_for_close())
            )
            self._emit_event(
                "WARNING",
                event="embedding_worker_close_canceled",
                operation="shutdown",
                outcome="canceled",
                duration_ms=self._elapsed_ms(shutdown_started_at),
                forced_stop=True,
            )
            raise
        finally:
            if acquired:
                self._operation_lock.release()

    async def _acquire_operation_lock(
        self,
        *,
        operation: str,
        timeout_s: float | None = None,
    ) -> bool:
        timeout = self._lock_timeout_s if timeout_s is None else timeout_s
        wait_started_at = perf_counter()
        try:
            await asyncio.wait_for(self._operation_lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._emit_event(
                "WARNING",
                event="embedding_worker_lock_timed_out",
                operation=operation,
                outcome="timeout",
                duration_ms=self._elapsed_ms(wait_started_at),
                timeout_s=timeout,
            )
            raise EmbeddingWorkerTimeoutError("lock", timeout) from exc
        return True

    def _ensure_open(self) -> None:
        if self._closed or self._close_requested:
            raise EmbeddingWorkerError("embedding worker client is closed")

    async def _ensure_started_locked(self) -> None:
        process = self._process
        if (
            self._state == WorkerState.READY
            and process is not None
            and process.returncode is None
        ):
            return

        if process is not None:
            await self._stop_process_locked(force=True)

        self._generation += 1
        self._state = WorkerState.STARTING
        startup_started_at = perf_counter()
        self._emit_event(
            "INFO",
            event="embedding_worker_starting",
            operation="startup",
            outcome="started",
            restart=self._generation > 1,
            model_version=self._expected_model_version,
        )
        try:
            await asyncio.wait_for(
                self._spawn_and_wait_ready_locked(self._generation),
                timeout=self._startup_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            worker_pid = self.worker_pid
            await self._stop_process_locked(force=True)
            self._state = WorkerState.NEW
            self._emit_event(
                "WARNING",
                event="embedding_worker_startup_timed_out",
                operation="startup",
                outcome="timeout",
                duration_ms=self._elapsed_ms(startup_started_at),
                timeout_s=self._startup_timeout_s,
                worker_pid=worker_pid,
                forced_stop=True,
            )
            raise EmbeddingWorkerTimeoutError(
                "startup",
                self._startup_timeout_s,
            ) from exc
        except asyncio.CancelledError:
            worker_pid = self.worker_pid
            await self._run_cleanup_shielded(
                asyncio.create_task(self._stop_process_locked(force=True))
            )
            self._state = WorkerState.NEW
            self._emit_event(
                "INFO",
                event="embedding_worker_startup_canceled",
                operation="startup",
                outcome="canceled",
                duration_ms=self._elapsed_ms(startup_started_at),
                worker_pid=worker_pid,
                forced_stop=True,
            )
            raise
        except EmbeddingWorkerError as exc:
            worker_pid = self.worker_pid
            await self._stop_process_locked(force=True)
            self._state = WorkerState.NEW
            self._emit_event(
                "ERROR",
                event="embedding_worker_startup_failed",
                operation="startup",
                outcome="failed",
                duration_ms=self._elapsed_ms(startup_started_at),
                worker_pid=worker_pid,
                forced_stop=True,
                **self._safe_error_fields(exc),
            )
            raise
        except Exception as exc:
            worker_pid = self.worker_pid
            await self._stop_process_locked(force=True)
            self._state = WorkerState.NEW
            self._emit_event(
                "ERROR",
                event="embedding_worker_startup_failed",
                operation="startup",
                outcome="failed",
                duration_ms=self._elapsed_ms(startup_started_at),
                worker_pid=worker_pid,
                forced_stop=True,
                error_type=type(exc).__name__,
            )
            raise EmbeddingWorkerError(
                "embedding worker failed to start",
                context={"generation": self._generation},
            ) from exc
        else:
            self._emit_event(
                "INFO",
                event="embedding_worker_ready",
                operation="startup",
                outcome="success",
                duration_ms=self._elapsed_ms(startup_started_at),
                dimension=self._dimension,
                model_version=self._expected_model_version,
            )

    async def _spawn_and_wait_ready_locked(self, generation: int) -> None:
        process = await asyncio.create_subprocess_exec(
            *self._command,
            "--generation",
            str(generation),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
            env=self._environment,
            limit=self._max_response_bytes + 1,
        )
        self._process = process
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))

        response = decode_worker_response(
            await self._read_stdout_line(process),
            max_bytes=self._max_response_bytes,
        )
        if isinstance(response, WorkerFailure):
            raise EmbeddingWorkerError(
                "embedding worker reported a startup failure",
                context={"error_type": response.error_type, "generation": generation},
            )
        if not isinstance(response, WorkerReady):
            raise EmbeddingWorkerProtocolError("worker did not send READY during startup")
        if response.generation != generation:
            raise EmbeddingWorkerProtocolError("worker READY generation does not match")
        if response.model_version != self._expected_model_version:
            raise EmbeddingWorkerProtocolError(
                "worker READY model_version does not match",
                context={
                    "expected_model_version": self._expected_model_version,
                    "actual_model_version": response.model_version,
                },
            )

        self._dimension = response.dimension
        self._state = WorkerState.READY

    def _build_request(self, texts: list[str]) -> EmbeddingRequest:
        try:
            return EmbeddingRequest(
                protocol_version=PROTOCOL_VERSION,
                type="embed",
                generation=self._generation,
                request_id=str(uuid4()),
                texts=tuple(texts),
            )
        except ValidationError as exc:
            raise EmbeddingWorkerProtocolError(
                "embedding request failed protocol validation",
                context={"validation_error_count": exc.error_count()},
            ) from exc

    async def _exchange_locked(self, request: EmbeddingRequest) -> WorkerResponse:
        process = self._require_live_process()
        if process.stdin is None:
            raise EmbeddingWorkerError("embedding worker stdin is unavailable")
        try:
            process.stdin.write(encode_message(
                request,
                max_bytes=self._max_request_bytes,
            ))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise EmbeddingWorkerError(
                "embedding worker request pipe closed unexpectedly",
                context={"generation": self._generation},
            ) from exc

        return decode_worker_response(
            await self._read_stdout_line(process),
            max_bytes=self._max_response_bytes,
        )

    async def _read_stdout_line(self, process: asyncio.subprocess.Process) -> bytes:
        if process.stdout is None:
            raise EmbeddingWorkerError("embedding worker stdout is unavailable")
        try:
            raw = await process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise EmbeddingWorkerProtocolError(
                "embedding worker response exceeds the stream limit"
            ) from exc
        if not raw:
            exit_code = await process.wait()
            raise EmbeddingWorkerError(
                "embedding worker exited before sending a response",
                context={"exit_code": exit_code, "generation": self._generation},
            )
        if not raw.endswith(b"\n"):
            raise EmbeddingWorkerProtocolError(
                "embedding worker response is not newline terminated"
            )
        return raw

    def _require_live_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is None or process.returncode is not None:
            raise EmbeddingWorkerError(
                "embedding worker is not running",
                context={"generation": self._generation},
            )
        return process

    async def _fail_protocol_locked(self, message: str) -> NoReturn:
        worker_pid = self.worker_pid
        await self._stop_process_locked(force=True)
        self._state = WorkerState.NEW
        self._emit_event(
            "ERROR",
            event="embedding_worker_protocol_violation",
            operation="protocol",
            outcome="failed",
            worker_pid=worker_pid,
            forced_stop=True,
            error_type="EmbeddingWorkerProtocolError",
        )
        raise EmbeddingWorkerProtocolError(
            message,
            context={"generation": self._generation},
        )

    async def _stop_process_locked(self, *, force: bool) -> None:
        process = self._process
        stderr_task = self._stderr_task
        if process is None:
            self._stderr_task = None
            self._dimension = None
            if stderr_task is not None:
                await self._finish_stderr_task(stderr_task)
            return

        if process.returncode is None and not force and process.stdin is not None:
            process.stdin.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()
            try:
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_s)
            except asyncio.TimeoutError:
                force = True

        if process.returncode is None and force:
            with suppress(ProcessLookupError):
                process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_s)
            except asyncio.TimeoutError as exc:
                raise EmbeddingWorkerTimeoutError(
                    "shutdown",
                    self._shutdown_timeout_s,
                ) from exc

        self._process = None
        self._stderr_task = None
        self._dimension = None
        if stderr_task is not None:
            await self._finish_stderr_task(stderr_task)

    async def _interrupt_current_process_for_close(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            process.kill()
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=self._shutdown_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise EmbeddingWorkerTimeoutError(
                "shutdown",
                self._shutdown_timeout_s,
            ) from exc

    @staticmethod
    async def _run_cleanup_shielded(cleanup: asyncio.Task[None]) -> None:
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        await cleanup

    async def _finish_stderr_task(self, task: asyncio.Task[None]) -> None:
        done, _ = await asyncio.wait(
            {task},
            timeout=self._shutdown_timeout_s,
        )
        if not done:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                return
            decoded = chunk.decode("utf-8", errors="replace")
            self._stderr_tail = (self._stderr_tail + decoded)[-self._max_stderr_chars:]

    def _emit_event(
        self,
        level: str,
        *,
        event: str,
        operation: str,
        outcome: str,
        **fields: Any,
    ) -> None:
        event_fields: dict[str, Any] = {
            "component": "process_embedding_client",
            "event": event,
            "operation": operation,
            "outcome": outcome,
            "generation": self._generation,
        }
        worker_pid = self.worker_pid
        if worker_pid is not None:
            event_fields["worker_pid"] = worker_pid
        event_fields.update(fields)
        logger.bind(**event_fields).log(
            level,
            "Embedding worker | event={event} | outcome={outcome}",
            event=event,
            outcome=outcome,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 3)

    @staticmethod
    def _safe_error_fields(exc: EmbeddingWorkerError) -> dict[str, Any]:
        fields: dict[str, Any] = {"error_type": type(exc).__name__}
        exit_code = exc.context.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            fields["exit_code"] = exit_code
        worker_error_type = exc.context.get("error_type")
        if isinstance(worker_error_type, str):
            fields["worker_error_type"] = worker_error_type
        return fields
