from __future__ import annotations

import argparse
import sys
import threading
from typing import Literal

from rag.embedding_worker_protocol import (
    PROTOCOL_VERSION,
    EmbeddingResult,
    WorkerFailure,
    WorkerReady,
    decode_worker_request,
    encode_message,
)

Mode = Literal[
    "success",
    "startup_hang",
    "startup_hang_once",
    "inference_hang",
    "inference_hang_once",
    "crash_before_ready",
    "crash_during_inference",
    "worker_error",
    "wrong_request_id",
    "wrong_generation",
    "wrong_generation_once",
    "malformed_json",
]

WRONG_REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _write_stdout(payload: bytes) -> None:
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _write_stderr(marker: str) -> None:
    sys.stderr.write(f"{marker}\n")
    sys.stderr.flush()


def _hang(marker: str) -> None:
    _write_stderr(marker)
    threading.Event().wait()


def _vectors_for(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple((float(index + 1), 1.0) for index, _ in enumerate(texts))


def run(*, mode: Mode, generation: int, model_version: str) -> int:
    if mode == "crash_before_ready":
        _write_stderr("CRASH_BEFORE_READY")
        return 21
    if mode == "startup_hang" or (mode == "startup_hang_once" and generation == 1):
        _hang("STARTUP_HANG")

    _write_stdout(encode_message(WorkerReady(
        protocol_version=PROTOCOL_VERSION,
        type="ready",
        generation=generation,
        model_version=model_version,
        dimension=2,
    )))

    for raw_request in sys.stdin.buffer:
        request = decode_worker_request(raw_request)

        if mode == "inference_hang" or (
            mode == "inference_hang_once" and generation == 1
        ):
            _hang("INFERENCE_HANG")
        if mode == "crash_during_inference":
            _write_stderr("CRASH_DURING_INFERENCE")
            return 22
        if mode == "malformed_json":
            _write_stdout(b"not-json\n")
            continue
        if mode == "worker_error":
            _write_stdout(encode_message(WorkerFailure(
                protocol_version=PROTOCOL_VERSION,
                type="error",
                generation=generation,
                request_id=request.request_id,
                error_type="model_inference_failed",
                message="deterministic fake worker failure",
            )))
            continue

        response_generation = (
            generation + 1
            if mode == "wrong_generation"
            or (mode == "wrong_generation_once" and generation == 1)
            else generation
        )
        response_request_id = (
            WRONG_REQUEST_ID if mode == "wrong_request_id" else request.request_id
        )
        _write_stdout(encode_message(EmbeddingResult(
            protocol_version=PROTOCOL_VERSION,
            type="result",
            generation=response_generation,
            request_id=response_request_id,
            vectors=_vectors_for(request.texts),
        )))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic test-only embedding worker")
    parser.add_argument(
        "--mode",
        choices=(
            "success",
            "startup_hang",
            "startup_hang_once",
            "inference_hang",
            "inference_hang_once",
            "crash_before_ready",
            "crash_during_inference",
            "worker_error",
            "wrong_request_id",
            "wrong_generation",
            "wrong_generation_once",
            "malformed_json",
        ),
        required=True,
    )
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    return run(
        mode=args.mode,
        generation=args.generation,
        model_version=args.model_version,
    )


if __name__ == "__main__":
    raise SystemExit(main())
