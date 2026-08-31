from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from typing import BinaryIO, Protocol, TextIO, cast

from pydantic import ValidationError

from errors import EmbeddingWorkerProtocolError
from rag.embedding_worker_protocol import (
    MAX_PROTOCOL_MESSAGE_BYTES,
    MAX_VECTOR_DIMENSION,
    PROTOCOL_VERSION,
    EmbeddingRequest,
    EmbeddingResult,
    WorkerFailure,
    WorkerReady,
    decode_worker_request,
    encode_message,
)

EXIT_OK = 0
EXIT_STARTUP_FAILURE = 20
EXIT_PROTOCOL_FAILURE = 30
EXIT_OUTPUT_FAILURE = 40


class EmbeddingModel(Protocol):
    """Narrow boundary around the synchronous model owned by this process."""

    def get_embedding_dimension(self) -> int | None:
        """Return the fixed output dimension, if the model exposes one."""
        ...

    def encode(
        self,
        sentences: list[str],
    ) -> object:
        """Synchronously encode one validated batch."""
        ...


ModelLoader = Callable[[str], EmbeddingModel]


class _SentenceTransformerAdapter:
    """Keep the third-party model shape behind our narrow worker boundary."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "embedding worker requires the rag extra: pip install -e '.[rag]'"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def get_embedding_dimension(self) -> int | None:
        dimension: int | None = self._model.get_embedding_dimension()
        return dimension

    def encode(
        self,
        sentences: list[str],
    ) -> object:
        output: object = self._model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return output


def _load_sentence_transformer(model_name: str) -> EmbeddingModel:
    return _SentenceTransformerAdapter(model_name)


def serve(
    *,
    generation: int,
    model_name: str,
    model_version: str,
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: TextIO,
    model_loader: ModelLoader = _load_sentence_transformer,
) -> int:
    """Load one model and serve bounded JSONL requests until stdin reaches EOF."""
    try:
        with redirect_stdout(stderr):
            model = model_loader(model_name)
            dimension = model.get_embedding_dimension()
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 1 <= dimension <= MAX_VECTOR_DIMENSION
        ):
            raise ValueError("embedding model returned an invalid dimension")
        ready = WorkerReady(
            protocol_version=PROTOCOL_VERSION,
            type="ready",
            generation=generation,
            model_version=model_version,
            dimension=dimension,
        )
    except Exception as exc:
        _write_diagnostic(stderr, "MODEL_STARTUP_FAILED", exc)
        failure = WorkerFailure(
            protocol_version=PROTOCOL_VERSION,
            type="error",
            generation=max(generation, 0),
            request_id=None,
            error_type="model_startup_failed",
            message="embedding model failed to initialize",
        )
        if not _emit(stdout, failure):
            return EXIT_OUTPUT_FAILURE
        return EXIT_STARTUP_FAILURE

    if not _emit(stdout, ready):
        return EXIT_OUTPUT_FAILURE

    while True:
        raw_request = stdin.readline(MAX_PROTOCOL_MESSAGE_BYTES + 1)
        if not raw_request:
            return EXIT_OK
        if (
            len(raw_request) > MAX_PROTOCOL_MESSAGE_BYTES
            or not raw_request.endswith(b"\n")
        ):
            return _terminate_for_protocol_error(
                stdout=stdout,
                stderr=stderr,
                generation=generation,
                request_id=None,
                cause=EmbeddingWorkerProtocolError(
                    "embedding worker request is oversized or not newline terminated"
                ),
            )

        try:
            request = decode_worker_request(raw_request)
        except EmbeddingWorkerProtocolError as exc:
            return _terminate_for_protocol_error(
                stdout=stdout,
                stderr=stderr,
                generation=generation,
                request_id=None,
                cause=exc,
            )

        if request.generation != generation:
            return _terminate_for_protocol_error(
                stdout=stdout,
                stderr=stderr,
                generation=generation,
                request_id=request.request_id,
                cause=EmbeddingWorkerProtocolError(
                    "embedding request generation does not match worker generation"
                ),
            )

        response = _run_inference(
            model=model,
            request=request,
            dimension=ready.dimension,
            stderr=stderr,
        )
        if not _emit(stdout, response):
            return EXIT_OUTPUT_FAILURE


def _run_inference(
    *,
    model: EmbeddingModel,
    request: EmbeddingRequest,
    dimension: int,
    stderr: TextIO,
) -> EmbeddingResult | WorkerFailure:
    try:
        with redirect_stdout(stderr):
            output = model.encode(list(request.texts))
        vectors = _coerce_vectors(
            output,
            expected_batch_size=len(request.texts),
            expected_dimension=dimension,
        )
        return EmbeddingResult(
            protocol_version=PROTOCOL_VERSION,
            type="result",
            generation=request.generation,
            request_id=request.request_id,
            vectors=vectors,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _write_diagnostic(stderr, "MODEL_OUTPUT_INVALID", exc)
        error_type = "model_output_invalid"
        message = "embedding model returned an invalid result"
    except Exception as exc:
        _write_diagnostic(stderr, "MODEL_INFERENCE_FAILED", exc)
        error_type = "model_inference_failed"
        message = "embedding model inference failed"

    return WorkerFailure(
        protocol_version=PROTOCOL_VERSION,
        type="error",
        generation=request.generation,
        request_id=request.request_id,
        error_type=error_type,
        message=message,
    )


def _coerce_vectors(
    output: object,
    *,
    expected_batch_size: int,
    expected_dimension: int,
) -> tuple[tuple[float, ...], ...]:
    tolist = getattr(output, "tolist", None)
    if not callable(tolist):
        raise TypeError("model output must expose tolist()")
    raw_vectors = tolist()
    if not isinstance(raw_vectors, list) or len(raw_vectors) != expected_batch_size:
        raise ValueError("model output batch size does not match request")

    vectors: list[tuple[float, ...]] = []
    for raw_vector in raw_vectors:
        if not isinstance(raw_vector, list) or len(raw_vector) != expected_dimension:
            raise ValueError("model output dimension does not match READY")
        vectors.append(tuple(raw_vector))
    return tuple(vectors)


def _terminate_for_protocol_error(
    *,
    stdout: BinaryIO,
    stderr: TextIO,
    generation: int,
    request_id: str | None,
    cause: Exception,
) -> int:
    _write_diagnostic(stderr, "PROTOCOL_ERROR", cause)
    failure = WorkerFailure(
        protocol_version=PROTOCOL_VERSION,
        type="error",
        generation=generation,
        request_id=request_id,
        error_type="protocol_error",
        message="embedding worker rejected an invalid request",
    )
    if not _emit(stdout, failure):
        return EXIT_OUTPUT_FAILURE
    return EXIT_PROTOCOL_FAILURE


def _emit(stdout: BinaryIO, message: WorkerReady | EmbeddingResult | WorkerFailure) -> bool:
    try:
        stdout.write(encode_message(message))
        stdout.flush()
    except (BrokenPipeError, OSError):
        return False
    return True


def _write_diagnostic(stderr: TextIO, marker: str, exc: Exception) -> None:
    stderr.write(f"{marker}: {type(exc).__name__}\n")
    stderr.flush()


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _bounded_non_blank(value: str) -> str:
    if not value.strip() or len(value) > 256:
        raise argparse.ArgumentTypeError("must contain 1 to 256 non-blank characters")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process-isolated embedding worker")
    parser.add_argument("--generation", type=_non_negative_int, required=True)
    parser.add_argument("--model-name", type=_bounded_non_blank, required=True)
    parser.add_argument("--model-version", type=_bounded_non_blank, required=True)
    args = parser.parse_args(argv)
    return serve(
        generation=args.generation,
        model_name=args.model_name,
        model_version=args.model_version,
        stdin=cast(BinaryIO, sys.stdin.buffer),
        stdout=cast(BinaryIO, sys.stdout.buffer),
        stderr=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
