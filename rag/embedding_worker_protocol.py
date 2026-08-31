from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, TypeAlias, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from errors import EmbeddingWorkerProtocolError

PROTOCOL_VERSION: Literal[1] = 1
MAX_PROTOCOL_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_BATCH_SIZE = 1024
MAX_TEXT_CHARS = 65_536
MAX_VECTOR_DIMENSION = 4096

Generation: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
StrictString: TypeAlias = Annotated[str, Field(strict=True)]
PositiveDimension: TypeAlias = Annotated[
    int,
    Field(strict=True, gt=0, le=MAX_VECTOR_DIMENSION),
]
ProtocolModelT = TypeVar("ProtocolModelT", bound=BaseModel)


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_request_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("request_id must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("request_id must be a canonical UUID")
    return value


class EmbeddingRequest(_ProtocolModel):
    protocol_version: Literal[1]
    type: Literal["embed"]
    generation: Generation
    request_id: StrictString
    texts: tuple[StrictString, ...]

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _validate_request_id(value)

    @field_validator("texts", mode="before")
    @classmethod
    def validate_texts(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("texts must be an array")
        if not value:
            raise ValueError("texts must not be empty")
        if len(value) > MAX_BATCH_SIZE:
            raise ValueError("embedding batch exceeds the protocol limit")
        for text in value:
            if not isinstance(text, str):
                raise ValueError("every embedding input must be a string")
            if not text.strip():
                raise ValueError("embedding inputs must not be blank")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError("embedding input exceeds the protocol limit")
        return value


class WorkerReady(_ProtocolModel):
    protocol_version: Literal[1]
    type: Literal["ready"]
    generation: Generation
    model_version: Annotated[
        str,
        Field(strict=True, min_length=1, max_length=256),
    ]
    dimension: PositiveDimension

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_version must not be blank")
        return value


class EmbeddingResult(_ProtocolModel):
    protocol_version: Literal[1]
    type: Literal["result"]
    generation: Generation
    request_id: StrictString
    vectors: tuple[tuple[float, ...], ...]

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _validate_request_id(value)

    @field_validator("vectors", mode="before")
    @classmethod
    def validate_vectors(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("vectors must be a non-empty array")
        if len(value) > MAX_BATCH_SIZE:
            raise ValueError("embedding result exceeds the protocol batch limit")

        expected_dimension: int | None = None
        for vector in value:
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError("every vector must be a non-empty array")
            if len(vector) > MAX_VECTOR_DIMENSION:
                raise ValueError("embedding vector exceeds the protocol dimension limit")
            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ValueError("embedding vector dimensions must match")

            has_nonzero_value = False
            for item in vector:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError("embedding vector values must be JSON numbers")
                try:
                    numeric = float(item)
                except OverflowError as exc:
                    raise ValueError(
                        "embedding vector values must fit in a finite float"
                    ) from exc
                if not math.isfinite(numeric):
                    raise ValueError("embedding vector values must be finite")
                has_nonzero_value = has_nonzero_value or numeric != 0.0
            if not has_nonzero_value:
                raise ValueError("embedding vectors must not be zero vectors")

        return value


class WorkerFailure(_ProtocolModel):
    protocol_version: Literal[1]
    type: Literal["error"]
    generation: Generation
    request_id: StrictString | None = None
    error_type: Annotated[
        str,
        Field(
            strict=True,
            min_length=1,
            max_length=128,
            pattern=r"^[a-z][a-z0-9_]*$",
        ),
    ]
    message: Annotated[
        str,
        Field(strict=True, min_length=1, max_length=1024),
    ]

    @field_validator("request_id")
    @classmethod
    def validate_optional_request_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_request_id(value)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("worker error message must not be blank")
        return value


WorkerResponse: TypeAlias = WorkerReady | EmbeddingResult | WorkerFailure
ProtocolMessage: TypeAlias = EmbeddingRequest | WorkerResponse


def encode_message(
    message: ProtocolMessage,
    *,
    max_bytes: int = MAX_PROTOCOL_MESSAGE_BYTES,
) -> bytes:
    """Serialize one validated protocol message as a bounded JSON line."""
    _validate_max_bytes(max_bytes)
    serialized = cast(str, message.model_dump_json())
    encoded = (serialized + "\n").encode("utf-8")
    if len(encoded) > max_bytes:
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message exceeds the configured size limit",
            context={"message_bytes": len(encoded), "max_bytes": max_bytes},
        )
    return encoded


def decode_worker_request(
    raw: bytes,
    *,
    max_bytes: int = MAX_PROTOCOL_MESSAGE_BYTES,
) -> EmbeddingRequest:
    """Decode a parent-to-worker request and reject response message types."""
    payload = _decode_json_object(raw, max_bytes=max_bytes)
    if payload.get("type") != "embed":
        raise _unexpected_message_type(payload.get("type"), direction="request")
    return _validate_model(EmbeddingRequest, payload)


def decode_worker_response(
    raw: bytes,
    *,
    max_bytes: int = MAX_PROTOCOL_MESSAGE_BYTES,
) -> WorkerResponse:
    """Decode a worker-to-parent response and reject request message types."""
    payload = _decode_json_object(raw, max_bytes=max_bytes)
    message_type = payload.get("type")
    response_type: type[WorkerReady] | type[EmbeddingResult] | type[WorkerFailure]
    if message_type == "ready":
        response_type = WorkerReady
    elif message_type == "result":
        response_type = EmbeddingResult
    elif message_type == "error":
        response_type = WorkerFailure
    else:
        raise _unexpected_message_type(message_type, direction="response")
    return _validate_model(response_type, payload)


def _decode_json_object(raw: bytes, *, max_bytes: int) -> dict[str, Any]:
    _validate_max_bytes(max_bytes)
    if not isinstance(raw, bytes):
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message must be bytes"
        )
    if not raw:
        raise EmbeddingWorkerProtocolError("embedding worker IPC message is empty")
    if len(raw) > max_bytes:
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message exceeds the configured size limit",
            context={"message_bytes": len(raw), "max_bytes": max_bytes},
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message is not valid UTF-8"
        ) from exc

    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message must be a JSON object"
        )
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _validate_model(
    model_type: type[ProtocolModelT],
    payload: dict[str, Any],
) -> ProtocolModelT:
    try:
        return cast(ProtocolModelT, model_type.model_validate(payload))
    except ValidationError as exc:
        raise EmbeddingWorkerProtocolError(
            "embedding worker IPC message failed schema validation",
            context={
                "message_type": str(payload.get("type"))[:64],
                "validation_error_count": exc.error_count(),
            },
        ) from exc


def _unexpected_message_type(value: object, *, direction: str) -> EmbeddingWorkerProtocolError:
    return EmbeddingWorkerProtocolError(
        f"unexpected embedding worker {direction} message type",
        context={"message_type": str(value)[:64]},
    )


def _validate_max_bytes(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
