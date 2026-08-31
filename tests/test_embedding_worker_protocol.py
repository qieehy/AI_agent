from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from errors import AgentError, EmbeddingWorkerError, EmbeddingWorkerProtocolError
from rag.embedding_worker_protocol import (
    MAX_BATCH_SIZE,
    MAX_TEXT_CHARS,
    PROTOCOL_VERSION,
    EmbeddingRequest,
    EmbeddingResult,
    WorkerFailure,
    WorkerReady,
    decode_worker_request,
    decode_worker_response,
    encode_message,
)

REQUEST_ID = "12345678-1234-5678-abcd-567812345678"


def _json_line(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _request_payload(**overrides) -> dict:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "embed",
        "generation": 2,
        "request_id": REQUEST_ID,
        "texts": ["query", "tool capability"],
    }
    payload.update(overrides)
    return payload


def test_protocol_error_has_a_stable_framework_taxonomy() -> None:
    assert issubclass(EmbeddingWorkerProtocolError, EmbeddingWorkerError)
    assert issubclass(EmbeddingWorkerError, AgentError)


def test_embedding_request_round_trip() -> None:
    request = EmbeddingRequest.model_validate(_request_payload())

    decoded = decode_worker_request(encode_message(request))

    assert decoded == request
    assert decoded.texts == ("query", "tool capability")
    assert encode_message(request).endswith(b"\n")


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        (
            WorkerReady(
                protocol_version=PROTOCOL_VERSION,
                type="ready",
                generation=2,
                model_version="model-v1",
                dimension=2,
            ),
            WorkerReady,
        ),
        (
            EmbeddingResult(
                protocol_version=PROTOCOL_VERSION,
                type="result",
                generation=2,
                request_id=REQUEST_ID,
                vectors=((1.0, 0.0), (0.5, 0.5)),
            ),
            EmbeddingResult,
        ),
        (
            WorkerFailure(
                protocol_version=PROTOCOL_VERSION,
                type="error",
                generation=2,
                request_id=None,
                error_type="model_load_failed",
                message="model unavailable",
            ),
            WorkerFailure,
        ),
    ],
)
def test_worker_response_round_trip(message, expected_type) -> None:
    decoded = decode_worker_response(encode_message(message))

    assert isinstance(decoded, expected_type)
    assert decoded == message


@pytest.mark.parametrize(
    "payload",
    [
        {
            "protocol_version": PROTOCOL_VERSION,
            "type": "ready",
            "generation": 2,
            "model_version": 123,
            "dimension": 2,
        },
        {
            "protocol_version": PROTOCOL_VERSION,
            "type": "error",
            "generation": 2,
            "request_id": None,
            "error_type": "model_load_failed",
            "message": 123,
        },
    ],
)
def test_worker_response_strings_are_not_coerced(payload) -> None:
    with pytest.raises(EmbeddingWorkerProtocolError, match="schema validation"):
        decode_worker_response(_json_line(payload))


@pytest.mark.parametrize(
    "payload",
    [
        _request_payload(protocol_version=2),
        _request_payload(generation=True),
        _request_payload(request_id="not-a-uuid"),
        _request_payload(texts=[]),
        _request_payload(texts=["   "]),
        _request_payload(texts=[1]),
        _request_payload(unexpected="field"),
    ],
)
def test_request_schema_rejects_invalid_payload(payload) -> None:
    with pytest.raises(EmbeddingWorkerProtocolError, match="schema validation"):
        decode_worker_request(_json_line(payload))


def test_request_id_must_be_canonical_uuid() -> None:
    noncanonical = str(UUID(REQUEST_ID)).upper()

    with pytest.raises(EmbeddingWorkerProtocolError, match="schema validation"):
        decode_worker_request(_json_line(_request_payload(request_id=noncanonical)))


def test_request_enforces_batch_and_text_limits() -> None:
    with pytest.raises(EmbeddingWorkerProtocolError, match="schema validation"):
        decode_worker_request(
            _json_line(_request_payload(texts=["x"] * (MAX_BATCH_SIZE + 1)))
        )

    with pytest.raises(EmbeddingWorkerProtocolError, match="schema validation"):
        decode_worker_request(
            _json_line(_request_payload(texts=["x" * (MAX_TEXT_CHARS + 1)]))
        )


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[]],
        [[0.0, 0.0]],
        [[1.0], [1.0, 2.0]],
        [[True, 1.0]],
        [["1", 1.0]],
        [[float("inf"), 1.0]],
        [[10**4000, 1.0]],
    ],
)
def test_result_rejects_invalid_vectors(vectors) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "result",
        "generation": 2,
        "request_id": REQUEST_ID,
        "vectors": vectors,
    }

    with pytest.raises(EmbeddingWorkerProtocolError):
        decode_worker_response(_json_line(payload))


def test_non_standard_json_constants_are_rejected() -> None:
    raw = (
        f'{{"protocol_version":1,"type":"result","generation":2,'
        f'"request_id":"{REQUEST_ID}","vectors":[[NaN,1.0]]}}\n'
    ).encode()

    with pytest.raises(EmbeddingWorkerProtocolError, match="not valid JSON"):
        decode_worker_response(raw)


def test_duplicate_json_keys_are_rejected() -> None:
    raw = (
        f'{{"protocol_version":1,"type":"embed","type":"embed",'
        f'"generation":2,"request_id":"{REQUEST_ID}","texts":["query"]}}\n'
    ).encode()

    with pytest.raises(EmbeddingWorkerProtocolError, match="not valid JSON"):
        decode_worker_request(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not-json\n",
        b"[]\n",
        b"\xff\n",
    ],
)
def test_invalid_wire_payloads_are_rejected(raw) -> None:
    with pytest.raises(EmbeddingWorkerProtocolError):
        decode_worker_request(raw)


def test_message_size_is_checked_before_json_parsing() -> None:
    raw = _json_line(_request_payload())

    with pytest.raises(EmbeddingWorkerProtocolError, match="size limit") as exc_info:
        decode_worker_request(raw, max_bytes=len(raw) - 1)

    assert exc_info.value.context == {
        "message_bytes": len(raw),
        "max_bytes": len(raw) - 1,
    }


def test_encode_rejects_oversized_message() -> None:
    request = EmbeddingRequest.model_validate(_request_payload())

    with pytest.raises(EmbeddingWorkerProtocolError, match="size limit"):
        encode_message(request, max_bytes=1)


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5])
def test_max_bytes_must_be_a_positive_integer(max_bytes) -> None:
    request = EmbeddingRequest.model_validate(_request_payload())

    with pytest.raises(ValueError, match="positive integer"):
        encode_message(request, max_bytes=max_bytes)


def test_message_direction_is_enforced() -> None:
    request = EmbeddingRequest.model_validate(_request_payload())
    ready = WorkerReady(
        protocol_version=PROTOCOL_VERSION,
        type="ready",
        generation=2,
        model_version="model-v1",
        dimension=2,
    )

    with pytest.raises(EmbeddingWorkerProtocolError, match="request message type"):
        decode_worker_request(encode_message(ready))
    with pytest.raises(EmbeddingWorkerProtocolError, match="response message type"):
        decode_worker_response(encode_message(request))


def test_failure_message_is_bounded_and_typed() -> None:
    with pytest.raises(ValidationError):
        WorkerFailure(
            protocol_version=PROTOCOL_VERSION,
            type="error",
            generation=2,
            error_type="Invalid Type",
            message="failure",
        )

    with pytest.raises(ValidationError):
        WorkerFailure(
            protocol_version=PROTOCOL_VERSION,
            type="error",
            generation=2,
            error_type="model_failed",
            message="x" * 1025,
        )
