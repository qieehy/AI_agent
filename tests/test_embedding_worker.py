from __future__ import annotations

from io import BytesIO, StringIO
from typing import cast

import pytest

from rag.embedding_worker import (
    EXIT_OK,
    EXIT_PROTOCOL_FAILURE,
    EXIT_STARTUP_FAILURE,
    EmbeddingModel,
    _SentenceTransformerAdapter,
    serve,
)
from rag.embedding_worker_protocol import (
    MAX_PROTOCOL_MESSAGE_BYTES,
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


class FakeArray:
    def __init__(self, value: object) -> None:
        self._value = value

    def tolist(self) -> object:
        return self._value


class FakeModel:
    def __init__(self, outputs: list[object], *, dimension: int | None = 2) -> None:
        self._outputs = outputs
        self._dimension = dimension
        self.calls: list[tuple[list[str], dict[str, bool]]] = []

    def get_embedding_dimension(self) -> int | None:
        return self._dimension

    def encode(self, sentences: list[str], **kwargs: bool) -> object:
        self.calls.append((sentences, kwargs))
        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return FakeArray(output)


class NoisyFakeModel(FakeModel):
    def get_embedding_dimension(self) -> int | None:
        print("model startup noise")
        return super().get_embedding_dimension()

    def encode(self, sentences: list[str], **kwargs: bool) -> object:
        print("model inference noise")
        return super().encode(sentences, **kwargs)


def test_sentence_transformer_adapter_exposes_protocol_dimension_method() -> None:
    upstream_model = FakeModel([[[1.0, 0.0]]], dimension=384)
    adapter = _SentenceTransformerAdapter.__new__(_SentenceTransformerAdapter)
    adapter._model = upstream_model

    assert adapter.get_embedding_dimension() == 384
    adapter.encode(["query"])
    assert upstream_model.calls == [
        (
            ["query"],
            {
                "normalize_embeddings": True,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        )
    ]


def _request(request_id: str, texts: tuple[str, ...], *, generation: int = 1) -> bytes:
    return encode_message(EmbeddingRequest(
        protocol_version=PROTOCOL_VERSION,
        type="embed",
        generation=generation,
        request_id=request_id,
        texts=texts,
    ))


def _responses(stdout: BytesIO) -> list[WorkerReady | EmbeddingResult | WorkerFailure]:
    return [decode_worker_response(line) for line in stdout.getvalue().splitlines(keepends=True)]


def _serve(
    *,
    model: FakeModel,
    requests: bytes = b"",
) -> tuple[int, list[WorkerReady | EmbeddingResult | WorkerFailure], str]:
    stdout = BytesIO()
    stderr = StringIO()
    exit_code = serve(
        generation=1,
        model_name="fake-model",
        model_version="fake-model@revision-1",
        stdin=BytesIO(requests),
        stdout=stdout,
        stderr=stderr,
        model_loader=lambda _: cast(EmbeddingModel, model),
    )
    return exit_code, _responses(stdout), stderr.getvalue()


def test_worker_sends_ready_runs_normalized_batch_and_exits_on_eof() -> None:
    model = FakeModel([[[1.0, 0.0], [0.0, 1.0]]])

    exit_code, responses, stderr = _serve(
        model=model,
        requests=_request(REQUEST_ID_1, ("first", "second")),
    )

    assert exit_code == EXIT_OK
    assert responses == [
        WorkerReady(
            protocol_version=PROTOCOL_VERSION,
            type="ready",
            generation=1,
            model_version="fake-model@revision-1",
            dimension=2,
        ),
        EmbeddingResult(
            protocol_version=PROTOCOL_VERSION,
            type="result",
            generation=1,
            request_id=REQUEST_ID_1,
            vectors=((1.0, 0.0), (0.0, 1.0)),
        ),
    ]
    assert model.calls == [
        (
            ["first", "second"],
            {},
        )
    ]
    assert stderr == ""


def test_startup_failure_is_sanitized_and_never_sends_ready() -> None:
    stdout = BytesIO()
    stderr = StringIO()

    def fail_to_load(_: str) -> EmbeddingModel:
        raise RuntimeError("secret model path")

    exit_code = serve(
        generation=1,
        model_name="fake-model",
        model_version="fake-v1",
        stdin=BytesIO(),
        stdout=stdout,
        stderr=stderr,
        model_loader=fail_to_load,
    )
    responses = _responses(stdout)

    assert exit_code == EXIT_STARTUP_FAILURE
    assert len(responses) == 1
    assert isinstance(responses[0], WorkerFailure)
    assert responses[0].request_id is None
    assert responses[0].error_type == "model_startup_failed"
    assert "secret model path" not in responses[0].message
    assert stderr.getvalue() == "MODEL_STARTUP_FAILED: RuntimeError\n"


def test_model_stdout_is_redirected_away_from_the_jsonl_channel() -> None:
    model = NoisyFakeModel([[[1.0, 0.0]]])

    exit_code, responses, stderr = _serve(
        model=model,
        requests=_request(REQUEST_ID_1, ("query",)),
    )

    assert exit_code == EXIT_OK
    assert len(responses) == 2
    assert isinstance(responses[0], WorkerReady)
    assert isinstance(responses[1], EmbeddingResult)
    assert stderr == "model startup noise\nmodel inference noise\n"


@pytest.mark.parametrize("dimension", [None, 0, 4097])
def test_invalid_model_dimension_fails_before_ready(dimension: int | None) -> None:
    exit_code, responses, stderr = _serve(model=FakeModel([], dimension=dimension))

    assert exit_code == EXIT_STARTUP_FAILURE
    assert len(responses) == 1
    assert isinstance(responses[0], WorkerFailure)
    assert responses[0].error_type == "model_startup_failed"
    assert "MODEL_STARTUP_FAILED" in stderr


def test_protocol_corruption_fails_closed_without_calling_model() -> None:
    model = FakeModel([])

    exit_code, responses, stderr = _serve(model=model, requests=b"not-json\n")

    assert exit_code == EXIT_PROTOCOL_FAILURE
    assert isinstance(responses[0], WorkerReady)
    assert isinstance(responses[1], WorkerFailure)
    assert responses[1].error_type == "protocol_error"
    assert model.calls == []
    assert "PROTOCOL_ERROR" in stderr


def test_oversized_request_is_rejected_before_model_inference() -> None:
    model = FakeModel([])
    oversized_request = b"{" + b"x" * MAX_PROTOCOL_MESSAGE_BYTES + b"\n"

    exit_code, responses, stderr = _serve(
        model=model,
        requests=oversized_request,
    )

    assert exit_code == EXIT_PROTOCOL_FAILURE
    assert isinstance(responses[1], WorkerFailure)
    assert responses[1].error_type == "protocol_error"
    assert model.calls == []
    assert "PROTOCOL_ERROR" in stderr


def test_wrong_request_generation_fails_closed_and_preserves_correlation_id() -> None:
    model = FakeModel([])

    exit_code, responses, _ = _serve(
        model=model,
        requests=_request(REQUEST_ID_1, ("query",), generation=2),
    )

    assert exit_code == EXIT_PROTOCOL_FAILURE
    failure = responses[1]
    assert isinstance(failure, WorkerFailure)
    assert failure.request_id == REQUEST_ID_1
    assert model.calls == []


def test_inference_failure_is_sanitized_and_worker_processes_next_request() -> None:
    model = FakeModel([
        RuntimeError("secret inference detail"),
        [[1.0, 0.0]],
    ])

    exit_code, responses, stderr = _serve(
        model=model,
        requests=(
            _request(REQUEST_ID_1, ("fail",))
            + _request(REQUEST_ID_2, ("recover",))
        ),
    )

    assert exit_code == EXIT_OK
    failure = responses[1]
    result = responses[2]
    assert isinstance(failure, WorkerFailure)
    assert failure.request_id == REQUEST_ID_1
    assert failure.error_type == "model_inference_failed"
    assert "secret inference detail" not in failure.message
    assert isinstance(result, EmbeddingResult)
    assert result.request_id == REQUEST_ID_2
    assert "MODEL_INFERENCE_FAILED: RuntimeError" in stderr


@pytest.mark.parametrize(
    "invalid_output",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0]],
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
    ],
)
def test_invalid_model_output_becomes_structured_failure(invalid_output: object) -> None:
    model = FakeModel([invalid_output])

    exit_code, responses, stderr = _serve(
        model=model,
        requests=_request(REQUEST_ID_1, ("query",)),
    )

    assert exit_code == EXIT_OK
    failure = responses[1]
    assert isinstance(failure, WorkerFailure)
    assert failure.request_id == REQUEST_ID_1
    assert failure.error_type == "model_output_invalid"
    assert "MODEL_OUTPUT_INVALID" in stderr
