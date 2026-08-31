from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from tools.router import EmbeddingRouter, ToolRoutingError, get_tool_name


def make_schema(
    name: str,
    description: str = "",
    parameters: dict | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


class FakeBatchEmbedder:
    def __init__(self, vector_for: Callable[[str], list[float]]) -> None:
        self._vector_for = vector_for
        self.batches: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [self._vector_for(text) for text in texts]


def semantic_vector(text: str) -> list[float]:
    if text in {"东京天气", "天气"} or "weather" in text:
        return [1.0, 0.0]
    if "search" in text:
        return [0.8, 0.6]
    if "calculator" in text:
        return [0.0, 1.0]
    return [0.1, 0.9]


def make_router(embedder, **overrides) -> EmbeddingRouter:
    options = {
        "model_version": "fake-v1",
        "threshold": 0.5,
        "top_k": 2,
        "cache_size": 32,
        "max_query_chars": 100,
    }
    options.update(overrides)
    return EmbeddingRouter(embedder, **options)


@pytest.mark.anyio
async def test_embedding_router_selects_ranked_top_k() -> None:
    schemas = [
        make_schema("weather", "query weather"),
        make_schema("search", "search internet"),
        make_schema("calculator", "calculate numbers"),
    ]
    router = make_router(FakeBatchEmbedder(semantic_vector))

    result = await router.route("东京天气", schemas)

    assert result.selected_names == ("weather", "search")
    assert [score.name for score in result.ranked_scores] == [
        "weather",
        "search",
        "calculator",
    ]
    assert result.model_version == "fake-v1"


@pytest.mark.anyio
async def test_no_match_is_fail_closed() -> None:
    embedder = FakeBatchEmbedder(
        lambda text: [1.0, 0.0] if text == "weather" else [0.0, 1.0]
    )
    router = make_router(embedder, threshold=0.8)

    result = await router.route(
        "weather",
        [make_schema("calculator", "calculate numbers")],
    )

    assert result.selected_schemas == ()
    assert len(result.ranked_scores) == 1


@pytest.mark.anyio
async def test_empty_input_does_not_call_embedder() -> None:
    embedder = FakeBatchEmbedder(semantic_vector)
    router = make_router(embedder)

    assert (await router.route("", [make_schema("weather")])).selected_schemas == ()
    assert (await router.route("weather", [])).selected_schemas == ()
    assert embedder.batches == []


@pytest.mark.anyio
async def test_tool_embeddings_are_batched_and_cached() -> None:
    embedder = FakeBatchEmbedder(semantic_vector)
    router = make_router(embedder)
    schemas = [make_schema("weather"), make_schema("calculator")]

    await router.route("天气", schemas)
    await router.route("天气", schemas)

    assert len(embedder.batches) == 2
    assert len(embedder.batches[0]) == 3  # query + two uncached tools
    assert embedder.batches[1] == ["天气"]  # query only


@pytest.mark.anyio
async def test_parameter_schema_participates_in_cache_identity() -> None:
    embedder = FakeBatchEmbedder(semantic_vector)
    router = make_router(embedder)
    schema = make_schema(
        "weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    )

    await router.route("天气", [schema])
    schema["function"]["parameters"]["properties"]["days"] = {"type": "integer"}
    await router.route("天气", [schema])

    assert len(embedder.batches[0]) == 2
    assert len(embedder.batches[1]) == 2
    assert embedder.batches[0][1] != embedder.batches[1][1]


@pytest.mark.anyio
async def test_embedding_failure_is_translated_with_cause() -> None:
    class BrokenEmbedder:
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model failed")

    router = make_router(BrokenEmbedder())

    with pytest.raises(ToolRoutingError) as exc_info:
        await router.route("weather", [make_schema("weather")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert exc_info.value.context["model_version"] == "fake-v1"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "vector",
    [[], [0.0, 0.0], [float("nan"), 1.0], [True, 1.0], ["1", 1.0]],
)
async def test_invalid_embedding_vectors_fail_closed(vector) -> None:
    router = make_router(FakeBatchEmbedder(lambda text: vector))

    with pytest.raises(ToolRoutingError):
        await router.route("weather", [make_schema("weather")])


@pytest.mark.anyio
async def test_dimension_mismatch_fails_closed() -> None:
    embedder = FakeBatchEmbedder(
        lambda text: [1.0, 0.0] if text == "weather" else [1.0]
    )
    router = make_router(embedder)

    with pytest.raises(ToolRoutingError, match="dimensions must match"):
        await router.route("weather", [make_schema("weather")])


@pytest.mark.anyio
async def test_query_size_is_bounded_before_embedding() -> None:
    embedder = FakeBatchEmbedder(semantic_vector)
    router = make_router(embedder, max_query_chars=3)

    with pytest.raises(ToolRoutingError, match="size limit"):
        await router.route("1234", [make_schema("weather")])

    assert embedder.batches == []


@pytest.mark.anyio
async def test_duplicate_tool_names_are_rejected() -> None:
    router = make_router(FakeBatchEmbedder(semantic_vector))

    with pytest.raises(ToolRoutingError, match="duplicate"):
        await router.route("天气", [make_schema("weather"), make_schema("weather")])


@pytest.mark.anyio
async def test_cache_must_hold_the_active_schema_set() -> None:
    embedder = FakeBatchEmbedder(semantic_vector)
    router = make_router(embedder, cache_size=1)

    with pytest.raises(ToolRoutingError, match="cache is too small"):
        await router.route(
            "天气",
            [make_schema("weather"), make_schema("calculator")],
        )

    assert embedder.batches == []


@pytest.mark.anyio
async def test_native_async_embedder_owns_cancellation_cleanup() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    class CancelAwareEmbedder:
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    router = make_router(CancelAwareEmbedder())
    task = asyncio.create_task(router.route("天气", [make_schema("weather")]))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleaned.is_set()


def test_get_tool_name_rejects_malformed_schema() -> None:
    with pytest.raises(ToolRoutingError):
        get_tool_name({"type": "function"})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_version": ""}, "model_version"),
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.1}, "threshold"),
        ({"top_k": 0}, "top_k"),
        ({"cache_size": 0}, "cache_size"),
        ({"max_query_chars": 0}, "max_query_chars"),
    ],
)
def test_configuration_validation(kwargs, message) -> None:
    defaults = {"model_version": "fake-v1"}
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        EmbeddingRouter(FakeBatchEmbedder(semantic_vector), **defaults)
