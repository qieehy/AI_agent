from __future__ import annotations

import asyncio
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from errors import ToolRoutingError

ToolSchema = dict[str, Any]
Vector = tuple[float, ...]


class AsyncBatchEmbedder(Protocol):
    """Asynchronous batch interface required by the Router."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a non-empty batch of texts."""
        ...



@dataclass(frozen=True, slots=True)
class ToolScore:
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class ToolRoute:
    """Immutable routing result used by Runtime and observability."""

    selected_schemas: tuple[ToolSchema, ...]
    ranked_scores: tuple[ToolScore, ...]
    model_version: str
    threshold: float

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(get_tool_name(schema) for schema in self.selected_schemas)


def get_tool_name(schema: ToolSchema) -> str:
    """Extract the stable tool name from an OpenAI-compatible tool schema."""
    try:
        name = schema["function"]["name"]
    except (KeyError, TypeError) as exc:
        raise ToolRoutingError("invalid tool schema: missing function.name") from exc

    if not isinstance(name, str) or not name:
        raise ToolRoutingError(
            "invalid tool schema: function.name must be a non-empty string"
        )

    return name


def _tool_text(schema: ToolSchema) -> str:
    """Build a deterministic capability representation for embedding."""
    function = schema.get("function")
    if not isinstance(function, dict):
        raise ToolRoutingError("invalid tool schema: function must be an object")

    name = get_tool_name(schema)
    description = function.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ToolRoutingError(
            "invalid tool schema: function.description must be a string"
        )

    parameters = function.get("parameters", {"type": "object"})
    if not isinstance(parameters, dict):
        raise ToolRoutingError(
            "invalid tool schema: function.parameters must be an object"
        )

    try:
        parameter_text = json.dumps(
            parameters,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ToolRoutingError(
            "invalid tool schema: function.parameters must be JSON serializable"
        ) from exc

    return f"name: {name}\ndescription: {description}\nparameters: {parameter_text}"


def _validated_vector(vector: object, *, source: str) -> Vector:
    if not isinstance(vector, list) or not vector:
        raise ToolRoutingError(
            "embedding provider returned an invalid vector",
            context={"source": source, "reason": "empty_or_non_list"},
        )

    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ToolRoutingError(
                "embedding provider returned an invalid vector",
                context={"source": source, "reason": "non_numeric_value"},
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise ToolRoutingError(
                "embedding provider returned an invalid vector",
                context={"source": source, "reason": "non_finite_value"},
            )
        values.append(converted)

    if not any(values):
        raise ToolRoutingError(
            "embedding provider returned an invalid vector",
            context={"source": source, "reason": "zero_vector"},
        )

    return tuple(values)


def _cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ToolRoutingError(
            "embedding dimensions must match",
            context={"query_dimension": len(left), "tool_dimension": len(right)},
        )

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    dot = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return dot / (left_norm * right_norm)


class ToolRouter(Protocol):
    """Select the only tool schemas an LLM may call during a run."""

    async def route(
        self,
        query: str,
        schemas: list[ToolSchema],
    ) -> ToolRoute:
        """Return a fail-closed candidate set for the query."""
        ...


class EmbeddingRouter(ToolRouter):
    """Embedding-only tool retrieval with a bounded, versioned LRU cache.

    The injected embedder owns its timeout, cancellation, and resource-cleanup
    semantics. The Router awaits that async boundary directly and serializes routes
    to keep its vector cache consistent.
    """

    def __init__(
        self,
        embedder: AsyncBatchEmbedder,
        *,
        model_version: str,
        threshold: float = 0.35,
        top_k: int = 3,
        cache_size: int = 256,
        max_query_chars: int = 4096,
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if cache_size <= 0:
            raise ValueError("cache_size must be greater than 0")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be greater than 0")

        self._embedder = embedder
        self._model_version = model_version
        self._threshold = threshold
        self._top_k = top_k
        self._cache_size = cache_size
        self._max_query_chars = max_query_chars
        self._tool_vectors: OrderedDict[str, Vector] = OrderedDict()
        self._route_lock = asyncio.Lock()

    async def route(
        self,
        query: str,
        schemas: list[ToolSchema],
    ) -> ToolRoute:
        if not schemas:
            return ToolRoute((), (), self._model_version, self._threshold)
        if not isinstance(query, str) or not query.strip():
            return ToolRoute((), (), self._model_version, self._threshold)
        if len(query) > self._max_query_chars:
            raise ToolRoutingError(
                "tool routing query exceeds the configured size limit",
                context={
                    "query_chars": len(query),
                    "max_query_chars": self._max_query_chars,
                },
            )

        names = [get_tool_name(schema) for schema in schemas]
        if len(names) != len(set(names)):
            raise ToolRoutingError("tool schemas contain duplicate function names")
        tool_texts = [_tool_text(schema) for schema in schemas]
        if len(tool_texts) > self._cache_size:
            raise ToolRoutingError(
                "tool embedding cache is too small for the active schema set",
                context={
                    "schema_count": len(schemas),
                    "cache_size": self._cache_size,
                },
            )

        async with self._route_lock:
            missing_texts = [text for text in tool_texts if text not in self._tool_vectors]
            vectors = await self._embed_safely([query, *missing_texts])
            query_vector = vectors[0]

            for text, vector in zip(missing_texts, vectors[1:], strict=True):
                self._tool_vectors[text] = vector
                self._tool_vectors.move_to_end(text)
                while len(self._tool_vectors) > self._cache_size:
                    self._tool_vectors.popitem(last=False)

            tool_vectors: list[Vector] = []
            for text in tool_texts:
                cached_vector = self._tool_vectors.get(text)
                if cached_vector is None:  # pragma: no cover - guarded by the route lock
                    raise ToolRoutingError("tool embedding cache invariant violated")
                self._tool_vectors.move_to_end(text)
                tool_vectors.append(cached_vector)

        scored = [
            (index, _cosine_similarity(query_vector, vector), schema, name)
            for index, (schema, name, vector) in enumerate(
                zip(schemas, names, tool_vectors, strict=True)
            )
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = [item for item in scored if item[1] >= self._threshold][
            : self._top_k
        ]

        return ToolRoute(
            selected_schemas=tuple(item[2] for item in selected),
            ranked_scores=tuple(ToolScore(name=item[3], score=item[1]) for item in scored),
            model_version=self._model_version,
            threshold=self._threshold,
        )

    async def _embed_safely(self, texts: list[str]) -> list[Vector]:
        try:
            raw_vectors = await self._embedder.embed_batch(texts)
        except Exception as exc:
            raise ToolRoutingError(
                "tool embedding failed",
                context={"batch_size": len(texts), "model_version": self._model_version},
            ) from exc

        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(texts):
            raise ToolRoutingError(
                "embedding provider returned an unexpected batch size",
                context={"expected": len(texts)},
            )

        return [
            _validated_vector(vector, source="query" if index == 0 else "tool")
            for index, vector in enumerate(raw_vectors)
        ]
