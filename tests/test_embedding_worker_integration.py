from __future__ import annotations

import math
import os

import pytest

from main import build_runtime
from rag.process_embeddings import WorkerState

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


@pytest.mark.anyio
async def test_production_runtime_uses_real_process_isolated_embedding_worker() -> None:
    if os.environ.get("RUN_REAL_EMBEDDING_INTEGRATION") != "1":
        pytest.skip("set RUN_REAL_EMBEDDING_INTEGRATION=1 to run the real model")

    application = build_runtime()
    client = application.embedding_client
    assert client.state == WorkerState.NEW
    assert client.worker_pid is None
    assert client.generation == 0

    try:
        vectors = await client.embed_batch(
            [
                "今天天气很好",
                "今日阳光明媚",
                "Python 装饰器怎么用",
            ]
        )

        first_pid = client.worker_pid
        assert first_pid is not None
        assert client.state == WorkerState.READY
        assert client.generation == 1
        assert len(vectors) == 3
        assert len(vectors[0]) > 0
        assert all(len(vector) == len(vectors[0]) for vector in vectors)
        assert all(
            all(math.isfinite(value) for value in vector)
            for vector in vectors
        )
        assert all(
            math.isclose(
                sum(value * value for value in vector),
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
            for vector in vectors
        )
        assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[2])

        second_vectors = await client.embed_batch(["再次使用同一个 Worker"])
        assert len(second_vectors) == 1
        assert len(second_vectors[0]) == len(vectors[0])
        assert client.worker_pid == first_pid
        assert client.generation == 1
    finally:
        await application.aclose()

    assert client.state == WorkerState.CLOSED
    assert client.worker_pid is None
