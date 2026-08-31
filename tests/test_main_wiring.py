from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from config.settings import Settings
from main import (
    PROJECT_ROOT,
    RuntimeApplication,
    _embedding_worker_environment,
    _run_application_repl,
    build_runtime,
)
from rag.process_embeddings import ProcessEmbeddingClient
from tools import EmbeddingRouter, ToolRegistry


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        pattern="react",
        tool_router_model="router-model-v1",
        tool_router_threshold=0.42,
        tool_router_top_k=2,
        tool_router_cache_size=16,
        tool_router_max_query_chars=512,
        tool_router_worker_startup_timeout_s=101.0,
        tool_router_worker_inference_timeout_s=11.0,
        tool_router_worker_lock_timeout_s=3.0,
        tool_router_worker_shutdown_timeout_s=7.0,
        tool_router_worker_max_request_bytes=123_456,
        tool_router_worker_max_response_bytes=654_321,
        tool_router_worker_max_stderr_chars=2048,
        planner_timeout_s=13.0,
        planner_max_tasks=9,
        planner_max_goal_chars=700,
    )


def test_build_runtime_wires_process_embedding_client_into_production_router() -> None:
    registry = ToolRegistry()

    @registry.register
    def search(query: str) -> str:
        """Search documents."""
        return query

    llm = MagicMock()
    memory = MagicMock()
    embedding_client = MagicMock(spec=ProcessEmbeddingClient)
    runtime_instance = MagicMock()
    handler = MagicMock()
    worker_environment = {"PYTHONUTF8": "1"}

    with (
        patch("main.get_settings", return_value=_settings()),
        patch("main.AsyncLLMClient", return_value=llm),
        patch("main.create_registry", return_value=registry),
        patch("main.create_memory_manager", return_value=memory),
        patch(
            "main._embedding_worker_environment",
            return_value=worker_environment,
        ),
        patch(
            "main.ProcessEmbeddingClient",
            return_value=embedding_client,
        ) as client_type,
        patch("main.Runtime", return_value=runtime_instance) as runtime_type,
    ):
        application = build_runtime(handlers=[handler])

    client_type.assert_called_once_with(
        command=(
            sys.executable,
            "-m",
            "rag.embedding_worker",
            "--model-name",
            "router-model-v1",
            "--model-version",
            "router-model-v1",
        ),
        cwd=PROJECT_ROOT,
        environment=worker_environment,
        expected_model_version="router-model-v1",
        startup_timeout_s=101.0,
        inference_timeout_s=11.0,
        lock_timeout_s=3.0,
        shutdown_timeout_s=7.0,
        max_request_bytes=123_456,
        max_response_bytes=654_321,
        max_stderr_chars=2048,
    )
    runtime_kwargs = runtime_type.call_args.kwargs
    router = runtime_kwargs["tool_router"]
    assert isinstance(router, EmbeddingRouter)
    assert router._embedder is embedding_client
    assert runtime_kwargs["handlers"] == [handler]
    assert runtime_kwargs["validator"] is not None
    assert runtime_kwargs["planner"] is None
    assert application.runtime is runtime_instance
    assert application.memory_manager is memory
    assert application.embedding_client is embedding_client
    assert application.executor.get_schemas()[0]["function"]["name"] == "search"


def test_build_runtime_wires_planner_only_for_plan_execute_pattern() -> None:
    registry = ToolRegistry()
    llm = MagicMock()

    with (
        patch("main.get_settings", return_value=_settings()),
        patch("main.AsyncLLMClient", return_value=llm),
        patch("main.create_registry", return_value=registry),
        patch("main.create_memory_manager", return_value=MagicMock()),
        patch("main.ProcessEmbeddingClient", return_value=MagicMock()),
        patch("main.Runtime", return_value=MagicMock()) as runtime_type,
    ):
        build_runtime(pattern="plan_execute")

    planner = runtime_type.call_args.kwargs["planner"]
    assert planner is not None
    assert planner._llm_call is llm
    assert planner._timeout_s == 13.0
    assert planner._max_tasks == 9
    assert planner._max_goal_chars == 700


def test_worker_environment_is_allowlisted_and_excludes_agent_secrets() -> None:
    environment = _embedding_worker_environment(
        {
            "API_KEY": "llm-secret",
            "TAVILY_API_KEY": "search-secret",
            "HF_HOME": "model-cache",
            "HF_TOKEN": "private-model-token",
            "PATH": "worker-path",
            "SYSTEMROOT": "windows-root",
        }
    )

    assert environment == {
        "HF_HOME": "model-cache",
        "HF_TOKEN": "private-model-token",
        "PATH": "worker-path",
        "SYSTEMROOT": "windows-root",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
    }
    assert "API_KEY" not in environment
    assert "TAVILY_API_KEY" not in environment


@pytest.mark.anyio
async def test_build_runtime_constructs_real_client_without_eager_worker_start() -> None:
    with (
        patch("main.get_settings", return_value=_settings()),
        patch("main.AsyncLLMClient", return_value=MagicMock()),
        patch("main.create_registry", return_value=ToolRegistry()),
        patch("main.create_memory_manager", return_value=MagicMock()),
        patch("main.Runtime", return_value=MagicMock()),
    ):
        application = build_runtime()

    assert isinstance(application.embedding_client, ProcessEmbeddingClient)
    assert application.embedding_client.worker_pid is None
    assert application.embedding_client.generation == 0
    await application.aclose()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_router_worker_startup_timeout_s", 0),
        ("tool_router_worker_startup_timeout_s", float("inf")),
        ("tool_router_worker_inference_timeout_s", float("nan")),
        ("tool_router_worker_inference_timeout_s", -1),
        ("tool_router_worker_lock_timeout_s", 0),
        ("tool_router_worker_shutdown_timeout_s", 0),
        ("tool_router_worker_max_request_bytes", 8 * 1024 * 1024 + 1),
        ("tool_router_worker_max_response_bytes", 8 * 1024 * 1024 + 1),
        ("tool_router_worker_max_stderr_chars", 65_537),
    ],
)
def test_worker_configuration_fails_fast(field: str, value: int | float) -> None:
    with pytest.raises(ValidationError):
        Settings(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1",
            _env_file=None,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("planner_timeout_s", 0),
        ("planner_timeout_s", float("inf")),
        ("planner_timeout_s", float("nan")),
        ("planner_max_tasks", 0),
        ("planner_max_tasks", 101),
        ("planner_max_goal_chars", 0),
        ("planner_max_goal_chars", 16_385),
    ],
)
def test_planner_configuration_fails_fast(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1",
            _env_file=None,
            **{field: value},
        )


def _application_with_close_mock() -> tuple[RuntimeApplication, AsyncMock]:
    embedding_client = MagicMock(spec=ProcessEmbeddingClient)
    close = AsyncMock()
    embedding_client.aclose = close
    application = RuntimeApplication(
        runtime=MagicMock(),
        memory_manager=MagicMock(),
        executor=MagicMock(),
        embedding_client=embedding_client,
    )
    return application, close


@pytest.mark.anyio
async def test_application_repl_closes_embedding_client_on_normal_exit() -> None:
    application, close = _application_with_close_mock()

    with patch("main._repl", new=AsyncMock()) as repl:
        await _run_application_repl(
            application=application,
            console=MagicMock(),
            streamed=[False],
        )

    repl.assert_awaited_once()
    close.assert_awaited_once_with()


@pytest.mark.anyio
async def test_application_repl_closes_embedding_client_on_failure() -> None:
    application, close = _application_with_close_mock()

    with (
        patch("main._repl", new=AsyncMock(side_effect=RuntimeError("repl failed"))),
        pytest.raises(RuntimeError, match="repl failed"),
    ):
        await _run_application_repl(
            application=application,
            console=MagicMock(),
            streamed=[False],
        )

    close.assert_awaited_once_with()


@pytest.mark.anyio
async def test_application_repl_closes_embedding_client_on_cancellation() -> None:
    application, close = _application_with_close_mock()
    started = asyncio.Event()

    async def blocking_repl(**_: object) -> None:
        started.set()
        await asyncio.Event().wait()

    with patch("main._repl", side_effect=blocking_repl):
        task = asyncio.create_task(
            _run_application_repl(
                application=application,
                console=MagicMock(),
                streamed=[False],
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    close.assert_awaited_once_with()
