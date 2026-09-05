from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from rich.markdown import Markdown

from config.settings import Settings
from main import (
    PROJECT_ROOT,
    RuntimeApplication,
    _embedding_worker_environment,
    _repl,
    _run_application_repl,
    build_runtime,
)
from rag.process_embeddings import ProcessEmbeddingClient
from runtime import RunStatus, RuntimeState, StopReason
from runtime.reflection import Critic
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
        critic_timeout_s=17.0,
        critic_max_feedback_chars=400,
        reflection_revision_rounds=3,
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
    assert runtime_kwargs["critic"] is None
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


def test_build_runtime_wires_critic_only_for_reflection_pattern() -> None:
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
        build_runtime(pattern="reflection")

    runtime_kwargs = runtime_type.call_args.kwargs
    critic = runtime_kwargs["critic"]
    assert isinstance(critic, Critic)
    assert critic._llm_call is llm
    assert critic._timeout_s == 17.0
    assert critic._max_feedback_chars == 400
    assert runtime_kwargs["planner"] is None
    assert runtime_kwargs["loop_guard"].policy.reflection_revision_rounds == 3


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


def test_reflection_configuration_has_bounded_defaults() -> None:
    settings = Settings(
        api_key="test-key",
        model="test-model",
        base_url="https://example.invalid/v1",
        _env_file=None,
    )

    assert settings.critic_timeout_s == 30.0
    assert settings.critic_max_feedback_chars == 2000
    assert settings.reflection_revision_rounds == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("critic_timeout_s", 0),
        ("critic_timeout_s", float("inf")),
        ("critic_timeout_s", float("nan")),
        ("critic_max_feedback_chars", 0),
        ("critic_max_feedback_chars", 16_385),
        ("reflection_revision_rounds", -1),
        ("reflection_revision_rounds", 11),
    ],
)
def test_reflection_configuration_fails_fast(
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


def test_critic_feedback_limit_cannot_exceed_router_query_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1",
            tool_router_max_query_chars=100,
            critic_max_feedback_chars=101,
            _env_file=None,
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


@pytest.mark.anyio
async def test_repl_does_not_render_memory_as_answer_for_non_success_terminal() -> None:
    state = RuntimeState(
        session_id="reflection-limit-session",
        status=RunStatus.REFLECTION_LIMIT,
        stop_reason=StopReason.REFLECTION_LIMIT,
    )
    runtime = MagicMock()
    runtime.run_async = AsyncMock(return_value=state)
    memory_manager = MagicMock()
    console = MagicMock()
    console.input.side_effect = ["question", "/q"]

    await _repl(
        runtime=runtime,
        memory_manager=memory_manager,
        executor=MagicMock(),
        console=console,
        streamed=[False],
    )

    memory_manager.get_or_create.assert_not_called()
    assert any(
        "reflection_limit" in str(call.args[0])
        for call in console.print.call_args_list
        if call.args
    )


@pytest.mark.anyio
async def test_repl_renders_accepted_buffered_reflection_answer_once() -> None:
    state = RuntimeState(
        session_id="accepted-reflection-session",
        status=RunStatus.FINISHED,
        stop_reason=StopReason.FINISH_NORMAL,
    )
    runtime = MagicMock()
    runtime.run_async = AsyncMock(return_value=state)
    memory_manager = MagicMock()
    memory_manager.get_or_create.return_value = SimpleNamespace(
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "approved answer"},
        ]
    )
    console = MagicMock()
    console.input.side_effect = ["question", "/q"]

    await _repl(
        runtime=runtime,
        memory_manager=memory_manager,
        executor=MagicMock(),
        console=console,
        streamed=[False],
    )

    memory_manager.get_or_create.assert_called_once_with("accepted-reflection-session")
    console.print.assert_called_once()
    rendered = console.print.call_args.args[0]
    assert isinstance(rendered, Markdown)
    assert rendered.markup == "approved answer"
