import asyncio
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import cast

from rich.console import Console
from rich.markdown import Markdown
from typer import Typer

from config import get_settings
from llm import AsyncLLMClient
from memory import MemoryManager, create_memory_manager
from observability import logger, setup_logging
from prompts import create_agent_profile
from prompts.base import PromptContext
from prompts.factory import PatternName
from rag.process_embeddings import ProcessEmbeddingClient
from runtime import Event, EventHandler, EventType, LoopGuard, Planner, RunStatus, Runtime
from runtime.reflection import Critic
from tools import EmbeddingRouter, Executor, ToolCallValidator, create_registry

app = Typer()

PROJECT_ROOT = Path(__file__).resolve().parent
_WORKER_ENV_ALLOWLIST = (
    "APPDATA",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_TOKEN",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PYTHONPATH",
    "REQUESTS_CA_BUNDLE",
    "SENTENCE_TRANSFORMERS_HOME",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
    "USERPROFILE",
    "VIRTUAL_ENV",
    "WINDIR",
)


@dataclass(frozen=True, slots=True)
class RuntimeApplication:
    """Own the Runtime and every async resource created by the composition root."""

    runtime: Runtime
    memory_manager: MemoryManager
    executor: Executor
    embedding_client: ProcessEmbeddingClient

    async def aclose(self) -> None:
        await self.embedding_client.aclose()

    async def __aenter__(self) -> "RuntimeApplication":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _embedding_worker_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source_environment = os.environ if source is None else source
    environment = {
        name: value
        for name in _WORKER_ENV_ALLOWLIST
        if (value := source_environment.get(name))
    }
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def build_runtime(
    *,
    model: str | None = None,
    pattern: str | None = None,
    handlers: list[EventHandler] | None = None,
) -> RuntimeApplication:
    """Build one process-local application and its owned embedding worker client."""
    settings = get_settings()
    async_llm = AsyncLLMClient(model=model) if model else AsyncLLMClient()
    registry = create_registry()
    memory_manager = create_memory_manager()
    executor = Executor(registry, mode="parallel")
    embedding_client = ProcessEmbeddingClient(
        command=(
            sys.executable,
            "-m",
            "rag.embedding_worker",
            "--model-name",
            settings.tool_router_model,
            "--model-version",
            settings.tool_router_model,
        ),
        cwd=PROJECT_ROOT,
        environment=_embedding_worker_environment(),
        expected_model_version=settings.tool_router_model,
        startup_timeout_s=settings.tool_router_worker_startup_timeout_s,
        inference_timeout_s=settings.tool_router_worker_inference_timeout_s,
        lock_timeout_s=settings.tool_router_worker_lock_timeout_s,
        shutdown_timeout_s=settings.tool_router_worker_shutdown_timeout_s,
        max_request_bytes=settings.tool_router_worker_max_request_bytes,
        max_response_bytes=settings.tool_router_worker_max_response_bytes,
        max_stderr_chars=settings.tool_router_worker_max_stderr_chars,
    )
    tool_router = EmbeddingRouter(
        embedding_client,
        model_version=settings.tool_router_model,
        threshold=settings.tool_router_threshold,
        top_k=settings.tool_router_top_k,
        cache_size=settings.tool_router_cache_size,
        max_query_chars=settings.tool_router_max_query_chars,
    )

    pattern = cast(PatternName, pattern or settings.pattern)
    profile = create_agent_profile(pattern=pattern)
    planner = None
    critic = None
    loop_policy = profile.loop_policy
    if pattern == "plan_execute":
        planner = Planner(
            async_llm,
            timeout_s=settings.planner_timeout_s,
            max_tasks=settings.planner_max_tasks,
            max_goal_chars=settings.planner_max_goal_chars,
        )
    if pattern == "reflection":
        critic = Critic(
            async_llm,
            timeout_s=settings.critic_timeout_s,
            max_feedback_chars=settings.critic_max_feedback_chars,
        )
        loop_policy = replace(
            profile.loop_policy,
            reflection_revision_rounds=settings.reflection_revision_rounds,
        )
    system_message = profile.pattern.build(PromptContext())
    runtime = Runtime(
        llm_call_async=async_llm,
        llm_stream_async=async_llm.stream,
        tool_executor=executor,
        memory_manager=memory_manager,
        handlers=handlers,
        loop_guard=LoopGuard(loop_policy),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=tool_router,
        system_prompt=system_message.content,
        planner=planner,
        critic=critic,
    )
    return RuntimeApplication(
        runtime=runtime,
        memory_manager=memory_manager,
        executor=executor,
        embedding_client=embedding_client,
    )


@app.command()
def chat(model: str | None = None, pattern: str | None = None):
    """CLI 聊天。--pattern 覆盖 .env 的 settings.pattern（D24: Prompt 可配置）。"""
    setup_logging()
    console = Console()
    streamed = [False]

    def on_llm_event(event: Event) -> None:
        """LLM_TOKEN 事件 → 实时打字机：文本逐字打印，工具轮显示 dim 状态。"""
        if event.type != EventType.LLM_TOKEN:
            return
        token = event.data.get("token")
        if token:
            if not streamed[0]:
                console.print("[cyan]Assistant[/cyan]", end=" ")
                streamed[0] = True
            console.print(token, end="")
        elif event.data.get("tool_calls"):
            names = [tc["function"]["name"] for tc in event.data["tool_calls"]]
            console.print(f"\n[dim]🔧 调用工具: {', '.join(names)}[/dim]")

    application = build_runtime(
        model=model,
        pattern=pattern,
        handlers=[on_llm_event],
    )
    asyncio.run(
        _run_application_repl(
            application=application,
            console=console,
            streamed=streamed,
        )
    )


async def _run_application_repl(
    *,
    application: RuntimeApplication,
    console: Console,
    streamed: list[bool],
) -> None:
    async with application:
        await _repl(
            runtime=application.runtime,
            memory_manager=application.memory_manager,
            executor=application.executor,
            console=console,
            streamed=streamed,
        )


async def _repl(runtime, memory_manager, executor, console, streamed)-> None:
    session_id = None
    while True:
        q = await asyncio.to_thread(console.input, ">")

        if q == "/q":
            break
        elif q == "/clear":
            session_id = None
            console.print("[dim]会话已重置[/dim]")
            continue
        elif q == "/history":
            msgs = memory_manager.get_or_create(session_id=session_id).messages
            for m in msgs:
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:120]
                color = {"user": "green", "assistant": "cyan", "tool": "yellow"}.get(role, "dim")
                console.print(f"  [{color}]{role:>10}[/{color}]  {content}")
            if not msgs:
                console.print("[dim]（无历史）[/dim]")
            continue
        elif q == "/tools":
            schemas = executor.get_schemas()
            for s in schemas:
                func = s["function"]
                console.print(f"  [bold]{func['name']}[/bold] — [dim]{func.get('description', '')}[/dim]")
            if not schemas:
                console.print("[dim]（无可用工具）[/dim]")
            continue

        streamed[0] = False
        state = await runtime.run_async(q, session_id=session_id)
        session_id = state.session_id
        if state.status is not RunStatus.FINISHED:
            error_info = state.error_info or {}
            message = error_info.get("message")

            logger.error(
                f"Run did not finish successfully | "
                f"status={state.status.value} | "
                f"source={state.error_source} | "
                f"message={message or '?'}"
            )

            detail = f": {message}" if isinstance(message, str) and message else ""
            console.print(f"[red]Run ended: {state.status.value}{detail}[/red]")
        elif streamed[0]:
            console.print()  # 流式：内容已实时打印，只补换行
        else:
            last = memory_manager.get_or_create(session_id).messages[-1]
            console.print(Markdown(last.get("content", "[no content]") if last else "[no content]"))

if __name__ == "__main__":
    app()
