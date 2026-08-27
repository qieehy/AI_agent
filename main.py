import asyncio

from rich.console import Console
from rich.markdown import Markdown
from typer import Typer

from config import get_settings
from llm import AsyncLLMClient
from memory import create_memory_manager
from observability import logger, setup_logging
from prompts import create_agent_profile
from prompts.base import PromptContext
from runtime import Event, EventType, LoopGuard, Runtime
from tools import Executor, ToolCallValidator, create_registry

app = Typer()
@app.command()
def chat(model: str | None = None, pattern: str | None = None):
    """CLI 聊天。--pattern 覆盖 .env 的 settings.pattern（D24: Prompt 可配置）。"""
    setup_logging()
    async_llm = AsyncLLMClient(model = model) if model else AsyncLLMClient()
    registry = create_registry()
    memory_manager = create_memory_manager()
    executor = Executor(registry, mode="parallel")

    # 组合根：AgentProfile = 提示模式 + 循环策略，一次工厂调用全部就位
    profile = create_agent_profile(pattern or get_settings().pattern)
    system_message = profile.pattern.build(
        PromptContext(tool_schemas=tuple(executor.get_schemas()))
    )
    loop_guard = LoopGuard(profile.loop_policy)
    validator = ToolCallValidator(executor.get_schemas())

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

    runtime = Runtime(
        llm_call_async=async_llm,
        llm_stream_async=async_llm.stream,
        tool_executor=executor,
        memory_manager=memory_manager,
        handlers=[on_llm_event],
        loop_guard=loop_guard,
        validator=validator,
        system_prompt=system_message.content,
    )
    asyncio.run(_repl(runtime=runtime, memory_manager=memory_manager, executor=executor, console=console, streamed=streamed))


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
        if state.status.value == "failed":
            err = state.error_info or {}
            logger.error(f"[{state.error_source}] {err.get('message', '?')}")
        elif streamed[0]:
            console.print()  # 流式：内容已实时打印，只补换行
        else:
            last = memory_manager.get_or_create(session_id).messages[-1]
            console.print(Markdown(last.get("content", "[no content]") if last else "[no content]"))

if __name__ == "__main__":
    app()
