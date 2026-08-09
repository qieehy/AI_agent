from rich.console import Console
from rich.markdown import Markdown
from typer import Typer

from llm import LLMClient
from memory import create_memory_manager
from observability import logger, setup_logging
from runtime import Runtime
from tools import Executor, create_registry

app = Typer()
@app.command()
def chat(model: str | None = None):
    setup_logging()
    llm = LLMClient(model = model) if model else LLMClient()
    registry = create_registry()
    memory_manager = create_memory_manager()
    executor = Executor(registry, mode="parallel")
    runtime = Runtime(llm_call=llm, tool_executor=executor, memory_manager=memory_manager)
    session_id = None

    console = Console()
    while True:
        q = input(">")

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

        state = runtime.run(q, session_id=session_id)
        session_id = state.session_id
        if state.status.value == "failed":
            err = state.error_info or {}
            logger.error(f"[{state.error_source}] {err.get('message', '?')}")
        else:
            last = memory_manager.get_or_create(session_id).messages[-1]
            console.print(Markdown(last.get("content", "[no content]") if last else "[no content]"))



if __name__ == "__main__":
    app()
