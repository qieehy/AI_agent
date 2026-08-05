
from llm import LLMClient
from memory import create_memory_manager
from observability import logger, setup_logging
from runtime import Runtime
from tools import Executor, create_registry


def main():
    setup_logging()
    llm = LLMClient()
    registry = create_registry()
    memory = create_memory_manager()
    executor = Executor(registry, mode="parallel")
    runtime = Runtime(llm_call=llm, tool_executor=executor, memory=memory)
    session_id = None

    while True:
        q = input(">")
        if q == "/q":
            break
        state = runtime.run(q, session_id=session_id)
        session_id = state.session_id
        if state.status.value == "failed":
            err = state.error_info or {}
            logger.error(f"[{state.error_source}] {err.get('message', '?')}")
        else:
            last = runtime.last_message
            print(last.get("content", "[no content]") if last else "[no content]")

if __name__ == "__main__":
    main()
