from llm import LLMClient
from tools import Executor, create_registry
from runtime import Runtime
from observability import setup_logging, logger


def main():
    setup_logging()
    llm = LLMClient()
    registry = create_registry()
    executor = Executor(registry, mode="parallel")
    runtime = Runtime(llm_call=llm, tool_executor=executor)

    while True:
        q = input(">")
        if q == "exit":
            break
        state = runtime.run(q)
        if state.status.value == "failed":
            err = state.error_info or {}
            logger.error(f"[{state.error_source}] {err.get('message', '?')}")
        else:
            last = state.messages[-1] if state.messages else {}
            print(last.get("content", "[no content]"))

if __name__ == "__main__":
    main()