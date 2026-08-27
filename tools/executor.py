import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from tools.registry import ToolRegistry

ExecutionMode = Literal["serial", "parallel"]

@dataclass
class ToolResult:
    tool_call: Any
    status: Literal["success", "failed"]
    content: Any | None = None
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = 0.0

class Executor:
    """tool_call执行器（D24 起仅异步执行路径）
     职责：
      - 决定怎么跑（serial / parallel）
      - 把工具执行中的 raise 转成 ToolResult
      - 永远不抛异常（除非程序错误，如 TypeError）
    """
    def __init__(self, registry: ToolRegistry, mode: ExecutionMode="parallel", max_workers: int = 4):
        self._registry = registry
        self._mode = mode
        self._max_workers = max_workers

    def get_schemas(self):
         return self._registry.get_schemas()


    async def execute_calls_async(self, tool_calls: list[Any]) -> list[ToolResult]:
        """阻塞型工具经 asyncio.to_thread 卸货到工作线程，不堵事件循环。

        parallel 模式用每次调用独立的 Semaphore(max_workers) 限流：
        Executor 实例跨会话共享，实例级信号量会把不同会话的工具执行
        互相串起来；每次调用独立信号量只约束单批内的并发度。
        """
        if not tool_calls:
            return []

        if self._mode == "serial":
            return [await asyncio.to_thread(self._run_one, tc) for tc in tool_calls]

        sem = asyncio.Semaphore(self._max_workers)

        async def _bounded(tc):
            async with sem:
                return await asyncio.to_thread(self._run_one, tc)

        return list(await asyncio.gather(*(_bounded(tc) for tc in tool_calls)))


    def _run_one(self, tool_call: Any) -> ToolResult:

        t0 = time.perf_counter()
        name = tool_call.function.name
        tool = self._registry.get_tool(name)

        if tool is None:
            return ToolResult(
                tool_call=tool_call,
                status="failed",
                error=f"tool {name} is not registered or disabled",
                error_type="ToolError",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_call=tool_call,
                status="failed",
                error=f"invalid JSON arguments for tool {name}: {e.msg}",
                error_type="ToolError",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            result = tool.func(**args)
            content = Executor.serialize_tool_result(result)
            return ToolResult(
                tool_call = tool_call,
                status = "success",
                content = content,
                duration_ms = (time.perf_counter() - t0) * 1000,
            )

        except Exception as e:
            return ToolResult(
                tool_call = tool_call,
                status = "failed",
                error = str(e),
                error_type = type(e).__name__,
                duration_ms = (time.perf_counter() - t0) * 1000,
            )

    @staticmethod
    def serialize_tool_result(content: Any) -> str:
        if isinstance(content, str):
            return content

        return json.dumps(
            content,
            ensure_ascii=False,
            default=str,
        )
