import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """tool_call执行器
     职责：
      - 决定怎么跑（serial / parallel）
      - 把 ToolManager.execute 的 raise 转成 ToolResult
      - 永远不抛异常（除非程序错误，如 TypeError）
    """
    def __init__(self, registry: ToolRegistry, mode: ExecutionMode="parallel", max_workers: int = 4):
        self._registry = registry
        self._mode = mode
        self._max_workers = max_workers

    def get_schemas(self):
         return self._registry.get_schemas()


    def execute_calls(self, tool_calls: list[Any]) -> list[ToolResult]:
        if not tool_calls:
            return []
        if self._mode == "serial":
            return self._execute_serial(tool_calls)
        return self._execute_parallel(tool_calls)


    async def execute_calls_async(self, tool_calls: list[Any]) -> list[ToolResult]:
        """async 路径走默认线程池，max_workers 仅约束 sync parallel"""
        if not tool_calls:
            return []

        if self._mode == "serial":
            results = []
            for tc in tool_calls:
                result = await asyncio.to_thread(self._run_one, tc)
                results.append(result)
            return results

        return list(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(self._run_one, tc)
                        for tc in tool_calls
                    )
                    )
                )


    def _execute_serial(self, tool_calls: list[Any]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tc in tool_calls:
            results.append(self._run_one(tc))
        return results


    def _execute_parallel(self, tool_calls: list[Any]) -> list[ToolResult]:
        indexed: list[tuple[int, ToolResult]] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._run_one, tc): i for i, tc in enumerate(tool_calls)}
            for fut in as_completed(futures):
                idx = futures[fut]
                res = fut.result()
                indexed.append((idx, res))
        indexed.sort(key = lambda x: x[0])
        return [r for _ , r in indexed]


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
