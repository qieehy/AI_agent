"""D23: Executor.execute_calls_async 契约测试。

策略：
- 工具函数保持同步（time.sleep 模拟阻塞型工具），async 路径靠 asyncio.to_thread 卸货
- 钉契约：空列表、成功、并行（总时长上界）、串行（总时长下界）、顺序保持、
  部分失败不抛、未注册/坏 JSON 翻译、不阻塞事件循环（事件顺序钉，不用计时）
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from tools.executor import Executor
from tools.registry import ToolRegistry

# ---------- 辅助函数 ----------

def _make_tool_call(name: str, arguments: str) -> MagicMock:
    """构造假的 tool_call 对象（OpenAI 风格）。"""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------- 基础契约 ----------

@pytest.mark.anyio
async def test_async_executor_empty_list():
    """空 tool_calls → 空结果。"""
    executor = Executor(ToolRegistry())
    assert await executor.execute_calls_async([]) == []


@pytest.mark.anyio
async def test_async_executor_success():
    """正常路径：注册 + 调 + 成功 → status="success" + content 序列化。"""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    results = await Executor(registry).execute_calls_async([_make_tool_call("add", '{"a": 1, "b": 2}')])

    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].content == "3"   # 非 str 返回值序列化
    assert results[0].error is None


@pytest.mark.anyio
async def test_async_executor_not_registered():
    """tool 不存在 → ToolResult failed，不抛异常。"""
    executor = Executor(ToolRegistry())
    results = await executor.execute_calls_async([_make_tool_call("not_exist", "{}")])

    assert results[0].status == "failed"
    assert "not registered" in results[0].error.lower()
    assert results[0].error_type == "ToolError"


@pytest.mark.anyio
async def test_async_executor_invalid_json():
    """JSON 解析失败 → ToolResult failed。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    results = await Executor(registry).execute_calls_async([_make_tool_call("add", "{bad json")])

    assert results[0].status == "failed"
    assert "json" in results[0].error.lower()
    assert results[0].error_type == "ToolError"


# ---------- 并发语义 ----------

@pytest.mark.anyio
async def test_async_executor_parallel_is_concurrent():
    """parallel 模式：3 个 0.3s 阻塞工具总时长 < 0.7s（串行要 0.9s，上界留足 margin）。"""
    registry = ToolRegistry()

    for name in ("slow_a", "slow_b", "slow_c"):
        def _slow() -> str:
            time.sleep(0.3)
            return "done"

        _slow.__name__ = name
        registry.register(_slow)

    tcs = [_make_tool_call(name, "{}") for name in ("slow_a", "slow_b", "slow_c")]
    t0 = time.perf_counter()
    results = await Executor(registry, mode="parallel").execute_calls_async(tcs)
    elapsed = time.perf_counter() - t0

    assert len(results) == 3
    assert all(r.status == "success" for r in results)
    assert elapsed < 0.7


@pytest.mark.anyio
async def test_async_executor_serial_is_sequential():
    """serial 模式：3 个 0.2s 工具总时长 >= 0.5s（time.sleep 不会提前返回，下界安全）。"""
    registry = ToolRegistry()

    for name in ("slow_a", "slow_b", "slow_c"):
        def _slow() -> str:
            time.sleep(0.2)
            return "done"

        _slow.__name__ = name
        registry.register(_slow)

    tcs = [_make_tool_call(name, "{}") for name in ("slow_a", "slow_b", "slow_c")]
    t0 = time.perf_counter()
    results = await Executor(registry, mode="serial").execute_calls_async(tcs)
    elapsed = time.perf_counter() - t0

    assert len(results) == 3
    assert all(r.status == "success" for r in results)
    assert elapsed >= 0.5


@pytest.mark.anyio
async def test_async_executor_preserves_input_order():
    """gather 返回按输入顺序：results[i].tool_call is tool_calls[i]，与完成先后无关。"""
    registry = ToolRegistry()

    @registry.register
    def echo(value: str) -> str:
        return value

    tcs = [
        _make_tool_call("echo", '{"value": "one"}'),
        _make_tool_call("echo", '{"value": "two"}'),
        _make_tool_call("echo", '{"value": "three"}'),
    ]
    results = await Executor(registry).execute_calls_async(tcs)

    assert [r.content for r in results] == ["one", "two", "three"]
    assert [r.tool_call for r in results] == tcs


@pytest.mark.anyio
async def test_async_executor_partial_failure_no_raise():
    """一个工具抛异常 + 两个正常 → 3 个结果都在，失败者翻译，整体不抛。"""
    registry = ToolRegistry()

    @registry.register
    def boom() -> str:
        raise ValueError("kaboom")

    @registry.register
    def ok() -> str:
        return "fine"

    tcs = [
        _make_tool_call("ok", "{}"),
        _make_tool_call("boom", "{}"),
        _make_tool_call("ok", "{}"),
    ]
    results = await Executor(registry).execute_calls_async(tcs)

    assert len(results) == 3
    assert [r.status for r in results] == ["success", "failed", "success"]
    assert results[1].error == "kaboom"
    assert results[1].error_type == "ValueError"


# ---------- 事件循环不阻塞 ----------

@pytest.mark.anyio
async def test_async_executor_does_not_block_event_loop():
    """阻塞型工具跑在工作线程：工具执行期间，同循环的其他任务照常推进。

    用事件顺序钉，不用计时（确定性）：tick 睡 0.05s，慢工具睡 0.3s。
    若实现堵了循环（如 async 函数里直接调 sync 工具），tick 必然后于
    tool-done，事件顺序反转，测试必红。
    """
    registry = ToolRegistry()

    @registry.register
    def slow() -> str:
        time.sleep(0.3)
        return "done"

    events: list[str] = []

    async def run_tools():
        results = await Executor(registry).execute_calls_async([_make_tool_call("slow", "{}")])
        events.append("tool-done")
        return results

    async def tick():
        await asyncio.sleep(0.05)
        events.append("tick")

    results, _ = await asyncio.gather(run_tools(), tick())

    assert results[0].status == "success"
    assert events == ["tick", "tool-done"]
