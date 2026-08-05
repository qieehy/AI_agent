"""测试 runtime/runtime.py 的错误隔离 + D3 集成。

D3 改动：
- Runtime 构造接收 tool_executor: Executor（不是 tool_call: Callable）
- Executor.execute_calls 返回 list[ToolResult]
- 失败时 Executor 装成 ToolResult（不抛），Runtime 看到 failed_count > 0 才 raise
- RuntimeState.error_info 形状统一

策略：
- 直接传假 llm_call / Executor
- Executor 用真 ToolRegistry + 注册假 tool
- 验证 state.error_source / state.error_info / state.status
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from runtime import Runtime
from runtime.state import RunStatus
from errors import LLMError, ToolError, AgentError
from tools import ToolRegistry, Executor
from memory import MemoryManager

from .conftest import make_memory as _make_memory, make_llm_response as _make_llm_response
from .conftest import make_tool_call as _make_tool_call


# ---------- 辅助函数 ----------

def _make_runtime(llm_call, registry: ToolRegistry | None = None, mode: str = "parallel",
                  memory: MemoryManager | None = None) -> Runtime:
    """构造 Runtime + Executor 的便捷 helper。"""
    registry = registry if registry is not None else ToolRegistry()
    executor = Executor(registry, mode=mode)
    memory = memory if memory is not None else _make_memory()
    return Runtime(llm_call=llm_call, tool_executor=executor, memory=memory)


# ---------- 测试：正常路径 ----------

def test_runtime_runs_to_finished():
    """正常路径：LLM 直接回答案 → state.status == FINISHED。"""
    def llm(messages, tools):
        return _make_llm_response("hello back")

    runtime = _make_runtime(llm)
    state = runtime.run("hi")

    assert state.status == RunStatus.FINISHED
    assert state.error_source is None
    assert state.error is None


# ---------- 测试：LLM 失败 ----------

def test_runtime_marks_llm_error_source():
    """LLM 抛 LLMError → state.error_source == 'llm' + state.status == FAILED。"""
    def bad_llm(messages, tools):
        raise LLMError("LLM timeout", context={"model": "gpt-4"})

    runtime = _make_runtime(bad_llm)
    state = runtime.run("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "llm"
    # error_info 来自 e.to_dict()
    assert state.error_info["type"] == "LLMError"
    assert state.error_info["message"] == "LLM timeout"
    assert state.error_info["context"] == {"model": "gpt-4"}


# ---------- 测试：Tool 失败 ----------

def test_runtime_marks_tool_error_when_tool_not_found():
    """LLM 调不存在的 tool → Executor 返回 failed → Runtime 标 FAILED + tool。"""
    def llm(messages, tools):
        return _make_llm_response(
            content="",
            tool_calls=[_make_tool_call("call_1", "bad_tool", "{}")],
        )

    # registry 空，"bad_tool" 不存在
    runtime = _make_runtime(llm)
    state = runtime.run("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "tool"
    # error_info 是 D3 重新构造的 dict（不再来自 e.to_dict()）
    assert state.error_info["type"] == "ToolError"
    assert "1/1 tool(s) failed" in state.error_info["message"]


def test_runtime_succeeds_after_tool_call():
    """正常 Tool 调用：tool 成功 + LLM 收尾 → state.status == FINISHED。"""
    registry = ToolRegistry()

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    call_count = [0]

    def llm(messages, tools):
        call_count[0] += 1
        # 第一次：返回 tool_call
        # 第二次：返回最终答案
        if call_count[0] == 1:
            return _make_llm_response(
                content="",
                tool_calls=[_make_tool_call("c1", "echo", '{"msg": "hi"}')],
            )
        return _make_llm_response("done")

    runtime = _make_runtime(llm, registry=registry)
    state = runtime.run("start")

    assert state.status == RunStatus.FINISHED
    assert state.error_source is None
    # 验证 tool_call 真的被调了
    assert call_count[0] == 2


# ---------- 测试：未知异常逃逸 ----------

def test_runtime_lets_non_agent_error_escape():
    """非 AgentError（如 KeyError）必须逃逸，不能被 runtime 吞掉。"""
    def bad_llm(messages, tools):
        raise KeyError("oops, not an AgentError")

    runtime = _make_runtime(bad_llm)

    with pytest.raises(KeyError) as excinfo:
        runtime.run("hi")
    assert "oops" in str(excinfo.value)


# ---------- 测试：事件流 ----------

def test_runtime_emits_run_start_and_finish():
    """正常路径：handler 应该收到 RUN_START + RUN_FINISH 事件。"""
    events_received = []

    def handler(event):
        events_received.append(event)

    def llm(messages, tools):
        return _make_llm_response("hi")

    runtime = _make_runtime(llm)
    # 重建 runtime with handlers
    registry = ToolRegistry()
    executor = Executor(registry, mode="parallel")
    runtime = Runtime(llm_call=llm, tool_executor=executor, handlers=[handler], memory=_make_memory())
    runtime.run("user input")

    types = [e.type.value for e in events_received]
    # 至少应该看到这些事件
    assert "run.start" in types
    assert "llm.request" in types
    assert "llm.response" in types
    assert "run.finish" in types


def test_runtime_emits_run_error_on_failure():
    """LLM 失败：handler 应该收到 RUN_ERROR 事件 + data.error_source == 'llm'。"""
    events_received = []

    def handler(event):
        events_received.append(event)

    def bad_llm(messages, tools):
        raise LLMError("boom")

    registry = ToolRegistry()
    executor = Executor(registry, mode="parallel")
    runtime = Runtime(llm_call=bad_llm, tool_executor=executor, handlers=[handler], memory=_make_memory())
    runtime.run("hi")

    error_events = [e for e in events_received if e.type.value == "run.error"]
    assert len(error_events) >= 1
    # 最终 emit 的 RUN_ERROR 事件应带 error_source
    final = error_events[-1]
    assert final.data.get("error_source") == "llm"
