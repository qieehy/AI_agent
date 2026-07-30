"""测试 runtime/runtime.py 的错误隔离。

策略：直接传假 llm_call / tool_call 函数（不用 mock），
验证 state.error_source / state.error_info 是否被正确设置。

注意：
- 假 LLM 响应必须让 message.model_dump 是真函数（不是 MagicMock），
  否则 hasattr 检查会失败。
- "未知异常逃逸" 测试用 pytest.raises 包 runtime.run，
  验证非 AgentError 不会被 runtime 吞掉。
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from runtime import Runtime
from runtime.state import RunStatus
from errors import LLMError, ToolError, AgentError


# ---------- 辅助函数 ----------

def _make_llm_response(content: str = "ok", tool_calls: list | None = None):
    """构造假的 LLM 响应对象（OpenAI 风格）。

    关键：model_dump 必须是真 lambda，不是 MagicMock。
    runtime 会 hasattr(message, "model_dump") 判断。
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls or []
    response.choices[0].message.model_dump = lambda: {"role": "assistant", "content": content}
    return response


def _make_tool_call(call_id: str, name: str, arguments: str):
    """构造假的 tool_call 对象。"""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------- 测试：正常路径 ----------

def test_runtime_runs_to_finished():
    """正常路径：LLM 一次性返回答案，state.status == FINISHED。"""
    def llm(messages, tools):
        return _make_llm_response("hello back")

    def tool(name, args):
        return "unused"

    runtime = Runtime(llm_call=llm, tool_call=tool)
    state = runtime.run("hi")

    assert state.status == RunStatus.FINISHED
    assert state.error_source is None
    assert state.error is None
    # 用户消息 + assistant 回复 = 2 条
    assert len(state.messages) == 2


# ---------- 测试：LLM 失败 ----------

def test_runtime_marks_llm_error_source():
    """LLM 抛 LLMError → state.error_source == 'llm' + state.status == FAILED。"""
    def bad_llm(messages, tools):
        raise LLMError("LLM timeout", context={"model": "gpt-4"})

    runtime = Runtime(llm_call=bad_llm, tool_call=lambda n, a: "x")
    state = runtime.run("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "llm"
    # error_info 来自 e.to_dict()
    assert state.error_info["type"] == "LLMError"
    assert state.error_info["message"] == "LLM timeout"
    assert state.error_info["context"] == {"model": "gpt-4"}


# ---------- 测试：Tool 失败 ----------

def test_runtime_marks_tool_error_source():
    """Tool 抛 ToolError → state.error_source == 'tool'。"""
    def llm(messages, tools):
        # LLM 返回带 tool_calls 的响应
        return _make_llm_response(
            content="",
            tool_calls=[_make_tool_call("call_1", "bad_tool", "{}")],
        )

    def bad_tool(name, args):
        raise ToolError("tool not found", context={"tool": name})

    runtime = Runtime(llm_call=llm, tool_call=bad_tool)
    state = runtime.run("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "tool"
    assert state.error_info["type"] == "ToolError"
    assert state.error_info["message"] == "tool not found"


def test_runtime_succeeds_after_tool_call():
    """正常 Tool 调用：tool 成功 + LLM 收尾 → state.status == FINISHED。"""
    def llm_responses(messages, tools):
        # 第一次：返回 tool_call
        # 第二次：返回最终答案
        if not any(m.get("role") == "tool" for m in messages):
            return _make_llm_response(
                content="",
                tool_calls=[_make_tool_call("c1", "echo", '{"msg": "hi"}')],
            )
        return _make_llm_response("done")

    def echo_tool(name, args):
        import json
        return args  # 原样返回 args（已是 dict）

    runtime = Runtime(llm_call=llm_responses, tool_call=echo_tool)
    state = runtime.run("start")

    assert state.status == RunStatus.FINISHED
    assert state.error_source is None


# ---------- 测试：未知异常逃逸 ----------

def test_runtime_lets_non_agent_error_escape():
    """非 AgentError（如 KeyError）必须逃逸，不能被 runtime 吞掉。"""
    def bad_llm(messages, tools):
        raise KeyError("oops, not an AgentError")

    runtime = Runtime(llm_call=bad_llm, tool_call=lambda n, a: "x")

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

    runtime = Runtime(
        llm_call=llm,
        tool_call=lambda n, a: "x",
        handlers=[handler],
    )
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

    runtime = Runtime(
        llm_call=bad_llm,
        tool_call=lambda n, a: "x",
        handlers=[handler],
    )
    runtime.run("hi")

    error_events = [e for e in events_received if e.type.value == "run.error"]
    assert len(error_events) >= 1
    # 最终 emit 的 RUN_ERROR 事件应带 error_source
    final = error_events[-1]
    assert final.data.get("error_source") == "llm"
