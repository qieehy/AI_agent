"""D23: Runtime.run_async 异步步进循环契约测试。

策略：
- async 假 callable（async generator 流 / async 非流），sync 槽传毒药函数防串槽
- 真 ToolRegistry + Executor + 真 MemoryManager（temp SQLite）
- 只钉行为：llm.token 事件按序、消息形状、工具照常执行、中途异常 FAILED、
  非流回退（零 token 事件）、接线错误 fail-fast
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from errors import ConfigError, LLMError
from runtime import Runtime
from runtime.state import RunStatus
from tools import Executor, ToolRegistry

from .conftest import make_llm_response as _make_llm_response
from .conftest import make_memory as _make_memory

# ---------- 辅助 ----------

def _chunk(content=None, tool_calls=None, finish_reason=None):
    """构造流式 chunk（StreamChunk 形状，duck typing）。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls, finish_reason=finish_reason)


def _sync_poison(messages, tools):
    raise AssertionError("async 路径不得调用 sync 槽")


def _make_runtime(*, llm_call_async=None, llm_stream_async=None,
                  registry=None, handlers=None, max_steps=100):
    """构造 Runtime（sync 槽全毒药）+ Executor + 真 MemoryManager。"""
    registry = registry if registry is not None else ToolRegistry()
    memory_manager = _make_memory()
    runtime = Runtime(
        llm_call=_sync_poison,
        llm_stream=_sync_poison,
        llm_call_async=llm_call_async,
        llm_stream_async=llm_stream_async,
        tool_executor=Executor(registry, mode="serial"),
        memory_manager=memory_manager,
        handlers=handlers or [],
        max_steps=max_steps,
    )
    return runtime, memory_manager


def _token_events(events):
    return [e for e in events if e.type.value == "llm.token"]


# ---------- 文本流 ----------

@pytest.mark.anyio
async def test_async_streaming_emits_token_events_and_finishes():
    """文本流：每个 content 碎片发一个 llm.token 事件；最终消息进 memory；状态 FINISHED。"""
    events = []

    async def stream(messages, tools):
        yield _chunk(content="你")
        yield _chunk(content="好")
        yield _chunk(finish_reason="stop")

    runtime, memory_manager = _make_runtime(llm_stream_async=stream, handlers=[events.append])
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FINISHED
    assert [e.data["token"] for e in _token_events(events)] == ["你", "好"]
    final_msg = memory_manager.get_or_create(state.session_id).messages[-1]
    assert final_msg["role"] == "assistant"
    assert final_msg["content"] == "你好"
    assert final_msg.get("tool_calls") in (None, [])


# ---------- 工具流 ----------

@pytest.mark.anyio
async def test_async_streaming_tool_call_executes_then_finishes():
    """工具流：重组后的 tool_calls 照常执行（echo），第二轮文本流收尾。"""
    registry = ToolRegistry()

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    events = []
    calls = [0]

    async def stream(messages, tools):
        calls[0] += 1
        if calls[0] == 1:
            yield _chunk(tool_calls=[{"id": "c1", "type": "function",
                                      "function": {"name": "echo", "arguments": '{"msg": "hi"}'}}],
                         finish_reason="tool_calls")
        else:
            yield _chunk(content="完成")
            yield _chunk(finish_reason="stop")

    runtime, memory_manager = _make_runtime(
        llm_stream_async=stream, registry=registry, handlers=[events.append]
    )
    state = await runtime.run_async("start")

    assert state.status == RunStatus.FINISHED
    assert calls[0] == 2
    # 消息序列：[user, assistant(tool_calls), tool, assistant(最终答案)]
    msgs = memory_manager.get_or_create(state.session_id).messages
    tool_round = msgs[1]
    assert tool_round["content"] is None
    assert tool_round["tool_calls"] == [{"id": "c1", "type": "function",
                                         "function": {"name": "echo", "arguments": '{"msg": "hi"}'}}]
    assert any(m.get("role") == "tool" and m.get("name") == "echo" for m in msgs)
    assert msgs[-1]["content"] == "完成"
    # 收尾 chunk 的 token 事件带 tool_calls 数据，供 CLI 显示"调用工具"状态
    tool_status_event = next(e for e in _token_events(events) if e.data.get("tool_calls"))
    assert tool_status_event.data["token"] is None
    assert [tc["function"]["name"] for tc in tool_status_event.data["tool_calls"]] == ["echo"]


# ---------- 异常 ----------

@pytest.mark.anyio
async def test_async_streaming_midstream_error_marks_failed():
    """流中途抛 LLMError → FAILED + error_source='llm'。"""
    async def stream(messages, tools):
        yield _chunk(content="半句")
        raise LLMError("stream broke")

    runtime, _ = _make_runtime(llm_stream_async=stream)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "llm"


@pytest.mark.anyio
async def test_async_tool_failure_becomes_tool_result_and_continues():
    """工具抛异常 → [ERROR ValueError] tool 消息进 memory，循环继续，第二轮收尾 FINISHED。"""
    registry = ToolRegistry()

    @registry.register
    def boom() -> str:
        raise ValueError("kaboom")

    calls = [0]

    async def stream(messages, tools):
        calls[0] += 1
        if calls[0] == 1:
            yield _chunk(tool_calls=[{"id": "c1", "type": "function",
                                      "function": {"name": "boom", "arguments": "{}"}}],
                         finish_reason="tool_calls")
        else:
            yield _chunk(content="收工")
            yield _chunk(finish_reason="stop")

    runtime, memory_manager = _make_runtime(llm_stream_async=stream, registry=registry)
    state = await runtime.run_async("start")

    assert state.status == RunStatus.FINISHED
    msgs = memory_manager.get_or_create(state.session_id).messages
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "[ERROR ValueError] kaboom"
    assert msgs[-1]["content"] == "收工"


# ---------- 非流回退 ----------

@pytest.mark.anyio
async def test_async_without_stream_uses_async_call_path():
    """不传 llm_stream_async：await llm_call_async 走非流路径，零 token 事件。"""
    events = []

    async def llm_async(messages, tools):
        return _make_llm_response("async answer")

    runtime, memory_manager = _make_runtime(llm_call_async=llm_async, handlers=[events.append])
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FINISHED
    assert _token_events(events) == []
    final_msg = memory_manager.get_or_create(state.session_id).messages[-1]
    assert final_msg["content"] == "async answer"


# ---------- 接线错误 ----------

@pytest.mark.anyio
async def test_run_async_without_async_slots_fails_fast():
    """接线错误 = 配置错误：run_async 入口直接抛 ConfigError，而不是吞成 FAILED state。"""
    runtime, _ = _make_runtime()
    with pytest.raises(ConfigError):
        await runtime.run_async("hi")


# ---------- 护栏 ----------

@pytest.mark.anyio
async def test_async_max_steps_guard():
    """max_steps=1 + 工具轮：第二轮 LLM 不发生，状态 MAX_STEP。"""
    registry = ToolRegistry()

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    calls = [0]

    async def stream(messages, tools):
        calls[0] += 1
        yield _chunk(tool_calls=[{"id": "c1", "type": "function",
                                  "function": {"name": "echo", "arguments": '{"msg": "hi"}'}}],
                     finish_reason="tool_calls")

    runtime, _ = _make_runtime(llm_stream_async=stream, registry=registry, max_steps=1)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.MAX_STEP
    assert calls[0] == 1
