"""D22: Runtime 流式接线契约测试（test-first：llm_stream 参数落地前为红）。

策略：
- llm_stream 传脚本化假生成器（chunk 是 SimpleNamespace，只钉属性契约，不 import StreamChunk）
- 真 ToolRegistry + Executor + 真 MemoryManager（temp SQLite）
- 只钉行为：llm.token 事件按序、消息形状与 OpenAI 格式一致、工具照常执行、
  中途异常标 FAILED、不传 llm_stream 走老路径（不发 token 事件）
"""
from __future__ import annotations

from types import SimpleNamespace

from errors import LLMError
from runtime import Runtime
from runtime.state import RunStatus
from tools import Executor, ToolRegistry

from .conftest import make_llm_response as _make_llm_response
from .conftest import make_memory as _make_memory

# ---------- 辅助 ----------

def _chunk(content=None, tool_calls=None, finish_reason=None):
    """构造流式 chunk（Step 1 的 StreamChunk 形状，duck typing）。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls, finish_reason=finish_reason)


def _make_runtime(llm_call, llm_stream=None, registry=None, handlers=None):
    """构造 Runtime + Executor + 真 MemoryManager，返回 (runtime, memory_manager)。"""
    registry = registry if registry is not None else ToolRegistry()
    memory_manager = _make_memory()
    runtime = Runtime(
        llm_call=llm_call,
        llm_stream=llm_stream,
        tool_executor=Executor(registry, mode="serial"),
        memory_manager=memory_manager,
        handlers=handlers or [],
    )
    return runtime, memory_manager


def _token_events(events):
    return [e for e in events if e.type.value == "llm.token"]


# ---------- 文本流 ----------

def test_streaming_emits_token_events_and_finishes():
    """文本流：每个 content 碎片发一个 llm.token 事件；最终消息进 memory；状态 FINISHED。"""
    events = []

    def stream(messages, tools):
        return iter([_chunk(content="你"), _chunk(content="好"), _chunk(finish_reason="stop")])

    def sync_llm(messages, tools):
        raise AssertionError("提供 llm_stream 时不得调用 llm_call")

    runtime, memory_manager = _make_runtime(sync_llm, llm_stream=stream, handlers=[events.append])
    state = runtime.run("hi")

    assert state.status == RunStatus.FINISHED
    assert [e.data["token"] for e in _token_events(events)] == ["你", "好"]
    final_msg = memory_manager.get_or_create(state.session_id).messages[-1]
    assert final_msg["role"] == "assistant"
    assert final_msg["content"] == "你好"
    assert final_msg.get("tool_calls") in (None, [])


# ---------- 工具流 ----------

def test_streaming_tool_call_executes_then_finishes():
    """工具流：重组后的 tool_calls 照常执行（echo），第二轮文本流收尾。"""
    registry = ToolRegistry()

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    events = []
    calls = [0]

    def stream(messages, tools):
        calls[0] += 1
        if calls[0] == 1:
            return iter([
                _chunk(tool_calls=[{"id": "c1", "type": "function",
                                    "function": {"name": "echo", "arguments": '{"msg": "hi"}'}}],
                       finish_reason="tool_calls"),
            ])
        return iter([_chunk(content="完成"), _chunk(finish_reason="stop")])

    def sync_llm(messages, tools):
        raise AssertionError("提供 llm_stream 时不得调用 llm_call")

    runtime, memory_manager = _make_runtime(
        sync_llm, llm_stream=stream, registry=registry, handlers=[events.append]
    )
    state = runtime.run("start")

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

def test_streaming_midstream_error_marks_failed():
    """流中途抛 LLMError → FAILED + error_source='llm'。"""
    def stream(messages, tools):
        def gen():
            yield _chunk(content="半句")
            raise LLMError("stream broke")
        return gen()

    runtime, _ = _make_runtime(lambda m, t: None, llm_stream=stream)
    state = runtime.run("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "llm"


# ---------- 老路径回归 ----------

def test_without_llm_stream_uses_sync_path():
    """不传 llm_stream：走老路径，不发 llm.token 事件。"""
    events = []

    def llm(messages, tools):
        return _make_llm_response("sync answer")

    runtime, _ = _make_runtime(llm, handlers=[events.append])
    state = runtime.run("hi")

    assert state.status == RunStatus.FINISHED
    assert _token_events(events) == []
