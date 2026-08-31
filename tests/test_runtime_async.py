"""D23: Runtime.run_async 异步步进循环契约测试。

策略：
- async 假 callable（async generator 流 / async 非流）；sync Runtime 已删除，只钉 async 槽
- 真 ToolRegistry + Executor + 真 MemoryManager（temp SQLite）
- 只钉行为：llm.token 事件按序、消息形状、工具照常执行、中途异常 FAILED、
  非流回退（零 token 事件）、接线错误 fail-fast
- 尾部区块移植自已删除的 test_runtime.py（sync 版独有的覆盖）
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from errors import ConfigError, LLMError
from runtime import LoopGuard, Runtime
from runtime.policy import LoopPolicy
from runtime.state import RunStatus
from tools import Executor, ToolCallValidator, ToolRegistry, ToolRoute, ToolRoutingError

from .conftest import AllToolsRouter
from .conftest import make_llm_response as _make_llm_response
from .conftest import make_memory as _make_memory
from .conftest import make_tool_call as _make_tool_call

# ---------- 辅助 ----------

def _chunk(content=None, tool_calls=None, finish_reason=None):
    """构造流式 chunk（StreamChunk 形状，duck typing）。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls, finish_reason=finish_reason)


def _make_runtime(*, llm_call_async=None, llm_stream_async=None,
                  registry=None, handlers=None, max_steps=100, tool_router=None):
    """构造 Runtime（仅 async 槽）+ Executor + LoopGuard + Validator + 真 MemoryManager。

    D24 起 Runtime 强制注入 loop_guard / validator：
    validator 基于 executor 暴露的 schemas 构建——测试与生产同一张校验表。
    """
    registry = registry if registry is not None else ToolRegistry()
    memory_manager = _make_memory()
    executor = Executor(registry, mode="serial")
    runtime = Runtime(
        llm_call_async=llm_call_async,
        llm_stream_async=llm_stream_async,
        tool_executor=executor,
        memory_manager=memory_manager,
        handlers=handlers or [],
        loop_guard=LoopGuard(LoopPolicy(max_steps=max_steps)),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=tool_router or AllToolsRouter(),
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
    """max_steps=1 + 工具轮：第二轮 LLM 不发生，状态 MAX_STEPS。"""
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

    assert state.status == RunStatus.MAX_STEPS
    assert calls[0] == 1


# ---------- 移植自 test_runtime.py（sync 版独有覆盖） ----------

@pytest.mark.anyio
async def test_async_llm_error_marks_error_info_from_to_dict():
    """LLM 抛 LLMError → FAILED + error_info 来自 e.to_dict() 三字段。"""
    async def bad_llm(messages, tools):
        raise LLMError("LLM timeout", context={"model": "gpt-4"})

    runtime, _ = _make_runtime(llm_call_async=bad_llm)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "llm"
    assert state.error_info["type"] == "LLMError"
    assert state.error_info["message"] == "LLM timeout"
    assert state.error_info["context"] == {"model": "gpt-4"}


@pytest.mark.anyio
async def test_async_unknown_tool_validation_then_terminates():
    """调不存在的 tool → D24 语义：validator 拦截（不再进 executor）→ 回喂一次 →
    第二轮仍违规 → VALIDATION_FAILED（validation_feedback_rounds=1 耗尽）。"""
    async def llm(messages, tools):
        return _make_llm_response(
            content="",
            tool_calls=[_make_tool_call("call_1", "bad_tool", "{}")],
        )

    runtime, memory_manager = _make_runtime(llm_call_async=llm)  # registry 空，bad_tool 不存在
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.VALIDATION_FAILED
    assert state.validation_failure_rounds == 2
    # 回喂消息带 tool_call_id（配对约束），内容带验证错误原因
    msgs = memory_manager.get_or_create(state.session_id).messages
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert all(m.get("tool_call_id") == "call_1" for m in tool_msgs)
    assert all("[VALIDATION ERROR]" in m["content"] for m in tool_msgs)


@pytest.mark.anyio
async def test_async_non_agent_error_escapes():
    """非 AgentError（如 KeyError）必须逃逸，不能被 runtime 吞掉。"""
    async def bad_llm(messages, tools):
        raise KeyError("oops, not an AgentError")

    runtime, _ = _make_runtime(llm_call_async=bad_llm)
    with pytest.raises(KeyError) as excinfo:
        await runtime.run_async("hi")
    assert "oops" in str(excinfo.value)


@pytest.mark.anyio
async def test_async_emits_run_start_and_finish():
    """正常路径：handler 收到 run.start / llm.request / llm.response / run.finish。"""
    events = []

    async def llm(messages, tools):
        return _make_llm_response("hi")

    runtime, _ = _make_runtime(llm_call_async=llm, handlers=[events.append])
    await runtime.run_async("hi")

    types = [e.type.value for e in events]
    assert "run.start" in types
    assert "llm.request" in types
    assert "llm.response" in types
    assert "run.finish" in types


@pytest.mark.anyio
async def test_async_emits_run_error_with_error_source():
    """LLM 失败：收到 run.error 事件，最终事件 data.error_source == 'llm'。"""
    events = []

    async def bad_llm(messages, tools):
        raise LLMError("boom")

    runtime, _ = _make_runtime(llm_call_async=bad_llm, handlers=[events.append])
    await runtime.run_async("hi")

    error_events = [e for e in events if e.type.value == "run.error"]
    assert len(error_events) >= 1
    assert error_events[-1].data.get("error_source") == "llm"


# ---------- D24 新护栏：循环检测 / 验证预算 / 工具错误预算 ----------

@pytest.mark.anyio
async def test_async_loop_detection_terminates_on_repeat_calls():
    """同一工具同一参数连续 3 次 → LOOP_DETECTED（循环守卫接线生效）。"""
    registry = ToolRegistry()

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    calls = [0]

    async def llm(messages, tools):
        calls[0] += 1
        return _make_llm_response(
            content="",
            tool_calls=[_make_tool_call(f"c{calls[0]}", "echo", '{"msg": "hi"}')],
        )

    runtime, _ = _make_runtime(llm_call_async=llm, registry=registry)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.LOOP_DETECTED
    assert state.stop_reason.value == "loop_detected"
    assert calls[0] == 3  # 第三次调用被拦下，未再执行


@pytest.mark.anyio
async def test_async_validation_feedback_self_heals_then_reset():
    """先违规后改对：违规回喂 → 第二轮修正 → FINISHED；validation_failure_rounds 清零。"""
    registry = ToolRegistry()

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    attempts = [0]

    async def llm(messages, tools):
        attempts[0] += 1
        if attempts[0] == 1:
            return _make_llm_response(content="", tool_calls=[_make_tool_call("a1", "bad_tool", "{}")])
        if attempts[0] == 2:
            return _make_llm_response(content="", tool_calls=[_make_tool_call("a2", "echo", '{"msg": "ok"}')])
        return _make_llm_response("done")

    runtime, memory_manager = _make_runtime(llm_call_async=llm, registry=registry)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FINISHED
    assert state.validation_failure_rounds == 0  # 干净一轮清零
    msgs = memory_manager.get_or_create(state.session_id).messages
    # 违规被回喂（带 tool_call_id），echo 真执行过
    assert any(m.get("tool_call_id") == "a1" and "[VALIDATION ERROR]" in m.get("content", "") for m in msgs)
    assert any(m.get("role") == "tool" and m.get("name") == "echo" for m in msgs)


@pytest.mark.anyio
async def test_async_tool_error_budget_terminates():
    """工具连续整轮全败：tool_error_feedback_rounds=1 → 第二轮全败终止 FAILED/ERROR/tool。"""
    registry = ToolRegistry()

    @registry.register
    def boom() -> str:
        raise ValueError("kaboom")

    calls = [0]

    async def llm(messages, tools):
        calls[0] += 1
        return _make_llm_response(content="", tool_calls=[_make_tool_call(f"b{calls[0]}", "boom", "{}")])

    runtime, _ = _make_runtime(llm_call_async=llm, registry=registry)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "tool"
    assert state.stop_reason.value == "error"
    assert state.tool_error_rounds == 2


@pytest.mark.anyio
async def test_async_tool_error_rounds_reset_on_partial_success():
    """整轮全败推预算，下一轮任一成功即清零（不是累计计数器）。"""
    registry = ToolRegistry()

    @registry.register
    def boom() -> str:
        raise ValueError("kaboom")

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    attempts = [0]

    async def llm(messages, tools):
        attempts[0] += 1
        if attempts[0] == 1:
            return _make_llm_response(content="", tool_calls=[_make_tool_call("c1", "boom", "{}")])
        if attempts[0] == 2:
            return _make_llm_response(content="", tool_calls=[_make_tool_call("c2", "echo", '{"msg": "ok"}')])
        return _make_llm_response("done")

    runtime, _ = _make_runtime(llm_call_async=llm, registry=registry)
    state = await runtime.run_async("hi")

    assert state.status == RunStatus.FINISHED
    assert state.tool_error_rounds == 0


@pytest.mark.anyio
async def test_async_only_valid_calls_execute():
    """一轮里 mixed：合法调用执行、非法调用被拦不执行、违规回喂消息配对。"""
    registry = ToolRegistry()
    executed = []

    @registry.register
    def echo(msg: str) -> str:
        """Echo."""
        executed.append(msg)
        return msg

    async def llm(messages, tools):
        return _make_llm_response(
            content="",
            tool_calls=[
                _make_tool_call("v1", "echo", '{"msg": "valid"}'),
                _make_tool_call("v2", "bad_tool", "{}"),
            ],
        )

    runtime, memory_manager = _make_runtime(llm_call_async=llm, registry=registry)
    state = await runtime.run_async("hi")

    assert executed == ["valid"]  # 非法调用未执行
    msgs = memory_manager.get_or_create(state.session_id).messages
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert any(m.get("tool_call_id") == "v2" and "[VALIDATION ERROR]" in m.get("content", "") for m in tool_msgs)
    assert any(m.get("tool_call_id") == "v1" and m.get("name") == "echo" for m in tool_msgs)


@pytest.mark.anyio
async def test_runtime_sends_only_routed_schemas_and_records_decision():
    registry = ToolRegistry()

    @registry.register
    def selected() -> str:
        """Selected tool."""
        return "selected"

    @registry.register
    def hidden() -> str:
        """Hidden tool."""
        return "hidden"

    class SelectedOnlyRouter:
        async def route(self, query, schemas):
            selected_schema = next(
                schema for schema in schemas
                if schema["function"]["name"] == "selected"
            )
            return ToolRoute(
                selected_schemas=(selected_schema,),
                ranked_scores=(),
                model_version="test-v1",
                threshold=0.5,
            )

    observed_tools = []

    async def llm(messages, tools):
        observed_tools.append(tools)
        return _make_llm_response("done")

    runtime, _ = _make_runtime(
        llm_call_async=llm,
        registry=registry,
        tool_router=SelectedOnlyRouter(),
    )
    state = await runtime.run_async("use selected")

    assert [schema["function"]["name"] for schema in observed_tools[0]] == [
        "selected"
    ]
    tool_routes = state.metadata["tool_routes"]
    assert len(tool_routes) == 1
    assert tool_routes[0]["selected_tools"] == ["selected"]
    assert tool_routes[0]["model_version"] == "test-v1"
    assert tool_routes[0]["threshold"] == 0.5


@pytest.mark.anyio
async def test_runtime_rejects_registered_tool_not_selected_for_run():
    registry = ToolRegistry()
    executed = []

    @registry.register
    def selected() -> str:
        """Selected tool."""
        return "selected"

    @registry.register
    def hidden() -> str:
        """Hidden tool."""
        executed.append("hidden")
        return "hidden"

    class SelectedOnlyRouter:
        async def route(self, query, schemas):
            selected_schema = next(
                schema for schema in schemas
                if schema["function"]["name"] == "selected"
            )
            return ToolRoute((selected_schema,), (), "test-v1", 0.5)

    async def llm(messages, tools):
        return _make_llm_response(
            content="",
            tool_calls=[_make_tool_call("hidden-1", "hidden", "{}")],
        )

    runtime, memory_manager = _make_runtime(
        llm_call_async=llm,
        registry=registry,
        tool_router=SelectedOnlyRouter(),
    )
    state = await runtime.run_async("use selected")

    assert state.status == RunStatus.VALIDATION_FAILED
    assert executed == []
    tool_messages = [
        message
        for message in memory_manager.get_or_create(state.session_id).messages
        if message.get("role") == "tool"
    ]
    assert any("tool_not_routed" in message["content"] for message in tool_messages)


@pytest.mark.anyio
async def test_router_failure_terminates_before_llm_call():
    class FailingRouter:
        async def route(self, query, schemas):
            raise ToolRoutingError("router unavailable")

    llm_called = False

    async def llm(messages, tools):
        nonlocal llm_called
        llm_called = True
        return _make_llm_response("unexpected")

    runtime, _ = _make_runtime(
        llm_call_async=llm,
        tool_router=FailingRouter(),
    )
    state = await runtime.run_async("hello")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "router"
    assert state.error_info["type"] == "ToolRoutingError"
    assert llm_called is False
