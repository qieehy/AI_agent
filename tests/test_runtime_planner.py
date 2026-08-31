from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from runtime import LoopGuard, Planner, Runtime
from runtime.policy import LoopPolicy
from runtime.state import RunStatus
from tools import Executor, ToolCallValidator, ToolRegistry

from .conftest import AllToolsRouter, make_llm_response, make_memory


def _runtime(planner_llm, execution_llm, *, events=None, router=None):
    registry = ToolRegistry()

    @registry.register
    def search(query: str) -> str:
        """Search evidence."""
        return query

    executor = Executor(registry, mode="serial")
    runtime = Runtime(
        llm_call_async=execution_llm,
        tool_executor=executor,
        memory_manager=make_memory(),
        loop_guard=LoopGuard(LoopPolicy()),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=router or AllToolsRouter(),
        handlers=[events.append] if events is not None else [],
        planner=Planner(planner_llm),
    )
    return runtime


@pytest.mark.anyio
async def test_runtime_uses_validated_plan_for_context_routing_and_event() -> None:
    events = []
    routed_queries = []

    class RecordingRouter(AllToolsRouter):
        async def route(self, query, schemas):
            routed_queries.append(query)
            return await super().route(query, schemas)

    async def planner_llm(messages, tools):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"goal":"answer with evidence","tasks":['
            '{"id":"find","goal":"search authoritative evidence","dependencies":[]}]}'
        ))])

    execution_messages = []

    async def execution_llm(messages, tools):
        execution_messages.append(messages)
        return make_llm_response("done")

    runtime = _runtime(
        planner_llm, execution_llm, events=events, router=RecordingRouter()
    )
    state = await runtime.run_async("question")

    assert state.status == RunStatus.FINISHED
    assert state.metadata["plan"]["tasks"][0]["id"] == "find"
    assert "search authoritative evidence" in routed_queries[0]
    assert any(
        message["role"] == "system" and "validated execution DAG" in message["content"]
        for message in execution_messages[0]
    )
    assert [event.type.value for event in events].count("plan.created") == 1


@pytest.mark.anyio
async def test_invalid_plan_fails_before_router_and_execution() -> None:
    calls = {"execution": 0}

    async def planner_llm(messages, tools):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="bad"))])

    async def execution_llm(messages, tools):
        calls["execution"] += 1
        return make_llm_response("must not run")

    runtime = _runtime(planner_llm, execution_llm)
    state = await runtime.run_async("question")

    assert state.status == RunStatus.FAILED
    assert state.error_source == "planner"
    assert state.error_info["type"] == "PlannerError"
    assert calls["execution"] == 0


@pytest.mark.anyio
async def test_planner_cancellation_uses_runtime_canceled_path() -> None:
    started = asyncio.Event()
    events = []
    calls = {"execution": 0}

    async def planner_llm(messages, tools):
        started.set()
        await asyncio.Event().wait()

    async def execution_llm(messages, tools):
        calls["execution"] += 1
        return make_llm_response("must not run")

    runtime = _runtime(planner_llm, execution_llm, events=events)
    task = asyncio.create_task(runtime.run_async("question", session_id="cancel-plan"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["execution"] == 0
    terminal = [event for event in events if event.type.value == "run.error"]
    assert len(terminal) == 1
    assert terminal[0].data["status"] == "canceled"
    assert terminal[0].data["error_source"] == "runtime"
