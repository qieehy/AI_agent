from __future__ import annotations

import asyncio

import pytest

from errors import SessionBusyError
from runtime import (
    Event,
    LoopGuard,
    LoopPolicy,
    Runtime,
    SessionCoordinator,
)
from tools import Executor, ToolCallValidator, ToolRegistry

from .conftest import AllToolsRouter, make_llm_response, make_memory


def _runtime(llm_call, *, handlers=None, coordinator=None):
    registry = ToolRegistry()
    executor = Executor(registry)
    memory = make_memory()
    runtime = Runtime(
        tool_executor=executor,
        memory_manager=memory,
        llm_call_async=llm_call,
        loop_guard=LoopGuard(LoopPolicy()),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=AllToolsRouter(),
        handlers=handlers,
        session_coordinator=coordinator,
    )
    return runtime, memory


@pytest.mark.anyio
async def test_same_session_rejects_overlapping_run_without_writing_memory():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def llm(messages, tools):
        entered.set()
        await release.wait()
        return make_llm_response("done")

    runtime, memory = _runtime(llm)
    first = asyncio.create_task(runtime.run_async("first", session_id="shared"))
    await entered.wait()

    with pytest.raises(SessionBusyError) as exc_info:
        await runtime.run_async("second", session_id="shared")

    assert exc_info.value.context["session_id"] == "shared"
    # The rejected request must not append even its user message.
    assert [m["content"] for m in memory.get_or_create("shared").messages] == ["first"]

    release.set()
    state = await first
    assert state.status.value == "finished"
    assert [m["content"] for m in memory.get_or_create("shared").messages] == [
        "first",
        "done",
    ]


@pytest.mark.anyio
async def test_different_sessions_can_run_concurrently():
    both_entered = asyncio.Event()
    release = asyncio.Event()
    active = 0

    async def llm(messages, tools):
        nonlocal active
        active += 1
        if active == 2:
            both_entered.set()
        await release.wait()
        return make_llm_response("done")

    runtime, _ = _runtime(llm)
    first = asyncio.create_task(runtime.run_async("one", session_id="s1"))
    second = asyncio.create_task(runtime.run_async("two", session_id="s2"))

    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release.set()
    states = await asyncio.gather(first, second)
    assert all(state.status.value == "finished" for state in states)


@pytest.mark.anyio
async def test_cancellation_emits_canceled_terminal_event_and_releases_lease():
    entered = asyncio.Event()
    calls = 0
    events: list[Event] = []

    async def llm(messages, tools):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()
        return make_llm_response("recovered")

    coordinator = SessionCoordinator()
    runtime, _ = _runtime(llm, handlers=[events.append], coordinator=coordinator)
    first = asyncio.create_task(runtime.run_async("first", session_id="shared"))
    await entered.wait()
    first.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first

    canceled = [event for event in events if event.data.get("status") == "canceled"]
    assert len(canceled) == 1
    assert canceled[0].data["stop_reason"] == "canceled"
    assert coordinator.tracked_session_count == 0

    state = await runtime.run_async("second", session_id="shared")
    assert state.status.value == "finished"
    assert coordinator.tracked_session_count == 0


@pytest.mark.anyio
async def test_wait_mode_times_out_and_reclaims_waiter_reference():
    coordinator = SessionCoordinator(conflict_mode="wait", acquire_timeout=0.01)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def owner():
        async with coordinator.lease("shared"):
            entered.set()
            await release.wait()

    task = asyncio.create_task(owner())
    await entered.wait()

    with pytest.raises(SessionBusyError) as exc_info:
        async with coordinator.lease("shared"):
            pytest.fail("timed-out waiter acquired the session")

    assert exc_info.value.context["retry_after_ms"] == 10
    assert coordinator.tracked_session_count == 1
    release.set()
    await task
    assert coordinator.tracked_session_count == 0


@pytest.mark.anyio
async def test_session_entries_do_not_accumulate():
    coordinator = SessionCoordinator()

    for index in range(100):
        async with coordinator.lease(f"session-{index}"):
            pass

    assert coordinator.tracked_session_count == 0
