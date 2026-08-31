from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from errors import PlannerError
from runtime.planner import Planner


def _response(content: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.anyio
async def test_planner_builds_valid_dag_and_exposes_capabilities() -> None:
    calls: list[tuple[list[dict], object]] = []

    async def llm(messages, tools):
        calls.append((messages, tools))
        return _response(
            '{"goal":"prepare report","tasks":['
            '{"id":"research","goal":"collect evidence","dependencies":[]},'
            '{"id":"write","goal":"write report","dependencies":["research"]}]}'
        )

    planner = Planner(llm)
    plan = await planner.plan(
        "prepare a report",
        [{"type": "function", "function": {"name": "search", "description": "Search."}}],
    )

    assert [task.id for task in plan.tasks] == ["research", "write"]
    assert plan.tasks[1].dependencies == ("research",)
    assert '"name":"search"' in calls[0][0][0]["content"]
    assert calls[0][1] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"goal":"g","tasks":[]}',
        '{"goal":"g","tasks":[{"id":"a","goal":"A","dependencies":["missing"]}]}',
        '{"goal":"g","tasks":['
        '{"id":"a","goal":"A","dependencies":["b"]},'
        '{"id":"b","goal":"B","dependencies":["a"]}]}',
    ],
)
async def test_planner_rejects_invalid_or_non_dag_output(content: str) -> None:
    async def llm(messages, tools):
        return _response(content)

    with pytest.raises(PlannerError):
        await Planner(llm).plan("request", [])


@pytest.mark.anyio
async def test_planner_timeout_is_structured_and_cancellation_propagates() -> None:
    async def slow(messages, tools):
        await asyncio.Event().wait()

    with pytest.raises(PlannerError, match="timed out") as captured:
        await Planner(slow, timeout_s=0.01).plan("request", [])
    assert captured.value.context == {"timeout_s": 0.01}

    task = asyncio.create_task(Planner(slow).plan("request", []))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
