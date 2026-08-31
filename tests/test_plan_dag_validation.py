from __future__ import annotations

import json

import pytest

from errors import PlannerError
from runtime.planner import parse_task_plan


def _plan(tasks: list[dict[str, object]]) -> str:
    return json.dumps({"goal": "Complete request", "tasks": tasks})


def _task(
    task_id: object,
    *,
    dependencies: list[object] | None = None,
) -> dict[str, object]:
    return {
        "id": task_id,
        "goal": f"Execute {task_id}",
        "dependencies": [] if dependencies is None else dependencies,
    }


@pytest.mark.parametrize(
    "task_id",
    [None, 1, "", "1starts_with_number", "contains space", "a" * 65],
)
def test_parse_task_plan_rejects_invalid_task_ids(task_id: object) -> None:
    with pytest.raises(PlannerError, match="task id is invalid") as captured:
        parse_task_plan(_plan([_task(task_id)]))

    assert captured.value.context == {"task_index": 0}


def test_parse_task_plan_accepts_bounded_ascii_task_ids() -> None:
    plan = parse_task_plan(
        _plan(
            [
                _task("research_sources"),
                _task("write-report_2", dependencies=["research_sources"]),
                _task("a" * 64),
            ]
        )
    )

    assert tuple(task.id for task in plan.tasks) == (
        "research_sources",
        "write-report_2",
        "a" * 64,
    )


def test_parse_task_plan_rejects_duplicate_task_ids() -> None:
    with pytest.raises(PlannerError, match="task ids must be unique"):
        parse_task_plan(_plan([_task("research"), _task("research")]))


def test_parse_task_plan_rejects_unknown_dependencies() -> None:
    with pytest.raises(PlannerError, match="unknown dependencies") as captured:
        parse_task_plan(_plan([_task("write", dependencies=["research"])]))

    assert captured.value.context == {
        "task_id": "write",
        "unknown": ["research"],
    }


def test_parse_task_plan_rejects_self_dependency() -> None:
    with pytest.raises(PlannerError, match="cannot depend on itself") as captured:
        parse_task_plan(_plan([_task("research", dependencies=["research"])]))

    assert captured.value.context == {"task_id": "research"}


def test_parse_task_plan_rejects_dependency_cycle() -> None:
    with pytest.raises(PlannerError, match="dependency cycle"):
        parse_task_plan(
            _plan(
                [
                    _task("research", dependencies=["write"]),
                    _task("write", dependencies=["research"]),
                ]
            )
        )


def test_parse_task_plan_accepts_dependency_declared_later() -> None:
    plan = parse_task_plan(
        _plan(
            [
                _task("write", dependencies=["research"]),
                _task("research"),
            ]
        )
    )

    assert plan.tasks[0].dependencies == ("research",)
