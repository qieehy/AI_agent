from __future__ import annotations

import pytest

from errors import PlannerError
from runtime.planner import parse_task_plan


def test_parse_task_plan_converts_valid_json_to_immutable_models() -> None:
    plan = parse_task_plan(
        '{"goal":" Prepare report ","tasks":['
        '{"id":"research","goal":" Collect evidence ","dependencies":[]},'
        '{"id":"write","goal":"Write report","dependencies":["research"]}]}'
    )

    assert plan.goal == "Prepare report"
    assert tuple(task.id for task in plan.tasks) == ("research", "write")
    assert plan.tasks[0].goal == "Collect evidence"
    assert plan.tasks[1].dependencies == ("research",)


@pytest.mark.parametrize("content", [None, 123, [], {}])
def test_parse_task_plan_requires_json_text(content: object) -> None:
    with pytest.raises(PlannerError, match="JSON text"):
        parse_task_plan(content)


def test_parse_task_plan_rejects_invalid_json() -> None:
    with pytest.raises(PlannerError, match="valid JSON"):
        parse_task_plan("not json")


@pytest.mark.parametrize(
    "content",
    [
        '{"goal":"g"}',
        '{"goal":"g","tasks":[],"unexpected":true}',
        '[{"goal":"g","tasks":[]}]',
    ],
)
def test_parse_task_plan_requires_exact_top_level_fields(content: str) -> None:
    with pytest.raises(PlannerError, match="goal and tasks"):
        parse_task_plan(content)


@pytest.mark.parametrize(
    "content",
    [
        '{"goal":"g","tasks":[{"id":"a","goal":"A"}]}',
        '{"goal":"g","tasks":['
        '{"id":"a","goal":"A","dependencies":[],"result":"invented"}]}',
        '{"goal":"g","tasks":["not an object"]}',
    ],
)
def test_parse_task_plan_requires_exact_task_fields(content: str) -> None:
    with pytest.raises(PlannerError, match="id, goal, and dependencies"):
        parse_task_plan(content)


@pytest.mark.parametrize(
    "content",
    [
        '{"goal":"g","tasks":[]}',
        '{"goal":"g","tasks":['
        '{"id":"a","goal":"A","dependencies":[]},'
        '{"id":"b","goal":"B","dependencies":[]}]}',
    ],
)
def test_parse_task_plan_enforces_configured_task_count(content: str) -> None:
    with pytest.raises(PlannerError, match="task count") as captured:
        parse_task_plan(content, max_tasks=1)

    assert captured.value.context == {"max_tasks": 1}


@pytest.mark.parametrize(
    "content",
    [
        '{"goal":" ","tasks":[{"id":"a","goal":"A","dependencies":[]}]}',
        '{"goal":"g","tasks":[{"id":"a","goal":" ","dependencies":[]}]}',
    ],
)
def test_parse_task_plan_rejects_blank_goals(content: str) -> None:
    with pytest.raises(PlannerError, match="non-empty text"):
        parse_task_plan(content)


def test_parse_task_plan_enforces_goal_size_limit() -> None:
    content = (
        '{"goal":"1234","tasks":['
        '{"id":"a","goal":"A","dependencies":[]}]}'
    )

    with pytest.raises(PlannerError, match="size limit") as captured:
        parse_task_plan(content, max_goal_chars=3)

    assert captured.value.context == {"max_goal_chars": 3}


@pytest.mark.parametrize(
    "dependencies",
    ["research", [1], ["research", "research"]],
)
def test_parse_task_plan_rejects_invalid_dependency_collections(
    dependencies: object,
) -> None:
    import json

    content = json.dumps(
        {
            "goal": "g",
            "tasks": [
                {"id": "write", "goal": "Write", "dependencies": dependencies}
            ],
        }
    )

    with pytest.raises(PlannerError, match="dependencies"):
        parse_task_plan(content)
