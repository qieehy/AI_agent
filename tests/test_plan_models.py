from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from runtime.planner import PlanTask, TaskPlan


def test_plan_task_has_stable_defaults_and_serialization() -> None:
    task = PlanTask(id="research", goal="Collect authoritative evidence")

    assert task.id == "research"
    assert task.goal == "Collect authoritative evidence"
    assert task.dependencies == ()
    assert task.to_dict() == {
        "id": "research",
        "goal": "Collect authoritative evidence",
        "dependencies": [],
    }


def test_task_plan_preserves_order_and_serializes_nested_tasks() -> None:
    research = PlanTask(id="research", goal="Collect evidence")
    write = PlanTask(
        id="write",
        goal="Write the report",
        dependencies=("research",),
    )
    plan = TaskPlan(goal="Prepare a report", tasks=(research, write))

    assert plan.tasks == (research, write)
    assert plan.to_dict() == {
        "goal": "Prepare a report",
        "tasks": [
            {"id": "research", "goal": "Collect evidence", "dependencies": []},
            {
                "id": "write",
                "goal": "Write the report",
                "dependencies": ["research"],
            },
        ],
    }


def test_plan_models_are_immutable_after_creation() -> None:
    task = PlanTask(id="research", goal="Collect evidence")
    plan = TaskPlan(goal="Prepare a report", tasks=(task,))

    with pytest.raises(FrozenInstanceError):
        task.goal = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.goal = "Changed"  # type: ignore[misc]


def test_task_plan_builds_bounded_execution_inputs_without_mutating_data() -> None:
    plan = TaskPlan(
        goal="Prepare a report",
        tasks=(
            PlanTask(id="research", goal="Collect evidence"),
            PlanTask(id="write", goal="Write report", dependencies=("research",)),
        ),
    )

    routing_query = plan.routing_query()
    execution_context = plan.execution_context()

    assert routing_query == "Prepare a report\nCollect evidence\nWrite report"
    assert "validated execution DAG" in execution_context
    assert '"dependencies":["research"]' in execution_context
    assert plan.tasks[1].dependencies == ("research",)
