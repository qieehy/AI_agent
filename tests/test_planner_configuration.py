from __future__ import annotations

import pytest

from runtime.planner import Planner


async def _unused_llm(messages, tools):
    raise AssertionError("configuration tests must not call the LLM")


def test_planner_accepts_safe_default_limits_without_calling_llm() -> None:
    planner = Planner(_unused_llm)

    assert planner is not None


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("timeout_s", 0, "timeout_s must be greater than 0"),
        ("timeout_s", -1, "timeout_s must be greater than 0"),
        ("timeout_s", float("inf"), "timeout_s must be finite"),
        ("timeout_s", float("nan"), "timeout_s must be finite"),
        ("max_tasks", 0, "max_tasks must be greater than 0"),
        ("max_tasks", -1, "max_tasks must be greater than 0"),
        ("max_goal_chars", 0, "max_goal_chars must be greater than 0"),
        ("max_goal_chars", -1, "max_goal_chars must be greater than 0"),
    ],
)
def test_planner_rejects_non_positive_limits_at_construction(
    argument: str,
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Planner(_unused_llm, **{argument: value})
