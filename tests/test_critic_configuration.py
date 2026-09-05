from __future__ import annotations

import pytest

from errors import ConfigError
from runtime.policy import LoopPolicy
from runtime.reflection import Critic


async def _unused_llm(messages, tools):
    raise AssertionError("configuration tests must not call the LLM")


def test_critic_accepts_safe_default_limits_without_calling_llm() -> None:
    critic = Critic(_unused_llm)

    assert critic is not None


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("timeout_s", 0, "timeout_s must be greater than 0"),
        ("timeout_s", -1, "timeout_s must be greater than 0"),
        ("timeout_s", float("inf"), "timeout_s must be finite"),
        ("timeout_s", float("nan"), "timeout_s must be finite"),
        (
            "max_feedback_chars",
            0,
            "max_feedback_chars must be greater than 0",
        ),
        (
            "max_feedback_chars",
            -1,
            "max_feedback_chars must be greater than 0",
        ),
    ],
)
def test_critic_rejects_invalid_limits_at_construction(
    argument: str,
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Critic(_unused_llm, **{argument: value})


@pytest.mark.parametrize("value", [-1, -2])
def test_loop_policy_rejects_negative_reflection_revision_budget(
    value: int,
) -> None:
    with pytest.raises(
        ConfigError,
        match="reflection_revision_rounds must be greater than or equal to 0",
    ):
        LoopPolicy(reflection_revision_rounds=value)
