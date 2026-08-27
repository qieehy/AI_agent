from typing import Literal

from errors import ConfigError
from runtime.policy import LoopPolicy

from .base import PromptPattern
from .plan_execute import PlanExecutePattern
from .profile import AgentProfile
from .react import ReActPattern
from .reflection import ReflectionPattern

PatternName = Literal[
    "react",
    "plan_execute",
    "reflection",
]


_PATTERN_TYPES: dict[str, type[PromptPattern]] = {
    "react": ReActPattern,
    "plan_execute": PlanExecutePattern,
    "reflection": ReflectionPattern,
}

_DEFAULT_POLICIES: dict[str, LoopPolicy] = {
    "react": LoopPolicy(
        max_steps=20,
        max_consecutive_repeats=3,
        validation_feedback_rounds=1,
        tool_error_feedback_rounds=1,
    ),
    "plan_execute": LoopPolicy(
        max_steps=50,
        max_consecutive_repeats=3,
        validation_feedback_rounds=1,
        tool_error_feedback_rounds=1,
    ),
    "reflection": LoopPolicy(
        max_steps=30,
        max_consecutive_repeats=3,
        validation_feedback_rounds=1,
        tool_error_feedback_rounds=1,
    ),
}

def create_agent_profile(
    pattern: PatternName,
) -> AgentProfile:
    pattern_type = _PATTERN_TYPES.get(pattern)

    if pattern_type is None:
        raise ConfigError(
            f"Unknown prompt pattern: {pattern!r}"
        )

    prompt_pattern = pattern_type()

    return AgentProfile(
        pattern=prompt_pattern,
        loop_policy=_DEFAULT_POLICIES[pattern],
    )
