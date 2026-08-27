from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .policy import LoopPolicy


@dataclass(frozen=True, slots=True)
class ToolCallSignature:
    """Canonical identity of a tool call."""

    name: str
    arguments: str


def _canonicalize_arguments(arguments: Any) -> str:
    """Return a deterministic representation of tool arguments.

    JSON objects are normalized so semantically equivalent argument
    objects produce the same signature regardless of key ordering.
    """

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments

        arguments = parsed

    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return repr(arguments)


def tool_call_signature(tool_call: Any) -> ToolCallSignature:
    """Build a stable signature from an SDK tool call."""

    name = tool_call.function.name
    arguments = _canonicalize_arguments(
        tool_call.function.arguments
    )

    return ToolCallSignature(
        name=name,
        arguments=arguments,
    )

def _is_consecutive_repeat(
    history: list[ToolCallSignature],
    max_repeats: int,
) -> bool:
    """Return True when the latest calls are identical consecutively."""

    if max_repeats <= 0:
        raise ValueError("max_repeats must be greater than 0")

    if len(history) < max_repeats:
        return False

    recent = history[-max_repeats:]

    return len(set(recent)) == 1


@dataclass(frozen=True)
class LoopCheckResult:
    detected: bool
    reason: str | None = None


class LoopGuard:
    """Detect unsafe or non-progressing agent execution."""

    def __init__(self, policy: LoopPolicy) -> None:
        self._policy = policy

    def check_steps(self, step_count: int) -> LoopCheckResult:
        if step_count >= self._policy.max_steps:
            return LoopCheckResult(
                detected=True,
                reason="max_steps",
            )

        return LoopCheckResult(detected=False)

    def check_tool_calls(
            self,
            history: list[ToolCallSignature],
    ) -> LoopCheckResult:
        if _is_consecutive_repeat(
                history,
                self._policy.max_consecutive_repeats,
        ):
            return LoopCheckResult(
                detected=True,
                reason="loop_detected",
            )

        return LoopCheckResult(detected=False)

    def check_validation_budget(self, rounds: int) -> LoopCheckResult:
        """连续验证失败轮数是否已超过可回喂预算。"""
        if rounds > self._policy.validation_feedback_rounds:
            return LoopCheckResult(
                detected=True,
                reason="validation_feedback_budget_exceeded",
            )

        return LoopCheckResult(detected=False)

    def check_tool_error_budget(self, rounds: int) -> LoopCheckResult:
        """连续整轮工具全败的轮数是否已超过可回喂预算。"""
        if rounds > self._policy.tool_error_feedback_rounds:
            return LoopCheckResult(
                detected=True,
                reason="tool_error_feedback_budget_exceeded",
            )

        return LoopCheckResult(detected=False)

    @property
    def policy(self) -> LoopPolicy:
        """只读暴露策略：供 Runtime 落结构化归因（如预算上限值）。"""
        return self._policy
