from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .step import Step
from .stop_reason import StopReason


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_TOOL = "awaiting_tool"
    FINISHED = "finished"
    MAX_STEPS = "max_steps"
    LOOP_DETECTED = "loop_detected"
    VALIDATION_FAILED = "validation_failed"
    FAILED = "failed"
    CANCELED = "canceled"
    REFLECTION_LIMIT = "reflection_limit"


@dataclass
class RuntimeState:
    session_id: str
    status: RunStatus = RunStatus.PENDING
    steps: list[Step] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_history: list[Any] = field(default_factory=list)
    step_count: int = 0
    stop_reason: StopReason | None = None
    # 连续失败轮数预算（feedback rounds 计数，成功一轮即清零）：
    # validation_failure_rounds = 连续出现非法 tool_call 的轮数
    # tool_error_rounds = 连续整轮工具全败的轮数
    validation_failure_rounds: int = 0
    tool_error_rounds: int = 0
    error: BaseException | None = None
    error_source: str | None = None
    error_info: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reflection_revision_rounds: int = 0

    def is_terminal(self) -> bool:
        return self.status in (
            RunStatus.FINISHED,
            RunStatus.FAILED,
            RunStatus.CANCELED,
            RunStatus.MAX_STEPS,
            RunStatus.LOOP_DETECTED,
            RunStatus.VALIDATION_FAILED,
            RunStatus.REFLECTION_LIMIT,
        )
