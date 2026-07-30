from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_TOOL = "awaiting_tool"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class RuntimeState:
    session_id: str
    status: RunStatus = RunStatus.PENDING
    messages:list[dict[str,Any]] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 100
    error: Exception | None = None
    error_source: str | None = None
    error_info: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in (RunStatus.FINISHED, RunStatus.FAILED, RunStatus.CANCELED)