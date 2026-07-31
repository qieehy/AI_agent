from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeAlias

class EventType(str, Enum):
    RUN_START = "run.start"
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    # TOOL_REQUEST = "tool.request"
    TOOL_RESPONSE = "tool.response"
    RUN_FINISH = "run.finish"
    RUN_ERROR = "run.error"

@dataclass
class Event:
    type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)
    step_index: int | None = None

EventHandler: TypeAlias = Callable[[Event], None]
"""定义EventHandler为一种函数类型: 参数类型Event, 无返回值"""