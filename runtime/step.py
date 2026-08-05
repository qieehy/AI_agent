from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_EXEC = "tool_exec"
    FINAL_ANSWER = "final_answer"

@dataclass
class Step:
    index: int
    kind: StepKind
    input: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "duration_ms": self.duration_ms,
            "output": self.output,
            "error": self.error,
            "input": self.input,
        }
