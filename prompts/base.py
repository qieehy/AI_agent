from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """A system message produced by a prompt pattern."""

    role: Literal["system"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class PromptContext:
    """Runtime context available to a prompt pattern."""

    tool_schemas: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class PromptPattern(ABC):
    """Base interface for agent prompt patterns"""

    @abstractmethod
    def build(self, context: PromptContext) -> PromptMessage:
        """build the system message"""


