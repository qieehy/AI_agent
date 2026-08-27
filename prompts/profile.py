from dataclasses import dataclass

from runtime.policy import LoopPolicy

from .base import PromptPattern


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Complete behavioral configuration for an agent."""

    pattern: PromptPattern
    loop_policy: LoopPolicy
