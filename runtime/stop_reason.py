from enum import Enum


class StopReason(str, Enum):
    """Why an agent run reached a terminal state."""

    FINISH_NORMAL = "finish_normal"
    MAX_STEPS = "max_steps"
    LOOP_DETECTED = "loop_detected"
    VALIDATION_FAILED = "validation_failed"
    CANCELED = "canceled"
    ERROR = "error"
    REFLECTION_LIMIT = "reflection_limit"
