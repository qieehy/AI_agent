from dataclasses import dataclass

from errors import ConfigError


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    """Runtime execution safeguards.

    Attributes:
        max_steps:
            Maximum number of runtime steps allowed in one run.

        max_consecutive_repeats:
            Number of identical consecutive tool calls required to
            trigger loop detection.

        validation_feedback_rounds:
            Number of times a tool-call validation error may be
            returned to the model before the run is terminated.

        tool_error_feedback_rounds:
            Number of times a tool execution error may be returned
            to the model before the run is terminated.
    """

    max_steps: int = 100
    max_consecutive_repeats: int = 3
    validation_feedback_rounds: int = 1
    tool_error_feedback_rounds: int = 1

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ConfigError("max_steps must be greater than 0")

        if self.max_consecutive_repeats <= 0:
            raise ConfigError(
                "max_consecutive_repeats must be greater than 0"
            )

        if self.validation_feedback_rounds < 0:
            raise ConfigError(
                "validation_feedback_rounds must be greater than or equal to 0"
            )

        if self.tool_error_feedback_rounds < 0:
            raise ConfigError(
                "tool_error_feedback_rounds must be greater than or equal to 0"
            )
