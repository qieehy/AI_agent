import json
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ToolCallViolation:
    """A validation failure for a single tool call."""

    tool_call_id: str
    name: str | None
    reason: str
    details: str | None = None


@dataclass(frozen=True)
class ToolCallValidationResult:
    """Result of validating a collection of tool calls."""

    valid_calls: list[Any]
    violations: list[ToolCallViolation]

    @property
    def is_valid(self) -> bool:
        return not self.violations


class ToolCallValidator:
    """Validate LLM-generated tool calls before execution."""

    def __init__(self, schemas: list[dict[str, Any]]) -> None:
        self._schemas = {
            schema["function"]["name"]: schema
            for schema in schemas
        }

    def validate(
            self,
            tool_calls: list[Any],
    ) -> ToolCallValidationResult:
        valid_calls: list[Any] = []
        violations: list[ToolCallViolation] = []

        for tool_call in tool_calls:
            violation = self._validate_one(tool_call)

            if violation is None:
                valid_calls.append(tool_call)
            else:
                violations.append(violation)

        return ToolCallValidationResult(
            valid_calls=valid_calls,
            violations=violations,
        )

    def _validate_one(
            self,
            tool_call: Any,
    ) -> ToolCallViolation | None:

        tool_call_id = getattr(tool_call, "id", "")
        function = getattr(tool_call, "function", None)

        if function is None:
            return ToolCallViolation(
                tool_call_id=tool_call_id,
                name=None,
                reason="malformed_tool_call",
                details="missing function",
            )

        name = getattr(function, "name", None)

        if not name:
            return ToolCallViolation(
                tool_call_id=tool_call_id,
                name=None,
                reason="malformed_tool_call",
                details="missing function name",
            )

        schema = self._schemas.get(name)

        if schema is None:
            return ToolCallViolation(
                tool_call_id=tool_call_id,
                name=name,
                reason="unknown_tool",
                details=f"tool {name!r} is not registered",
            )

        arguments = getattr(function, "arguments", None)

        parsed_arguments = self._parse_arguments(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )

        if isinstance(parsed_arguments, ToolCallViolation):
            return parsed_arguments

        return self._validate_schema(
            tool_call_id=tool_call_id,
            name=name,
            arguments=parsed_arguments,
            schema=schema,
        )

    def _parse_arguments(
            self,
            *,
            tool_call_id: str,
            name: str,
            arguments: Any,
    ) -> dict[str, Any] | ToolCallViolation:

        if not isinstance(arguments, str):
            return ToolCallViolation(
                tool_call_id=tool_call_id,
                name=name,
                reason="invalid_arguments",
                details="arguments must be a JSON string",
            )

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return ToolCallViolation(
                tool_call_id=tool_call_id,
                name=name,
                reason="invalid_arguments",
                details=f"invalid JSON: {exc.msg}",
            )

        if not isinstance(parsed, dict):
            return ToolCallViolation(
                tool_call_id=tool_call_id,
                name=name,
                reason="invalid_arguments",
                details="tool arguments must be a JSON object",
            )

        return parsed

    def _validate_schema(
            self,
            *,
            tool_call_id: str,
            name: str,
            arguments: dict[str, Any],
            schema: dict[str, Any],
    ) -> ToolCallViolation | None:

        parameters = schema["function"].get(
            "parameters",
            {"type": "object"},
        )

        validator = Draft202012Validator(parameters)

        errors = sorted(
            validator.iter_errors(arguments),
            key=lambda x: list(x.path),
        )

        if not errors:
            return None

        error = errors[0]

        return ToolCallViolation(
            tool_call_id=tool_call_id,
            name=name,
            reason="schema_validation_failed",
            details=error.message,
        )
