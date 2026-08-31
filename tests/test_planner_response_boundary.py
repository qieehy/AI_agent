from __future__ import annotations

from types import SimpleNamespace

import pytest

from errors import PlannerError
from runtime.planner import Planner


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=None),
        SimpleNamespace(choices=[SimpleNamespace(message=None)]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())]),
    ],
)
async def test_planner_translates_invalid_response_shape(
    response: object,
) -> None:
    async def llm(messages, tools):
        return response

    with pytest.raises(PlannerError, match="invalid shape") as captured:
        await Planner(llm).plan("request", [])

    assert captured.value.__cause__ is not None


@pytest.mark.anyio
async def test_planner_delegates_non_text_content_to_parser_boundary() -> None:
    async def llm(messages, tools):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

    with pytest.raises(PlannerError, match="JSON text"):
        await Planner(llm).plan("request", [])


@pytest.mark.anyio
async def test_planner_translates_llm_call_failure_and_preserves_cause() -> None:
    provider_error = RuntimeError("provider unavailable")

    async def llm(messages, tools):
        raise provider_error

    with pytest.raises(PlannerError, match="planner request failed") as captured:
        await Planner(llm).plan("request", [])

    assert captured.value.__cause__ is provider_error
