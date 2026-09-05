from __future__ import annotations

from types import SimpleNamespace

import pytest

from errors import ReflectionError
from runtime.reflection import Critic


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
async def test_critic_translates_invalid_response_shape(response: object) -> None:
    async def llm(messages, tools):
        return response

    with pytest.raises(ReflectionError, match="invalid shape") as captured:
        await Critic(llm).review(
            context_messages=[{"role": "user", "content": "question"}],
            candidate_answer="candidate",
        )

    assert captured.value.__cause__ is not None


@pytest.mark.anyio
async def test_critic_delegates_non_text_content_to_parser_boundary() -> None:
    async def llm(messages, tools):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    with pytest.raises(ReflectionError, match="JSON text"):
        await Critic(llm).review(
            context_messages=[{"role": "user", "content": "question"}],
            candidate_answer="candidate",
        )


@pytest.mark.anyio
async def test_critic_translates_llm_failure_and_preserves_cause() -> None:
    provider_error = RuntimeError("provider unavailable")

    async def llm(messages, tools):
        raise provider_error

    with pytest.raises(ReflectionError, match="critic request failed") as captured:
        await Critic(llm).review(
            context_messages=[{"role": "user", "content": "question"}],
            candidate_answer="candidate",
        )

    assert captured.value.__cause__ is provider_error
