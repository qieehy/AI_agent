from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from errors import ReflectionError
from runtime.reflection import Critic, CritiqueDecision


def _response(content: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.anyio
async def test_critic_reviews_candidate_without_tools_and_returns_validated_result() -> None:
    calls: list[tuple[list[dict], object]] = []

    async def llm(messages, tools):
        calls.append((messages, tools))
        return _response('{"decision":"revise","feedback":"Correct the calculation."}')

    critic = Critic(llm)
    context_messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "tool", "content": "4", "tool_call_id": "call-1"},
    ]

    result = await critic.review(
        context_messages=context_messages,
        candidate_answer="The answer is 5.",
    )

    assert result.decision is CritiqueDecision.REVISE
    assert result.feedback == "Correct the calculation."
    assert len(calls) == 1

    messages, tools = calls[0]
    assert tools is None
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Return JSON only" in messages[0]["content"]
    assert "do not execute tools" in messages[0]["content"].lower()

    payload = json.loads(messages[1]["content"])
    assert payload == {
        "context_messages": context_messages,
        "candidate_answer": "The answer is 5.",
    }


@pytest.mark.anyio
async def test_critic_passes_feedback_limit_to_parser() -> None:
    async def llm(messages, tools):
        return _response('{"decision":"revise","feedback":"1234"}')

    critic = Critic(llm, max_feedback_chars=3)

    with pytest.raises(ReflectionError, match="size limit") as captured:
        await critic.review(
            context_messages=[{"role": "user", "content": "question"}],
            candidate_answer="candidate",
        )

    assert captured.value.context == {"max_feedback_chars": 3}


@pytest.mark.anyio
async def test_critic_timeout_is_structured_and_cleans_up_llm_task() -> None:
    cleaned_up = asyncio.Event()

    async def slow_llm(messages, tools):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    critic = Critic(slow_llm, timeout_s=0.01)

    with pytest.raises(ReflectionError, match="timed out") as captured:
        await critic.review(
            context_messages=[{"role": "user", "content": "question"}],
            candidate_answer="candidate",
        )

    assert captured.value.context == {"timeout_s": 0.01}
    assert isinstance(captured.value.__cause__, asyncio.TimeoutError)
    assert cleaned_up.is_set()


@pytest.mark.anyio
async def test_critic_cancellation_propagates_and_cleans_up_llm_task() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def slow_llm(messages, tools):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    critic = Critic(slow_llm)
    task = asyncio.create_task(
        critic.review(
            context_messages=[{"role": "user", "content": "question"}],
            candidate_answer="candidate",
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleaned_up.is_set()
