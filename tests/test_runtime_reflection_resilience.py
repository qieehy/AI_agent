from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from errors import ToolRoutingError
from runtime import LoopGuard, Runtime, StepKind
from runtime.policy import LoopPolicy
from runtime.reflection import Critic
from runtime.state import RunStatus
from tools import Executor, ToolCallValidator, ToolRegistry

from .conftest import AllToolsRouter, make_llm_response, make_memory, make_tool_call


def _critic_response(decision: str, feedback: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({"decision": decision, "feedback": feedback})
                )
            )
        ]
    )


def _runtime(
    *,
    critic_llm,
    execution_llm=None,
    execution_stream=None,
    registry: ToolRegistry | None = None,
    handlers=None,
    router=None,
    max_steps: int = 100,
):
    registry = registry or ToolRegistry()
    executor = Executor(registry, mode="serial")
    memory_manager = make_memory()
    runtime = Runtime(
        llm_call_async=execution_llm,
        llm_stream_async=execution_stream,
        tool_executor=executor,
        memory_manager=memory_manager,
        handlers=handlers or [],
        loop_guard=LoopGuard(
            LoopPolicy(
                max_steps=max_steps,
                reflection_revision_rounds=1,
            )
        ),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=router or AllToolsRouter(),
        critic=Critic(critic_llm),
    )
    return runtime, memory_manager


@pytest.mark.anyio
async def test_runtime_cancellation_during_critique_cleans_up_without_committing_draft() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    events = []

    async def execution_llm(messages, tools):
        return make_llm_response("unreviewed draft")

    async def critic_llm(messages, tools):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    runtime, memory_manager = _runtime(
        execution_llm=execution_llm,
        critic_llm=critic_llm,
        handlers=[events.append],
    )
    task = asyncio.create_task(runtime.run_async("question", session_id="cancel-critique"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleaned_up.is_set()
    assert memory_manager.get_or_create("cancel-critique").messages == [
        {"role": "user", "content": "question"},
    ]
    terminal_events = [event for event in events if event.type.value == "run.error"]
    assert len(terminal_events) == 1
    assert terminal_events[0].data["status"] == "canceled"
    assert terminal_events[0].step_index == 2


@pytest.mark.anyio
@pytest.mark.parametrize("candidate", [None, "", "   "])
async def test_runtime_rejects_non_text_or_blank_candidate_before_critique(
    candidate: object,
) -> None:
    critic_calls = 0

    async def execution_llm(messages, tools):
        return make_llm_response(candidate)  # type: ignore[arg-type]

    async def critic_llm(messages, tools):
        nonlocal critic_calls
        critic_calls += 1
        return _critic_response("accept", None)

    runtime, memory_manager = _runtime(
        execution_llm=execution_llm,
        critic_llm=critic_llm,
    )

    state = await runtime.run_async("question", session_id="empty-candidate")

    assert state.status is RunStatus.FAILED
    assert state.error_source == "llm"
    assert state.error_info is not None
    assert state.error_info["type"] == "LLMError"
    assert critic_calls == 0
    assert memory_manager.get_or_create(state.session_id).messages == [
        {"role": "user", "content": "question"},
    ]


@pytest.mark.anyio
async def test_streaming_reflection_never_emits_rejected_draft_tokens() -> None:
    stream_calls = 0
    critiques = iter(
        [
            ("revise", "Correct the answer."),
            ("accept", None),
        ]
    )
    emitted_tokens: list[str] = []

    async def execution_stream(messages, tools):
        nonlocal stream_calls
        stream_calls += 1
        chunks = ["bad ", "draft"] if stream_calls == 1 else ["good ", "answer"]
        for content in chunks:
            yield SimpleNamespace(
                content=content,
                tool_calls=None,
                finish_reason=None,
            )
        yield SimpleNamespace(
            content=None,
            tool_calls=None,
            finish_reason="stop",
        )

    async def critic_llm(messages, tools):
        decision, feedback = next(critiques)
        return _critic_response(decision, feedback)

    def handler(event) -> None:
        if event.type.value == "llm.token" and event.data.get("token"):
            emitted_tokens.append(event.data["token"])

    runtime, memory_manager = _runtime(
        execution_stream=execution_stream,
        critic_llm=critic_llm,
        handlers=[handler],
    )

    state = await runtime.run_async("question", session_id="stream-reflection")

    assert state.status is RunStatus.FINISHED
    assert "bad draft" not in "".join(emitted_tokens)
    assert "".join(emitted_tokens) in {"", "good answer"}
    assert memory_manager.get_or_create(state.session_id).messages[-1]["content"] == "good answer"


@pytest.mark.anyio
async def test_revision_context_survives_tool_execution_until_next_candidate() -> None:
    registry = ToolRegistry()

    @registry.register
    def calculate(expression: str) -> str:
        """Return test calculation evidence."""
        return "4"

    execution_messages: list[list[dict]] = []
    execution_calls = 0

    async def execution_llm(messages, tools):
        nonlocal execution_calls
        execution_calls += 1
        execution_messages.append(messages)
        if execution_calls == 1:
            return make_llm_response("The answer is 5.")
        if execution_calls == 2:
            return make_llm_response(
                "",
                tool_calls=[
                    make_tool_call(
                        "calculate-1",
                        "calculate",
                        '{"expression":"2+2"}',
                    )
                ],
            )
        return make_llm_response("The answer is 4.")

    critique_calls = 0

    async def critic_llm(messages, tools):
        nonlocal critique_calls
        critique_calls += 1
        if critique_calls == 1:
            return _critic_response("revise", "Use calculation evidence.")
        return _critic_response("accept", None)

    runtime, memory_manager = _runtime(
        execution_llm=execution_llm,
        critic_llm=critic_llm,
        registry=registry,
    )

    state = await runtime.run_async("What is 2 + 2?", session_id="revision-tool")

    assert state.status is RunStatus.FINISHED
    assert execution_calls == 3
    for messages in execution_messages[1:]:
        reflection_contexts = [
            message["content"]
            for message in messages
            if message.get("role") == "system"
            and "validated critic feedback" in message.get("content", "")
        ]
        assert len(reflection_contexts) == 1
        assert "The answer is 5." in reflection_contexts[0]
        assert "Use calculation evidence." in reflection_contexts[0]

    stored_messages = memory_manager.get_or_create(state.session_id).messages
    assert not any(
        message.get("role") == "assistant" and message.get("content") == "The answer is 5."
        for message in stored_messages
    )
    assert stored_messages[-1]["content"] == "The answer is 4."


@pytest.mark.anyio
async def test_router_receives_critic_feedback_when_selecting_revision_tools() -> None:
    queries: list[str] = []

    class RecordingRouter(AllToolsRouter):
        async def route(self, query, schemas):
            queries.append(query)
            return await super().route(query, schemas)

    answers = iter(["draft", "corrected"])

    async def execution_llm(messages, tools):
        return make_llm_response(next(answers))

    critiques = iter(
        [
            ("revise", "Use calculator evidence."),
            ("accept", None),
        ]
    )

    async def critic_llm(messages, tools):
        decision, feedback = next(critiques)
        return _critic_response(decision, feedback)

    runtime, _ = _runtime(
        execution_llm=execution_llm,
        critic_llm=critic_llm,
        router=RecordingRouter(),
    )

    state = await runtime.run_async("question", session_id="revision-routing")

    assert state.status is RunStatus.FINISHED
    assert len(queries) == 2
    assert "Use calculator evidence." in queries[1]


@pytest.mark.anyio
async def test_revision_routing_query_does_not_expand_past_an_accepted_limit() -> None:
    max_query_chars = 32
    queries: list[str] = []

    class BoundedRecordingRouter(AllToolsRouter):
        async def route(self, query, schemas):
            queries.append(query)
            if len(query) > max_query_chars:
                raise ToolRoutingError(
                    "tool routing query exceeds the configured size limit",
                    context={
                        "query_chars": len(query),
                        "max_query_chars": max_query_chars,
                    },
                )
            return await super().route(query, schemas)

    answers = iter(["draft", "corrected"])

    async def execution_llm(messages, tools):
        return make_llm_response(next(answers))

    critiques = iter(
        [
            ("revise", "Use calculator evidence."),
            ("accept", None),
        ]
    )

    async def critic_llm(messages, tools):
        decision, feedback = next(critiques)
        return _critic_response(decision, feedback)

    runtime, _ = _runtime(
        execution_llm=execution_llm,
        critic_llm=critic_llm,
        router=BoundedRecordingRouter(),
    )

    original_query = "q" * max_query_chars
    state = await runtime.run_async(original_query, session_id="bounded-revision-routing")

    assert state.status is RunStatus.FINISHED
    assert queries == [original_query, "Use calculator evidence."]


@pytest.mark.anyio
async def test_runtime_does_not_start_critic_without_remaining_step_budget() -> None:
    critic_calls = 0

    async def execution_llm(messages, tools):
        return make_llm_response("candidate")

    async def critic_llm(messages, tools):
        nonlocal critic_calls
        critic_calls += 1
        return _critic_response("accept", None)

    runtime, memory_manager = _runtime(
        execution_llm=execution_llm,
        critic_llm=critic_llm,
        max_steps=1,
    )

    state = await runtime.run_async("question", session_id="critic-step-budget")

    assert state.status is RunStatus.MAX_STEPS
    assert state.step_count == 1
    assert [step.kind for step in state.steps] == [StepKind.LLM_CALL]
    assert critic_calls == 0
    assert memory_manager.get_or_create(state.session_id).messages == [
        {"role": "user", "content": "question"},
    ]
