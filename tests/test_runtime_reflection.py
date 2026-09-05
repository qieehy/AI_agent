from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime import LoopGuard, Runtime, StepKind
from runtime.policy import LoopPolicy
from runtime.reflection import Critic
from runtime.state import RunStatus
from tools import Executor, ToolCallValidator, ToolRegistry

from .conftest import AllToolsRouter, make_llm_response, make_memory


def _critic_response(decision: str, feedback: str | None) -> SimpleNamespace:
    import json

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
    execution_llm,
    *,
    critic_llm=None,
    reflection_revision_rounds: int = 1,
    events: list | None = None,
):
    registry = ToolRegistry()
    executor = Executor(registry, mode="serial")
    memory_manager = make_memory()
    runtime = Runtime(
        llm_call_async=execution_llm,
        tool_executor=executor,
        memory_manager=memory_manager,
        handlers=[events.append] if events is not None else [],
        loop_guard=LoopGuard(
            LoopPolicy(
                reflection_revision_rounds=reflection_revision_rounds,
            )
        ),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=AllToolsRouter(),
        critic=Critic(critic_llm) if critic_llm is not None else None,
    )
    return runtime, memory_manager


@pytest.mark.anyio
async def test_runtime_commits_candidate_only_after_critic_accepts() -> None:
    events: list = []

    async def execution_llm(messages, tools):
        return make_llm_response("approved answer")

    async def critic_llm(messages, tools):
        return _critic_response("accept", None)

    runtime, memory_manager = _runtime(
        execution_llm,
        critic_llm=critic_llm,
        events=events,
    )

    state = await runtime.run_async("question", session_id="accept-session")

    assert state.status is RunStatus.FINISHED
    assert state.reflection_revision_rounds == 0
    assert state.step_count == 2
    assert [step.index for step in state.steps] == [0, 1]
    assert [step.kind for step in state.steps] == [StepKind.LLM_CALL, StepKind.CRITIQUE]
    critique_step = state.steps[1]
    assert critique_step.input == {"candidate_step_index": 0, "round": 1}
    assert critique_step.output == {"decision": "accept"}
    assert critique_step.error is None
    assert critique_step.duration_ms >= 0
    assert memory_manager.get_or_create(state.session_id).messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "approved answer"},
    ]

    critique_events = [event for event in events if event.type.value == "critique.completed"]
    assert len(critique_events) == 1
    assert critique_events[0].data["decision"] == "accept"
    assert critique_events[0].data["round"] == 1
    assert critique_events[0].step_index == 1
    assert critique_events[0].data["duration_ms"] >= 0
    assert "feedback" not in critique_events[0].data
    terminal_events = [event for event in events if event.type.value == "run.finish"]
    assert len(terminal_events) == 1
    assert terminal_events[0].data["final"] == "approved answer"


@pytest.mark.anyio
async def test_runtime_revises_with_transient_feedback_then_commits_final_answer() -> None:
    execution_calls: list[list[dict]] = []
    events: list = []
    execution_answers = iter(["draft answer", "corrected answer"])
    critique_results = iter(
        [
            ("revise", "Correct the arithmetic."),
            ("accept", None),
        ]
    )

    async def execution_llm(messages, tools):
        execution_calls.append(messages)
        return make_llm_response(next(execution_answers))

    async def critic_llm(messages, tools):
        decision, feedback = next(critique_results)
        return _critic_response(decision, feedback)

    runtime, memory_manager = _runtime(
        execution_llm,
        critic_llm=critic_llm,
        events=events,
    )

    state = await runtime.run_async("question", session_id="revise-session")

    assert state.status is RunStatus.FINISHED
    assert state.reflection_revision_rounds == 1
    assert [step.index for step in state.steps] == [0, 1, 2, 3]
    assert [step.kind for step in state.steps] == [
        StepKind.LLM_CALL,
        StepKind.CRITIQUE,
        StepKind.LLM_CALL,
        StepKind.CRITIQUE,
    ]
    critique_events = [event for event in events if event.type.value == "critique.completed"]
    assert [event.step_index for event in critique_events] == [1, 3]
    assert [event.data["candidate_step_index"] for event in critique_events] == [0, 2]
    assert len(execution_calls) == 2

    revision_context = [
        message
        for message in execution_calls[1]
        if message.get("role") == "system"
        and "validated critic feedback" in message.get("content", "")
    ]
    assert len(revision_context) == 1
    assert "draft answer" in revision_context[0]["content"]
    assert "Correct the arithmetic." in revision_context[0]["content"]

    messages = memory_manager.get_or_create(state.session_id).messages
    assert messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "corrected answer"},
    ]


@pytest.mark.anyio
async def test_runtime_fails_closed_when_revision_budget_is_exhausted() -> None:
    execution_calls = 0
    critic_calls = 0
    events: list = []

    async def execution_llm(messages, tools):
        nonlocal execution_calls
        execution_calls += 1
        return make_llm_response(f"draft {execution_calls}")

    async def critic_llm(messages, tools):
        nonlocal critic_calls
        critic_calls += 1
        return _critic_response("revise", "Still incorrect.")

    runtime, memory_manager = _runtime(
        execution_llm,
        critic_llm=critic_llm,
        reflection_revision_rounds=1,
        events=events,
    )

    state = await runtime.run_async("question", session_id="budget-session")

    assert state.status.value == "reflection_limit"
    assert state.stop_reason is not None
    assert state.stop_reason.value == "reflection_limit"
    assert state.reflection_revision_rounds == 1
    assert execution_calls == 2
    assert critic_calls == 2
    assert memory_manager.get_or_create(state.session_id).messages == [
        {"role": "user", "content": "question"},
    ]
    terminal_events = [event for event in events if event.type.value == "run.error"]
    assert len(terminal_events) == 1
    assert terminal_events[0].data["final"] is None


@pytest.mark.anyio
async def test_runtime_critic_failure_does_not_commit_candidate() -> None:
    events: list = []

    async def execution_llm(messages, tools):
        return make_llm_response("unreviewed draft")

    async def critic_llm(messages, tools):
        raise RuntimeError("critic provider unavailable")

    runtime, memory_manager = _runtime(
        execution_llm,
        critic_llm=critic_llm,
        events=events,
    )

    state = await runtime.run_async("question", session_id="failure-session")

    assert state.status is RunStatus.FAILED
    assert state.error_source == "critic"
    assert state.error_info is not None
    assert state.error_info["type"] == "ReflectionError"
    assert state.steps[-1].kind is StepKind.CRITIQUE
    assert state.steps[-1].output is None
    assert state.steps[-1].error == "ReflectionError"
    assert memory_manager.get_or_create(state.session_id).messages == [
        {"role": "user", "content": "question"},
    ]
    terminal_events = [event for event in events if event.type.value == "run.error"]
    assert len(terminal_events) == 1
    assert terminal_events[0].data["final"] is None


@pytest.mark.anyio
async def test_runtime_without_critic_preserves_existing_completion_path() -> None:
    async def execution_llm(messages, tools):
        return make_llm_response("ordinary answer")

    runtime, memory_manager = _runtime(execution_llm)

    state = await runtime.run_async("question", session_id="ordinary-session")

    assert state.status is RunStatus.FINISHED
    assert memory_manager.get_or_create(state.session_id).messages[-1] == {
        "role": "assistant",
        "content": "ordinary answer",
    }
