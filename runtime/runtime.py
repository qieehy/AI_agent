from __future__ import annotations
import time
import uuid
from typing import Any, Callable, TypeAlias

from runtime.state import RuntimeState, RunStatus
from runtime.step import Step, StepKind
from runtime.event import Event, EventType, EventHandler

LLMCallable: TypeAlias = Callable[[list[dict], list[dict] | None], Any]
ToolCallable: TypeAlias = Callable[[str, dict], Any]

class Runtime:
    """Agent Runtime 调度器"""

    def __init__(
            self,
            llm_call: LLMCallable,
            tool_call: ToolCallable,
            handlers: list[EventHandler] | None = None,
            max_steps: int = 100,
    ):
        self._llm_call = llm_call
        self._tool_call = tool_call
        self._handlers = handlers or []
        self._max_steps = max_steps

    def run(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        state = self._init_state(user_input, session_id)
        self._emit(Event(type=EventType.RUN_START, data={"user_input": user_input}, step_index=0))

        while not state.is_terminal():
            try:
                self._step(state)
            except Exception as e:
                state.status = RunStatus.FAILED
                state.error = e
                self._emit(Event(type=EventType.RUN_ERROR, data={"error": str(e)}, step_index=state.step_count))
                break

        self._emit(Event(
            type=EventType.RUN_FINISH if state.status == RunStatus.FINISHED else EventType.RUN_ERROR,
            data={"final": state.messages[-1].get("content") if state.messages else None},
            step_index=state.step_count
        ))
        return state



    def _step(self, state: RuntimeState) -> None:
        if state.step_count >= state.max_steps:
            state.status = RunStatus.FINISHED
            return

        t0 = time.perf_counter()
        self._emit(Event(type=EventType.LLM_REQUEST, step_index=state.step_count))
        response = self._llm_call(state.messages, None)
        self._emit(Event(type=EventType.LLM_RESPONSE, data={"response": response}, step_index=state.step_count))

        message = response.choices[0].message
        Step(index=state.step_count, kind=StepKind.LLM_CALL,
             output=message.model_dump() if hasattr(message, "model_dump") else dict(message),
             duration_ms=(time.perf_counter() - t0) * 1000,)
        state.step_count += 1

        if message.tool_calls:
            state.status = RunStatus.AWAITING_TOOL
            state.messages.append(message.model_dump() if hasattr(message, "model_dump") else dict(message))

            for tool_call in message.tool_calls:
                self._emit(Event(
                    type=EventType.TOOL_REQUEST,
                    data={"name": tool_call.function_name, "args": tool_call.function.arguments},
                    step_index=state.step_count
                ))
                t1 = time.perf_counter()
                result = self._tool_call(tool_call.function.name, tool_call.function.arguments)
                Step(index=state.step_count, kind=StepKind.TOOL_EXEC,
                     input={"name": tool_call.function.name}, output=result, duration_ms=(time.perf_counter() - t1) * 1000)
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(result),
                })
                self._emit(Event(
                    type=EventType.TOOL_RESPONSE,
                    data={"name": tool_call.function.name, "result": str(result)[:200]},
                    step_index=state.step_count,
                ))
            state.status = RunStatus.RUNNING
        else:
            state.messages.append(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            state.status = RunStatus.FINISHED




    def _init_state(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        return RuntimeState(
            session_id=session_id or str(uuid.uuid4()),
            status=RunStatus.RUNNING,
            messages=[{"role": "user", "content": user_input}],
            max_steps=self._max_steps,
        )

    def _emit(self, event: Event)->None:
        """广播事件"""
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass