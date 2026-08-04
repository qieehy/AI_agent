from __future__ import annotations
import time
import uuid
from typing import Any, Callable, TypeAlias

from errors import AgentError
from observability import logger, set_trace_id
from runtime.state import RuntimeState, RunStatus
from runtime.step import Step, StepKind
from runtime.event import Event, EventType, EventHandler
from tools import Executor

LLMCallable: TypeAlias = Callable[[list[dict], list[dict] | None], Any]


class Runtime:
    """Agent Runtime 调度器"""

    def __init__(
            self,
            llm_call: LLMCallable,
            tool_executor: Executor,
            handlers: list[EventHandler] | None = None,
            max_steps: int = 100,
    ):
        self._llm_call = llm_call
        self._handlers = handlers or []
        self._max_steps = max_steps
        self._tool_executor = tool_executor

    def run(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        state = self._init_state(user_input, session_id)
        set_trace_id(state.session_id)
        logger.info(f"Run started | session={state.session_id} | input={user_input[:100]}")
        self._emit(Event(type=EventType.RUN_START, data={"user_input": user_input}, step_index=0))

        while not state.is_terminal():
            try:
                self._step(state)
            except AgentError:
                logger.error(f"Run aborted | step={state.step_count} | status={state.status.value}")
                break

        self._emit(Event(
            type=EventType.RUN_FINISH if state.status == RunStatus.FINISHED else EventType.RUN_ERROR,
            data={"final": state.messages[-1].get("content") if state.messages else None,
                  "error_source": state.error_source if state.status == RunStatus.FAILED else None
                  },
            step_index=state.step_count
        ))
        logger.info(f"Run finished | status={state.status.value} | steps={state.step_count}")
        return state


    def _step(self, state: RuntimeState) -> None:
        if state.step_count >= state.max_steps:
            state.status = RunStatus.FINISHED
            return

        t0 = time.perf_counter()
        logger.debug(f"LLM request | step={state.step_count} | messages={len(state.messages)}")
        self._emit(Event(type=EventType.LLM_REQUEST, step_index=state.step_count))
        try:
            response = self._llm_call(state.messages, self._tool_executor.get_schemas())

        except AgentError as e:
            self._mark_failed(state=state, error=e, info=e.to_dict(), source="llm")
            raise

        except Exception as e:
            self._mark_failed(state=state, error=e)
            raise


        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM response | step={state.step_count} | duration={duration_ms:.0f}ms")
        self._emit(Event(type=EventType.LLM_RESPONSE , data={"response": response}, step_index=state.step_count))

        message = response.choices[0].message
        Step(index=state.step_count, kind=StepKind.LLM_CALL,
             output=message.model_dump() if hasattr(message, "model_dump") else dict(message),
             duration_ms=(time.perf_counter() - t0) * 1000,)
        state.step_count += 1

        if message.tool_calls:
            state.status = RunStatus.AWAITING_TOOL
            state.messages.append(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            logger.info(f"Tool calls | step={state.step_count} | count={len(message.tool_calls)}")

            try:
                results = self._tool_executor.execute_calls(message.tool_calls)

            except AgentError as e:
                self._mark_failed(state=state, source="tool", error=e, info=e.to_dict())
                raise

            except Exception as e:
                self._mark_failed(state, error=e, source="tool")
                raise

            for r in results:
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": r.tool_call.id,
                    "name": r.tool_call.function.name,
                    "content": r.content if r.status == "success" else f"[ERROR {r.error_type}] {r.error}",
                })
                if r.status == "failed":
                    logger.warning(f"Tool failed | name={r.tool_call.function.name} "
                                   f"error={r.error_type}: {r.error}")
                else:
                    logger.info(f"Tool success | name={r.tool_call.function.name} "
                                f"duration={r.duration_ms:.0f}ms")
                self._emit(Event(type=EventType.TOOL_RESPONSE, data={
                    "name": r.tool_call.function.name,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "result": str(r.content)[:200] if r.status == "success" else None,
                    "error": r.error,
                    "error_type": r.error_type,
                }, step_index=state.step_count))

            failed_count = sum(1 for r in results if r.status == "failed")
            if failed_count > 0:
                info = {
                    "type": "ToolError",
                    "message": f"{failed_count}/{len(results)} tool(s) failed",
                    "context": {
                        "failed_calls": [
                            {"name": r.tool_call.function.name, "error": r.error, "error_type": r.error_type}
                            for r in results if r.status == "failed"
                        ],
                    },
                    "cause": None,
                }
                self._mark_failed(state=state, source="tool", info=info)
                return
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
            except Exception as e:
                logger.error(f"Event handler error | handler={type(handler).__name__} "
                             f"error={type(e).__name__}: {e}")

    @staticmethod
    def _mark_failed(state, info: dict[str, Any]=None, source="unknown", error=None):
        """集中标记 FAILED 状态。"""

        state.error_info = info or {
            "type": type(error).__name__,
            "message": None,
            "context": None,
            "cause": None,
        }
        state.status = RunStatus.FAILED
        state.error = error
        state.error_source = source

