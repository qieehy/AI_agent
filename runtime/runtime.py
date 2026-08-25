from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from types import SimpleNamespace
from typing import Any, TypeAlias

from errors import AgentError, ConfigError, LLMError
from memory import BufferMemory, MemoryManager
from observability import logger, set_trace_id
from runtime.event import Event, EventHandler, EventType
from runtime.state import RunStatus, RuntimeState
from runtime.step import Step, StepKind
from tools import Executor, ToolResult

LLMCallable: TypeAlias = Callable[[list[dict], list[dict] | None], Any]
LLMStreamCallable: TypeAlias = Callable[[list[dict], list[dict] | None], Iterator[Any]]
AsyncLLMCallable: TypeAlias = Callable[[list[dict], list[dict] | None], Awaitable[Any]]
AsyncLLMStreamCallable: TypeAlias = Callable[[list[dict], list[dict] | None], AsyncIterator[Any]]


class Runtime:
    """Agent Runtime 调度器"""

    def __init__(
            self,
            *,
            llm_call: LLMCallable,
            tool_executor: Executor,
            memory_manager: MemoryManager,
            llm_stream: LLMStreamCallable | None = None,
            llm_call_async: AsyncLLMCallable | None = None,
            llm_stream_async: AsyncLLMStreamCallable | None = None,
            handlers: list[EventHandler] | None = None,
            max_steps: int = 100,
    ):
        self._llm_stream = llm_stream
        self._llm_call = llm_call
        self._llm_call_async = llm_call_async
        self._llm_stream_async = llm_stream_async
        self._handlers = handlers or []
        self._max_steps = max_steps
        self._tool_executor = tool_executor
        self._memory_manager = memory_manager


    def run(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        state = self._init_state(user_input, session_id)
        set_trace_id(state.session_id)
        _memory = self._memory_manager.get_or_create(state.session_id)
        logger.info(f"Run started | session={state.session_id} | input={user_input[:100]}")
        self._emit(Event(type=EventType.RUN_START, data={"user_input": user_input}, step_index=0))

        while not state.is_terminal():
            try:
                self._run_steps(state)
            except AgentError:
                logger.error(f"Run aborted | step={state.step_count} | status={state.status.value}")
                break

        self._emit(Event(
            type=EventType.RUN_FINISH if state.status == RunStatus.FINISHED else EventType.RUN_ERROR,
            data={"final":_memory.messages[-1].get("content") if _memory.messages else None,
                  "error_source": state.error_source if state.status == RunStatus.FAILED else None
                  },
            step_index=state.step_count
        ))
        logger.info(f"Run finished | status={state.status.value} | steps={state.step_count}")
        return state





    def _run_steps(self, state: RuntimeState) -> None:
        if state.step_count >= state.max_steps:
            state.status = RunStatus.MAX_STEP
            return
        _memory = self._memory_manager.get_or_create(state.session_id)
        t0 = time.perf_counter()
        logger.debug(f"LLM request | step={state.step_count} | messages={len(_memory.messages)}")
        self._emit(Event(type=EventType.LLM_REQUEST, step_index=state.step_count))
        try:
            message = self._get_llm_message(state=state, messages=_memory.get_context(), schemas=self._tool_executor.get_schemas())
        except AgentError as e:
            self._mark_failed(state=state, error=e, info=e.to_dict(), source="llm")
            raise

        except Exception as e:
            self._mark_failed(state=state, error=e, source="llm")
            raise


        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM response | step={state.step_count} | duration={duration_ms:.0f}ms")
        self._emit(Event(type=EventType.LLM_RESPONSE , data={"message": message}, step_index=state.step_count))

        state.steps.append(Step(index=state.step_count, kind=StepKind.LLM_CALL,
             output=message.model_dump() if hasattr(message, "model_dump") else dict(message),
             duration_ms=duration_ms))
        state.step_count += 1

        if message.tool_calls:
            state.status = RunStatus.AWAITING_TOOL
            _memory.add_message(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            logger.info(f"Tool calls | step={state.step_count} | count={len(message.tool_calls)}")

            try:
                results = self._tool_executor.execute_calls(message.tool_calls)

            except AgentError as e:
                self._mark_failed(state=state, source="tool", error=e, info=e.to_dict())
                raise

            except Exception as e:
                self._mark_failed(state, error=e, source="tool")
                raise

            self._process_tool_results(memory=_memory, state=state, results=results)

        else:
            _memory.add_message(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            state.status = RunStatus.FINISHED

    async def run_async(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        if self._llm_call_async is None and self._llm_stream_async is None:
            raise ConfigError("run_async 需要 llm_call_async 或 llm_stream_async (接线错误)")

        state = self._init_state(user_input, session_id)
        set_trace_id(state.session_id)
        _memory = self._memory_manager.get_or_create(state.session_id)
        logger.info(f"Run started | session={state.session_id} | input={user_input[:100]}")
        self._emit(Event(type=EventType.RUN_START, data={"user_input": user_input}, step_index=0))

        while not state.is_terminal():
            try:
                await self._run_steps_async(state)
            except AgentError:
                logger.error(f"Run aborted | step={state.step_count} | status={state.status.value}")
                break

        self._emit(Event(
            type=EventType.RUN_FINISH if state.status == RunStatus.FINISHED else EventType.RUN_ERROR,
            data={"final":_memory.messages[-1].get("content") if _memory.messages else None,
                  "error_source": state.error_source if state.status == RunStatus.FAILED else None
                  },
            step_index=state.step_count
        ))
        logger.info(f"Run finished | status={state.status.value} | steps={state.step_count}")
        return state


    async def _run_steps_async(self, state: RuntimeState) -> None:
        if state.step_count >= state.max_steps:
            state.status = RunStatus.MAX_STEP
            return
        _memory = self._memory_manager.get_or_create(state.session_id)
        t0 = time.perf_counter()
        logger.debug(f"LLM request | step={state.step_count} | messages={len(_memory.messages)}")
        self._emit(Event(type=EventType.LLM_REQUEST, step_index=state.step_count))
        try:
            message = await self._get_llm_message_async(state=state, messages=_memory.get_context(), schemas=self._tool_executor.get_schemas())
        except AgentError as e:
            self._mark_failed(state=state, error=e, info=e.to_dict(), source="llm")
            raise

        except Exception as e:
            self._mark_failed(state=state, error=e, source="llm")
            raise


        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM response | step={state.step_count} | duration={duration_ms:.0f}ms")
        self._emit(Event(type=EventType.LLM_RESPONSE , data={"message": message}, step_index=state.step_count))

        state.steps.append(Step(index=state.step_count, kind=StepKind.LLM_CALL,
             output=message.model_dump() if hasattr(message, "model_dump") else dict(message),
             duration_ms=duration_ms))
        state.step_count += 1

        if message.tool_calls:
            state.status = RunStatus.AWAITING_TOOL
            _memory.add_message(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            logger.info(f"Tool calls | step={state.step_count} | count={len(message.tool_calls)}")

            try:
                results = await self._tool_executor.execute_calls_async(message.tool_calls)

            except AgentError as e:
                self._mark_failed(state=state, source="tool", error=e, info=e.to_dict())
                raise

            except Exception as e:
                self._mark_failed(state, error=e, source="tool")
                raise

            self._process_tool_results(memory=_memory, state=state, results=results)

        else:
            _memory.add_message(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            state.status = RunStatus.FINISHED

    def _process_tool_results(self, state, memory: BufferMemory, results: list[ToolResult]) -> None:
        for r in results:
            tool_msg = {
                "role": "tool",
                "tool_call_id": r.tool_call.id,
                "name": r.tool_call.function.name,
                "content": r.content if r.status == "success" else f"[ERROR {r.error_type}] {r.error}",
            }
            state.steps.append(Step(index=state.step_count, kind=StepKind.TOOL_EXEC,
                                    output=tool_msg,
                                    duration_ms=r.duration_ms))
            state.step_count += 1
            memory.add_message(tool_msg)
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
            logger.warning(f"Tool partial failure | {failed_count}/{len(results)} failed | will retry")
        state.status = RunStatus.RUNNING


    def _init_state(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        session_id = session_id or str(uuid.uuid4())
        _current_memory = self._memory_manager.get_or_create(session_id)
        _current_memory.add_message({"role": "user", "content": user_input})
        return RuntimeState(
            session_id=session_id,
            status=RunStatus.RUNNING,
            messages=[],
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
    def _mark_failed(state, info: dict[str, Any] | None = None, source="unknown", error=None):
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

    def _get_llm_message(
            self,
            state: RuntimeState,
            messages: list[dict],
            schemas: list[dict] | None,
    ):
        if self._llm_stream is None:
            response = self._llm_call(messages, schemas)
            return response.choices[0].message

        parts: list[str] = []

        for chunk in self._llm_stream(messages, schemas):
            if chunk.content:
                parts.append(chunk.content)

                self._emit(Event(
                    type=EventType.LLM_TOKEN,
                    data={"token": chunk.content},
                    step_index=state.step_count,
                ))

            if chunk.finish_reason is not None:
                text = "".join(parts) or None
                tool_calls = chunk.tool_calls

                if tool_calls:
                    self._emit(Event(
                        type=EventType.LLM_TOKEN,
                        data={
                            "token": None,
                            "tool_calls": tool_calls,
                        },
                        step_index=state.step_count,
                    ))

                return self._stream_message(text=text, tool_calls=tool_calls)

        raise LLMError(
            "LLM stream ended without finish chunk"
        )

    async def _get_llm_message_async(
            self,
            state: RuntimeState,
            messages: list[dict],
            schemas: list[dict] | None,
    ):
        if self._llm_stream_async is None:
            if self._llm_call_async is None:
                raise ConfigError("run_async 需要 llm_call_async 或 llm_stream_async (接线错误)")
            response = await self._llm_call_async(messages, schemas)
            return response.choices[0].message

        parts: list[str] = []

        async for chunk in self._llm_stream_async(messages, schemas):
            if chunk.content:
                parts.append(chunk.content)

                self._emit(Event(
                    type=EventType.LLM_TOKEN,
                    data={"token": chunk.content},
                    step_index=state.step_count,
                ))

            if chunk.finish_reason is not None:
                text = "".join(parts) or None
                tool_calls = chunk.tool_calls

                if tool_calls:
                    self._emit(Event(
                        type=EventType.LLM_TOKEN,
                        data={
                            "token": None,
                            "tool_calls": tool_calls,
                        },
                        step_index=state.step_count,
                    ))

                return self._stream_message(text=text, tool_calls=tool_calls)

        raise LLMError(
            "LLM stream ended without finish chunk"
        )

    @staticmethod
    def _stream_message(text: str | None, tool_calls: list[dict] | None) -> SimpleNamespace:
        """流式产物 → 鸭子 message（.content/.tool_calls/.model_dump() 对齐 SDK message）。"""
        return SimpleNamespace(
            content=text,
            tool_calls=[
                SimpleNamespace(
                    id=tc["id"],
                    type=tc["type"],
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ] if tool_calls else None,
            model_dump=lambda: {
                "role": "assistant",
                "content": text,
                "tool_calls": tool_calls,
            },
        )
