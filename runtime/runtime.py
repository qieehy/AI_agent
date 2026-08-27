from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from typing import Any, TypeAlias

from errors import AgentError, ConfigError, LLMError, ToolError
from memory import BufferMemory, MemoryManager
from observability import logger, set_trace_id
from runtime.event import Event, EventHandler, EventType
from runtime.loop_guard import LoopGuard, tool_call_signature
from runtime.state import RunStatus, RuntimeState
from runtime.step import Step, StepKind
from runtime.stop_reason import StopReason
from tools import Executor, ToolCallValidator, ToolResult
from tools.validator import ToolCallViolation

AsyncLLMCallable: TypeAlias = Callable[[list[dict], list[dict] | None], Awaitable[Any]]
AsyncLLMStreamCallable: TypeAlias = Callable[[list[dict], list[dict] | None], AsyncIterator[Any]]


class Runtime:
    """Agent Runtime 调度器"""

    def __init__(
            self,
            *,
            tool_executor: Executor,
            memory_manager: MemoryManager,
            llm_call_async: AsyncLLMCallable,
            llm_stream_async: AsyncLLMStreamCallable | None = None,
            handlers: list[EventHandler] | None = None,
            loop_guard: LoopGuard,
            validator: ToolCallValidator,
            system_prompt: str | None = None,
    ):
        self._llm_call_async = llm_call_async
        self._llm_stream_async = llm_stream_async
        self._handlers = handlers or []
        self._loop_guard = loop_guard
        self._validator = validator
        self._tool_executor = tool_executor
        self._memory_manager = memory_manager
        self._system_prompt = system_prompt

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
                  "error_source": state.error_source if state.status == RunStatus.FAILED else None,
                  "stop_reason": state.stop_reason.value if state.stop_reason else None,
                  },
            step_index=state.step_count
        ))
        logger.info(f"Run finished | status={state.status.value} | steps={state.step_count}")
        return state


    async def _run_steps_async(self, state: RuntimeState) -> None:
        step_result = self._loop_guard.check_steps(state.step_count)
        if step_result.detected:
            self._terminate(state, status=RunStatus.MAX_STEPS, stop_reason=StopReason.MAX_STEPS)
            return

        _memory = self._memory_manager.get_or_create(state.session_id)
        t0 = time.perf_counter()
        logger.debug(f"LLM request | step={state.step_count} | messages={len(_memory.messages)}")
        self._emit(Event(type=EventType.LLM_REQUEST, step_index=state.step_count))
        try:
            message = await self._get_llm_message_async(state=state, messages=_memory.get_context(), schemas=self._tool_executor.get_schemas())
        except AgentError as e:
            self._terminate(state, status=RunStatus.FAILED, stop_reason=StopReason.ERROR,
                            error=e, error_info=e.to_dict(), error_source="llm")
            raise

        except Exception as e:
            self._terminate(state, status=RunStatus.FAILED, stop_reason=StopReason.ERROR,
                            error=e, error_source="llm")
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

            # 循环守卫：记录本次尝试的调用签名，检测死循环
            state.tool_call_history.extend(
                tool_call_signature(tc) for tc in message.tool_calls
            )
            loop = self._loop_guard.check_tool_calls(state.tool_call_history)
            if loop.detected:
                self._terminate(state, status=RunStatus.LOOP_DETECTED,
                                stop_reason=StopReason.LOOP_DETECTED)
                return

            _memory.add_message(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            logger.info(f"Tool calls | step={state.step_count} | count={len(message.tool_calls)}")

            validation = self._validator.validate(message.tool_calls)

            if validation.violations:
                if not self._handle_validation_failure(
                        state=state,
                        memory=_memory,
                        violations=validation.violations,
                ):
                    return  # 预算耗尽，已终止 VALIDATION_FAILED
            else:
                state.validation_failure_rounds = 0

            if not validation.valid_calls:
                # 全违规且预算未耗尽：违规已回喂，模型下一轮自愈
                state.status = RunStatus.RUNNING
                return

            try:
                results = await self._tool_executor.execute_calls_async(validation.valid_calls)

            except AgentError as e:
                self._terminate(state, status=RunStatus.FAILED, stop_reason=StopReason.ERROR,
                                error=e, error_info=e.to_dict(), error_source="tool")
                raise

            except Exception as e:
                self._terminate(state, status=RunStatus.FAILED, stop_reason=StopReason.ERROR,
                                error=e, error_source="tool")
                raise

            self._process_tool_results(memory=_memory, state=state, results=results)

        else:
            _memory.add_message(message.model_dump() if hasattr(message, "model_dump") else dict(message))
            self._terminate(state, status=RunStatus.FINISHED, stop_reason=StopReason.FINISH_NORMAL)

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

        if results and failed_count == len(results):
            # 整轮工具全败：推进 tool error 预算，超限即终止（不再回喂）
            state.tool_error_rounds += 1
            if self._loop_guard.check_tool_error_budget(state.tool_error_rounds).detected:
                budget_error = ToolError(
                    f"tool error feedback budget exceeded: {state.tool_error_rounds} "
                    f"consecutive failed rounds (limit "
                    f"{self._loop_guard.policy.tool_error_feedback_rounds})"
                )
                self._terminate(state, status=RunStatus.FAILED, stop_reason=StopReason.ERROR,
                                error=budget_error, error_info=budget_error.to_dict(),
                                error_source="tool")
                return
            logger.warning(
                f"Tool full failure | {failed_count}/{len(results)} failed | "
                f"rounds={state.tool_error_rounds} | will retry"
            )
        else:
            state.tool_error_rounds = 0
            if failed_count > 0:
                logger.warning(f"Tool partial failure | {failed_count}/{len(results)} failed | will retry")
        state.status = RunStatus.RUNNING


    def _init_state(self, user_input: str, session_id: str | None = None) -> RuntimeState:
        session_id = session_id or str(uuid.uuid4())
        _current_memory = self._memory_manager.get_or_create(session_id)
        # 会话开始时注入 system prompt（后续轮次消息已存在，不再重复叠加）
        if self._system_prompt and not _current_memory.messages:
            _current_memory.add_message({"role": "system", "content": self._system_prompt})
        _current_memory.add_message({"role": "user", "content": user_input})
        return RuntimeState(
            session_id=session_id,
            status=RunStatus.RUNNING,
            messages=[],
        )

    def _emit(self, event: Event)->None:
        """广播事件"""
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error | handler={type(handler).__name__} "
                             f"error={type(e).__name__}: {e}")

    def _terminate(
            self,
            state: RuntimeState,
            *,
            status: RunStatus,
            stop_reason: StopReason,
            error: Exception | None = None,
            error_info: dict[str, Any] | None = None,
            error_source: str | None = None,
    ) -> None:
        """唯一终止出口：设置终态 + 归因，落一条结构化日志。

        所有终态（FINISHED / FAILED / MAX_STEPS / LOOP_DETECTED /
        VALIDATION_FAILED）都必须从这里出去，保证 stop_reason 永不遗漏。

        注意：本方法只负责状态与归因；终态事件（RUN_FINISH / RUN_ERROR）
        由 run_async 在生命周期端点统一发出。
        """
        state.status = status
        state.stop_reason = stop_reason
        state.error = error
        state.error_source = error_source
        if error_info is None and error is not None:
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "context": None,
                "cause": None,
            }
        state.error_info = error_info
        logger.info(
            f"Run terminated | status={status.value} | reason={stop_reason.value} "
            f"| steps={state.step_count} | session={state.session_id}"
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

    def _handle_validation_failure(
            self,
            *,
            state: RuntimeState,
            memory: BufferMemory,
            violations: list[ToolCallViolation],
    ) -> bool:
        """回喂验证违规给模型自愈；返回 False 表示预算耗尽已终止。

        每条违规对应一条带 tool_call_id 的 tool 消息——OpenAI 消息配对约束：
        assistant 的每个 tool_call 都必须有匹配的 tool 消息，否则下一轮请求被拒。
        非法调用不执行，只把原因喂回，模型下一轮自纠。
        """
        state.validation_failure_rounds += 1

        for v in violations:
            details = f": {v.details}" if v.details else ""
            memory.add_message({
                "role": "tool",
                "tool_call_id": v.tool_call_id,
                "name": v.name or "",
                "content": f"[VALIDATION ERROR] {v.reason}{details}",
            })
            logger.warning(
                f"Validation violation | call_id={v.tool_call_id} | "
                f"tool={v.name or '?'} | reason={v.reason} | "
                f"rounds={state.validation_failure_rounds}"
            )

        if self._loop_guard.check_validation_budget(state.validation_failure_rounds).detected:
            self._terminate(
                state,
                status=RunStatus.VALIDATION_FAILED,
                stop_reason=StopReason.VALIDATION_FAILED,
            )
            return False

        logger.info(
            f"Validation feedback sent | violations={len(violations)} | "
            f"rounds={state.validation_failure_rounds}"
        )
        return True
