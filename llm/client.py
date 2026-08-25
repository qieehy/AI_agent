import asyncio
import random
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Literal

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config import get_settings
from errors import LLMError
from observability import logger


@dataclass
class StreamChunk:
    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None

_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (APITimeoutError, RateLimitError, APIError)

def _backoff_delay(attempt: int, base: float = 1.0) -> float:
    """指数退避, 分步重试"""
    exponential: float = base * (2 ** attempt)
    jitter: float = random.uniform(0, base * 0.3)
    return exponential + jitter

@dataclass(frozen=True)
class _ErrorDecision:
    kind: Literal["raise_auth", "retry", "raise_exhausted", "raise_unexpected"]
    error: LLMError | None = None
    delay: float | None = None


def _decide_on_error(e: Exception, *, attempt: int, max_retries: int, model: str) -> _ErrorDecision:
    if isinstance(e, AuthenticationError):
        error = LLMError("LLM authentication failed - check API key", context= {"model": model})
        return _ErrorDecision(kind="raise_auth", error = error)

    if isinstance(e, _RETRYABLE_EXCEPTIONS) :
        if attempt < max_retries:
            delay = _backoff_delay(attempt)
            logger.warning(f"LLM retry {attempt + 1}/{max_retries} "
                           f"| error={type(e).__name__} | delay={delay:.1f}s")
            return _ErrorDecision(kind="retry",  delay=delay)

        else:
            logger.error(f"LLM all retries exhausted "
                         f"| error={type(e).__name__}: {e}")
            error = LLMError(f"LLM request failed after {max_retries} retries",
                           context={"model": model, "error_type": type(e).__name__},
                           )
            return _ErrorDecision(kind="raise_exhausted", error = error)

    error = LLMError("LLM call failed unexpectedly",
             context={
                 "model": model,
                 "exception_type": type(e).__name__,
             },
             )
    return _ErrorDecision(kind = "raise_unexpected", error = error)


class LLMClient:
    def __init__(self, max_retries: int=3, model: str | None = None):
        self._settings = get_settings()
        self._model = model or self._settings.model
        self.client = OpenAI(api_key=self._settings.api_key, base_url=self._settings.base_url)
        self.max_retries = max_retries

    def __call__(self, message, tools=None):
        return self.chat(message, tools)

    def chat(self, messages, tools=None):
        for attempt in range(self.max_retries+1):
            try:
                response = self.client.chat.completions.create(model=self._model, messages=messages, tools=tools)
                return response
            except Exception as e:
                decision = _decide_on_error(e, attempt=attempt, max_retries=self.max_retries, model=self._model)
                if decision.kind == "retry":
                    time.sleep(decision.delay)
                    continue
                raise decision.error from e

    def _create_stream_with_retry(self, messages, tools):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                )

            except Exception as e:
                decision = _decide_on_error(e, attempt=attempt, max_retries=self.max_retries, model=self._model)
                if decision.kind == "retry":
                    time.sleep(decision.delay)
                    continue
                raise decision.error from e


    def stream(self, messages, tools=None) -> Iterator[StreamChunk]:
        """
            Stream LLM output.

            The LLM request is started lazily when the returned iterator
            is first consumed.
        """
        response = self._create_stream_with_retry(messages, tools)

        try:
            tool_frags: dict[int, dict] = {}

            for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                if delta.content:
                    yield StreamChunk(content=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_frags.setdefault(
                            tc.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            },
                        )
                        if tc.id:
                            slot["id"] = tc.id

                        if tc.function.name:
                            slot["function"]["name"] = tc.function.name

                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments


                if finish_reason is not None:
                    tool_calls = (
                        [tool_frags[index] for index in sorted(tool_frags)]
                        if tool_frags
                        else None
                    )

                    yield StreamChunk(
                        content=None,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                    )
                    return

        except Exception as e:
            raise LLMError(
                "LLM stream failed unexpectedly",
                context={
                    "model": self._model,
                    "exception_type": type(e).__name__,
                },
            ) from e


class AsyncLLMClient:
    def __init__(self, max_retries: int=3, model: str | None = None):
        self._settings = get_settings()
        self._model = model or self._settings.model
        self.client = AsyncOpenAI(api_key=self._settings.api_key, base_url=self._settings.base_url)
        self.max_retries = max_retries

    async def __call__(self, message, tools=None):
        return await self.chat(message, tools)

    async def chat(self, messages, tools=None):
        for attempt in range(self.max_retries+1):
            try:
                return await self.client.chat.completions.create(model=self._model, messages=messages, tools=tools)

            except Exception as e:
                decision = _decide_on_error(e, attempt=attempt, max_retries=self.max_retries, model=self._model)
                if decision.kind == "retry":
                    await asyncio.sleep(decision.delay)
                    continue
                raise decision.error from e

    async def stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        """
            Stream LLM output.

            The LLM request is started lazily when the returned iterator
            is first consumed.
        """
        response = await self._create_stream_with_retry(messages, tools)

        try:
            tool_frags: dict[int, dict] = {}

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                if delta.content:
                    yield StreamChunk(content=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_frags.setdefault(
                            tc.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            },
                        )
                        if tc.id:
                            slot["id"] = tc.id

                        if tc.function.name:
                            slot["function"]["name"] = tc.function.name

                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments


                if finish_reason is not None:
                    tool_calls = (
                        [tool_frags[index] for index in sorted(tool_frags)]
                        if tool_frags
                        else None
                    )

                    yield StreamChunk(
                        content=None,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                    )
                    return

        except Exception as e:
            raise LLMError(
                "LLM stream failed unexpectedly",
                context={
                    "model": self._model,
                    "exception_type": type(e).__name__,
                },
            ) from e


    async def _create_stream_with_retry(self, messages, tools):
        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                )

            except Exception as e:
                decision = _decide_on_error(e, attempt=attempt, max_retries=self.max_retries, model=self._model)
                if decision.kind == "retry":
                    await asyncio.sleep(decision.delay)
                    continue
                raise decision.error from e
