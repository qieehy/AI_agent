import random
import time

from openai import APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from config import get_settings
from errors import LLMError
from observability import logger


class LLMClient:
    def __init__(self, max_retries: int=3, model: str | None = None):
        self._settings = get_settings()
        self._model = model or self._settings.model
        self.client = OpenAI(api_key=self._settings.api_key, base_url=self._settings.base_url)
        self.max_retries = max_retries

    def __call__(self, message, tools):
        return self.chat(message, tools)

    def chat(self, messages, tools=None):
        for attempt in range(self.max_retries+1):
            try:
                response = self.client.chat.completions.create(model=self._model, messages=messages, tools=tools)
                return response
            except AuthenticationError as e:
                raise LLMError("LLM authentication failed - check API key",
                                context= {"model": self._model},
                ) from e
            except (APITimeoutError, RateLimitError, APIError) as e:
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(f"LLM retry {attempt + 1}/{self.max_retries} "
                                   f"| error={type(e).__name__} | delay={delay:.1f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"LLM all retries exhausted "
                                 f"| error={type(e).__name__}: {e}")
                    raise LLMError(f"LLM request failed after {self.max_retries} retries",
                                   context={"model": self._model, "error_type": type(e).__name__},
                                   ) from e
            except Exception as e:
                raise LLMError("LLM call failed unexpectedly",
                        context={
                            "model": self._model,
                            "exception_type": type(e).__name__,
                        },
                ) from e


    @staticmethod
    def _backoff_delay(attempt: int, base: float = 1.0) -> float:
        exponential: float = base * (2 ** attempt)
        jitter: float = random.uniform(0, base * 0.3)
        return exponential + jitter
