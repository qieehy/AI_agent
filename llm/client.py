import time
import random

from openai import OpenAI
from config import settings
from errors import LLMError
from openai import APIError, APITimeoutError, RateLimitError, AuthenticationError


class LLMClient:
    def __init__(self, max_retries: int=3):
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.max_retries = max_retries

    def __call__(self, message, tools):
        return self.chat(message, tools)

    def chat(self, messages, tools=None):
        for attempt in range(self.max_retries+1):
            try:
                response = self.client.chat.completions.create(model=settings.model, messages=messages, tools=tools)
                return response
            except AuthenticationError as e:
                raise LLMError("LLM authentication failed - check API key",
                                context= {"model": settings.model},
                ) from e
            except (APITimeoutError, RateLimitError, APIError) as e:
                if attempt < self.max_retries:
                    time.sleep(self._backoff_delay(attempt))
                else:
                    raise LLMError(f"LLM request failed after {self.max_retries} retries",
                                   context={"model": settings.model, "error_type": type(e).__name__},
                                   ) from e
            except Exception as e:
                raise LLMError("LLM call failed unexpectedly",
                        context={
                            "model": settings.model,
                            "exception_type": type(e).__name__,
                        },
                ) from e


    @staticmethod
    def _backoff_delay(attempt: int, base: float = 1.0) -> float:
        return base * (2 ** attempt) + random.uniform(0, base*0.3)