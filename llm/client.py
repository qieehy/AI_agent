from openai import OpenAI
from config import settings
from errors import LLMError
from openai import APIError, APITimeoutError, RateLimitError, AuthenticationError


class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def __call__(self, message, tools):
        return self.chat(message, tools)

    def chat(self, messages, tools=None):
        try:
            response = self.client.chat.completions.create(model=settings.model, messages=messages, tools=tools)
            return response
        except AuthenticationError as e:
            raise LLMError("LLM authentication failed - check API key",
                            context= {"model": settings.model},
            ) from e
        except APITimeoutError as e:
            raise LLMError("LLM request timeout",
                        context={
                            "model": settings.model,
                            "messages": messages[-20:]
                        },
            ) from e
        except RateLimitError as e:
            raise LLMError("LLM rate limit exceeded",
                        context={
                           "model": settings.model,
                        }
            ) from e

        except APIError as e:
            raise LLMError("LLM API error",
                            context={"model": settings.model},
                           ) from e
        except Exception as e:
            raise LLMError("LLM call failed unexpectedly",
                    context={
                        "model": settings.model,
                        "exception_type": type(e).__name__,
                    },
            ) from e