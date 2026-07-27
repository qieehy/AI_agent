from openai import OpenAI
from config import settings
from errors import LLMError
from openai import APIError, APITimeoutError, RateLimitError, AuthenticationError


class LLMClient:
    def __init__(self):
       self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def chat(self, messages, tools=None):
        try:
            response = self.client.chat.completions.create(model=settings.model, messages=messages, tools=tools)
            return response
        except APITimeoutError as e:
            raise LLMError("API Timeout error") from e
        except APIError as e:
            raise LLMError("API Error error") from e
        except AuthenticationError as e:
            raise LLMError("Authentication error") from e
        except RateLimitError as e:
            raise LLMError("API请求过于频繁") from e

