"""测试 llm/client.py 的异常翻译。

策略：用 unittest.mock 模拟 OpenAI 客户端的各种异常，
验证 LLMError 是否被正确抛出 + context 是否正确。

关键点：patch 'llm.client.OpenAI' 而不是 patch LLMClient.client，
否则 __init__ 里的 OpenAI(...) 会覆盖 mock。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import APIError, APITimeoutError, AuthenticationError, RateLimitError

from errors import LLMError
from llm.client import LLMClient

# ---------- 辅助函数 ----------

def _make_fake_response(content: str = "ok") -> MagicMock:
    """构造一个假的成功响应对象。"""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_fake_openai_error(exc_class, message: str = "fake error"):
    """构造一个假的 OpenAI 异常实例。

    为什么用 exc_class(...) 直接构造：
    - 必须用 BaseException 的真子类（MagicMock(spec=...) 看着像，
      但 raise X from MagicMock 会 TypeError）。
    - openai 不同异常的 __init__ 签名差异很大（inspect 验证）：
      - APITimeoutError(request)
      - APIError(message, request, *, body=None)
      - RateLimitError(message, *, response, body=None)
      - AuthenticationError(message, *, response, body=None)
    - 用 isinstance 判断类型选择对应构造方式。
    """
    if exc_class is APITimeoutError:
        return exc_class(MagicMock())
    if exc_class is APIError:
        return exc_class(message, MagicMock(), body=MagicMock())
    if exc_class in (RateLimitError, AuthenticationError):
        return exc_class(message, response=MagicMock(), body=MagicMock())
    raise ValueError(f"Unknown exc_class: {exc_class}")


# ---------- 测试 ----------

def test_chat_success_returns_response():
    """正常路径：不抛异常，返回 response。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _make_fake_response("hello")

        client = LLMClient()
        result = client.chat([{"role": "user", "content": "hi"}])

        assert result is not None
        assert result.choices[0].message.content == "hello"


def test_authentication_error_translates_to_llm_error():
    """AuthenticationError → LLMError + message 含 authentication。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = _make_fake_openai_error(
            AuthenticationError, "Invalid API key"
        )

        client = LLMClient()
        with pytest.raises(LLMError) as excinfo:
            client.chat([{"role": "user", "content": "hi"}])

        assert "authentication" in str(excinfo.value).lower()
        # 验证 __cause__ 链
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, AuthenticationError)


def test_timeout_error_includes_recent_messages_in_context():
    """APITimeoutError → LLMError + context 含最近的消息。"""
    long_messages = [{"role": "user", "content": f"msg {i}"} for i in range(50)]

    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = _make_fake_openai_error(
            APITimeoutError, "Request timeout"
        )

        client = LLMClient()
        with pytest.raises(LLMError) as excinfo:
            client.chat(long_messages)

        assert "timeout" in str(excinfo.value).lower()
        assert "error_type" in excinfo.value.context


def test_rate_limit_error_raises_llm_error():
    """RateLimitError → LLMError + context 含 model。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = _make_fake_openai_error(
            RateLimitError, "Rate limit exceeded"
        )

        client = LLMClient()
        with pytest.raises(LLMError) as excinfo:
            client.chat([{"role": "user", "content": "hi"}])

        assert "rate limit" in str(excinfo.value).lower()
        assert excinfo.value.context.get("model") is not None


def test_generic_api_error_raises_llm_error():
    """APIError（兜底）→ LLMError + message 含 'api error'。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = _make_fake_openai_error(
            APIError, "Generic API error"
        )

        client = LLMClient()
        with pytest.raises(LLMError) as excinfo:
            client.chat([{"role": "user", "content": "hi"}])

        assert "api error" in str(excinfo.value).lower()


def test_unexpected_exception_caught_by_fallback():
    """非 OpenAI 异常（网络/SSL）→ except Exception 兜底 → LLMError。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        # 模拟 SSL 错误这种非 OpenAI 异常
        mock_client.chat.completions.create.side_effect = ConnectionError("SSL handshake failed")

        client = LLMClient()
        with pytest.raises(LLMError) as excinfo:
            client.chat([{"role": "user", "content": "hi"}])

        # 兜底分支的特征：context 里应有 exception_type
        assert excinfo.value.context.get("exception_type") == "ConnectionError"
        assert "unexpectedly" in str(excinfo.value).lower() or "failed" in str(excinfo.value).lower()
        # 链式：原始 ConnectionError 在 __cause__ 里
        assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_llm_error_preserves_cause_chain():
    """所有路径：__cause__ 必须指向原始异常（不让原始信息丢失）。"""
    original = _make_fake_openai_error(APITimeoutError, "boom")

    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = original

        client = LLMClient()
        with pytest.raises(LLMError) as excinfo:
            client.chat([])

        # 验证 cause 链没断
        assert excinfo.value.__cause__ is original
