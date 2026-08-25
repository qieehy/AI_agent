"""D23: AsyncLLMClient 契约测试（async chat / async stream）。

策略：
- patch 'llm.client.AsyncOpenAI'（同 test_llm_client.py：patch 构造器而不是实例属性）
- create 是 AsyncMock：await 后返回假 chunk 迭代器（async generator），
  chunk 形状对齐真实 SDK：chunk.choices[0].delta.{content, tool_calls} / finish_reason
- 只钉契约：文本按序 yield、None 跳过、tool_call 按 index 重组、
  重试只包连接建立（中途断流不重试）、异常翻译成 LLMError、finish 后终止
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from openai import APIError, AuthenticationError, RateLimitError

from errors import LLMError
from llm.client import AsyncLLMClient, _decide_on_error

# ---------- 假 SDK ----------

def _chunk(content=None, tool_calls=None, finish_reason=None):
    """构造一个 OpenAI 流式 chunk：delta 带 content/tool_calls，choice 带 finish_reason。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _tc_frag(index, *, call_id=None, name=None, arguments=None):
    """构造一个 tool_call 碎片（真实 SDK：name 每 fragment 重复完整值，arguments 分片）。"""
    return SimpleNamespace(
        index=index,
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _astream_of(*chunks):
    for c in chunks:
        yield c


def _api_error(message: str = "fake stream error"):
    return APIError(message, MagicMock(), body=MagicMock())


def _rate_limit_error(message: str = "fake rate limit"):
    return RateLimitError(message, response=MagicMock(), body=MagicMock())


def _auth_error(message: str = "fake auth error"):
    return AuthenticationError(message, response=MagicMock(), body=MagicMock())


async def _collect(client, messages, tools=None):
    """消费 async stream 并收集 chunk 列表。"""
    return [c async for c in client.stream(messages, tools)]


def _mock_async_client(mock_openai) -> AsyncMock:
    """把构造器 mock 的 return_value 换成 AsyncMock，使 create 链可 await。"""
    client = AsyncMock()
    mock_openai.return_value = client
    return client


# ---------- chat ----------

@pytest.mark.anyio
async def test_async_chat_success_returns_response():
    """正常路径：create 被 await，返回 response。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai:
        mock_client = _mock_async_client(mock_openai)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "hello"
        mock_client.chat.completions.create.return_value = response

        client = AsyncLLMClient(model="test-model")
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "hello"
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_async_chat_retries_then_succeeds():
    """可重试异常：退避后重试成功，sleep 收到递增 delay。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai, \
         patch("llm.client.asyncio.sleep", new=AsyncMock()) as mock_sleep, \
         patch("llm.client._backoff_delay", side_effect=[0.25, 0.5]) as mock_backoff:
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _rate_limit_error(),
            MagicMock(),
        ]

        client = AsyncLLMClient(model="test-model")
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert result is not None
    assert mock_client.chat.completions.create.await_count == 3
    assert mock_sleep.await_args_list == [call(0.25), call(0.5)]
    assert mock_backoff.call_count == 2


@pytest.mark.anyio
async def test_async_auth_error_no_retry():
    """AuthenticationError 是永久异常：直译 LLMError，不重试。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai:
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.side_effect = _auth_error()

        client = AsyncLLMClient(model="test-model")
        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": "hi"}])

    assert "authentication" in str(excinfo.value).lower()
    assert isinstance(excinfo.value.__cause__, AuthenticationError)
    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_async_chat_exhausts_retries():
    """可重试异常耗尽 max_retries：LLMError + context 含 error_type，cause 链不断。"""
    original = _api_error("always down")

    with patch("llm.client.AsyncOpenAI") as mock_openai, \
         patch("llm.client.asyncio.sleep", new=AsyncMock()):
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.side_effect = original

        client = AsyncLLMClient(max_retries=2, model="test-model")
        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": "hi"}])

    assert excinfo.value.context.get("error_type") == "APIError"
    assert excinfo.value.__cause__ is original
    assert mock_client.chat.completions.create.await_count == 3


# ---------- 纯决策函数 ----------

def test_decide_on_error_all_branches():
    """_decide_on_error 四分支语义：auth / retry / exhausted / unexpected。"""
    auth = _decide_on_error(_auth_error(), attempt=0, max_retries=3, model="m")
    assert auth.kind == "raise_auth"
    assert auth.error is not None

    retry = _decide_on_error(_rate_limit_error(), attempt=0, max_retries=3, model="m")
    assert retry.kind == "retry"
    assert retry.error is None
    assert 1.0 <= retry.delay <= 1.3  # attempt=0: base 1.0 + jitter [0, 0.3]

    exhausted = _decide_on_error(_rate_limit_error(), attempt=3, max_retries=3, model="m")
    assert exhausted.kind == "raise_exhausted"
    assert exhausted.error.context["error_type"] == "RateLimitError"

    unexpected = _decide_on_error(ConnectionError("SSL"), attempt=0, max_retries=3, model="m")
    assert unexpected.kind == "raise_unexpected"
    assert unexpected.error.context["exception_type"] == "ConnectionError"


# ---------- 异步流 ----------

@pytest.mark.anyio
async def test_async_stream_yields_content_chunks_in_order():
    """文本碎片按到达顺序 yield；None content 跳过；收尾 chunk 带 finish_reason。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai:
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.return_value = _astream_of(
            _chunk(content="你"),
            _chunk(content="好"),
            _chunk(),  # content/tool_calls/finish 全空：跳过
            _chunk(finish_reason="stop"),
        )

        chunks = await _collect(AsyncLLMClient(model="test-model"),
                                [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["你", "好", None]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.anyio
async def test_async_empty_choices_chunk_is_skipped():
    """choices 为空的 usage chunk 不崩、不产出。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai:
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.return_value = _astream_of(
            _chunk(content="前面"),
            SimpleNamespace(choices=[]),
            _chunk(content="后面"),
            _chunk(finish_reason="stop"),
        )

        chunks = await _collect(AsyncLLMClient(model="test-model"),
                                [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["前面", "后面", None]


@pytest.mark.anyio
async def test_async_tool_calls_reassembled_by_index_across_fragments():
    """工具碎片按 index 归位；name 每 fragment 重复携带（覆盖语义，不得累加翻倍）；
    arguments 分片拼接；id 取一次；与 content 交错不影响。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai:
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.return_value = _astream_of(
            _chunk(tool_calls=[_tc_frag(0, call_id="call_1", name="get_weather")]),
            _chunk(content="查到了"),
            _chunk(tool_calls=[
                _tc_frag(1, call_id="call_2", name="calc"),
                _tc_frag(0, name="get_weather", arguments='{"ci'),
            ]),
            _chunk(tool_calls=[
                _tc_frag(0, name="get_weather", arguments='ty": "上海"}'),
                _tc_frag(1, name="calc", arguments='{"expr": "1+2"}'),
            ]),
            _chunk(finish_reason="tool_calls"),
        )

        chunks = await _collect(AsyncLLMClient(model="test-model"),
                                [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["查到了", None]
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].tool_calls == [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "上海"}'}},
        {"id": "call_2", "type": "function",
         "function": {"name": "calc", "arguments": '{"expr": "1+2"}'}},
    ]


@pytest.mark.anyio
async def test_async_connect_rate_limit_retries_then_succeeds():
    """只有连接建立（create 调用）失败才重试：RateLimitError 退避后第二次成功。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai, \
         patch("llm.client.asyncio.sleep", new=AsyncMock()):
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _astream_of(_chunk(content="重试成功"), _chunk(finish_reason="stop")),
        ]

        chunks = await _collect(AsyncLLMClient(model="test-model"),
                                [{"role": "user", "content": "hi"}])

    assert mock_client.chat.completions.create.await_count == 2
    assert chunks[0].content == "重试成功"


@pytest.mark.anyio
async def test_async_mid_stream_error_wraps_llm_error_without_retry():
    """中途断流不重试：已打出的文字收不回，直接翻译成 LLMError。

    异常在迭代时抛（async generator 语义），pytest.raises 必须包住 async for。
    """
    async def broken():
        yield _chunk(content="前半句")
        raise _api_error("connection reset")

    with patch("llm.client.AsyncOpenAI") as mock_openai, \
         patch("llm.client.asyncio.sleep", new=AsyncMock()):
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.return_value = broken()

        client = AsyncLLMClient(model="test-model")
        with pytest.raises(LLMError):
            async for _c in client.stream([{"role": "user", "content": "hi"}]):
                pass

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.anyio
async def test_async_stream_stops_after_finish_chunk():
    """finish chunk 后迭代终止：继续 anext 得 StopAsyncIteration，且 create 只有 1 次。"""
    with patch("llm.client.AsyncOpenAI") as mock_openai:
        mock_client = _mock_async_client(mock_openai)
        mock_client.chat.completions.create.return_value = _astream_of(
            _chunk(content="答"),
            _chunk(finish_reason="stop"),
        )

        client = AsyncLLMClient(model="test-model")
        stream = client.stream([{"role": "user", "content": "hi"}])
        chunks = [c async for c in stream]
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    assert [c.content for c in chunks] == ["答", None]
    assert mock_client.chat.completions.create.await_count == 1
