"""D22: LLMClient.stream 契约测试（test-first：stream 方法落地前为红）。

策略：
- patch 'llm.client.OpenAI'（同 test_llm_client.py：patch 构造器而不是实例属性）
- create(stream=True) 返回假 chunk 迭代器，chunk 形状对齐真实 SDK：
  chunk.choices[0].delta.{content, tool_calls} / chunk.choices[0].finish_reason
- 只钉契约：文本按序 yield、None 跳过、tool_call 按 index 重组、
  重试只包连接建立（中途断流不重试）、异常翻译成 LLMError
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import APIError, AuthenticationError, RateLimitError

from errors import LLMError
from llm.client import LLMClient

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


def _stream_of(*chunks):
    return iter(chunks)


def _api_error(message: str = "fake stream error"):
    return APIError(message, MagicMock(), body=MagicMock())


def _rate_limit_error(message: str = "fake rate limit"):
    return RateLimitError(message, response=MagicMock(), body=MagicMock())


def _auth_error(message: str = "fake auth error"):
    return AuthenticationError(message, response=MagicMock(), body=MagicMock())


def _collect(client, messages, tools=None):
    """迭代 stream 并收集 chunk 列表。"""
    return list(client.stream(messages, tools))


# ---------- 纯文本流 ----------

def test_stream_yields_content_chunks_in_order():
    """文本碎片按到达顺序 yield；None content 跳过；收尾 chunk 带 finish_reason。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _stream_of(
            _chunk(content="你"),
            _chunk(content="好"),
            _chunk(),  # content/tool_calls/finish 全空：跳过
            _chunk(finish_reason="stop"),
        )

        chunks = _collect(LLMClient(model="test-model"), [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["你", "好", None]
    assert chunks[-1].finish_reason == "stop"


def test_empty_choices_chunk_is_skipped():
    """choices 为空的 usage chunk 不崩、不产出。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        empty = SimpleNamespace(choices=[])
        mock_client.chat.completions.create.return_value = _stream_of(
            _chunk(content="前面"),
            empty,
            _chunk(content="后面"),
            _chunk(finish_reason="stop"),
        )

        chunks = _collect(LLMClient(model="test-model"), [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["前面", "后面", None]


def test_final_chunk_tool_calls_none_when_no_tools():
    """没有工具调用时，收尾 chunk 的 tool_calls 必须是 None（不是空 list）。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _stream_of(
            _chunk(content="纯文本答案"),
            _chunk(finish_reason="stop"),
        )

        chunks = _collect(LLMClient(model="test-model"), [{"role": "user", "content": "hi"}])

    assert chunks[-1].tool_calls is None


def test_content_and_finish_reason_on_same_chunk():
    """真实 SDK：最后一段 content 与 finish_reason 可能同 chunk —— content 先 yield，finish 收尾。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _stream_of(
            _chunk(content="答"),
            _chunk(content="案", finish_reason="stop"),
        )

        chunks = _collect(LLMClient(model="test-model"), [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["答", "案", None]
    assert chunks[-1].finish_reason == "stop"


# ---------- 请求参数 ----------

def test_create_receives_stream_true_model_messages_tools():
    """create 必须收到 stream=True；model 在请求参数里（SDK 约定），messages/tools 透传。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = _stream_of(_chunk(finish_reason="stop"))

        tools = [{"type": "function", "function": {"name": "echo"}}]
        messages = [{"role": "user", "content": "hi"}]
        _collect(LLMClient(model="test-model"), messages, tools=tools)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == messages
        assert kwargs["tools"] == tools


# ---------- tool_call 重组 ----------

def test_tool_calls_reassembled_by_index_across_fragments():
    """工具碎片按 index 归位；name 每 fragment 重复携带（覆盖语义，不得累加翻倍）；
    arguments 分片拼接；id 取一次；与 content 交错不影响。"""
    with patch("llm.client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _stream_of(
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

        chunks = _collect(LLMClient(model="test-model"), [{"role": "user", "content": "hi"}])

    assert [c.content for c in chunks] == ["查到了", None]
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].tool_calls == [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "上海"}'}},
        {"id": "call_2", "type": "function",
         "function": {"name": "calc", "arguments": '{"expr": "1+2"}'}},
    ]


# ---------- 重试边界 ----------

def test_connect_rate_limit_retries_then_succeeds():
    """只有连接建立（create 调用）失败才重试：RateLimitError 退避后第二次成功。"""
    with patch("llm.client.OpenAI") as mock_openai, patch("llm.client.time.sleep"):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _stream_of(_chunk(content="重试成功"), _chunk(finish_reason="stop")),
        ]

        chunks = _collect(LLMClient(model="test-model"), [{"role": "user", "content": "hi"}])

    assert mock_client.chat.completions.create.call_count == 2
    assert chunks[0].content == "重试成功"


def test_mid_stream_error_wraps_llm_error_without_retry():
    """中途断流不重试：已打出的文字收不回，直接翻译成 LLMError。

    异常在迭代时抛（生成器语义），pytest.raises 必须包住 for 循环而不是 stream() 调用。
    """
    def broken():
        yield _chunk(content="前半句")
        raise _api_error("connection reset")

    with patch("llm.client.OpenAI") as mock_openai, patch("llm.client.time.sleep"):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = broken()

        client = LLMClient(model="test-model")
        with pytest.raises(LLMError):
            for _c in client.stream([{"role": "user", "content": "hi"}]):
                pass

    assert mock_client.chat.completions.create.call_count == 1


def test_authentication_error_no_retry():
    """AuthenticationError 是永久异常：直译 LLMError，不重试。"""
    with patch("llm.client.OpenAI") as mock_openai, patch("llm.client.time.sleep"):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = _auth_error()

        client = LLMClient(model="test-model")
        with pytest.raises(LLMError) as excinfo:
            for _c in client.stream([{"role": "user", "content": "hi"}]):
                pass

    assert "authentication" in str(excinfo.value).lower()
    assert mock_client.chat.completions.create.call_count == 1


# ---------- 公共类型 ----------

def test_stream_chunk_dataclass_defaults():
    """StreamChunk 是流式产物的公共类型：三个字段都有 None 默认值。"""
    from llm.client import StreamChunk

    chunk = StreamChunk(content="hi")
    assert chunk.content == "hi"
    assert chunk.tool_calls is None
    assert chunk.finish_reason is None
