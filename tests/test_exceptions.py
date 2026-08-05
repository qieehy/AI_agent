"""测试 errors 模块。

覆盖：
1. 4 个异常类都能正常构造和 raise
2. 都能继承 AgentError（runtime 一把抓）
3. __str__ / to_dict 输出稳定
4. __cause__ 链 / context 字段正确传递
"""
from __future__ import annotations

import pytest

from errors import AgentError, ConfigError, LLMError, ToolError


def test_all_inherit_agent_error():
    """4 个异常类必须继承 AgentError，runtime 才能统一 catch。"""
    for cls in (LLMError, ToolError, ConfigError):
        assert issubclass(cls, AgentError)
        assert issubclass(cls, Exception)


def test_agent_error_is_base():
    """AgentError 本身是 Exception 子类。"""
    assert issubclass(AgentError, Exception)


def test_basic_construction():
    """只传 message 也能用，__cause__ 默认为 None。"""
    e = LLMError("API timeout")
    assert e.message == "API timeout"
    assert e.__cause__ is None
    assert e.context == {}


def test_full_construction():
    """raise ... from 触发 __cause__ 链。"""
    cause = ValueError("bad json")
    try:
        raise ToolError("tool call failed", context={"tool": "search", "attempt": 3}) from cause
    except ToolError as e:
        assert e.message == "tool call failed"
        assert e.__cause__ is cause
        assert e.context == {"tool": "search", "attempt": 3}


def test_can_be_raised_and_caught_by_base():
    """runtime 用 except AgentError 能 catch 所有子类。"""
    with pytest.raises(AgentError) as excinfo:
        raise LLMError("rate limited")
    assert excinfo.value.message == "rate limited"

    with pytest.raises(AgentError):
        raise ToolError("tool not found")

    with pytest.raises(AgentError):
        raise ConfigError("API key missing")


def test_str_includes_message():
    """__str__ 必含 message。"""
    assert "API timeout" in str(LLMError("API timeout"))


def test_str_includes_context_when_present():
    """__str__ 包含 context。"""
    e = LLMError("timeout", context={"api": "openai", "status": 429})
    s = str(e)
    assert "timeout" in s
    assert "context=" in s
    assert "openai" in s


def test_str_includes_cause_when_present():
    """__cause__ 存在时，__str__ 包含 cause 信息（含类型名）。"""
    cause = RuntimeError("network down")
    try:
        raise LLMError("call failed") from cause
    except LLMError as e:
        s = str(e)
        assert "call failed" in s
        assert "network down" in s
        assert "RuntimeError" in s


def test_to_dict_shape():
    """to_dict 输出的 cause 字段是 __cause__ 的 str。"""
    cause = ValueError("parse fail")
    try:
        raise ToolError("parse fail", context={"tool": "calc"}) from cause
    except ToolError as e:
        d = e.to_dict()
        assert d["type"] == "ToolError"
        assert d["message"] == "parse fail"
        assert d["context"] == {"tool": "calc"}
        assert d["cause"] == "parse fail"


def test_to_dict_no_cause():
    """没有 cause 时 cause 字段是 None（不是空字符串）。"""
    e = ConfigError("bad config")
    d = e.to_dict()
    assert d["cause"] is None
    assert d["context"] == {}


def test_context_is_default_dict():
    """context 默认值是空 dict，且不会在不同实例间共享。"""
    e1 = LLMError("a")
    e2 = LLMError("b")
    e1.context["k"] = "v"
    assert "k" not in e2.context  # 不共享


def test_config_error_distinct_type():
    """ConfigError 应该是独立的类（runtime 可能做特殊处理）。"""
    assert ConfigError is not LLMError
    assert ConfigError is not ToolError
