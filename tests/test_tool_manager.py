"""测试 tools/manager.py 的异常翻译。

策略：用 MagicMock 模拟 OpenAI 的 tool_call 对象，
验证 ToolError 是否被正确抛出 + context 是否正确 + __cause__ 链是否正确。

注意：
- 工具不存在时 raise ToolError 不带 from（__cause__ 应为 None）
- JSON 解析失败和执行失败都带 from（__cause__ 应指向原异常）
"""
from __future__ import annotations
import json as json_module
import pytest
from unittest.mock import MagicMock

from tools.manager import ToolManager
from errors import ToolError


# ---------- 辅助函数 ----------

def _make_tool_call(name: str, arguments: str) -> MagicMock:
    """构造假的 tool_call 对象（OpenAI 风格）。"""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------- 测试 ----------

def test_execute_success():
    """正常路径：注册 add 工具，传合法参数，返回 3。"""
    manager = ToolManager()

    @manager.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tc = _make_tool_call("add", '{"a": 1, "b": 2}')
    result = manager.execute(tc)
    assert result == 3


def test_execute_success_no_arguments():
    """正常路径：注册无参工具。"""
    manager = ToolManager()

    @manager.register
    def hello() -> str:
        """Say hello."""
        return "hi"

    tc = _make_tool_call("hello", "{}")
    result = manager.execute(tc)
    assert result == "hi"


def test_invalid_json_arguments_raises_tool_error():
    """JSON 解析失败 → ToolError + context 含 raw_arguments + cause 链。"""
    manager = ToolManager()
    bad_json = "this is not json {"

    tc = _make_tool_call("anything", bad_json)
    with pytest.raises(ToolError) as excinfo:
        manager.execute(tc)

    # message 应说明是 JSON 问题
    assert "json" in str(excinfo.value).lower()
    # context 应含原始参数（截 200 字符）
    assert excinfo.value.context.get("raw_arguments") == bad_json
    assert excinfo.value.context.get("tool") == "anything"
    # cause 链：原 JSONDecodeError 在 __cause__ 里
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, json_module.JSONDecodeError)


def test_tool_not_registered_raises_tool_error():
    """工具不存在 → ToolError + context 含 available_tools + 无 cause 链。"""
    manager = ToolManager()

    @manager.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    tc = _make_tool_call("does_not_exist", "{}")
    with pytest.raises(ToolError) as excinfo:
        manager.execute(tc)

    # message 应说明工具未注册
    assert "not registered" in str(excinfo.value).lower()
    # context 应含可用工具列表
    assert "add" in excinfo.value.context.get("available_tools", [])
    assert excinfo.value.context.get("tool") == "does_not_exist"
    # 注意：第二处 raise 没有 from e，__cause__ 应为 None
    assert excinfo.value.__cause__ is None


def test_tool_execution_failure_raises_tool_error():
    """工具内部抛异常 → ToolError + context 含 exception_type + cause 链。"""
    manager = ToolManager()

    @manager.register
    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    tc = _make_tool_call("divide", '{"a": 10, "b": 0}')
    with pytest.raises(ToolError) as excinfo:
        manager.execute(tc)

    # message 应说明执行失败
    assert "failed" in str(excinfo.value).lower()
    # context 应含原异常类型
    assert excinfo.value.context.get("exception_type") == "ZeroDivisionError"
    assert excinfo.value.context.get("tool") == "divide"
    # cause 链：原 ZeroDivisionError 在 __cause__ 里
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_tool_execution_type_error_raises_tool_error():
    """工具参数类型错（TypeError）→ ToolError。"""
    manager = ToolManager()

    @manager.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    # 传字符串而非 int，触发 TypeError
    tc = _make_tool_call("add", '{"a": "not_a_number", "b": 2}')
    with pytest.raises(ToolError) as excinfo:
        manager.execute(tc)

    assert excinfo.value.context.get("exception_type") == "TypeError"
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_raw_arguments_truncated_at_200():
    """raw_arguments 超过 200 字符时会被截断（防爆）。"""
    manager = ToolManager()
    bad_long_args = "x" * 500  # 不是 JSON

    tc = _make_tool_call("anything", bad_long_args)

    with pytest.raises(ToolError) as excinfo:
        manager.execute(tc)

    # 截断到 200 字符
    assert len(excinfo.value.context["raw_arguments"]) == 200
