"""测试 tools/registry.py + tools/executor.py 的 SRP 拆分。

策略：
- 单独测 ToolRegistry：注册 / 禁用 / schemas
- 单独测 Executor：通过 registry.get_tool() 拿 tool 直接执行
- 测 5 类错误：tool 不存在 / JSON 错 / 执行错 / 空列表 / 多 tool 批量

注意：
- ToolRegistry 只管注册（不执行）—— 测 register / disable / schemas
- Executor 只管执行（不注册）—— 测 execute_calls / ToolResult
- Executor 永远不抛异常——所有失败都转成 ToolResult(status="failed")
"""
from __future__ import annotations

import time
from typing import Annotated, Literal
from unittest.mock import MagicMock

import pytest

from tools.executor import Executor
from tools.registry import ToolRegistry

# ---------- 辅助函数 ----------

def _make_tool_call(name: str, arguments: str) -> MagicMock:
    """构造假的 tool_call 对象（OpenAI 风格）。"""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------- ToolRegistry 测试 ----------

def test_registry_register_and_list():
    """注册 + list_all：装饰器风格注册，list 包含 tool 名。"""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert "add" in registry.list_all()
    assert "add" in registry.list_enabled()
    assert len(registry.get_schemas()) == 1


def test_registry_register_many():
    """register_many：批量注册。"""
    registry = ToolRegistry()

    def add(a, b):
        return a + b

    def sub(a, b):
        return a - b

    registry.register_many([add, sub])

    assert set(registry.list_all()) == {"add", "sub"}


def test_registry_duplicate_raises():
    """重复注册同名 tool 抛 ValueError。"""
    registry = ToolRegistry()

    def add(a, b):
        return a + b

    registry.register(add)

    with pytest.raises(ValueError) as excinfo:
        registry.register(add)
    assert "already registered" in str(excinfo.value)


def test_registry_disable_excludes_from_schemas():
    """disable 后 tool 还在 list_all，但不在 get_schemas / list_enabled。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    registry.disable("add")

    assert "add" in registry.list_all()      # 代码还在
    assert "add" not in registry.list_enabled()
    assert len(registry.get_schemas()) == 0  # LLM 看不到


def test_registry_get_tool_returns_none_if_missing():
    """get_tool 不存在时返回 None（不抛 KeyError）。"""
    registry = ToolRegistry()
    assert registry.get_tool("not_exist") is None


def test_registry_get_tool_returns_none_if_disabled():
    """get_tool 对 disabled tool 也返回 None。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    registry.disable("add")
    assert registry.get_tool("add") is None


def test_registry_get_tool_returns_tool():
    """get_tool 正常返回 Tool 数据类。"""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    tool = registry.get_tool("add")
    assert tool is not None
    assert tool.name == "add"
    assert tool.description == "Add."
    assert tool.func(1, 2) == 3


# ---------- Executor 测试 ----------

def test_executor_empty_list():
    """空 tool_calls → 空结果。"""
    registry = ToolRegistry()
    executor = Executor(registry)
    assert executor.execute_calls([]) == []


def test_executor_success():
    """正常路径：注册 + 调 + 成功 → status="success" + content 是结果。"""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    executor = Executor(registry)
    tc = _make_tool_call("add", '{"a": 1, "b": 2}')
    results = executor.execute_calls([tc])

    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].content == "3"   # D15: executor 序列化非 str 返回值
    assert results[0].error is None


def test_executor_tool_not_registered():
    """tool 不存在 → ToolResult failed，不抛异常。"""
    registry = ToolRegistry()
    executor = Executor(registry)

    tc = _make_tool_call("not_exist", "{}")
    results = executor.execute_calls([tc])

    assert len(results) == 1
    assert results[0].status == "failed"
    assert "not registered" in results[0].error.lower()
    assert results[0].error_type == "ToolError"
    assert results[0].content is None


def test_executor_tool_disabled():
    """tool 被 disable → ToolResult failed。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    registry.disable("add")
    executor = Executor(registry)

    tc = _make_tool_call("add", '{"a": 1, "b": 2}')
    results = executor.execute_calls([tc])

    assert results[0].status == "failed"


def test_executor_invalid_json():
    """JSON 解析失败 → ToolResult failed + error 含 "json"。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    executor = Executor(registry)
    tc = _make_tool_call("add", "this is not json {")
    results = executor.execute_calls([tc])

    assert results[0].status == "failed"
    assert "json" in results[0].error.lower()


def test_executor_execution_error():
    """工具执行抛异常（ZeroDivisionError）→ ToolResult failed + error_type。"""
    registry = ToolRegistry()

    @registry.register
    def divide(a, b):
        return a / b

    executor = Executor(registry)
    tc = _make_tool_call("divide", '{"a": 10, "b": 0}')
    results = executor.execute_calls([tc])

    assert results[0].status == "failed"
    assert results[0].error_type == "ZeroDivisionError"


def test_executor_batch_serial():
    """serial 模式：多 tool_calls 按顺序执行，结果保序。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    @registry.register
    def multiply(a, b):
        return a * b

    executor = Executor(registry, mode="serial")
    tcs = [
        _make_tool_call("add", '{"a": 1, "b": 2}'),
        _make_tool_call("multiply", '{"a": 3, "b": 4}'),
    ]
    results = executor.execute_calls(tcs)

    assert len(results) == 2
    assert results[0].content == "3"   # add(1,2) = 3, 序列化为 "3"
    assert results[1].content == "12"  # multiply(3,4) = 12


def test_executor_batch_parallel_preserves_order():
    """parallel 模式：完成顺序无关，但返回结果按输入顺序。"""
    registry = ToolRegistry()

    @registry.register
    def slow(x: int) -> int:
        """Slow tool."""
        time.sleep(0.05)
        return x * 2

    @registry.register
    def fast(x: int) -> int:
        """Fast tool."""
        return x + 100

    executor = Executor(registry, mode="parallel", max_workers=2)
    tcs = [
        _make_tool_call("slow", '{"x": 1}'),     # 应返回 2
        _make_tool_call("fast", '{"x": 2}'),     # 应返回 102
    ]
    results = executor.execute_calls(tcs)

    # 即使 fast 先完成，结果顺序仍是 [slow, fast]
    assert results[0].content == "2"
    assert results[1].content == "102"


def test_executor_batch_partial_failure():
    """批量：1 成功 + 1 失败 → 各自 status。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    # 不注册 "sub" —— 它会失败
    executor = Executor(registry)
    tcs = [
        _make_tool_call("add", '{"a": 1, "b": 2}'),
        _make_tool_call("sub", '{}'),
    ]
    results = executor.execute_calls(tcs)

    assert results[0].status == "success"
    assert results[0].content == "3"
    assert results[1].status == "failed"


def test_executor_get_schemas_forwards():
    """Executor.get_schemas 转发到 registry.get_schemas。"""
    registry = ToolRegistry()

    @registry.register
    def add(a, b):
        return a + b

    registry.disable("add")
    executor = Executor(registry)

    # disable 后 Executor.get_schemas 也应该返回空
    assert executor.get_schemas() == []


# ---------- D5: _generate_schema 类型增强 ----------


def test_schema_list_str_items():
    """list[str] → {"type": "array", "items": {"type": "string"}}"""
    registry = ToolRegistry()

    @registry.register
    def search(keywords: list[str]) -> list[str]:
        """Search."""
        ...

    schema = registry.get_schemas()[0]
    kw = schema["function"]["parameters"]["properties"]["keywords"]
    assert kw == {"type": "array", "items": {"type": "string"}}


def test_schema_optional_not_required():
    """Optional[str] → nullable + 不在 required。"""
    registry = ToolRegistry()

    @registry.register
    def greet(name: str, title: str | None) -> str:
        """Greet."""
        ...

    schema = registry.get_schemas()[0]
    params = schema["function"]["parameters"]
    assert "title" not in params["required"]
    assert params["properties"]["title"] == {"type": "string", "nullable": True}


def test_schema_literal_enum():
    """Literal["a", "b"] → {"type": "string", "enum": ["a", "b"]}"""
    registry = ToolRegistry()

    @registry.register
    def set_mode(mode: Literal["fast", "slow"]) -> str:
        """Set mode."""
        ...

    schema = registry.get_schemas()[0]
    mode = schema["function"]["parameters"]["properties"]["mode"]
    assert mode == {"type": "string", "enum": ["fast", "slow"]}


def test_schema_annotated_description():
    """Annotated[int, "desc"] → {"type": "integer", "description": "desc"}"""
    registry = ToolRegistry()

    @registry.register
    def add(a: Annotated[int, "第一个加数"], b: Annotated[int, "第二个加数"]) -> int:
        """Add."""
        return a + b

    schema = registry.get_schemas()[0]
    a = schema["function"]["parameters"]["properties"]["a"]
    b = schema["function"]["parameters"]["properties"]["b"]
    assert a == {"type": "integer", "description": "第一个加数"}
    assert b == {"type": "integer", "description": "第二个加数"}


def test_schema_default_value():
    """x: int = 10 → default + 不在 required。"""
    registry = ToolRegistry()

    @registry.register
    def repeat(text: str, times: int = 3) -> str:
        """Repeat text."""
        ...

    schema = registry.get_schemas()[0]
    params = schema["function"]["parameters"]
    assert "times" not in params["required"]
    assert params["properties"]["times"] == {"type": "integer", "default": 3}
