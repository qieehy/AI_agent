"""测试 tools/registry.py 的注册 / 禁用 / schemas（SRP：Executor 契约见 test_executor_async.py）。

注意：
- ToolRegistry 只管注册（不执行）—— 测 register / disable / schemas
- Executor 测试（execute_calls_async / ToolResult）在 tests/test_executor_async.py
"""
from __future__ import annotations

from typing import Annotated, Literal

import pytest

from tools.registry import ToolRegistry

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
    """Optional[str] → type: ["string", "null"]（Draft202012 联合 type）+ 不在 required。"""
    registry = ToolRegistry()

    @registry.register
    def greet(name: str, title: str | None) -> str:
        """Greet."""
        ...

    schema = registry.get_schemas()[0]
    params = schema["function"]["parameters"]
    assert "title" not in params["required"]
    assert params["properties"]["title"] == {"type": ["string", "null"]}


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


def test_schema_literal_int_enum():
    """Literal[1, 2] → {"type": "integer", "enum": [1, 2]}（旧实现硬编码 string 不可满足）。"""
    registry = ToolRegistry()

    @registry.register
    def pick(level: Literal[1, 2]) -> str:
        """Pick."""
        ...

    schema = registry.get_schemas()[0]
    level = schema["function"]["parameters"]["properties"]["level"]
    assert level == {"type": "integer", "enum": [1, 2]}


def test_schema_nullable_validates_against_null():
    """Optional[str] 显式传 null → Draft202012 不再报 "None is not of type 'string'"。"""
    from jsonschema import Draft202012Validator

    registry = ToolRegistry()

    @registry.register
    def greet(name: str, title: str | None) -> str:
        """Greet."""
        ...

    schema = registry.get_schemas()[0]
    parameters = schema["function"]["parameters"]
    errors = list(Draft202012Validator(parameters).iter_errors({"name": "hi", "title": None}))
    assert errors == []


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
