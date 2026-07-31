from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """单个 tool 的元信息 + 执行函数。

    不可变——注册后不能改。
    未来扩展：required_permissions / timeout_ms / retry_policy / tags
    """
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Tool 中央注册表。

    企业级特性：
    - 集中管理（一个对象管所有 tool）
    - 可禁用（disable/enable）
    - 可审计（list_all / list_enabled）
    - 不可变 schemas（get_schemas 返回拷贝）
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._disabled: set[str] = set()

    def register(self, func) -> Tool:
        """注册一个 tool 函数。"""
        tool = self._make_tool(func)
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} already registered")
        self._tools[tool.name] = tool
        return tool

    def register_many(self, funcs) -> list[Tool]:
        """批量注册。"""
        return [self.register(f) for f in funcs]

    def disable(self, name: str) -> None:
        """禁用一个 tool（代码保留，但不暴露给 LLM）。"""
        if name not in self._tools:
            raise KeyError(f"tool {name} not registered")
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """启用一个 tool。"""
        self._disabled.discard(name)

    def get_schemas(self) -> list[dict]:
        """暴露给 LLM 的 schemas（只含未禁用的，返回拷贝）。"""
        return [
            t.to_openai_schema()
            for t in self._tools.values()
            if t.name not in self._disabled
        ]

    def list_all(self) -> list[str]:
        """列出所有 tool 名字（审计用）。"""
        return list(self._tools.keys())

    def list_enabled(self) -> list[str]:
        """列出已启用的 tool 名字。"""
        return [n for n in self._tools if n not in self._disabled]

    def get_tool(self, name: str) -> Tool | None:
        """为 Executor 提供单个tool"""
        if name in self._disabled:
            return None
        return self._tools.get(name)

    def _make_tool(self, func) -> Tool:
        return Tool(
            name=func.__name__,
            description=func.__doc__ or "",
            parameters=self._generate_schema(func),
            func=func,
        )

    def _generate_schema(self, func) -> dict:
        sig = inspect.signature(func)
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            properties[name] = {"type": self._py_to_json(param.annotation)}
            if param.default == inspect.Parameter.empty:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @staticmethod
    def _py_to_json(py_type) -> str:
        return {
            int: "integer", float: "number",
            str: "string", bool: "boolean",
        }.get(py_type, "string")