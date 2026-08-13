from . import calculator  # 让注册可见
from .builtin import file_tools, network_tools, shell_tools
from .executor import ExecutionMode, Executor, ToolResult
from .registry import Tool, ToolRegistry


def create_registry() -> ToolRegistry:
    """工厂函数——集中注册 + 返回新实例。"""
    registry = ToolRegistry()
    registry.register_many(calculator.TOOLS)
    registry.register_many(network_tools.TOOLS)
    registry.register_many(file_tools.TOOLS)
    registry.register_many(shell_tools.TOOLS)
    return registry

__all__ = [
  "ToolRegistry", "Tool",
  "ExecutionMode", "Executor", "ToolResult",
  "create_registry",   # ← 工厂函数
]
