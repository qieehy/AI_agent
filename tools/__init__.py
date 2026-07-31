from .registry import ToolRegistry, Tool
from . import calculator  # 让注册可见
from .executor import Executor, ToolResult, ExecutionMode


def create_registry() -> ToolRegistry:
    """工厂函数——集中注册 + 返回新实例。"""
    registry = ToolRegistry()
    registry.register_many(calculator.TOOLS)
    # registry.register_many(http_get.TOOLS)   # D16
    return registry

__all__ = [
  "ToolRegistry", "Tool",
  "ExecutionMode", "Executor", "ToolResult",
  "create_registry",   # ← 工厂函数
]
