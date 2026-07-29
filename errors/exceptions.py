"""异常体系 - Agent Framework 的所有自定义异常。

设计原则：
- AgentError 是基类，runtime 内部 try/except AgentError 一把抓
- 每个异常带 message / context 二个字段
- message: 人类可读描述
- context: dict，附加上下文（api_name / status_code / retry_count 等）
- __str__ 输出格式：<ClassName>: <message> | context=<dict> | cause=<cause>
"""
from __future__ import annotations
from typing import Any


class AgentError(Exception):
    """所有框架异常的基类。

    用途：runtime 内部用 except AgentError 统一兜底，
    防止未预期异常逃逸到 main 循环。
    """
    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    )->None:
        super().__init__(message)
        self.message = message
        self.context = context or {}     #默认赋值None, 后无参数时创建空dict, 避免默认context共享

    def __str__(self) -> str:
        parts = [self.message]
        if self.context:
            parts.append(f"context={self.context}")
        if self.__cause__:
            parts.append(f"cause={type(self.__cause__).__name__}: {self.__cause__}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """序列化用于日志/事件上报。"""
        return {
            "type": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "cause": str(self.__cause__) if self.__cause__ else None,
        }


class LLMError(AgentError):
    """LLM 调用相关错误：网络、限流、解析、auth。

    注意：不细分超时/限流/auth 错误 - 业务复杂度先压住，
    未来如有需要再拆 LLMTimeoutError / LLMRateLimitError。
    """


class ToolError(AgentError):
    """工具调用相关错误：工具不存在、参数非法、执行超时、执行异常。"""


class ConfigError(AgentError):
    """配置错误：缺 API key、env 变量未设、配置项非法值。

    ConfigError 不应重试，直接 FAIL。
    """
