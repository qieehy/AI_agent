"""短期记忆：滑动窗口 BufferMemory。

一个实例 = 一个会话。token 超限自动淘汰旧消息，保证：
1. system 永不被淘汰（独立字段，不在窗口里）
2. 最新消息永不删（循环 guard len > 1）
3. 单条超大消息允许超预算（不删光不崩）
4. tool 回合块整体淘汰（assistant tool_calls + 它的 tool 结果一起删）
"""
from __future__ import annotations
from typing import Any

from observability import logger
from .token_counter import count_message


class BufferMemory:
    """短期记忆：滑动窗口，token 超限自动淘汰。"""

    def __init__(self, system_message: dict[str, Any] | None = None, max_tokens: int = 4000):
        """
        Args:
            system_message: system 提示（独立字段，不在淘汰窗口里）
            max_tokens: token 预算上限（默认 4000，给 D9 的检索+tool schema+输出留余量）
        """
        self._system_message = system_message
        self._messages: list[dict[str, Any]] = []
        self._max_tokens = max_tokens
        self._total_tokens = 0
        if system_message:
            self._total_tokens += count_message(system_message)



    @property
    def messages(self) -> list[dict[str, Any]]:
        """防御性拷贝：调用者改返回值不影响内部状态。"""
        return list(self._messages)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def add_message(self, message: dict[str, Any]) -> None:
        """添加消息 → 计数 → 自动截断。"""
        self._messages.append(message)
        self._total_tokens += count_message(message)
        self._trim()

    def get_context(self) -> list[dict[str, Any]]:
        """返回 [system] + 窗口（OpenAI-ready）。防御性拷贝。"""
        ctx = [self._system_message] if self._system_message else []
        return ctx + self.messages

    def get_system_message(self) -> dict[str, Any] | None:
        return self._system_message

    def clear(self) -> None:
        """重置窗口，重新计入 system。"""
        self._messages = []
        self._total_tokens = 0
        if self._system_message:
            self._total_tokens += count_message(self._system_message)

    def _trim(self) -> None:
        """截断：4 条不变量。

        1. system 永不被淘汰（独立字段，不在 _messages 里）
        2. 最新消息永不删（guard len > 1）
        3. 单条超大消息允许超预算（len == 1 时停止）
        4. tool 回合块整体淘汰（assistant tool_calls + 它的 tool 结果一起删）
        """
        # 不变量 2/3：至少留 1 条（最新消息），否则允许超预算
        while self._total_tokens > self._max_tokens and len(self._messages) > 1:
            # 不变量 4：按回合块淘汰，不按单条
            block_size = self._count_turn_block_size(0)
            # 如果整个块会删到只剩 0 条（块大小 == 当前长度），停止（不变量 2/3）
            if block_size >= len(self._messages):
                break
            popped = self._messages[:block_size]
            self._messages = self._messages[block_size:]
            self._total_tokens -= sum(count_message(m) for m in popped)
            logger.debug(f"Memory trim | evicted={block_size} messages "
                         f"| remaining={len(self._messages)} "
                         f"| tokens={self._total_tokens}/{self._max_tokens}")

    def _count_turn_block_size(self, start_idx: int) -> int:
        """从 start_idx 起，一个"回合块"包含多少条消息。

        回合块 = 触发消息（user 或 assistant-with-tool_calls）+ 它引发的 tool 结果。
        OpenAI 要求每条 tool 消息的 tool_call_id 引用前面的 assistant tool_calls 消息，
        所以淘汰时必须整块一起删，否则 D9 接 LLM 时 API 400。
        """
        msg = self._messages[start_idx]
        role = msg.get("role")

        # assistant 发了 tool_call → 跟随的所有 tool 结果算同一块
        if role == "assistant" and msg.get("tool_calls"):
            n = 1
            i = start_idx + 1
            while i < len(self._messages) and self._messages[i].get("role") == "tool":
                n += 1
                i += 1
            return n

        # user / 普通 assistant / 孤立的 tool 消息 → 单条成块
        return 1
