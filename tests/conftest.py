"""pytest 共享 fixtures + helpers。

所有测试文件都能用 `make_memory` fixture，以及 `make_llm_response` / `make_tool_call` helper。
"""
from __future__ import annotations
import tempfile
import os
from unittest.mock import MagicMock

from memory import MemoryManager, SessionStore


# ========== helpers ==========

def make_memory(db_path: str | None = None) -> MemoryManager:
    """创建独立 SQLite 数据库的 MemoryManager（临时文件），函数级隔离。

    可以当普通函数调用（test_runtime.py 的 _make_runtime 默认参数），
    也可被 pytest fixture 包装使用。
    """
    if db_path:
        return MemoryManager(SessionStore(db_path))
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return MemoryManager(SessionStore(path))

def make_llm_response(content: str = "ok", tool_calls: list | None = None):
    """构造假的 OpenAI LLM 响应对象。

    model_dump 必须是真 lambda（runtime 用 hasattr 判断），不能是 MagicMock。
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls or []
    response.choices[0].message.model_dump = lambda: {"role": "assistant", "content": content}
    return response


def make_tool_call(call_id: str, name: str, arguments: str):
    """构造假的 tool_call 对象（OpenAI 风格）。"""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc
