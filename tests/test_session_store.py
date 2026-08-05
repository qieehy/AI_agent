"""测试 memory/session_store.py — SQLite 会话持久化 CRUD。"""
from __future__ import annotations

import contextlib
import os
import tempfile

import pytest

from memory import SessionStore

# ========== fixture ==========

@pytest.fixture
def store():
    """每个测试独立的临时数据库。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = SessionStore(path)
    yield s
    with contextlib.suppress(OSError):
        os.unlink(path)


# ========== save + load ==========

def test_save_and_load_roundtrip(store):
    """保存→加载 数据一致。"""
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    system = {"role": "system", "content": "you are helpful"}

    store.save("s1", system, messages)
    result = store.load("s1")

    assert result is not None
    loaded_system, loaded_messages = result
    assert loaded_system == system
    assert loaded_messages == messages


def test_save_overwrites_existing(store):
    """同一个 session_id 第二次 save 覆盖旧数据。"""
    store.save("s1", None, [{"role": "user", "content": "v1"}])
    store.save("s1", {"role": "system", "content": "sys"}, [{"role": "user", "content": "v2"}])

    _, messages = store.load("s1")
    assert len(messages) == 1
    assert messages[0]["content"] == "v2"


def test_load_nonexistent_returns_none(store):
    """不存在的 session 返回 None。"""
    assert store.load("no-such-id") is None


def test_save_without_system_message(store):
    """system_message=None 时保存+加载正常。"""
    messages = [{"role": "user", "content": "bare"}]
    store.save("s1", None, messages)

    sys_msg, msgs = store.load("s1")
    assert sys_msg is None
    assert msgs == messages


# ========== list ==========

def test_list_sessions_returns_ids(store):
    """list_sessions 返回所有 session ID。"""
    store.save("a", None, [{"role": "user", "content": "1"}])
    store.save("b", None, [{"role": "user", "content": "2"}])

    ids = store.list_sessions()
    assert "a" in ids
    assert "b" in ids


def test_list_sessions_empty(store):
    """空库返回 []。"""
    assert store.list_sessions() == []


def test_list_sessions_order(store):
    """list_sessions 按 updated_at DESC 排序——最新更新的在前面。"""
    import time

    store.save("older", None, [{"role": "user", "content": "first"}])
    time.sleep(1.1)  # CURRENT_TIMESTAMP 精度是秒级，必须跨秒
    store.save("newer", None, [{"role": "user", "content": "second"}])

    ids = store.list_sessions()
    assert ids[0] == "newer"
    assert ids[1] == "older"


# ========== delete ==========

def test_delete_removes_session(store):
    """删除后 load 返回 None。"""
    store.save("s1", None, [{"role": "user", "content": "x"}])
    store.delete("s1")

    assert store.load("s1") is None


def test_delete_nonexistent_does_not_raise(store):
    """删除不存在的 session 不抛异常。"""
    store.delete("ghost")  # 不应抛


def test_delete_one_keeps_others(store):
    """删除一个 session 不影响其他。"""
    store.save("a", None, [{"role": "user", "content": "a"}])
    store.save("b", None, [{"role": "user", "content": "b"}])

    store.delete("a")

    assert store.load("a") is None
    assert store.load("b") is not None
    assert store.list_sessions() == ["b"]


# ========== edge cases ==========

def test_save_empty_messages(store):
    """空消息列表也能正常保存+加载。"""
    store.save("empty", None, [])
    _, msgs = store.load("empty")
    assert msgs == []


def test_save_messages_with_special_chars(store):
    """消息中含有特殊字符（引号、换行、Unicode）正常保存。"""
    messages = [
        {"role": "user", "content": 'he said "hello"\n中文测试 🎉'}
    ]
    store.save("s1", None, messages)
    _, loaded = store.load("s1")
    assert loaded[0]["content"] == messages[0]["content"]
