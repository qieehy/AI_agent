"""测试 memory/manager.py — MemoryManager 缓存 + 持久化协调。"""
from __future__ import annotations
import tempfile
import os

import pytest

from memory import MemoryManager, SessionStore, BufferMemory
from memory import create_memory_manager
from errors import MemoryError


# ========== fixtures ==========

@pytest.fixture
def mgr():
    """每个测试独立的 MemoryManager。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = MemoryManager(SessionStore(path))
    yield m
    try:
        os.unlink(path)
    except OSError:
        pass


def _make_msg(role: str = "user", content: str = "hello"):
    return {"role": role, "content": content}


# ========== get_or_create ==========

def test_get_or_create_returns_buffer_memory(mgr):
    """首次调用创建新的 BufferMemory，类型正确。"""
    mem = mgr.get_or_create("s1")
    assert isinstance(mem, BufferMemory)
    assert mem.messages == []


def test_get_or_create_reuses_cache(mgr):
    """同一 session_id 两次调用返回同一个对象（缓存命中）。"""
    m1 = mgr.get_or_create("s1")
    m2 = mgr.get_or_create("s1")
    assert m1 is m2


def test_get_or_create_different_sessions(mgr):
    """不同 session_id 返回不同的 BufferMemory。"""
    m1 = mgr.get_or_create("s1")
    m2 = mgr.get_or_create("s2")
    assert m1 is not m2


def test_get_or_create_with_system_message(mgr):
    """system_message 传入后，BufferMemory 里能拿到。"""
    sys = {"role": "system", "content": "be helpful"}
    mem = mgr.get_or_create("s1", system_message=sys)
    assert mem.get_system_message() == sys


def test_get_or_create_system_message_only_first_time(mgr):
    """第二次 get_or_create 不传 system_message，应仍保留第一次的。"""
    sys = {"role": "system", "content": "v1"}
    mgr.get_or_create("s1", system_message=sys)
    # 第二次不传 system_message
    mem = mgr.get_or_create("s1")
    assert mem.get_system_message() == sys


# ========== save + load roundtrip ==========

def test_save_and_load_roundtrip(mgr):
    """保存到磁盘 → 新建 MemoryManager 实例 → load 回来，消息一致。"""
    mem = mgr.get_or_create("s1")
    mem.add_message(_make_msg("user", "hello"))
    mem.add_message(_make_msg("assistant", "hi"))
    mgr.save("s1")

    # 新建 MemoryManager（模拟进程重启）
    new_mgr = MemoryManager(mgr._store, max_tokens=4000)
    loaded = new_mgr.load("s1")

    assert loaded is not None
    assert len(loaded.messages) == 2
    assert loaded.messages[0]["content"] == "hello"
    assert loaded.messages[1]["content"] == "hi"


def test_load_returns_none_for_unknown_session(mgr):
    """不存在的 session load 返回 None。"""
    assert mgr.load("no-such-id") is None


def test_load_uses_cache_when_present(mgr):
    """缓存命中时不查 SQLite——已经 get_or_create 过就直接返回缓存。"""
    mem = mgr.get_or_create("s1")
    mem.add_message(_make_msg("user", "cached"))
    # 没有 save——纯缓存
    loaded = mgr.load("s1")
    assert loaded is mem
    assert loaded.messages[0]["content"] == "cached"


def test_load_from_disk_when_not_cached(mgr):
    """缓存未命中时从 SQLite 加载——用于进程重启场景。"""
    mem = mgr.get_or_create("s1")
    mem.add_message(_make_msg("user", "persisted"))
    mgr.save("s1")

    # 新建实例（缓存为空）
    new_mgr = MemoryManager(mgr._store)
    loaded = new_mgr.load("s1")

    assert loaded is not None
    assert loaded.messages[0]["content"] == "persisted"


def test_load_caches_after_disk_hit(mgr):
    """从磁盘加载后，下一次直接命中缓存。"""
    mem = mgr.get_or_create("s1")
    mem.add_message(_make_msg("user", "x"))
    mgr.save("s1")

    new_mgr = MemoryManager(mgr._store)
    first = new_mgr.load("s1")
    second = new_mgr.load("s1")
    assert first is second


def test_load_with_system_message_from_disk(mgr):
    """从磁盘加载时，system_message 也恢复。"""
    sys = {"role": "system", "content": "I am a bot"}
    mem = mgr.get_or_create("s1", system_message=sys)
    mem.add_message(_make_msg("user", "q"))
    mgr.save("s1")

    new_mgr = MemoryManager(mgr._store)
    loaded = new_mgr.load("s1")
    assert loaded.get_system_message() == sys


# ========== save error ==========

def test_save_nonexistent_session_raises(mgr):
    """save 未 get_or_create 过的 session 抛 MemoryError。"""
    with pytest.raises(MemoryError, match="not found"):
        mgr.save("ghost-session")


# ========== delete ==========

def test_delete_removes_from_cache_and_store(mgr):
    """删除后缓存和 SQLite 里都没有了。"""
    mem = mgr.get_or_create("s1")
    mem.add_message(_make_msg("user", "x"))
    mgr.save("s1")

    mgr.delete_session("s1")
    # 磁盘
    assert mgr._store.load("s1") is None
    # 缓存：新 get_or_create 应该是新对象
    new_mem = mgr.get_or_create("s1")
    assert new_mem.messages == []


def test_delete_nonexistent_does_not_raise(mgr):
    """删除不存在的 session 不抛异常。"""
    mgr.delete_session("no-such")  # 不应抛


# ========== list ==========

def test_list_sessions_delegates_to_store(mgr):
    """list_sessions 返回 store 中的 session ID 列表。"""
    for sid in ["a", "b", "c"]:
        mem = mgr.get_or_create(sid)
        mem.add_message(_make_msg("user", "msg"))
        mgr.save(sid)

    ids = mgr.list_sessions()
    assert set(ids) == {"a", "b", "c"}


# ========== factory ==========

def test_create_memory_manager():
    """工厂函数 create_memory_manager 返回 MemoryManager + db_path 生效。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        mm = create_memory_manager(db_path=path, max_tokens=2000)
        assert isinstance(mm, MemoryManager)
        # 验证数据库文件确实创建了
        assert os.path.exists(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
