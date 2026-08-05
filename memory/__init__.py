from .manager import MemoryManager
from .session_store import SessionStore
from .short_term import BufferMemory


def create_memory_manager(db_path: str = "data/sessions.db", max_tokens: int = 4000) -> MemoryManager:
    """组合根工厂——main.py 调这个创建 MemoryManager。

    不放单例：交给调用者管理生命周期。
    """
    store = SessionStore(db_path)
    return MemoryManager(store, max_tokens=max_tokens)


__all__ = ["BufferMemory", "SessionStore", "MemoryManager", "create_memory_manager"]
