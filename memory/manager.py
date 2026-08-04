from .short_term import BufferMemory
from .session_store import SessionStore
from errors import MemoryError

class MemoryManager:
    def __init__(self, store: SessionStore, max_tokens: int = 4000):
        self._store = store
        self._max_tokens = max_tokens
        self._cache: dict[str, BufferMemory] = {}

    def get_or_create(self, session_id: str, system_message: dict | None = None) -> BufferMemory:
        """拿一个BufferMemory"""
        if session_id not in self._cache:
            self._cache[session_id] = BufferMemory(system_message=system_message, max_tokens=self._max_tokens)
        return self._cache[session_id]

    def save(self, session_id):
        """缓存里的 BufferMemory SQLite 持久化"""
        mem = self._cache.get(session_id)
        if mem is None:
            raise MemoryError(f"Session {session_id} not found")
        self._store.save(session_id, system_message=mem.get_system_message(), messages=mem.messages)

    def load(self, session_id) -> BufferMemory | None:
        if session_id in self._cache:
            return self._cache[session_id]   #缓存里有就直接返回

        data = self._store.load(session_id)   #SOLite 里找
        if data is None:
            return None

        system_msg, messages = data
        mem = BufferMemory(system_message=system_msg, max_tokens=self._max_tokens)
        for msg in messages:
            mem.add_message(msg)
        self._cache[session_id] = mem
        return mem

    def delete_session(self, session_id):
        """删除会话, 缓存 ➕ SQLite"""
        self._cache.pop(session_id, None)   #key不存在不要抛异常
        self._store.delete(session_id)

    def list_sessions(self):
        return self._store.list_sessions()