from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from errors import SessionBusyError

SessionConflictMode = Literal["reject", "wait"]


@dataclass
class _SessionEntry:
    lock: asyncio.Lock
    references: int = 0   #当前持有者数量 + 正在等待的调用数量


class SessionCoordinator:
    """Serialize complete Runtime runs that share a session id.

    The coordinator owns only run leases; BufferMemory remains deliberately
    unaware of asyncio.  Entries are reference-counted so arbitrary session
    ids cannot grow the lock table forever.
    """

    def __init__(
        self,
        *,
        conflict_mode: SessionConflictMode = "reject",
        acquire_timeout: float | None = None,
    ) -> None:
        if conflict_mode not in ("reject", "wait"):
            raise ValueError("conflict_mode must be 'reject' or 'wait'")
        if acquire_timeout is not None and acquire_timeout <= 0:
            raise ValueError("acquire_timeout must be greater than 0")

        self._conflict_mode = conflict_mode
        self._acquire_timeout = acquire_timeout
        self._entries: dict[str, _SessionEntry] = {}
        self._entries_lock = asyncio.Lock()    #sessioncoordinator的锁, 保证一个session只被一个coordinator处理

    @asynccontextmanager
    async def lease(self, session_id: str) -> AsyncIterator[None]:
        if not session_id:
            raise ValueError("session_id must be a non-empty string")

        entry = await self._reserve(session_id)
        acquired = False
        try:
            if self._conflict_mode == "reject":
                # _reserve performs this check while holding _entries_lock, so
                # no competing reject-mode caller can pass the check as well.
                await entry.lock.acquire()
            elif self._acquire_timeout is None:
                await entry.lock.acquire()
            else:
                try:
                    await asyncio.wait_for(
                        entry.lock.acquire(),
                        timeout=self._acquire_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    raise SessionBusyError(
                        session_id,
                        retry_after_ms=int(self._acquire_timeout * 1000),
                    ) from exc

            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()     #可能在等待期间超时或取消，根本没拿到锁. 防止 release 了别的 session 的 lock
            await self._release_reference(session_id, entry)

    async def _reserve(self, session_id: str) -> _SessionEntry:
        async with self._entries_lock:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _SessionEntry(lock=asyncio.Lock())
                self._entries[session_id] = entry

            if self._conflict_mode == "reject" and entry.references > 0:
                raise SessionBusyError(session_id)

            entry.references += 1
            return entry

    async def _release_reference(
        self,
        session_id: str,
        entry: _SessionEntry,
    ) -> None:
        async with self._entries_lock:
            entry.references -= 1
            if entry.references == 0 and not entry.lock.locked():
                self._entries.pop(session_id, None)

    @property
    def tracked_session_count(self) -> int:
        """Number of live/queued session entries (primarily for diagnostics)."""
        return len(self._entries)
