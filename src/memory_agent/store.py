"""Memory stores: an in-process store and a local append-only JSONL store shared across sessions.

``JsonlMemoryStore`` is the cross-session sync mechanism. Every write is one JSON line appended
under an exclusive file lock; every read first tails any bytes other processes have appended
since the last read. Two agents pointed at the same file therefore see each other's memories
without restarting, with no server involved.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from .models import MemoryRecord

try:  # POSIX advisory locks; on other platforms we fall back to the in-process lock only.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class MemoryStore(Protocol):
    def upsert(self, record: MemoryRecord) -> None: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def delete(self, memory_id: str) -> bool: ...

    def list(self, user_id: str) -> list[MemoryRecord]: ...

    def sync(self) -> None:
        """Pull in changes made by other sessions/processes (no-op for in-process stores)."""


class InMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def upsert(self, record: MemoryRecord) -> None:
        self._records[record.id] = record.model_copy(deep=True)

    def get(self, memory_id: str) -> MemoryRecord | None:
        rec = self._records.get(memory_id)
        return rec.model_copy(deep=True) if rec else None

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def list(self, user_id: str) -> list[MemoryRecord]:
        return [r.model_copy(deep=True) for r in self._records.values() if r.user_id == user_id]

    def sync(self) -> None:
        return None


class JsonlMemoryStore:
    """Append-only operation log (``upsert`` / ``delete`` lines) replayed into a dict."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[str, MemoryRecord] = {}
        self._offset = 0
        self._inode: int | None = None
        self._mutex = threading.RLock()

    @contextlib.contextmanager
    def _locked(self, exclusive: bool) -> Iterator[None]:
        with self._mutex:
            if fcntl is None:  # pragma: no cover
                yield
                return
            with open(self._lock_path, "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                try:
                    yield
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    def _refresh(self) -> None:
        """Replay log lines appended since our last read. Caller holds the lock."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._records, self._offset, self._inode = {}, 0, None
            return
        if st.st_ino != self._inode or st.st_size < self._offset:
            # File was compacted/replaced (or truncated): rebuild from scratch.
            self._records, self._offset, self._inode = {}, 0, st.st_ino
        if st.st_size == self._offset:
            return
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        end = data.rfind(b"\n") + 1  # ignore a partially written trailing line
        for line in data[:end].splitlines():
            self._apply(line)
        self._offset += end

    def _apply(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            op = json.loads(line)
            if op["op"] == "upsert":
                rec = MemoryRecord.model_validate(op["record"])
                self._records[rec.id] = rec
            elif op["op"] == "delete":
                self._records.pop(op["id"], None)
        except (ValueError, KeyError, TypeError):
            log.warning("skipping corrupt memory log line: %r", line[:80])

    def _append(self, op: dict) -> None:
        line = json.dumps(op, separators=(",", ":"), default=str) + "\n"
        with open(self.path, "ab") as fh:
            fh.write(line.encode())
            fh.flush()
            os.fsync(fh.fileno())

    def sync(self) -> None:
        with self._locked(exclusive=False):
            self._refresh()

    def upsert(self, record: MemoryRecord) -> None:
        with self._locked(exclusive=True):
            self._refresh()
            self._append({"op": "upsert", "record": json.loads(record.model_dump_json())})
            self._refresh()

    def delete(self, memory_id: str) -> bool:
        with self._locked(exclusive=True):
            self._refresh()
            existed = memory_id in self._records
            if existed:
                self._append({"op": "delete", "id": memory_id})
                self._refresh()
            return existed

    def get(self, memory_id: str) -> MemoryRecord | None:
        self.sync()
        rec = self._records.get(memory_id)
        return rec.model_copy(deep=True) if rec else None

    def list(self, user_id: str) -> list[MemoryRecord]:
        self.sync()
        return [r.model_copy(deep=True) for r in self._records.values() if r.user_id == user_id]

    def compact(self) -> None:
        """Rewrite the log as one upsert per live record. Other readers detect the new inode."""
        with self._locked(exclusive=True):
            self._refresh()
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".compact-")
            try:
                with os.fdopen(fd, "wb") as fh:
                    for rec in self._records.values():
                        op = {"op": "upsert", "record": json.loads(rec.model_dump_json())}
                        fh.write((json.dumps(op, separators=(",", ":")) + "\n").encode())
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise
            self._refresh()
