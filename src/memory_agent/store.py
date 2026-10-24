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
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from .models import MemoryRecord

try:  # POSIX advisory locks; on other platforms we fall back to the in-process lock only.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_GEN_PREFIX = b'{"op":"gen",'


class MemoryStore(Protocol):
    def upsert(self, record: MemoryRecord) -> None: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def delete(self, memory_id: str) -> bool: ...

    def list(self, user_id: str) -> list[MemoryRecord]: ...

    def transaction(self) -> AbstractContextManager[None]:
        """Hold the store exclusively, with other sessions' changes pulled in, for a
        read-modify-write that must not interleave with another writer."""


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

    def transaction(self) -> AbstractContextManager[None]:
        return contextlib.nullcontext()


class JsonlMemoryStore:
    """Append-only operation log (``upsert`` / ``delete`` lines) replayed into a dict."""

    def __init__(self, path: str | Path, *, compact_slack: int = 1000) -> None:
        self.path = Path(path)
        self.compact_slack = compact_slack
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[str, MemoryRecord] = {}
        self._offset = 0
        self._ops = 0
        self._gen: str | None = None
        self._mutex = threading.RLock()
        self._held = False

    @contextlib.contextmanager
    def _locked(self, exclusive: bool) -> Iterator[None]:
        with self._mutex:
            if fcntl is None or self._held:
                yield
                return
            with open(self._lock_path, "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                self._held = True
                try:
                    yield
                finally:
                    self._held = False
                    fcntl.flock(fh, fcntl.LOCK_UN)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        with self._locked(exclusive=True):
            self._refresh()
            yield

    def _refresh(self) -> None:
        """Replay log lines appended since our last read. Caller holds the lock."""
        try:
            fh = open(self.path, "rb")
        except FileNotFoundError:
            self._records, self._offset, self._ops, self._gen = {}, 0, 0, None
            return
        with fh:
            gen = _generation(fh.readline())
            size = os.fstat(fh.fileno()).st_size
            if gen != self._gen or size < self._offset:
                # File was compacted/replaced (or truncated): rebuild from scratch.
                self._records, self._offset, self._ops, self._gen = {}, 0, 0, gen
            if size == self._offset:
                return
            fh.seek(self._offset)
            data = fh.read()
        end = data.rfind(b"\n") + 1  # ignore a partially written trailing line
        for line in data[:end].splitlines():
            self._apply(line)
            self._ops += 1
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
        """Append one op under the exclusive lock, compacting once the log is mostly stale."""
        line = json.dumps(op, separators=(",", ":"), default=str) + "\n"
        with open(self.path, "ab") as fh:
            fh.write(line.encode())
            fh.flush()
            os.fsync(fh.fileno())
        self._refresh()
        if self._ops > 2 * len(self._records) + self.compact_slack:
            self._rewrite()

    def sync(self) -> None:
        with self._locked(exclusive=False):
            self._refresh()

    def upsert(self, record: MemoryRecord) -> None:
        with self._locked(exclusive=True):
            self._refresh()
            self._append({"op": "upsert", "record": json.loads(record.model_dump_json())})

    def delete(self, memory_id: str) -> bool:
        with self._locked(exclusive=True):
            self._refresh()
            existed = memory_id in self._records
            if existed:
                self._append({"op": "delete", "id": memory_id})
            return existed

    def get(self, memory_id: str) -> MemoryRecord | None:
        self.sync()
        rec = self._records.get(memory_id)
        return rec.model_copy(deep=True) if rec else None

    def list(self, user_id: str) -> list[MemoryRecord]:
        self.sync()
        return [r.model_copy(deep=True) for r in self._records.values() if r.user_id == user_id]

    def compact(self) -> None:
        """Rewrite the log as one upsert per live record under a fresh generation header,
        which tells other readers to rebuild."""
        with self._locked(exclusive=True):
            self._refresh()
            self._rewrite()

    def _rewrite(self) -> None:
        """Caller holds the exclusive lock and has just refreshed."""
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".compact-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(_GEN_PREFIX + b'"id":"' + uuid.uuid4().hex.encode() + b'"}\n')
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


def _generation(first_line: bytes) -> str | None:
    """The log's generation id from its header line; ``None`` for a never-compacted log."""
    if not first_line.startswith(_GEN_PREFIX):
        return None
    try:
        return json.loads(first_line)["id"]
    except (ValueError, KeyError, TypeError):
        return None
