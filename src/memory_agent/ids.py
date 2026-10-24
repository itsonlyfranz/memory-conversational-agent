"""Validation for user/session ids that end up in file names and checkpoint thread ids."""

from __future__ import annotations

import re

_SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def safe_id(value: str, what: str = "id") -> str:
    if not _SAFE.match(value) or value in {".", ".."}:
        raise ValueError(f"invalid {what} {value!r}: use letters, digits, '_', '-', '.' (max 64)")
    return value
