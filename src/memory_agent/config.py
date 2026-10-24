"""Environment-driven settings and a tiny .env loader (no extra dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

DEFAULT_MODEL = "claude-sonnet-5-5"


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing variables."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


class Settings(BaseModel):
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    data_dir: Path = Path(".memory_data")
    max_context_tokens: int = 1200

    @classmethod
    def from_env(cls, dotenv: str | Path | None = ".env") -> Settings:
        if dotenv is not None:
            load_dotenv(dotenv)
        env = os.environ
        return cls(
            api_key=env.get("ANTHROPIC_API_KEY") or None,
            model=env.get("MEMORY_AGENT_MODEL") or DEFAULT_MODEL,
            data_dir=Path(env.get("MEMORY_AGENT_DATA_DIR") or ".memory_data"),
            max_context_tokens=int(env.get("MEMORY_AGENT_MAX_CONTEXT_TOKENS") or 1200),
        )
