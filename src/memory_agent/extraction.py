"""LLM-driven extraction of durable memories from a conversation turn."""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from .llm import LLMClient, LLMError
from .models import ChatMessage, ExtractedMemory
from .prompts import EXTRACT_SYSTEM

log = logging.getLogger(__name__)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_extraction(raw: str) -> list[ExtractedMemory]:
    """Parse the model's JSON array, tolerating code fences and malformed items."""
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except ValueError:
        return []
    out: list[ExtractedMemory] = []
    for item in items if isinstance(items, list) else []:
        try:
            mem = ExtractedMemory.model_validate(item)
        except ValidationError:
            continue
        if mem.text.strip():
            out.append(mem)
    return out


class MemoryExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, user_text: str, assistant_text: str) -> list[ExtractedMemory]:
        exchange = f"User: {user_text}\nAssistant: {assistant_text}"
        try:
            raw = self.llm.complete(
                system=EXTRACT_SYSTEM,
                messages=[ChatMessage(role="user", content=exchange)],
                max_tokens=512,
            )
        except LLMError as exc:
            log.warning("memory extraction failed: %s", exc)
            return []
        return parse_extraction(raw)
