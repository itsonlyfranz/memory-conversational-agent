"""Offline LLM stand-ins: a rule-based ``HeuristicLLM`` and a recording ``ScriptedLLM``.

They let the demo and the whole test suite run with no network or API key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from .models import ChatMessage
from .prompts import TASK_CHAT, TASK_COMPRESS, TASK_EXTRACT

_END = r"(?=[.,!?;]| and | but | because |$)"


def _rule(pattern: str, template: str, **extra: object) -> tuple[re.Pattern[str], str, dict]:
    return re.compile(pattern + r"\s+(?P<v>.+?)" + _END, re.I), template, extra


_RULES = [
    _rule(r"\bmy name is", "User's name is {v}", importance=0.9, slot="name"),
    _rule(r"\bi(?: now)? live in", "User lives in {v}", importance=0.8, slot="residence"),
    _rule(r"\bi(?:'ve| have)? moved to", "User lives in {v}", importance=0.8, slot="residence"),
    _rule(r"\bi work as an?", "User works as a {v}", importance=0.7, slot="job"),
    _rule(r"\bi work at", "User works at {v}", importance=0.7, slot="employer"),
    _rule(r"\bi(?:'m| am) allergic to", "User is allergic to {v}", importance=1.0),
    _rule(r"\bi (?:really )?(?:love|like|enjoy)", "User likes {v}", kind="preference"),
    _rule(r"\bi (?:hate|dislike|can't stand)", "User dislikes {v}", kind="preference"),
    _rule(r"\bmy dog is (?:named|called)", "User's dog is named {v}", importance=0.6),
    _rule(r"\bmy cat is (?:named|called)", "User's cat is named {v}", importance=0.6),
]


class HeuristicLLM:
    """Deterministic regex-based model that honours the three task prompts of this project."""

    def complete(
        self, *, system: str, messages: Sequence[ChatMessage], max_tokens: int = 1024
    ) -> str:
        if TASK_EXTRACT in system:
            return self._extract(messages[-1].content)
        if TASK_COMPRESS in system:
            return self._compress(messages[-1].content, max_tokens // 2)
        if TASK_CHAT in system:
            return self._chat(system, messages[-1].content)
        return "OK."

    def _extract(self, exchange: str) -> str:
        user_text = exchange.split("\nAssistant:")[0].removeprefix("User:").strip()
        found = []
        for pattern, template, extra in _RULES:
            for m in pattern.finditer(user_text):
                found.append({"text": template.format(v=m.group("v").strip()), **extra})
        return json.dumps(found)

    def _compress(self, content: str, max_tokens: int) -> str:
        prev, _, new = content.removeprefix("Existing summary:\n").partition("\n\nNew messages:\n")
        points = [] if prev.strip() == "(none)" else [prev.strip()]
        points += [
            line.removeprefix("User: ").strip()
            for line in new.splitlines()
            if line.startswith("User: ")
        ]
        text = " | ".join(p for p in points if p)
        budget = max(40, max_tokens * 4)
        return text if len(text) <= budget else "…" + text[-(budget - 1) :]

    def _chat(self, system: str, user_text: str) -> str:
        block = system.split("## Relevant memories\n", 1)[-1].strip()
        memories = [
            ln[2:].rsplit(" (relevance", 1)[0]
            for ln in block.splitlines()
            if ln[:2] == "- " and "Earlier conversation" not in ln
        ]
        if user_text.rstrip().endswith("?"):
            if memories:
                return "From what I remember: " + "; ".join(memories) + "."
            return "I don't have anything about that yet."
        return "Thanks, I'll remember that."


class ScriptedLLM:
    """Replays queued responses (or calls a function) and records every call for assertions."""

    def __init__(self, responses: Sequence[str] | Callable[..., str] = ()) -> None:
        self._responses = responses
        self._i = 0
        self.calls: list[dict] = []

    def complete(
        self, *, system: str, messages: Sequence[ChatMessage], max_tokens: int = 1024
    ) -> str:
        self.calls.append({"system": system, "messages": list(messages), "max_tokens": max_tokens})
        if callable(self._responses):
            return self._responses(system=system, messages=messages, max_tokens=max_tokens)
        resp = self._responses[min(self._i, len(self._responses) - 1)] if self._responses else ""
        self._i += 1
        return resp
