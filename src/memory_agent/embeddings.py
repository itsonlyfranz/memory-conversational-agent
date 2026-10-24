"""Local, deterministic text embeddings.

``HashingEmbedder`` uses the hashing trick over stemmed word unigrams, bigrams and character
trigrams, so it needs no model download or network. It captures lexical overlap (including
morphology like live/lives/living), not deep semantics; swap in any ``Embedder`` to change that.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_WORD = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """a an the is are was were be been am do does did you your i me my we our it its to of in on
    at and or but for with that this these those what which who whom where when why how can could
    would should will about please tell know remember user he she they them his her""".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def _stem(word: str) -> str:
    """Light suffix stripping so live/lives/living/lived share a stem."""
    word = word.replace("'s", "").replace("'", "")
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + repl
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _add(self, vec: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[h % self.dim] += sign * weight

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        for tok in tokens:
            self._add(vec, "w:" + tok, 1.0)
            padded = f"#{tok}#"
            for i in range(len(padded) - 2):
                self._add(vec, "c:" + padded[i : i + 3], 0.3)
        for a, b in zip(tokens, tokens[1:], strict=False):
            self._add(vec, f"b:{a}_{b}", 0.5)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [round(x / norm, 5) for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
