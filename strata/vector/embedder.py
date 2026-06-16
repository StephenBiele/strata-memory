"""Embedder interface + deterministic default (locked decision: pluggable, offline default).

The whole MVP runs offline and reproducibly: the default embedder hashes tokens into a fixed
vector with no model download. A host may inject a real model later. Every vector is tagged
with the embedder's ``model_id`` so mixed-generation comparisons are detectable (spec §8).
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, text: str) -> list[float]:
        ...


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        # Different embedding generations are not directly comparable (spec §8).
        raise ValueError("cosine across different embedding dimensions is undefined")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_TOKEN = __import__("re").compile(r"[\w]+", __import__("re").UNICODE)


class DeterministicHashEmbedder:
    """Reproducible bag-of-hashed-tokens embedder. Identical text -> identical vector."""

    def __init__(self, model_id: str = "det-hash-v1", dim: int = 64, seed: int = 0) -> None:
        self.model_id = model_id
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.sha1(f"{self.seed}:{tok}".encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        return vec
