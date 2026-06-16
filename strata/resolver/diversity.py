"""Post-hydration diversity (spec §9).

TurboVec discards full-precision vectors after 2–4 bit quantization, so max-marginal-relevance
and other diversity-aware reranking cannot run on the index vectors. Diversity is therefore
applied AFTER hydrating candidates from the canonical store — over canonical text, not index
vectors. This module provides a simple token-Jaccard de-duplication used for that pass.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def diversify(texts_in_rank_order: list[str], *, threshold: float = 0.85) -> list[int]:
    """Return the indices to KEEP: drop any item too similar to an already-kept higher-ranked
    item. Operates on hydrated canonical text (never on quantized vectors)."""
    kept: list[int] = []
    kept_texts: list[str] = []
    for i, text in enumerate(texts_in_rank_order):
        if any(jaccard(text, k) >= threshold for k in kept_texts):
            continue
        kept.append(i)
        kept_texts.append(text)
    return kept
