"""Memory record model and lifecycle enums (spec §8).

The canonical store is the *only* place record content lives. A ``MemoryRecord`` is the
head-row view of a record; version history, supersession/contradiction edges, source spans,
and per-adapter index acknowledgements live in their own tables (see ``schema.sql``).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from strata.ids import content_hash as _content_hash
from strata.ids import new_record_id


class RecordType(str, enum.Enum):
    FACT = "fact"
    EPISODE = "episode"
    PROFILE = "profile"
    GUARDRAIL = "guardrail"
    REFLECTION = "reflection"
    AGGREGATION = "aggregation"
    SKILL = "skill"          # post-MVP extension record type (represented, not tiered)
    ROUTINE = "routine"      # post-MVP extension record type
    SYSTEM = "system"


class Tier(str, enum.Enum):
    L0 = "L0"        # raw events
    L1 = "L1"        # atomic facts
    L1_5 = "L1.5"    # aggregation buffer
    L2 = "L2"        # episodes/scenarios
    L3 = "L3"        # continuity model
    L4 = "L4"        # deterministic guards


class Status(str, enum.Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    REINFORCED = "reinforced"
    STALE = "stale"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Sensitivity(str, enum.Enum):
    """Minimal hard-coded sensitivity classes (spec §13 enforcement floor)."""

    NORMAL = "normal"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


# Relationship edge kinds stored in the ``dependencies`` table.
class Relation(str, enum.Enum):
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    SOURCE_OF = "source_of"
    AGGREGATES = "aggregates"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class MemoryRecord:
    """Head-row view of a canonical record (spec §8 fields).

    ``id`` is allocated lazily on first write if not supplied. Timestamps are epoch
    milliseconds. ``content_hash`` is recomputed from ``content`` whenever the record is
    constructed via :meth:`create` so it always matches the stored text.
    """

    content: str
    record_type: RecordType = RecordType.FACT
    tier: Tier = Tier.L1
    status: Status = Status.ACTIVE
    id: Optional[int] = None
    record_subtype: Optional[str] = None
    confidence: Optional[float] = None
    confidence_reason: Optional[str] = None
    salience: Optional[float] = None
    sensitivity: Sensitivity = Sensitivity.NORMAL
    content_hash: str = ""
    embedding_model_id: Optional[str] = None
    embedding_generation: Optional[int] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    created_at: int = field(default_factory=_now_ms)
    updated_at: int = field(default_factory=_now_ms)
    current_version_id: Optional[int] = None
    model_side: Optional[str] = None  # nullable; fwd-compat with Strata Persona

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _content_hash(self.content)

    @classmethod
    def create(cls, content: str, **kwargs) -> "MemoryRecord":
        """Build a new record with a fresh ID and a content hash matching ``content``."""
        rec = cls(content=content, **kwargs)
        if rec.id is None:
            rec.id = new_record_id()
        rec.content_hash = _content_hash(rec.content)
        return rec

    def with_content(self, content: str, *, status: Optional[Status] = None) -> "MemoryRecord":
        """Return a copy with new content (hash + updated_at refreshed). Keeps the same id."""
        now = _now_ms()
        return replace(
            self,
            content=content,
            content_hash=_content_hash(content),
            status=status or self.status,
            updated_at=now,
        )
