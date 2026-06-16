"""Belief bundle contract (spec §9).

The belief bundle is the handoff from Strata to the host model: small enough for live voice
use, explicit enough for predictable behavior, source-linked enough for audit. Retrieval
candidates are never passed through unfiltered as truth. Default wire format is JSON; the
flat ordered list of 5–15 entries is the baseline compatibility contract.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from typing import Optional


class Category(str, enum.Enum):
    CURRENT_BELIEF = "current_belief"
    RECENT_CONTEXT = "recent_context"
    INTERACTION_GUIDANCE = "interaction_guidance"
    OPEN_CONFLICT = "open_conflict"
    HYPOTHESIS = "hypothesis"
    POLICY_FLAG = "policy_flag"


class HostInstruction(str, enum.Enum):
    USE_DIRECTLY = "use_directly"
    USE_CAUTIOUSLY = "use_cautiously"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    DO_NOT_SURFACE = "do_not_surface"


# Default per-category budgets (spec §9 belief bundle contract).
DEFAULT_BUDGETS = {
    Category.CURRENT_BELIEF: (3, 8),
    Category.RECENT_CONTEXT: (3, 6),
    Category.INTERACTION_GUIDANCE: (1, 5),
    Category.OPEN_CONFLICT: (0, 3),
    Category.HYPOTHESIS: (0, 3),
}


@dataclass
class BundleEntry:
    id: int
    category: Category
    claim: str
    confidence: str = "medium"            # low | medium | high | numeric if requested
    source_ids: list[int] = field(default_factory=list)
    status: str = "active"                # active|hypothesis|contradicted|superseded|sensitive|policy-only
    time_valid: Optional[dict] = None     # {valid_from, valid_until} or recency note
    superseded_by: Optional[int] = None
    host_instruction: Optional[HostInstruction] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        if self.host_instruction is not None:
            d["host_instruction"] = self.host_instruction.value
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class BeliefBundle:
    current_beliefs: list[BundleEntry] = field(default_factory=list)
    recent_context: list[BundleEntry] = field(default_factory=list)
    interaction_guidance: list[BundleEntry] = field(default_factory=list)
    open_conflicts: list[BundleEntry] = field(default_factory=list)
    hypotheses: list[BundleEntry] = field(default_factory=list)
    policy_flags: list[BundleEntry] = field(default_factory=list)

    def all_entries(self) -> list[BundleEntry]:
        return (
            self.current_beliefs + self.recent_context + self.interaction_guidance
            + self.open_conflicts + self.hypotheses + self.policy_flags
        )

    def evidence_refs(self) -> dict[int, list[int]]:
        """Entry id -> supporting canonical source ids (required for durable claims)."""
        return {e.id: e.source_ids for e in self.all_entries() if e.source_ids}

    def to_dict(self) -> dict:
        return {
            "current_beliefs": [e.to_dict() for e in self.current_beliefs],
            "recent_context": [e.to_dict() for e in self.recent_context],
            "interaction_guidance": [e.to_dict() for e in self.interaction_guidance],
            "open_conflicts": [e.to_dict() for e in self.open_conflicts],
            "hypotheses": [e.to_dict() for e in self.hypotheses],
            "policy_flags": [e.to_dict() for e in self.policy_flags],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def serialized_size(self) -> int:
        return len(self.to_json().encode("utf-8"))
