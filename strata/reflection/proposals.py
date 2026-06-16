"""Reflection / conflict proposals and their review states (spec §10).

Reflections and conflict decisions are stored as reversible proposals with evidence and
confidence. Deletion, privacy, and guardrail operations are NEVER overridden by proposals.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from strata.ids import new_operation_id


class ProposalState(str, enum.Enum):
    PROPOSED = "proposed"
    AUTO_ACCEPTED = "auto_accepted"
    USER_REVIEW_REQUIRED = "user_review_required"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class ProposalKind(str, enum.Enum):
    MERGE = "merge"                  # de-duplicate / consolidate L1 records
    SUPERSEDE = "supersede"          # newer evidence replaces older
    CLARIFY_CONFLICT = "clarify_conflict"  # unresolved contradiction
    PROFILE_UPDATE = "profile_update"


@dataclass
class Proposal:
    kind: ProposalKind
    record_ids: list[int]
    confidence: float
    evidence_ids: list[int] = field(default_factory=list)
    rationale: str = ""
    target_content: Optional[str] = None      # e.g. merged content for MERGE
    state: ProposalState = ProposalState.PROPOSED
    id: str = field(default_factory=new_operation_id)


class ProposalStore:
    """In-memory proposal store; also a managed surface (reflection proposals + evidence)."""

    name = "reflection_proposals"

    def __init__(self) -> None:
        self._by_id: dict[str, Proposal] = {}

    def add(self, proposal: Proposal) -> str:
        self._by_id[proposal.id] = proposal
        return proposal.id

    def get(self, proposal_id: str) -> Optional[Proposal]:
        return self._by_id.get(proposal_id)

    def list(self, *, state: Optional[ProposalState] = None) -> list[Proposal]:
        return [p for p in self._by_id.values() if state is None or p.state is state]

    def set_state(self, proposal_id: str, state: ProposalState) -> None:
        self._by_id[proposal_id].state = state

    # -- managed surface (deletion must purge proposals derived from deleted records) ------
    def remove(self, ids):
        ids = set(ids)
        out = {}
        for i in ids:
            for p in list(self._by_id.values()):
                if i in p.record_ids or i in p.evidence_ids:
                    self._by_id.pop(p.id, None)
            out[i] = self.absent(i)
        return out

    def absent(self, record_id: int) -> bool:
        return not any(
            record_id in p.record_ids or record_id in p.evidence_ids
            for p in self._by_id.values()
        )
