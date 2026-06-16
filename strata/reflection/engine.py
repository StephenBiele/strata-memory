"""Background reflection engine (spec §10).

Runs outside the live response path. Improves continuity without inventing unsupported claims:
nightly consolidation (dedup/merge) and contradiction audit. It is **enqueue-only** with
respect to index mutation — it proposes and applies via the engine, which routes index writes
through the single Write Coordinator (it never holds a vector write lock itself).

Safety rails:
* Reflection only considers records within the sensitivity ceiling (policy.can_reflect_on).
* A deterministic guard skips tombstoned records so deletions never re-enter recall.
* Deletion / privacy / guardrail records are never modified by proposals.
"""

from __future__ import annotations

from typing import Optional

from strata.canonical.records import MemoryRecord, RecordType, Status, Tier
from strata.reflection.buffer import cluster_l1, review_worthy
from strata.reflection.proposals import (
    Proposal,
    ProposalKind,
    ProposalState,
    ProposalStore,
)

_ACTIVE = (Status.ACTIVE, Status.REINFORCED)


class ReflectionEngine:
    def __init__(
        self,
        engine,
        proposals: Optional[ProposalStore] = None,
        *,
        auto_accept_threshold: float = 0.9,
    ) -> None:
        self.engine = engine
        self.store = engine.store
        self.policy = engine.policy
        self.proposals = proposals or ProposalStore()
        self.auto_accept_threshold = auto_accept_threshold

    # -- inputs ----------------------------------------------------------------
    def _reflectable_l1_facts(self) -> list[MemoryRecord]:
        out = []
        for rec in self.store.query(tier=Tier.L1, record_type=RecordType.FACT):
            if rec.status not in _ACTIVE:
                continue
            if self.store.is_tombstoned(rec.id):
                continue
            if not self.policy.can_reflect_on(rec):
                continue
            out.append(rec)
        return out

    # -- consolidation ---------------------------------------------------------
    def consolidate(self) -> list[str]:
        """Cluster duplicate/related L1 facts and propose merges; auto-accept low-risk dups."""
        records = self._reflectable_l1_facts()
        by_id = {r.id: r for r in records}
        proposal_ids: list[str] = []
        for cluster in review_worthy(cluster_l1(records)):
            members = [by_id[i] for i in cluster.record_ids]
            # Primary = highest salience, then longest content (most informative).
            primary = max(members, key=lambda r: ((r.salience or 0), len(r.content)))
            dups = [m.id for m in members if m.id != primary.id]
            # Exact-content duplicates (shared content_hash) are low-risk -> auto-merge.
            # Otherwise confidence rises with intra-cluster agreement and requires review.
            has_exact_dup = len({m.content_hash for m in members}) < len(members)
            confidence = 0.95 if has_exact_dup else min(0.89, 0.6 + 0.1 * len(dups))
            prop = Proposal(
                kind=ProposalKind.MERGE,
                record_ids=[primary.id, *dups],
                evidence_ids=cluster.record_ids,
                confidence=confidence,
                rationale=f"{len(members)} near-duplicate L1 facts consolidated",
                target_content=primary.content,
            )
            self.proposals.add(prop)
            proposal_ids.append(prop.id)
            if confidence >= self.auto_accept_threshold:
                self.apply(prop.id)
            else:
                self.proposals.set_state(prop.id, ProposalState.USER_REVIEW_REQUIRED)
        return proposal_ids

    # -- contradiction audit ---------------------------------------------------
    def contradiction_audit(self) -> list[str]:
        """Flag unresolved explicit contradictions for review (deterministic; spec §10)."""
        proposal_ids: list[str] = []
        seen: set[frozenset] = set()
        for rec in self.store.query():
            if rec.status not in _ACTIVE or self.store.is_tombstoned(rec.id):
                continue
            for other_id in self.store.contradictors_of(rec.id):
                other = self.store.read_one(other_id)
                if other is None or other.status not in _ACTIVE or self.store.is_tombstoned(other_id):
                    continue
                key = frozenset({rec.id, other_id})
                if key in seen:
                    continue
                seen.add(key)
                prop = Proposal(
                    kind=ProposalKind.CLARIFY_CONFLICT,
                    record_ids=[rec.id, other_id],
                    evidence_ids=[rec.id, other_id],
                    confidence=0.5,
                    rationale="unresolved contradiction between active records",
                    state=ProposalState.USER_REVIEW_REQUIRED,
                )
                self.proposals.add(prop)
                proposal_ids.append(prop.id)
        return proposal_ids

    # -- apply / review --------------------------------------------------------
    def apply(self, proposal_id: str) -> None:
        prop = self.proposals.get(proposal_id)
        if prop is None or prop.state in (ProposalState.APPLIED, ProposalState.REJECTED):
            return
        # Guardrails / deletions are never overridden by proposals.
        if prop.kind is ProposalKind.MERGE:
            primary, *dups = prop.record_ids
            self.engine.merge_records(primary, dups)
            self.proposals.set_state(proposal_id, ProposalState.APPLIED)
        elif prop.kind is ProposalKind.CLARIFY_CONFLICT:
            # Clarifications require a user decision; applying is a no-op here.
            self.proposals.set_state(proposal_id, ProposalState.USER_REVIEW_REQUIRED)

    def reject(self, proposal_id: str) -> None:
        self.proposals.set_state(proposal_id, ProposalState.REJECTED)
