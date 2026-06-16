"""Resolver — turn retrieval candidates into a defined belief bundle (spec §9).

Live recall uses *deterministic* status, validity, and supersession rules (spec §10); no LLM
in this path. The resolver:

1. Merges duplicate candidate IDs (repeated evidence strengthens confidence).
2. Hydrates from canonical and DROPS anything that cannot be backed by canonical truth:
   missing rows (dangling IDs), tombstoned records, or content_hash drift. (Invariant: never
   surface a dangling ID; every recalled ID resolves to canonical text or is dropped.)
3. Filters by time validity and host permission.
4. Resolves conflicts deterministically: a record superseded by an active record is excluded
   from current truth (a single explicit correction overrides repeated old evidence); records
   with an unresolved ``contradicts`` edge move to open_conflicts; non-contradictory scoped
   claims (no edge) both survive.
5. Routes survivors into belief-bundle categories and trims to per-category budgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from strata.canonical.records import MemoryRecord, RecordType, Status, Tier
from strata.canonical.store import CanonicalStore
from strata.resolver.bundle import (
    DEFAULT_BUDGETS,
    BeliefBundle,
    BundleEntry,
    Category,
    HostInstruction,
)

_ACTIVE = (Status.ACTIVE, Status.REINFORCED)
_STATUS_WEIGHT = {Status.REINFORCED: 1.1, Status.ACTIVE: 1.0}


@dataclass
class Candidate:
    """A retrieval candidate from zvec / TurboVec / FTS5 / structured filters."""

    record_id: int
    score: float = 0.0
    source: str = "structured"
    content_hash: Optional[str] = None  # what the index believes; checked against canonical


@dataclass
class _Merged:
    record_id: int
    best_score: float
    sources: list[str]
    content_hashes: set[str]
    evidence_count: int = field(default=1)


def _confidence_label(value: Optional[float], bump: int = 0) -> str:
    base = 1 if value is None else (0 if value < 0.4 else (2 if value >= 0.75 else 1))
    level = max(0, min(2, base + bump))
    return ("low", "medium", "high")[level]


class Resolver:
    def __init__(self, store: CanonicalStore, *, now_fn: Callable[[], int] = lambda: int(time.time() * 1000)):
        self.store = store
        self.now_fn = now_fn

    # -- public ----------------------------------------------------------------
    def resolve(
        self,
        candidates: Iterable[Candidate],
        *,
        now: Optional[int] = None,
        permit: Optional[Callable[[MemoryRecord], bool]] = None,
        budgets: dict = DEFAULT_BUDGETS,
        diversity: bool = False,
        diversity_threshold: float = 0.85,
    ) -> BeliefBundle:
        now = now if now is not None else self.now_fn()
        merged = self._merge(candidates)
        survivors = self._hydrate_and_filter(merged, now, permit)
        return self._build_bundle(
            survivors, budgets, diversity=diversity, diversity_threshold=diversity_threshold
        )

    # -- pipeline --------------------------------------------------------------
    def _merge(self, candidates: Iterable[Candidate]) -> dict[int, _Merged]:
        out: dict[int, _Merged] = {}
        for c in candidates:
            m = out.get(c.record_id)
            if m is None:
                out[c.record_id] = _Merged(
                    record_id=c.record_id,
                    best_score=c.score,
                    sources=[c.source],
                    content_hashes={c.content_hash} if c.content_hash else set(),
                )
            else:
                m.best_score = max(m.best_score, c.score)
                m.sources.append(c.source)
                m.evidence_count += 1
                if c.content_hash:
                    m.content_hashes.add(c.content_hash)
        return out

    def _hydrate_and_filter(
        self,
        merged: dict[int, _Merged],
        now: int,
        permit: Optional[Callable[[MemoryRecord], bool]],
    ) -> list[tuple[MemoryRecord, _Merged]]:
        records = {r.id: r for r in self.store.read(merged.keys())}
        survivors: list[tuple[MemoryRecord, _Merged]] = []
        for rid, m in merged.items():
            rec = records.get(rid)
            if rec is None:                                  # dangling ID -> drop
                continue
            if self.store.is_tombstoned(rid):                # deletion invariant -> drop
                continue
            if m.content_hashes and rec.content_hash not in m.content_hashes:
                continue                                     # hydration integrity -> drop
            if not self._time_valid(rec, now):               # validity window -> drop
                continue
            if permit is not None and not permit(rec):       # permission policy -> drop
                continue
            survivors.append((rec, m))
        return survivors

    @staticmethod
    def _time_valid(rec: MemoryRecord, now: int) -> bool:
        if rec.valid_from is not None and now < rec.valid_from:
            return False
        if rec.valid_until is not None and now >= rec.valid_until:
            return False
        return True

    def _is_superseded_by_active(self, record_id: int) -> bool:
        for sid in self.store.superseders_of(record_id):
            sup = self.store.read_one(sid)
            if sup is not None and sup.status in _ACTIVE and not self.store.is_tombstoned(sid):
                return True
        return False

    def _active_contradictors(self, record_id: int, present_ids: set[int]) -> list[int]:
        out = []
        for cid in self.store.contradictors_of(record_id):
            other = self.store.read_one(cid)
            if other is not None and other.status in _ACTIVE and not self.store.is_tombstoned(cid):
                out.append(cid)
        return out

    def _build_bundle(
        self,
        survivors: list[tuple[MemoryRecord, _Merged]],
        budgets: dict,
        *,
        diversity: bool = False,
        diversity_threshold: float = 0.85,
    ) -> BeliefBundle:
        bundle = BeliefBundle()
        present_ids = {rec.id for rec, _ in survivors}

        # Rank by status weight, confidence, recency, retrieval score, evidence count.
        def sort_key(item):
            rec, m = item
            return (
                _STATUS_WEIGHT.get(rec.status, 0.0),
                rec.confidence or 0.0,
                rec.updated_at,
                m.best_score,
                m.evidence_count,
            )

        ranked = sorted(survivors, key=sort_key, reverse=True)
        if diversity:
            # Post-hydration diversity over canonical text (never on quantized vectors).
            from strata.resolver.diversity import diversify
            keep = set(diversify([rec.content for rec, _ in ranked], threshold=diversity_threshold))
            ranked = [item for i, item in enumerate(ranked) if i in keep]

        for rec, m in ranked:
            # Reflections are hypotheses, never facts (spec §9).
            if rec.record_type is RecordType.REFLECTION:
                bundle.hypotheses.append(self._entry(rec, m, Category.HYPOTHESIS))
                continue
            if rec.record_type is RecordType.GUARDRAIL or rec.tier is Tier.L4:
                bundle.policy_flags.append(self._entry(rec, m, Category.POLICY_FLAG))
                continue

            # Only active/reinforced records can be current truth.
            if rec.status not in _ACTIVE:
                continue
            # A single explicit correction (active superseder) overrides repeated old evidence.
            if self._is_superseded_by_active(rec.id):
                continue
            # Unresolved contradiction -> open conflict, not current truth.
            if self._active_contradictors(rec.id, present_ids):
                bundle.open_conflicts.append(
                    self._entry(rec, m, Category.OPEN_CONFLICT, host=HostInstruction.USE_CAUTIOUSLY,
                                status="contradicted")
                )
                continue

            category = self._category_for(rec)
            getattr(bundle, _CATEGORY_FIELD[category]).append(self._entry(rec, m, category))

        self._apply_budgets(bundle, budgets)
        return bundle

    @staticmethod
    def _category_for(rec: MemoryRecord) -> Category:
        if rec.record_subtype == "interaction_guidance" or rec.tier is Tier.L3 and rec.record_subtype == "guidance":
            return Category.INTERACTION_GUIDANCE
        if rec.tier in (Tier.L0, Tier.L2) or rec.record_type is RecordType.EPISODE:
            return Category.RECENT_CONTEXT
        return Category.CURRENT_BELIEF

    def _entry(
        self,
        rec: MemoryRecord,
        m: _Merged,
        category: Category,
        *,
        host: Optional[HostInstruction] = None,
        status: Optional[str] = None,
    ) -> BundleEntry:
        bump = 1 if (m.evidence_count >= 2 or (rec.salience or 0) >= 0.8) else 0
        superseders = self.store.superseders_of(rec.id)
        time_valid = None
        if rec.valid_from is not None or rec.valid_until is not None:
            time_valid = {"valid_from": rec.valid_from, "valid_until": rec.valid_until}
        return BundleEntry(
            id=rec.id,
            category=category,
            claim=rec.content,
            confidence=_confidence_label(rec.confidence, bump),
            source_ids=[rec.id],
            status=status or (
                "hypothesis" if category is Category.HYPOTHESIS
                else "policy-only" if category is Category.POLICY_FLAG
                else rec.status.value
            ),
            time_valid=time_valid,
            superseded_by=superseders[0] if superseders else None,
            host_instruction=host or (
                HostInstruction.USE_CAUTIOUSLY if category is Category.HYPOTHESIS else None
            ),
        )

    @staticmethod
    def _apply_budgets(bundle: BeliefBundle, budgets: dict) -> None:
        for cat, field_name in _CATEGORY_FIELD.items():
            if cat not in budgets:
                continue
            _, max_n = budgets[cat]
            entries = getattr(bundle, field_name)
            if len(entries) > max_n:
                setattr(bundle, field_name, entries[:max_n])


_CATEGORY_FIELD = {
    Category.CURRENT_BELIEF: "current_beliefs",
    Category.RECENT_CONTEXT: "recent_context",
    Category.INTERACTION_GUIDANCE: "interaction_guidance",
    Category.OPEN_CONFLICT: "open_conflicts",
    Category.HYPOTHESIS: "hypotheses",
    Category.POLICY_FLAG: "policy_flags",
}
