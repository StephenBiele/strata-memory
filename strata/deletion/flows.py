"""Deletion + correction flows (spec §11).

Deletion is a multi-step state machine, not a best-effort helper. Canonical-first ordering is
mandatory: a record is tombstoned in the canonical store *before* any index mutation, so
resolver/hydration reject it immediately even while indexes still hold stale entries.

States (spec §11 Deletion Failure Semantics):
    requested → canonical_tombstoned → index_deleting → (verified | partial_failure)
    partial_failure → (verified after retry | manual_attention)

This synchronous service performs the canonical-first ordering, per-ID surface purge, and
verification. Durable retry-with-backoff is owned by the Write Coordinator (step 5); here we
do bounded in-line retries and escalate to ``manual_attention`` when exhausted.
"""

from __future__ import annotations

import enum
import time
from typing import Optional

from strata.canonical.records import MemoryRecord, Relation, Status
from strata.canonical.store import CanonicalStore
from strata.deletion.managed_surfaces import ManagedSurfaceRegistry
from strata.deletion.verify import verify_not_retrievable
from strata.ids import new_operation_id
from strata.lexical.fts import LexicalStore


class DeletionState(str, enum.Enum):
    REQUESTED = "requested"
    CANONICAL_TOMBSTONED = "canonical_tombstoned"
    INDEX_DELETING = "index_deleting"
    PARTIAL_FAILURE = "partial_failure"
    VERIFIED = "verified"
    MANUAL_ATTENTION = "manual_attention"


def _now_ms() -> int:
    return int(time.time() * 1000)


class DeletionService:
    def __init__(
        self,
        store: CanonicalStore,
        registry: ManagedSurfaceRegistry,
        *,
        lexical: Optional[LexicalStore] = None,
        max_retries: int = 3,
    ) -> None:
        self.store = store
        self.registry = registry
        self.lexical = lexical
        self.max_retries = max_retries

    # -- deletion --------------------------------------------------------------
    def request_deletion(
        self,
        record_id: int,
        *,
        mode: str = "logical",
        include_dependents: bool = True,
    ) -> str:
        """Tombstone canonical-first, then purge + verify managed surfaces.

        Returns a job_id usable with :meth:`deletion_status`.
        """
        job_id = new_operation_id()
        ids = {record_id}
        if include_dependents:
            ids |= self.store.dependency_graph(record_id)

        # 1) canonical-first: block from recall before mutating any index.
        for rid in ids:
            self.store.tombstone(
                rid, mode=mode, job_id=job_id, state=DeletionState.CANONICAL_TOMBSTONED.value
            )

        self._purge_and_verify(job_id, ids, mode, attempt=0)
        return job_id

    def retry(self, job_id: str) -> str:
        """Re-attempt purge/verify for a job currently in partial_failure/manual_attention."""
        rows = self.store._conn().execute(
            "SELECT record_id, mode FROM tombstones WHERE job_id = ?", (job_id,)
        ).fetchall()
        if not rows:
            raise KeyError(f"unknown deletion job {job_id}")
        ids = {r["record_id"] for r in rows}
        mode = rows[0]["mode"]
        self._purge_and_verify(job_id, ids, mode, attempt=0)
        return job_id

    def _purge_and_verify(self, job_id: str, ids: set[int], mode: str, *, attempt: int) -> None:
        for rid in ids:
            self.store.set_tombstone_state(rid, DeletionState.INDEX_DELETING.value)

        acks = self.registry.purge(ids)  # {surface: {id: ok}}
        for surface_name, per_id in acks.items():
            for rid, ok in per_id.items():
                self.store.record_index_ack(
                    rid, surface_name, job_id, "success" if ok else "failure"
                )

        failed_ids = {
            rid for per_id in acks.values() for rid, ok in per_id.items() if not ok
        }

        if not failed_ids:
            # Hard mode erases canonical content first, then verifies the row is physically
            # gone; logical mode keeps the (tombstoned) row and verifies it stays blocked.
            if mode == "hard":
                self._hard_erase(ids)
            if self._verify(ids, mode):
                now = _now_ms()
                for rid in ids:
                    self.store.set_tombstone_state(
                        rid, DeletionState.VERIFIED.value, verified_at=now
                    )
                return

        # Failure path: bounded retry, else escalate.
        if attempt < self.max_retries:
            for rid in ids:
                self.store.set_tombstone_state(rid, DeletionState.PARTIAL_FAILURE.value)
            self._purge_and_verify(job_id, ids, mode, attempt=attempt + 1)
        else:
            for rid in ids:
                self.store.set_tombstone_state(rid, DeletionState.MANUAL_ATTENTION.value)

    def _verify(self, ids: set[int], mode: str) -> bool:
        """Confirm deleted content is not retrievable through supported recall APIs."""
        return verify_not_retrievable(self.store, self.registry, ids, mode=mode)

    def _hard_erase(self, ids: set[int]) -> None:
        """Physically erase canonical content + scrub the lexical shadow (spec §11)."""
        for rid in ids:
            self.store.hard_delete(rid)
        if self.lexical is not None:
            # Rebuild scrubs the FTS shadow of any residue from the erased rows.
            self.lexical.rebuild()

    # -- status ----------------------------------------------------------------
    def deletion_status(self, job_id: str) -> dict:
        """Aggregate the job's per-record tombstone states into one job status (spec §11)."""
        rows = self.store._conn().execute(
            "SELECT record_id, mode, state, verified_at FROM tombstones WHERE job_id = ?",
            (job_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f"unknown deletion job {job_id}")
        states = [r["state"] for r in rows]
        # Job rolls up to the least-complete member state.
        order = [
            DeletionState.MANUAL_ATTENTION,
            DeletionState.PARTIAL_FAILURE,
            DeletionState.INDEX_DELETING,
            DeletionState.CANONICAL_TOMBSTONED,
            DeletionState.REQUESTED,
            DeletionState.VERIFIED,
        ]
        job_state = next(s.value for s in order if s.value in states)
        return {
            "job_id": job_id,
            "mode": rows[0]["mode"],
            "state": job_state,
            "record_ids": [r["record_id"] for r in rows],
            "verified": job_state == DeletionState.VERIFIED.value,
        }

    # -- correction ------------------------------------------------------------
    def correct(
        self,
        old_id: int,
        new_record: MemoryRecord,
        *,
        relation: Relation = Relation.SUPERSEDES,
        keep_history_in_index: bool = False,
    ) -> MemoryRecord:
        """Correction flow (spec §11): write new belief, mark old, drop old from active
        indexes (unless historical recall is explicitly allowed), preserve audit links.
        """
        written = self.store.write(new_record)
        old_status = Status.SUPERSEDED if relation is Relation.SUPERSEDES else Status.CONTRADICTED
        self.store.set_status(old_id, old_status)
        self.store.add_dependency(written.id, old_id, relation)
        if self.lexical is not None:
            self.lexical.index(written)
        if not keep_history_in_index:
            # Remove the old record from active indexes; canonical history is preserved.
            self.registry.purge({old_id})
        return written
