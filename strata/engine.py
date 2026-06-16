"""MemoryEngine — core wiring of canonical store, lexical, vector, coordinator, resolver,
and deletion into the recall/write/delete/correct flows (spec §6, §9, §11, §12).

All index mutation flows through the Write Coordinator (single writer). Recall fans out to
vector + lexical, hydrates through the resolver, and returns a belief bundle. This is the
object the gateway/CLI (step 9) sit on top of.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from strata.canonical.records import MemoryRecord, Relation
from strata.canonical.store import CanonicalStore
from strata.coordinator.coordinator import WriteCoordinator
from strata.coordinator.ops import Operation, OpType
from strata.deletion.flows import DeletionService
from strata.deletion.managed_surfaces import (
    CoordinatedVectorSurface,
    DerivedArtifactSurface,
    LexicalSurface,
    ManagedSurfaceRegistry,
)
from strata.lexical.fts import LexicalStore
from strata.resolver.bundle import BeliefBundle
from strata.resolver.resolver import Candidate, Resolver
from strata.vector.base import MemoryVectorStore, VectorRecord


def _now_ms() -> int:
    return int(time.time() * 1000)


class MemoryEngine:
    def __init__(
        self,
        store: CanonicalStore,
        lexical: LexicalStore,
        vector_stores: dict[str, MemoryVectorStore],
        coordinator: WriteCoordinator,
        resolver: Resolver,
        embedder,
        *,
        active_generation: int = 1,
        artifacts: Optional[DerivedArtifactSurface] = None,
        policy=None,
    ) -> None:
        from strata.policy.sensitivity import SensitivityPolicy

        self.store = store
        self.lexical = lexical
        self.vector_stores = vector_stores
        self.coordinator = coordinator
        self.resolver = resolver
        self.embedder = embedder
        self.active_generation = active_generation
        self.artifacts = artifacts or DerivedArtifactSurface()
        self.policy = policy or SensitivityPolicy()

        # Register the active embedding generation.
        self.store._conn().execute(
            "INSERT OR IGNORE INTO embedding_generations (generation, embedding_model_id, status, created_at) "
            "VALUES (?, ?, 'active', ?)",
            (active_generation, embedder.model_id, _now_ms()),
        )
        self.store._conn().commit()

        # Wire coordinator handlers + managed-surface registry.
        self.registry = ManagedSurfaceRegistry([LexicalSurface(lexical)])
        for name, vstore in vector_stores.items():
            coordinator.register_handler(name, self._make_handler(vstore))
            self.registry.register(CoordinatedVectorSurface(name, vstore, coordinator))
        self.registry.register(self.artifacts)

        self.deletion = DeletionService(store, self.registry, lexical=lexical)

    # -- coordinator handler ---------------------------------------------------
    def _make_handler(self, vstore: MemoryVectorStore):
        def handler(op: Operation) -> dict[int, bool]:
            if op.op_type is OpType.UPSERT_INDEX:
                records = [VectorRecord(**r) for r in (op.payload or {}).get("records", [])]
                return vstore.upsert(records)
            if op.op_type in (OpType.REMOVE_INDEX, OpType.HARD_PURGE):
                return vstore.remove(op.target_ids)
            raise ValueError(f"vector handler cannot apply {op.op_type}")
        return handler

    # -- write -----------------------------------------------------------------
    def write_memory(self, record: MemoryRecord, *, generation: Optional[int] = None) -> MemoryRecord:
        gen = generation if generation is not None else self.active_generation
        written = self.store.write(record)
        self.lexical.index(written)
        self._enqueue_vector_upsert(written, gen)
        self.coordinator.run_until_idle()
        return written

    def _enqueue_vector_upsert(self, record: MemoryRecord, generation: int) -> None:
        vrec = {
            "id": record.id,
            "vector": self.embedder.embed(record.content),
            "content_hash": record.content_hash,
            "embedding_model_id": self.embedder.model_id,
            "embedding_generation": generation,
            "metadata": {
                "status": record.status.value,
                "tier": record.tier.value,
                "sensitivity": record.sensitivity.value,
            },
        }
        for name in self.vector_stores:
            self.coordinator.enqueue(
                Operation(
                    op_type=OpType.UPSERT_INDEX,
                    target_ids=[record.id],
                    target=name,
                    payload={"records": [vrec]},
                    expected_generation=generation,
                )
            )

    # -- recall ----------------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        generation: Optional[int] = None,
        permit: Optional[Callable[[MemoryRecord], bool]] = None,
        diversity: bool = False,
    ) -> BeliefBundle:
        gen = generation if generation is not None else self.active_generation
        qv = self.embedder.embed(query)
        candidates: list[Candidate] = []
        for name, vstore in self.vector_stores.items():
            for hit in vstore.search(qv, top_k=top_k, generation=gen):
                candidates.append(
                    Candidate(hit.id, score=hit.score, source=name, content_hash=hit.content_hash)
                )
        for rec, score in self.lexical.search(query, top_k=top_k):
            candidates.append(
                Candidate(rec.id, score=-score, source="fts5", content_hash=rec.content_hash)
            )
        # Default permission filter is the sensitivity policy (MVP enforcement floor).
        permit = permit if permit is not None else self.policy.recall_permit()
        return self.resolver.resolve(candidates, permit=permit, diversity=diversity)

    # -- correction / deletion -------------------------------------------------
    def correct(self, old_id: int, new_record: MemoryRecord, *, relation: Relation = Relation.SUPERSEDES) -> MemoryRecord:
        written = self.deletion.correct(old_id, new_record, relation=relation)
        self._enqueue_vector_upsert(written, self.active_generation)
        self.coordinator.run_until_idle()
        return written

    def delete_memory(self, record_id: int, *, mode: str = "logical") -> str:
        return self.deletion.request_deletion(record_id, mode=mode)

    def merge_records(self, primary_id: int, duplicate_ids: list[int]) -> None:
        """Consolidate duplicates into a primary record (spec §7 L1.5 output).

        Duplicates are marked superseded, linked to the primary for audit, and removed from
        active indexes via the managed surfaces (vector removal is coordinator-routed). A
        deterministic guard skips any tombstoned id so reflection never resurrects deletions.
        """
        from strata.canonical.records import Status

        to_remove = []
        for d in duplicate_ids:
            if d == primary_id or self.store.is_tombstoned(d):
                continue
            self.store.set_status(d, Status.SUPERSEDED)
            self.store.add_dependency(primary_id, d, Relation.SUPERSEDES)
            to_remove.append(d)
        if to_remove:
            self.deletion.registry.purge(set(to_remove))

    def deletion_status(self, job_id: str) -> dict:
        return self.deletion.deletion_status(job_id)

    # -- migration hooks -------------------------------------------------------
    def open_shadow_generation(self, generation: int, embedder) -> None:
        """Begin a new embedding generation as a shadow index (spec §8 migration)."""
        self.store._conn().execute(
            "INSERT OR IGNORE INTO embedding_generations (generation, embedding_model_id, status, created_at) "
            "VALUES (?, ?, 'shadow', ?)",
            (generation, embedder.model_id, _now_ms()),
        )
        self.store._conn().commit()

    def reindex_into(self, generation: int, embedder, record_ids: list[int]) -> None:
        """Re-embed canonical records into a (shadow) generation via the coordinator.

        Parity checks must re-embed source text from canonical (spec §8); we read content from
        the canonical store, never from any index.
        """
        for rec in self.store.read(record_ids):
            if self.store.is_tombstoned(rec.id):
                continue  # deletion preservation across generations
            vrec = {
                "id": rec.id,
                "vector": embedder.embed(rec.content),
                "content_hash": rec.content_hash,
                "embedding_model_id": embedder.model_id,
                "embedding_generation": generation,
                "metadata": {
                    "status": rec.status.value,
                    "tier": rec.tier.value,
                    "sensitivity": rec.sensitivity.value,
                },
            }
            for name in self.vector_stores:
                self.coordinator.enqueue(
                    Operation(
                        op_type=OpType.UPSERT_INDEX,
                        target_ids=[rec.id],
                        target=name,
                        payload={"records": [vrec]},
                        expected_generation=generation,
                    )
                )
        self.coordinator.run_until_idle()

    def promote_generation(self, generation: int) -> None:
        """Promote a shadow generation to active after parity passes (spec §8)."""
        conn = self.store._conn()
        conn.execute("UPDATE embedding_generations SET status='retired' WHERE status='active'")
        conn.execute("UPDATE embedding_generations SET status='active' WHERE generation=?", (generation,))
        conn.commit()
        self.active_generation = generation
