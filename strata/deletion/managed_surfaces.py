"""Managed-surfaces registry (spec §11).

The deletion guarantee applies only to surfaces Strata manages. To make the guarantee
precise, deletion/reconciliation/verification iterate an explicit registry of surfaces that
mirror or derive canonical content. Anything not registered is out of scope for the guarantee.

A surface implements per-ID removal and a per-ID absence check, so callers always learn
*which* IDs a surface acknowledged (spec §11: "acknowledge remove(ids) with per-ID
success/failure, not only a global success flag").

The canonical store itself is NOT a registry surface — it is the authority that drives the
others (tombstone-first). The registry holds the derived/index surfaces: FTS5, zvec,
TurboVec, derived JSON/MD/JSONL artifacts, caches, reflection proposals, and bundle logs.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from strata.lexical.fts import LexicalStore


@runtime_checkable
class ManagedSurface(Protocol):
    name: str

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        """Remove records from this surface's active recall; return per-ID success."""
        ...

    def absent(self, record_id: int) -> bool:
        """True if the record cannot be surfaced from this surface's active recall."""
        ...


class LexicalSurface:
    """FTS5 lexical index, including its shadow/index tables."""

    name = "fts5"

    def __init__(self, lexical: LexicalStore) -> None:
        self._lex = lexical

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        ids = list(ids)
        self._lex.remove(ids)
        # Per-ID success = posting no longer present.
        return {i: not self._lex.contains(i) for i in ids}

    def absent(self, record_id: int) -> bool:
        return not self._lex.contains(record_id)


class VectorSurface:
    """Wraps any MemoryVectorStore (the fake, zvec, or TurboVec) as a managed surface.

    Used from step 6 onward. The wrapped store must expose ``remove(ids) -> dict[int, bool]``
    and ``contains(id) -> bool`` (soft-deleted IDs count as absent).
    """

    def __init__(self, name: str, store) -> None:
        self.name = name
        self._store = store

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        ids = list(ids)
        acks = self._store.remove(ids)
        # Normalize: a store may return a global bool or per-ID dict.
        if isinstance(acks, dict):
            return {i: bool(acks.get(i, False)) for i in ids}
        return {i: not self._store.contains(i) for i in ids}

    def absent(self, record_id: int) -> bool:
        return not self._store.contains(record_id)


class CoordinatedVectorSurface:
    """Vector index as a managed surface whose removals flow through the Write Coordinator.

    Honors the single-writer invariant: removal enqueues a destructive REMOVE_INDEX op and
    drives the coordinator to apply it, so the underlying vector store is only ever mutated
    inside the one writer (never directly from the deletion path).
    """

    def __init__(self, name: str, vector_store, coordinator) -> None:
        self.name = name
        self._store = vector_store
        self._coord = coordinator

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        from strata.coordinator.ops import Operation, OpType

        ids = list(ids)
        op = Operation(op_type=OpType.REMOVE_INDEX, target_ids=ids, target=self.name)
        self._coord.enqueue(op)
        self._coord.run_until_idle()
        return {i: not self._store.contains(i) for i in ids}

    def absent(self, record_id: int) -> bool:
        return not self._store.contains(record_id)


class DerivedArtifactSurface:
    """In-memory stand-in for derived JSON/MD/JSONL artifacts, caches, reflection proposals,
    and bundle logs that mirror canonical content. Holds a map of record_id -> derived blobs;
    removal drops them so the deleted content cannot leak through a derived surface.
    """

    def __init__(self, name: str = "derived_artifacts") -> None:
        self.name = name
        self._by_id: dict[int, list] = {}

    def put(self, record_id: int, blob) -> None:
        self._by_id.setdefault(record_id, []).append(blob)

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        out = {}
        for i in ids:
            self._by_id.pop(i, None)
            out[i] = i not in self._by_id
        return out

    def absent(self, record_id: int) -> bool:
        return record_id not in self._by_id


class ManagedSurfaceRegistry:
    def __init__(self, surfaces: Iterable[ManagedSurface] = ()) -> None:
        self._surfaces: list[ManagedSurface] = list(surfaces)

    def register(self, surface: ManagedSurface) -> None:
        self._surfaces.append(surface)

    def __iter__(self):
        return iter(self._surfaces)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._surfaces]

    def purge(self, ids: Iterable[int]) -> dict[str, dict[int, bool]]:
        """Remove ids from every surface; return per-surface per-ID acknowledgements."""
        ids = list(ids)
        return {s.name: s.remove(ids) for s in self._surfaces}

    def verify(self, ids: Iterable[int]) -> dict[str, dict[int, bool]]:
        """Per-surface per-ID absence after purge (used by deletion verification)."""
        ids = list(ids)
        return {s.name: {i: s.absent(i) for i in ids} for s in self._surfaces}
