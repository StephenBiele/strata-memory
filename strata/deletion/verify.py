"""Deletion verification (spec §11 deletion step 8).

A deleted record must not be retrievable through any supported recall API. Verification
iterates the managed-surfaces registry (every derived/index surface) and the canonical store,
confirming per-ID absence from active recall.
"""

from __future__ import annotations

from typing import Iterable

from strata.canonical.store import CanonicalStore
from strata.deletion.managed_surfaces import ManagedSurfaceRegistry


def verify_not_retrievable(
    store: CanonicalStore,
    registry: ManagedSurfaceRegistry,
    ids: Iterable[int],
    *,
    mode: str = "logical",
) -> bool:
    """Return True iff every id is absent from all managed surfaces and canonical recall.

    * Logical mode: canonical row may remain but must be tombstoned (blocked from recall).
    * Hard mode: canonical row must be physically gone in addition to surface absence.
    """
    ids = set(ids)
    surface_results = registry.verify(ids)
    for per_id in surface_results.values():
        if not all(per_id.values()):
            return False
    for rid in ids:
        if not store.is_tombstoned(rid):
            return False
        if mode == "hard" and store.read_one(rid) is not None:
            return False
    return True
