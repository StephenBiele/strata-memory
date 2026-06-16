"""Stable identity helpers shared across Strata.

Record IDs must be uint64-compatible (spec §8): vector indexes (zvec, TurboVec IdMapIndex)
key on them, and SQLite stores them as signed 64-bit INTEGER. We allocate IDs in the
range [1, 2**63 - 1] so the value is representable both as a Python int, a signed SQLite
INTEGER, and an unsigned uint64 for the vector backends without sign reinterpretation.
"""

from __future__ import annotations

import hashlib
import os
import threading
import uuid

# Upper bound for allocated record IDs: fits in a signed SQLite INTEGER *and* a uint64.
MAX_RECORD_ID = (1 << 63) - 1


def content_hash(content: str) -> str:
    """Stable sha256 hex of canonical content (spec §8 content_hash).

    Used for duplicate detection and rebuild/hydration verification. Normalizes to UTF-8
    bytes only; callers decide whether to normalize text (casing/whitespace) beforehand.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def new_operation_id() -> str:
    """Idempotency key for Write Coordinator operations and deletion jobs (spec §12)."""
    return uuid.uuid4().hex


class IdAllocator:
    """Thread-safe random uint64-range ID allocator.

    Random (not sequential) so IDs don't leak volume/order, while staying within the
    signed range. Callers must treat collisions as caller-handled: CanonicalStore relies
    on the PRIMARY KEY constraint and retries on the rare clash.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def new_id(self) -> int:
        with self._lock:
            # 8 random bytes, masked into [1, MAX_RECORD_ID].
            value = int.from_bytes(os.urandom(8), "big") & MAX_RECORD_ID
            return value or 1


_default_allocator = IdAllocator()


def new_record_id() -> int:
    """Allocate a new uint64-compatible record ID using the process default allocator."""
    return _default_allocator.new_id()
