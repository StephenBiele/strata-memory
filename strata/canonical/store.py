"""CanonicalStore — SQLite source of truth (spec §12 interface).

Required methods: write, read, query, tombstone, dependency_graph. Canonical text lives only
in these tables. The store keeps a per-thread connection (WAL mode) so any number of reader
threads run concurrently while the Write Coordinator serializes mutations elsewhere.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from strata.canonical.migrations import apply_schema
from strata.canonical.records import (
    MemoryRecord,
    Relation,
    RecordType,
    Sensitivity,
    Status,
    Tier,
)

_COLUMNS = [
    "id", "record_type", "record_subtype", "tier", "content", "status",
    "confidence", "confidence_reason", "salience", "sensitivity", "content_hash",
    "embedding_model_id", "embedding_generation", "valid_from", "valid_until",
    "created_at", "updated_at", "current_version_id", "model_side",
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        record_type=RecordType(row["record_type"]),
        record_subtype=row["record_subtype"],
        tier=Tier(row["tier"]),
        content=row["content"],
        status=Status(row["status"]),
        confidence=row["confidence"],
        confidence_reason=row["confidence_reason"],
        salience=row["salience"],
        sensitivity=Sensitivity(row["sensitivity"]),
        content_hash=row["content_hash"],
        embedding_model_id=row["embedding_model_id"],
        embedding_generation=row["embedding_generation"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        current_version_id=row["current_version_id"],
        model_side=row["model_side"],
    )


class CanonicalStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) == ":memory:":
            self.path = ":memory:"
        else:
            resolved = Path(path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(resolved)
        self._local = threading.local()
        # In-memory DBs are not shared across connections; pin a single shared connection.
        self._shared = self._new_connection() if self.path == ":memory:" else None
        # Apply schema once on a connection.
        apply_schema(self._conn())

    # -- connection management -------------------------------------------------
    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close()
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- writes ----------------------------------------------------------------
    def write(self, record: MemoryRecord) -> MemoryRecord:
        """Upsert the head row and append a version row (audit history).

        The ``trg_sync_current_version`` trigger updates ``current_version_id`` after the
        version insert, so we re-read it onto the returned record.
        """
        if record.id is None:
            raise ValueError("record.id must be set before write(); use MemoryRecord.create")
        conn = self._conn()
        now = _now_ms()
        values = [
            record.id, record.record_type.value, record.record_subtype, record.tier.value,
            record.content, record.status.value, record.confidence, record.confidence_reason,
            record.salience, record.sensitivity.value, record.content_hash,
            record.embedding_model_id, record.embedding_generation, record.valid_from,
            record.valid_until, record.created_at or now, now, record.current_version_id,
            record.model_side,
        ]
        placeholders = ", ".join("?" for _ in _COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in _COLUMNS if c not in ("id", "created_at"))
        conn.execute(
            f"INSERT INTO records ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            values,
        )
        conn.execute(
            "INSERT INTO versions (record_id, content, content_hash, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (record.id, record.content, record.content_hash, record.status.value, now),
        )
        conn.commit()
        return self.read_one(record.id) or record

    def add_dependency(self, parent_id: int, child_id: int, rel: Relation) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO dependencies (parent_id, child_id, rel) VALUES (?, ?, ?)",
            (parent_id, child_id, rel.value),
        )
        conn.commit()

    def set_status(self, record_id: int, status: Status) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE records SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, _now_ms(), record_id),
        )
        conn.commit()

    def supersede(self, old_id: int, new_record: MemoryRecord) -> MemoryRecord:
        """Replace a current belief while preserving history (spec §11 correction flow).

        Writes ``new_record``, marks ``old_id`` superseded, and records the supersedes edge.
        Index removal of the old record is the caller's job via the Write Coordinator.
        """
        written = self.write(new_record)
        self.set_status(old_id, Status.SUPERSEDED)
        self.add_dependency(written.id, old_id, Relation.SUPERSEDES)
        return written

    # -- reads -----------------------------------------------------------------
    def read_one(self, record_id: int) -> Optional[MemoryRecord]:
        row = self._conn().execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def read(self, ids: Iterable[int]) -> list[MemoryRecord]:
        """Read records by id, skipping any that do not exist (no dangling rows)."""
        ids = list(ids)
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        rows = self._conn().execute(
            f"SELECT * FROM records WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["id"]: _row_to_record(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def query(
        self,
        *,
        status: Optional[Status] = None,
        tier: Optional[Tier] = None,
        record_type: Optional[RecordType] = None,
        sensitivity: Optional[Sensitivity] = None,
        exclude_tombstoned: bool = True,
        limit: Optional[int] = None,
    ) -> list[MemoryRecord]:
        clauses, params = [], []
        if status is not None:
            clauses.append("r.status = ?"); params.append(status.value)
        if tier is not None:
            clauses.append("r.tier = ?"); params.append(tier.value)
        if record_type is not None:
            clauses.append("r.record_type = ?"); params.append(record_type.value)
        if sensitivity is not None:
            clauses.append("r.sensitivity = ?"); params.append(sensitivity.value)
        if exclude_tombstoned:
            clauses.append("r.id NOT IN (SELECT record_id FROM tombstones)")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT r.* FROM records r{where} ORDER BY r.updated_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn().execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    # -- deletion --------------------------------------------------------------
    def tombstone(
        self,
        record_id: int,
        *,
        mode: str = "logical",
        job_id: str,
        state: str = "canonical_tombstoned",
    ) -> None:
        """Mark a record's deletion state AUTHORITATIVELY (canonical-first invariant).

        Setting the tombstone and flipping status to ``deleted`` is done in one transaction
        so a record is never half-deleted: once committed, hydration/resolver must reject it
        even if an index still holds a stale entry.
        """
        conn = self._conn()
        now = _now_ms()
        conn.execute(
            "INSERT INTO tombstones (record_id, mode, state, job_id, requested_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(record_id) DO UPDATE SET mode = excluded.mode, state = excluded.state",
            (record_id, mode, state, job_id, now),
        )
        conn.execute(
            "UPDATE records SET status = ?, updated_at = ? WHERE id = ?",
            (Status.DELETED.value, now, record_id),
        )
        conn.commit()

    def set_tombstone_state(self, record_id: int, state: str, *, verified_at: Optional[int] = None) -> None:
        conn = self._conn()
        if verified_at is not None:
            conn.execute(
                "UPDATE tombstones SET state = ?, verified_at = ? WHERE record_id = ?",
                (state, verified_at, record_id),
            )
        else:
            conn.execute(
                "UPDATE tombstones SET state = ? WHERE record_id = ?", (state, record_id)
            )
        conn.commit()

    def get_tombstone(self, record_id: int) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM tombstones WHERE record_id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None

    def is_tombstoned(self, record_id: int) -> bool:
        """Authoritative deletion check used by hydration/resolver filters."""
        return self._conn().execute(
            "SELECT 1 FROM tombstones WHERE record_id = ?", (record_id,)
        ).fetchone() is not None

    def tombstoned_ids(self) -> set[int]:
        return {
            r[0] for r in self._conn().execute("SELECT record_id FROM tombstones").fetchall()
        }

    def hard_delete(self, record_id: int) -> None:
        """Physically remove canonical rows for a record (hard-delete mode).

        Cascades to versions/dependencies/source_spans via ON DELETE CASCADE. The tombstone
        row is preserved (state stays authoritative) so the ID can never be resurrected.
        """
        conn = self._conn()
        conn.execute("PRAGMA secure_delete = ON")
        conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()

    # -- dependency graph ------------------------------------------------------
    def dependency_graph(self, record_id: int) -> set[int]:
        """Return all record IDs reachable from ``record_id`` via dependency edges.

        Used to discover derived records that must be re-evaluated or removed when a record
        is corrected or deleted (spec §11 deletion step 1). Traversal follows edges in both
        directions and is cycle-safe.
        """
        conn = self._conn()
        seen: set[int] = set()
        frontier = [record_id]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            rows = conn.execute(
                "SELECT child_id AS other FROM dependencies WHERE parent_id = ? "
                "UNION SELECT parent_id AS other FROM dependencies WHERE child_id = ?",
                (current, current),
            ).fetchall()
            for r in rows:
                if r["other"] not in seen:
                    frontier.append(r["other"])
        seen.discard(record_id)
        return seen

    def derived_from(self, record_id: int) -> list[int]:
        """Records derived from ``record_id`` — parent ids of a derived_from edge.

        Used to find the facts (L1) distilled from a raw event (L0): a fact links
        to its source via ``add_dependency(fact_id, event_id, Relation.DERIVED_FROM)``.
        """
        rows = self._conn().execute(
            "SELECT parent_id FROM dependencies WHERE child_id = ? AND rel = ?",
            (record_id, Relation.DERIVED_FROM.value),
        ).fetchall()
        return [r[0] for r in rows]

    def superseders_of(self, record_id: int) -> list[int]:
        """Records that supersede ``record_id`` (parent ids of a supersedes edge)."""
        rows = self._conn().execute(
            "SELECT parent_id FROM dependencies WHERE child_id = ? AND rel = ?",
            (record_id, Relation.SUPERSEDES.value),
        ).fetchall()
        return [r[0] for r in rows]

    def contradictors_of(self, record_id: int) -> list[int]:
        """Records linked to ``record_id`` by a contradicts edge, in either direction."""
        rows = self._conn().execute(
            "SELECT child_id AS other FROM dependencies WHERE parent_id = ? AND rel = ? "
            "UNION SELECT parent_id AS other FROM dependencies WHERE child_id = ? AND rel = ?",
            (record_id, Relation.CONTRADICTS.value, record_id, Relation.CONTRADICTS.value),
        ).fetchall()
        return [r[0] for r in rows]

    def record_index_ack(
        self, record_id: int, adapter: str, op_id: str, ack: str, generation: Optional[int] = None
    ) -> None:
        """Record a per-ID per-adapter index acknowledgement (spec §11)."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO index_acknowledgements (record_id, adapter, op_id, ack, generation, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(record_id, adapter, op_id) DO UPDATE SET ack = excluded.ack, "
            "generation = excluded.generation, updated_at = excluded.updated_at",
            (record_id, adapter, op_id, ack, generation, _now_ms()),
        )
        conn.commit()
