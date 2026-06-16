"""LexicalStore — exact lookup over canonical text via SQLite FTS5/BM25 (spec §6, §12).

Design decisions (proven empirically against SQLite 3.53.1, see docs/PINS.md):

* **Contentful FTS5 keyed by record id.** External-content FTS5 corrupted ("database disk
  image is malformed") under manual per-row INSERT+DELETE on this SQLite build, and its
  removal semantics drift when the external text changes first. A contentful FTS5 table
  stores a tokenized copy in its ``%_content`` shadow, which makes per-row DELETE/UPDATE
  self-contained and drift-proof. That shadow is a *derived lexical-index surface* — it is
  explicitly enumerated in the spec §11 managed-surfaces registry ("FTS5 lexical index,
  including its shadow/index tables"), scrubbed by secure-delete — not a second authoritative
  copy of memory. Canonical truth still lives solely in the ``records`` table.
* **secure-delete enabled at creation** so removed postings AND the shadow content copy are
  scrubbed from ``records_fts_data``/``_idx``/``_content`` on delete.
* **search filters tombstoned/inactive rows at query time** (JOIN to ``records``), enforcing
  the active-recall deletion guarantee at the lexical surface regardless of index state.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from strata.canonical.records import MemoryRecord, Status
from strata.canonical.store import CanonicalStore, _row_to_record

# Statuses that may appear in active recall. Superseded/contradicted/stale/deleted are excluded.
ACTIVE_STATUSES = (Status.ACTIVE, Status.REINFORCED)

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _safe_match(text: str) -> str:
    """Build a safe FTS5 MATCH expression from arbitrary host text.

    FTS5 MATCH is its own query language; passing raw user text risks syntax errors or
    operator injection. We extract word tokens and AND them as quoted strings, which covers
    exact-name/date/phrase lookup without exposing FTS5 operators.
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


class LexicalStore:
    def __init__(self, store: CanonicalStore, table: str = "records_fts") -> None:
        self.store = store
        self.table = table
        self._ensure_schema()

    # -- schema ----------------------------------------------------------------
    def _ensure_schema(self) -> None:
        conn = self.store._conn()
        existed = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (self.table,)
        ).fetchone()
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING fts5("
            f"content, tokenize='unicode61')"
        )
        if not existed:
            conn.execute(f"INSERT INTO {self.table}({self.table}, rank) VALUES('secure-delete', 1)")
        conn.commit()

    # -- mutation --------------------------------------------------------------
    def index(self, record: MemoryRecord) -> None:
        """Insert/refresh postings for ``record`` (delete-then-insert; drift-proof)."""
        conn = self.store._conn()
        conn.execute(f"DELETE FROM {self.table} WHERE rowid = ?", (record.id,))
        conn.execute(
            f"INSERT INTO {self.table}(rowid, content) VALUES (?, ?)",
            (record.id, record.content),
        )
        conn.commit()

    def remove(self, ids: Iterable[int]) -> None:
        """Remove postings for ids. Drift-proof: FTS holds its own content copy."""
        ids = list(ids)
        if not ids:
            return
        conn = self.store._conn()
        placeholders = ", ".join("?" for _ in ids)
        conn.execute(f"DELETE FROM {self.table} WHERE rowid IN ({placeholders})", ids)
        conn.commit()

    def rebuild(self) -> None:
        """Resync the index from canonical records (fallback / rebuild-from-canonical).

        Contentful FTS5 cannot use the 'rebuild' command (no external content table), so we
        clear and re-index every canonical record's current content.
        """
        conn = self.store._conn()
        conn.execute(f"DELETE FROM {self.table}")
        rows = conn.execute("SELECT id, content FROM records").fetchall()
        conn.executemany(
            f"INSERT INTO {self.table}(rowid, content) VALUES (?, ?)",
            [(r["id"], r["content"]) for r in rows],
        )
        conn.commit()

    # -- query -----------------------------------------------------------------
    def search(
        self,
        text: str,
        *,
        top_k: int = 10,
        statuses: Optional[Sequence[Status]] = ACTIVE_STATUSES,
        exclude_tombstoned: bool = True,
    ) -> list[tuple[MemoryRecord, float]]:
        """BM25-ranked exact/lexical search over active canonical records.

        Tombstoned and non-active rows are filtered at query time regardless of index state.
        """
        match = _safe_match(text)
        clauses = [f"{self.table} MATCH ?"]
        params: list = [match]
        if exclude_tombstoned:
            clauses.append("r.id NOT IN (SELECT record_id FROM tombstones)")
        if statuses:
            clauses.append("r.status IN (%s)" % ", ".join("?" for _ in statuses))
            params.extend(s.value for s in statuses)
        sql = (
            f"SELECT r.*, bm25({self.table}) AS score "
            f"FROM {self.table} JOIN records r ON r.id = {self.table}.rowid "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY bm25({self.table}) LIMIT ?"
        )
        params.append(top_k)
        rows = self.store._conn().execute(sql, params).fetchall()
        return [(_row_to_record(row), float(row["score"])) for row in rows]

    def contains(self, record_id: int) -> bool:
        """Whether any posting exists for ``record_id`` (used by deletion verification)."""
        return self.store._conn().execute(
            f"SELECT 1 FROM {self.table} WHERE rowid = ?", (record_id,)
        ).fetchone() is not None

    def stats(self) -> dict:
        n = self.store._conn().execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        return {"table": self.table, "postings_rows": n}
