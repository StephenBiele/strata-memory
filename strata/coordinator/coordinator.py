"""WriteCoordinator — single durable writer for all index mutation (spec §12).

All index mutations (live writes, reflection jobs, deletion, reindex) enqueue operations
instead of mutating indexes directly. Recall stays concurrent and read-optimized; writes are
serialized and replayable. This single-writer model is mandatory: zvec enforces
single-process-exclusive writes, so live sessions and the reflection engine must never both
hold the zvec write lock. Here, exactly one worker applies operations under ``write_lock``.

Ordering (spec §12):
* FIFO per canonical record id — an op waits behind any earlier not-done op touching a shared id.
* Global ordering for destructive ops — a destructive op waits for everything before it, and
  acts as a barrier so nothing after it runs until it completes.

Idempotency: every op has an op_id; handlers are expected to be idempotent (upsert/remove),
so crash recovery can safely replay interrupted ops.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from strata.canonical.store import CanonicalStore
from strata.coordinator.ops import DESTRUCTIVE, Operation, OpState, OpType

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Handler applies an op to its target adapter and returns per-ID acknowledgements.
Handler = Callable[[Operation], dict[int, bool]]


def _now_ms() -> int:
    return int(time.time() * 1000)


class WriteCoordinator:
    def __init__(
        self,
        store: CanonicalStore,
        *,
        max_attempts: int = 5,
        backoff_base_ms: int = 50,
        ack_recorder: Optional[Callable[[int, str, str, str], None]] = None,
    ) -> None:
        self.store = store
        self.max_attempts = max_attempts
        self.backoff_base_ms = backoff_base_ms
        # Default ack recorder writes per-ID acks into the canonical index_acknowledgements.
        self._ack_recorder = ack_recorder or self._default_ack_recorder
        self.write_lock = threading.Lock()         # the single-writer lock
        self._handlers: dict[str, Handler] = {}
        self._wake = threading.Event()
        self._apply_schema()
        self.recover()

    def _apply_schema(self) -> None:
        conn = self.store._conn()
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()

    def _default_ack_recorder(self, record_id: int, target: str, op_id: str, ack: str) -> None:
        self.store.record_index_ack(record_id, target, op_id, ack)

    # -- registration ----------------------------------------------------------
    def register_handler(self, target: str, handler: Handler) -> None:
        self._handlers[target] = handler

    # -- enqueue ---------------------------------------------------------------
    def enqueue(self, op: Operation) -> str:
        """Append an operation to the durable log (idempotent on op_id)."""
        conn = self.store._conn()
        now = _now_ms()
        conn.execute(
            "INSERT OR IGNORE INTO op_log "
            "(op_id, op_type, target, target_ids, payload, expected_generation, state, "
            " is_destructive, attempts, next_attempt_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)",
            (
                op.op_id, op.op_type.value, op.target, json.dumps(op.target_ids),
                json.dumps(op.payload) if op.payload is not None else None,
                op.expected_generation, OpState.PENDING.value,
                1 if op.is_destructive else 0, now, now,
            ),
        )
        conn.commit()
        self._wake.set()
        return op.op_id

    # -- recovery --------------------------------------------------------------
    def recover(self) -> int:
        """Reset interrupted in_progress ops to pending for replay after a crash."""
        conn = self.store._conn()
        cur = conn.execute(
            "UPDATE op_log SET state = ?, updated_at = ? WHERE state = ?",
            (OpState.PENDING.value, _now_ms(), OpState.IN_PROGRESS.value),
        )
        conn.commit()
        return cur.rowcount

    # -- scheduling ------------------------------------------------------------
    def _select_eligible(self, now: int) -> Optional[dict]:
        conn = self.store._conn()
        # All not-done ops, in global order, drive both candidacy and blocking.
        rows = conn.execute(
            "SELECT seq, op_id, op_type, target, target_ids, payload, expected_generation, "
            "state, is_destructive, attempts FROM op_log "
            "WHERE state NOT IN (?, ?) ORDER BY seq",
            (OpState.DONE.value, OpState.FAILED.value),
        ).fetchall()
        not_done = [dict(r) for r in rows]
        for i, op in enumerate(not_done):
            # Candidate must be runnable now.
            if op["state"] not in (OpState.PENDING.value, OpState.PARTIAL_FAILURE.value):
                continue
            ready = conn.execute(
                "SELECT next_attempt_at FROM op_log WHERE seq = ?", (op["seq"],)
            ).fetchone()[0]
            if ready is not None and ready > now:
                continue
            earlier = not_done[:i]
            if self._blocked(op, earlier):
                continue
            return op
        return None

    def _blocked(self, op: dict, earlier: list[dict]) -> bool:
        if not earlier:
            return False
        # A pending destructive op earlier in the log is a global barrier.
        if any(e["is_destructive"] for e in earlier):
            return True
        # A destructive candidate waits for everything before it.
        if op["is_destructive"]:
            return True
        # FIFO per record id: wait behind any earlier op sharing a target id.
        ids = set(json.loads(op["target_ids"]))
        for e in earlier:
            if ids & set(json.loads(e["target_ids"])):
                return True
        return False

    # -- apply -----------------------------------------------------------------
    def apply_next(self) -> Optional[str]:
        """Apply the next eligible operation. Returns its op_id, or None if none ready."""
        with self.write_lock:
            now = _now_ms()
            op_row = self._select_eligible(now)
            if op_row is None:
                return None
            self._set_state(op_row["seq"], OpState.IN_PROGRESS)
            op = Operation(
                op_type=OpType(op_row["op_type"]),
                target_ids=json.loads(op_row["target_ids"]),
                target=op_row["target"],
                payload=json.loads(op_row["payload"]) if op_row["payload"] else None,
                expected_generation=op_row["expected_generation"],
                op_id=op_row["op_id"],
            )
            self._run(op_row, op)
            return op.op_id

    def _run(self, op_row: dict, op: Operation) -> None:
        handler = self._handlers.get(op.target)
        try:
            if handler is None:
                raise KeyError(f"no handler registered for target {op.target!r}")
            acks = handler(op)
        except Exception as exc:  # handler hard failure: all ids failed
            acks = {rid: False for rid in op.target_ids}
            self._record_acks(op, acks)
            self._fail_or_retry(op_row, op, reason=str(exc))
            return
        self._record_acks(op, acks)
        if all(acks.get(rid, False) for rid in op.target_ids):
            self._finish(op_row, op, acks, OpState.DONE)
        else:
            self._fail_or_retry(op_row, op, reason="partial_ack", acks=acks)

    def _record_acks(self, op: Operation, acks: dict[int, bool]) -> None:
        for rid in op.target_ids:
            self._ack_recorder(
                rid, op.target or "unknown", op.op_id,
                "success" if acks.get(rid, False) else "failure",
            )

    def _finish(self, op_row: dict, op: Operation, acks: dict, state: OpState) -> None:
        conn = self.store._conn()
        conn.execute(
            "UPDATE op_log SET state = ?, ack_state = ?, updated_at = ? WHERE seq = ?",
            (state.value, json.dumps({str(k): v for k, v in acks.items()}), _now_ms(), op_row["seq"]),
        )
        conn.commit()

    def _fail_or_retry(self, op_row: dict, op: Operation, *, reason: str, acks: Optional[dict] = None) -> None:
        attempts = op_row["attempts"] + 1
        conn = self.store._conn()
        if attempts >= self.max_attempts:
            conn.execute(
                "UPDATE op_log SET state = ?, attempts = ?, ack_state = ?, updated_at = ? WHERE seq = ?",
                (OpState.FAILED.value, attempts, json.dumps({"reason": reason}), _now_ms(), op_row["seq"]),
            )
        else:
            backoff = self.backoff_base_ms * (2 ** (attempts - 1))
            conn.execute(
                "UPDATE op_log SET state = ?, attempts = ?, next_attempt_at = ?, ack_state = ?, "
                "updated_at = ? WHERE seq = ?",
                (OpState.PARTIAL_FAILURE.value, attempts, _now_ms() + backoff,
                 json.dumps({"reason": reason}), _now_ms(), op_row["seq"]),
            )
        conn.commit()

    def _set_state(self, seq: int, state: OpState) -> None:
        conn = self.store._conn()
        conn.execute(
            "UPDATE op_log SET state = ?, updated_at = ? WHERE seq = ?",
            (state.value, _now_ms(), seq),
        )
        conn.commit()

    # -- drain helpers ---------------------------------------------------------
    def run_until_idle(self, *, max_steps: int = 10_000) -> int:
        """Apply ops until none are immediately eligible. Returns count applied.

        Note: ops in backoff (future next_attempt_at) are not 'idle-eligible'; callers that
        need them should wait for the backoff window or call :meth:`retry_due`.
        """
        applied = 0
        for _ in range(max_steps):
            if self.apply_next() is None:
                break
            applied += 1
        return applied

    def retry(self, op_id: str) -> None:
        """Force a failed/partial op back to pending for immediate retry."""
        conn = self.store._conn()
        conn.execute(
            "UPDATE op_log SET state = ?, next_attempt_at = NULL, updated_at = ? "
            "WHERE op_id = ? AND state IN (?, ?)",
            (OpState.PENDING.value, _now_ms(), op_id,
             OpState.PARTIAL_FAILURE.value, OpState.FAILED.value),
        )
        conn.commit()
        self._wake.set()

    def rebuild_index(self, target: str, record_ids: list[int], *, payload_fn=None) -> str:
        """Enqueue a full re-upsert of ``record_ids`` into ``target`` (reindex/migration)."""
        op = Operation(
            op_type=OpType.UPSERT_INDEX,
            target_ids=list(record_ids),
            target=target,
            payload=payload_fn(record_ids) if payload_fn else None,
        )
        return self.enqueue(op)

    # -- inspection ------------------------------------------------------------
    def op_status(self, op_id: str) -> Optional[dict]:
        row = self.store._conn().execute(
            "SELECT * FROM op_log WHERE op_id = ?", (op_id,)
        ).fetchone()
        return dict(row) if row else None

    def audit_log(self, *, limit: int = 100) -> list[dict]:
        rows = self.store._conn().execute(
            "SELECT seq, op_id, op_type, target, target_ids, state, attempts, ack_state "
            "FROM op_log ORDER BY seq LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def pending_count(self) -> int:
        return self.store._conn().execute(
            "SELECT COUNT(*) FROM op_log WHERE state NOT IN (?, ?)",
            (OpState.DONE.value, OpState.FAILED.value),
        ).fetchone()[0]
