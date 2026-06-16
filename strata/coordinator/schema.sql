-- Write Coordinator durable operation log (spec §12). Lives in the same SQLite file as the
-- canonical store so enqueue + canonical writes are atomic and the queue is one of the
-- managed surfaces (spec §11). This is a durable op log, not an in-memory thread queue:
-- it gives crash recovery, replay, and idempotent retries.

CREATE TABLE IF NOT EXISTS op_log (
  seq                 INTEGER PRIMARY KEY AUTOINCREMENT,  -- global order
  op_id               TEXT NOT NULL UNIQUE,               -- idempotency key
  op_type             TEXT NOT NULL,   -- upsert_index|remove_index|tombstone|reindex_promote|hard_purge
  target              TEXT,            -- adapter/handler name (zvec|turbovec|fts5|...)
  target_ids          TEXT NOT NULL,   -- JSON array of canonical record ids
  payload             TEXT,            -- JSON op payload (e.g. vectors), optional
  expected_generation INTEGER,         -- embedding generation guard
  state               TEXT NOT NULL,   -- pending|in_progress|partial_failure|done|failed
  is_destructive      INTEGER NOT NULL DEFAULT 0,         -- global-order barrier
  attempts            INTEGER NOT NULL DEFAULT 0,
  next_attempt_at     INTEGER,         -- earliest retry time (exponential backoff)
  ack_state           TEXT,            -- JSON {record_id: success|failure}
  created_at          INTEGER NOT NULL,
  updated_at          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_op_log_state ON op_log(state);
