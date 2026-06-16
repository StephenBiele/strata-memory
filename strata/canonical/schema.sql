-- Strata canonical schema (spec §8, §11 managed-surfaces registry).
-- Applied idempotently by migrations.apply_schema(). WAL mode + foreign_keys are set as
-- PRAGMAs by the store at connect time (PRAGMAs are connection-scoped, not schema-scoped).

-- records: current head row per memory id. canonical content lives ONLY here.
CREATE TABLE IF NOT EXISTS records (
  id                 INTEGER PRIMARY KEY,        -- uint64-compatible stable ID
  record_type        TEXT NOT NULL,
  record_subtype     TEXT,
  tier               TEXT NOT NULL,
  content            TEXT NOT NULL,
  status             TEXT NOT NULL,
  confidence         REAL,
  confidence_reason  TEXT,
  salience           REAL,
  sensitivity        TEXT NOT NULL DEFAULT 'normal',
  content_hash       TEXT NOT NULL,
  embedding_model_id TEXT,
  embedding_generation INTEGER,
  valid_from         INTEGER,
  valid_until        INTEGER,
  created_at         INTEGER NOT NULL,
  updated_at         INTEGER NOT NULL,
  current_version_id INTEGER,                    -- -> versions.version_id
  model_side         TEXT                        -- nullable; fwd-compat with Strata Persona
);

CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_tier   ON records(tier);
CREATE INDEX IF NOT EXISTS idx_records_hash   ON records(content_hash);

-- versions: append-only history for corrections/supersession + audit.
CREATE TABLE IF NOT EXISTS versions (
  version_id   INTEGER PRIMARY KEY,
  record_id    INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
  content      TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  status       TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  op_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_versions_record ON versions(record_id);

-- Keep records.current_version_id pointing at the newest version row.
-- The WHEN guard is mandatory: without it the UPDATE inside the trigger body re-enters the
-- write path and can recurse; it fires only when current_version_id is actually stale.
CREATE TRIGGER IF NOT EXISTS trg_sync_current_version
AFTER INSERT ON versions
WHEN (SELECT current_version_id FROM records WHERE id = NEW.record_id) IS NOT NEW.version_id
BEGIN
  UPDATE records
     SET current_version_id = NEW.version_id,
         updated_at = NEW.created_at
   WHERE id = NEW.record_id;
END;

-- dependencies: derivation/supersession/contradiction edges. Drives dependency_graph()
-- and cascade-deletion discovery.
CREATE TABLE IF NOT EXISTS dependencies (
  parent_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
  child_id  INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
  rel       TEXT NOT NULL,
  PRIMARY KEY (parent_id, child_id, rel)
);
CREATE INDEX IF NOT EXISTS idx_dependencies_child ON dependencies(child_id);

-- tombstones: AUTHORITATIVE deletion state (canonical-first invariant). A row here means
-- hydration/resolver must reject the record regardless of any stale index entry.
-- Intentionally NO foreign key: the tombstone is a permanent gravestone that must survive
-- hard-deletion of the records row so a hard-deleted ID can never be resurrected.
CREATE TABLE IF NOT EXISTS tombstones (
  record_id    INTEGER PRIMARY KEY,
  mode         TEXT NOT NULL,        -- logical | hard
  state        TEXT NOT NULL,        -- requested | canonical_tombstoned | index_deleting
                                     -- | partial_failure | verified | manual_attention
  job_id       TEXT NOT NULL,
  requested_at INTEGER NOT NULL,
  verified_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tombstones_job ON tombstones(job_id);

CREATE TABLE IF NOT EXISTS embedding_generations (
  generation         INTEGER PRIMARY KEY,
  embedding_model_id TEXT NOT NULL,
  status             TEXT NOT NULL,   -- active | shadow | retired
  created_at         INTEGER NOT NULL,
  notes              TEXT
);

-- index_acknowledgements: per-ID per-adapter ack (spec §11: ack must be per-ID, not global).
CREATE TABLE IF NOT EXISTS index_acknowledgements (
  record_id  INTEGER NOT NULL,
  adapter    TEXT NOT NULL,           -- zvec | turbovec | fts5
  op_id      TEXT NOT NULL,
  ack        TEXT NOT NULL,           -- pending | success | failure
  generation INTEGER,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (record_id, adapter, op_id)
);

CREATE TABLE IF NOT EXISTS source_spans (
  span_id    INTEGER PRIMARY KEY,
  record_id  INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
  event_id   INTEGER,
  char_start INTEGER,
  char_end   INTEGER,
  raw_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_source_spans_record ON source_spans(record_id);

CREATE TABLE IF NOT EXISTS policy_epochs (
  epoch             INTEGER PRIMARY KEY,
  sensitivity_rules TEXT NOT NULL,
  created_at        INTEGER NOT NULL
);

-- schema_meta: single-row schema version marker for migrations.
CREATE TABLE IF NOT EXISTS schema_meta (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  schema_version INTEGER NOT NULL
);
