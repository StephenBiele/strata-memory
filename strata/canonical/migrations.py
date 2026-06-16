"""Schema application / versioning for the canonical store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def apply_schema(conn: sqlite3.Connection) -> None:
    """Idempotently apply the canonical schema and record the schema version.

    Safe to call on every connect: all DDL uses ``IF NOT EXISTS``. The connection is
    expected to already have ``foreign_keys`` enabled by the store.
    """
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO schema_meta(id, schema_version) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET schema_version = excluded.schema_version",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT schema_version FROM schema_meta WHERE id = 1").fetchone()
    return row[0] if row else None
