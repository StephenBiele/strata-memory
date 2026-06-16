"""Step 1 unit suite: canonical schema + CanonicalStore."""

import pytest

from strata.canonical.migrations import SCHEMA_VERSION, get_schema_version
from strata.canonical.records import MemoryRecord, Relation, RecordType, Status, Tier
from strata.ids import new_operation_id


@pytest.fixture
def store():
    from strata.canonical.store import CanonicalStore
    s = CanonicalStore(":memory:")
    yield s
    s.close()


def test_schema_version_recorded(store):
    assert get_schema_version(store._conn()) == SCHEMA_VERSION


def test_write_and_read_roundtrip(store):
    rec = MemoryRecord.create("user prefers tea", tier=Tier.L1)
    store.write(rec)
    got = store.read_one(rec.id)
    assert got is not None
    assert got.content == "user prefers tea"
    assert got.content_hash == rec.content_hash
    assert got.status is Status.ACTIVE
    # current_version_id is set by the trigger after the version insert.
    assert got.current_version_id is not None


def test_trigger_does_not_recurse_and_tracks_latest_version(store):
    rec = MemoryRecord.create("v1")
    store.write(rec)
    first_version = store.read_one(rec.id).current_version_id
    store.write(rec.with_content("v2"))
    second_version = store.read_one(rec.id).current_version_id
    assert second_version is not None and second_version != first_version
    # exactly two version rows, no runaway recursion.
    n = store._conn().execute(
        "SELECT COUNT(*) FROM versions WHERE record_id = ?", (rec.id,)
    ).fetchone()[0]
    assert n == 2


def test_read_skips_missing_ids(store):
    rec = MemoryRecord.create("present")
    store.write(rec)
    out = store.read([rec.id, 999_999_999])
    assert [r.id for r in out] == [rec.id]


def test_query_filters_and_excludes_tombstoned(store):
    a = MemoryRecord.create("a", tier=Tier.L1)
    b = MemoryRecord.create("b", tier=Tier.L2)
    store.write(a)
    store.write(b)
    assert {r.id for r in store.query(tier=Tier.L1)} == {a.id}
    store.tombstone(a.id, job_id=new_operation_id())
    assert a.id not in {r.id for r in store.query()}
    assert a.id in {r.id for r in store.query(exclude_tombstoned=False)}


def test_tombstone_is_authoritative(store):
    rec = MemoryRecord.create("secret")
    store.write(rec)
    assert not store.is_tombstoned(rec.id)
    store.tombstone(rec.id, job_id=new_operation_id())
    assert store.is_tombstoned(rec.id)
    assert store.read_one(rec.id).status is Status.DELETED
    assert rec.id in store.tombstoned_ids()


def test_supersede_marks_old_and_links(store):
    old = MemoryRecord.create("prefers coffee")
    store.write(old)
    new = MemoryRecord.create("prefers tea")
    store.supersede(old.id, new)
    assert store.read_one(old.id).status is Status.SUPERSEDED
    # new supersedes old via dependency edge.
    assert old.id in store.dependency_graph(new.id)


def test_dependency_graph_is_cycle_safe(store):
    a = MemoryRecord.create("a"); b = MemoryRecord.create("b"); c = MemoryRecord.create("c")
    for r in (a, b, c):
        store.write(r)
    store.add_dependency(a.id, b.id, Relation.DERIVED_FROM)
    store.add_dependency(b.id, c.id, Relation.DERIVED_FROM)
    store.add_dependency(c.id, a.id, Relation.DERIVED_FROM)  # cycle
    assert store.dependency_graph(a.id) == {b.id, c.id}


def test_hard_delete_cascades(store):
    rec = MemoryRecord.create("to erase")
    store.write(rec)
    store.tombstone(rec.id, mode="hard", job_id=new_operation_id())
    store.hard_delete(rec.id)
    assert store.read_one(rec.id) is None
    # version rows are gone too.
    n = store._conn().execute(
        "SELECT COUNT(*) FROM versions WHERE record_id = ?", (rec.id,)
    ).fetchone()[0]
    assert n == 0


def test_model_side_defaults_to_null(store):
    # Forward-compat column for Strata Persona MUST default to NULL — never auto-tag records
    # (e.g. as 'user') or a future schema would silently mislabel all existing rows.
    rec = MemoryRecord.create("untagged record")
    assert rec.model_side is None
    store.write(rec)
    stored = store._conn().execute(
        "SELECT model_side FROM records WHERE id = ?", (rec.id,)
    ).fetchone()[0]
    assert stored is None


def test_per_id_index_ack(store):
    rec = MemoryRecord.create("indexed")
    store.write(rec)
    op = new_operation_id()
    store.record_index_ack(rec.id, "zvec", op, "pending")
    store.record_index_ack(rec.id, "zvec", op, "success")
    row = store._conn().execute(
        "SELECT ack FROM index_acknowledgements WHERE record_id = ? AND adapter = 'zvec'",
        (rec.id,),
    ).fetchone()
    assert row["ack"] == "success"
