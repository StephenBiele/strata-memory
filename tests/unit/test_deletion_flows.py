"""Step 3 unit suite: deletion state machine, managed surfaces, correction flow."""

import pytest

from strata.canonical.records import MemoryRecord, Relation, Status
from strata.deletion.flows import DeletionService, DeletionState
from strata.deletion.managed_surfaces import (
    DerivedArtifactSurface,
    LexicalSurface,
    ManagedSurfaceRegistry,
)


class FlakyIndexSurface:
    """A managed surface that can be told to fail removal for specific ids (to drive the
    partial_failure / manual_attention states)."""

    name = "flaky_index"

    def __init__(self):
        self._ids: set[int] = set()
        self.fail_ids: set[int] = set()

    def add(self, record_id):
        self._ids.add(record_id)

    def remove(self, ids):
        out = {}
        for i in ids:
            if i in self.fail_ids:
                out[i] = False  # refuse to remove
            else:
                self._ids.discard(i)
                out[i] = True
        return out

    def absent(self, record_id):
        return record_id not in self._ids


@pytest.fixture
def env():
    from strata.canonical.store import CanonicalStore
    from strata.lexical.fts import LexicalStore
    cs = CanonicalStore(":memory:")
    lx = LexicalStore(cs)
    flaky = FlakyIndexSurface()
    artifacts = DerivedArtifactSurface()
    registry = ManagedSurfaceRegistry([LexicalSurface(lx), flaky, artifacts])
    svc = DeletionService(cs, registry, lexical=lx, max_retries=2)
    yield cs, lx, flaky, artifacts, svc
    cs.close()


def _add(cs, lx, content, **kw):
    rec = MemoryRecord.create(content, **kw)
    cs.write(rec)
    lx.index(rec)
    return rec


def test_canonical_first_tombstone_before_index(env):
    cs, lx, flaky, artifacts, svc = env
    rec = _add(cs, lx, "delete me alpha")
    flaky.add(rec.id)
    flaky.fail_ids = {rec.id}  # index will NOT acknowledge removal
    svc.request_deletion(rec.id)
    # Even though the flaky index still holds the id, canonical is tombstoned => blocked.
    assert cs.is_tombstoned(rec.id)
    assert not flaky.absent(rec.id)  # stale index entry still present
    assert lx.search("alpha") == []  # recall path still refuses it


def test_full_verified_deletion(env):
    cs, lx, flaky, artifacts, svc = env
    rec = _add(cs, lx, "clean delete beta")
    artifacts.put(rec.id, {"summary": "beta"})
    job = svc.request_deletion(rec.id)
    status = svc.deletion_status(job)
    assert status["state"] == DeletionState.VERIFIED.value
    assert status["verified"] is True
    assert artifacts.absent(rec.id)
    assert not lx.contains(rec.id)


def test_partial_failure_then_manual_attention(env):
    cs, lx, flaky, artifacts, svc = env
    rec = _add(cs, lx, "stuck delete gamma")
    flaky.add(rec.id)
    flaky.fail_ids = {rec.id}
    job = svc.request_deletion(rec.id)
    status = svc.deletion_status(job)
    assert status["state"] == DeletionState.MANUAL_ATTENTION.value
    assert cs.is_tombstoned(rec.id)  # stays blocked regardless


def test_retry_succeeds_after_surface_recovers(env):
    cs, lx, flaky, artifacts, svc = env
    rec = _add(cs, lx, "recoverable delete delta")
    flaky.add(rec.id)
    flaky.fail_ids = {rec.id}
    job = svc.request_deletion(rec.id)
    assert svc.deletion_status(job)["state"] == DeletionState.MANUAL_ATTENTION.value
    flaky.fail_ids = set()  # surface recovers
    svc.retry(job)
    assert svc.deletion_status(job)["state"] == DeletionState.VERIFIED.value


def test_deletion_includes_dependents(env):
    cs, lx, flaky, artifacts, svc = env
    parent = _add(cs, lx, "parent record")
    child = _add(cs, lx, "derived summary")
    cs.add_dependency(parent.id, child.id, Relation.SOURCE_OF)
    job = svc.request_deletion(parent.id)
    st = svc.deletion_status(job)
    assert set(st["record_ids"]) == {parent.id, child.id}
    assert cs.is_tombstoned(child.id)


def test_hard_delete_erases_canonical_row(env):
    cs, lx, flaky, artifacts, svc = env
    rec = _add(cs, lx, "physically erase epsilon")
    job = svc.request_deletion(rec.id, mode="hard")
    assert svc.deletion_status(job)["state"] == DeletionState.VERIFIED.value
    assert cs.read_one(rec.id) is None  # row physically gone
    assert cs.is_tombstoned(rec.id)     # tombstone preserved (cannot resurrect)
    assert lx.search("epsilon", statuses=None) == []


def test_per_id_index_acknowledgements_recorded(env):
    cs, lx, flaky, artifacts, svc = env
    rec = _add(cs, lx, "ack record zeta")
    flaky.add(rec.id)
    svc.request_deletion(rec.id)
    rows = cs._conn().execute(
        "SELECT adapter, ack FROM index_acknowledgements WHERE record_id = ?", (rec.id,)
    ).fetchall()
    acks = {r["adapter"]: r["ack"] for r in rows}
    assert acks.get("fts5") == "success"
    assert acks.get("flaky_index") == "success"


def test_correction_supersedes_and_drops_old_from_active(env):
    cs, lx, flaky, artifacts, svc = env
    old = _add(cs, lx, "prefers coffee strongly")
    new = MemoryRecord.create("prefers tea now")
    svc.correct(old.id, new)
    assert cs.read_one(old.id).status is Status.SUPERSEDED
    assert lx.search("coffee") == []          # old gone from active recall
    assert {r.id for r, _ in lx.search("tea")} == {new.id}
    assert old.id in cs.dependency_graph(new.id)  # audit link preserved
