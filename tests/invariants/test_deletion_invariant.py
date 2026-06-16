"""I1 — Deletion invariant: once tombstoned, a record is never returned by recall or
hydration, even if a stale index entry exists."""

import pytest

from strata.deletion.flows import DeletionState
from strata.ids import new_operation_id
from strata.vector.base import VectorRecord

pytestmark = pytest.mark.invariant


def _claims(bundle):
    return {e.claim for e in bundle.all_entries()}


def test_tombstoned_never_recalled(h):
    rec = h.write("user lives in Berlin")
    assert "user lives in Berlin" in _claims(h.engine.recall("user lives in Berlin"))
    h.engine.delete_memory(rec.id)
    assert "user lives in Berlin" not in _claims(h.engine.recall("user lives in Berlin"))


def test_tombstoned_never_hydrated_with_stale_index_entry(h):
    rec = h.write("secret passphrase orange")
    # Tombstone canonical-first WITHOUT purging the index -> a stale index entry remains.
    h.store.tombstone(rec.id, job_id=new_operation_id())
    assert h.engine.vector_stores["hot"].contains(rec.id)  # stale entry still present
    bundle = h.engine.recall("secret passphrase orange")
    assert rec.id not in {e.id for e in bundle.all_entries()}  # hydration/resolver reject it


def test_deletion_state_machine_reaches_verified(h):
    rec = h.write("ephemeral note zulu")
    job = h.engine.delete_memory(rec.id)
    status = h.engine.deletion_status(job)
    assert status["state"] == DeletionState.VERIFIED.value
    assert status["verified"] is True


def test_per_id_acknowledgements_recorded(h):
    rec = h.write("acked deletion record")
    h.engine.delete_memory(rec.id)
    rows = h.store._conn().execute(
        "SELECT adapter, ack FROM index_acknowledgements WHERE record_id = ?", (rec.id,)
    ).fetchall()
    acks = {r["adapter"]: r["ack"] for r in rows}
    # Each managed surface acknowledged removal per-ID (not a single global flag).
    assert acks.get("fts5") == "success"
    assert acks.get("hot") == "success"


def test_hard_delete_blocks_and_erases(h):
    rec = h.write("physically forget me")
    job = h.engine.delete_memory(rec.id, mode="hard")
    assert h.engine.deletion_status(job)["state"] == DeletionState.VERIFIED.value
    assert h.store.read_one(rec.id) is None
    assert h.store.is_tombstoned(rec.id)
    assert "physically forget me" not in _claims(h.engine.recall("physically forget me"))
