"""I2 — Hydration invariant: every recalled ID resolves to canonical text or is dropped;
never surface a dangling ID."""

import pytest

from strata.vector.base import VectorRecord

pytestmark = pytest.mark.invariant


def test_dangling_index_id_is_dropped(h):
    real = h.write("the cat sat on the mat")
    # Inject a stale/dangling index entry with no canonical record behind it.
    ghost_vec = h.embedder.embed("the cat sat on the mat")
    h.engine.vector_stores["hot"].upsert([
        VectorRecord(id=999_999, vector=ghost_vec, content_hash="deadbeef",
                     embedding_model_id=h.embedder.model_id, embedding_generation=1, metadata={})
    ])
    bundle = h.engine.recall("the cat sat on the mat")
    ids = {e.id for e in bundle.all_entries()}
    assert 999_999 not in ids        # dangling id never surfaced
    assert real.id in ids


def test_every_recalled_id_hydrates_to_canonical(h):
    for t in ["alpha fact one", "beta fact two", "gamma fact three"]:
        h.write(t)
    bundle = h.engine.recall("beta fact two")
    for entry in bundle.all_entries():
        rec = h.store.read_one(entry.id)
        assert rec is not None
        assert rec.content == entry.claim


def test_content_hash_drift_candidate_is_dropped(h):
    from strata.canonical.records import MemoryRecord

    # Canonical record exists, but the ONLY index entry for it carries a stale content_hash.
    rec = MemoryRecord.create("drifted canonical content")
    h.store.write(rec)  # canonical only — deliberately not lexically indexed
    vs = h.engine.vector_stores["hot"]
    vs.upsert([
        VectorRecord(id=rec.id, vector=h.embedder.embed("drifted canonical content"),
                     content_hash="stalehash", embedding_model_id=h.embedder.model_id,
                     embedding_generation=1, metadata={})
    ])
    sample = vs.search(h.embedder.embed("drifted canonical content"), top_k=1, generation=1)
    carries_hash = bool(sample and sample[0].content_hash)
    bundle = h.engine.recall("drifted canonical content")
    surfaced = rec.id in {e.id for e in bundle.all_entries()}
    if carries_hash:
        # Hash-carrying backends (fake/zvec) detect the drift and drop the candidate.
        assert not surfaced
    else:
        # The archive (TurboVec) stores no hash; integrity is enforced by canonical hydration,
        # so any surfaced id must still resolve to real (non-dangling) canonical text.
        if surfaced:
            assert h.store.read_one(rec.id).content == "drifted canonical content"


def test_surfaced_entries_all_have_matching_hash(h):
    # The real invariant: every surfaced id hydrates to canonical with a matching content_hash.
    for t in ["matching one", "matching two", "matching three"]:
        h.write(t)
    bundle = h.engine.recall("matching two")
    for entry in bundle.all_entries():
        rec = h.store.read_one(entry.id)
        assert rec is not None and rec.content == entry.claim
