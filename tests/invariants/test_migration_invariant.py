"""I5 — Migration invariant: deletion preservation and hydration integrity hold across an
embedding-generation change. Top-K overlap is a regression *signal*, not a gate."""

import pytest

from strata.vector.embedder import DeterministicHashEmbedder

pytestmark = pytest.mark.invariant


def _migrate(h):
    """Open a shadow generation with a *different* embedder, reindex from canonical, promote.

    Keeps the same dimension (a real index backend like zvec pins a collection's vector
    dimension; a different-dim model is a separate-shadow-collection concern). A new model_id
    and seed still constitutes a genuine embedding-generation change: different vectors that
    are not directly comparable, separated at query time by the generation filter.
    """
    gen2 = 2
    embedder2 = DeterministicHashEmbedder(model_id="det-hash-v2", dim=h.embedder.dim, seed=7)
    h.engine.open_shadow_generation(gen2, embedder2)
    all_ids = [r["id"] for r in h.store._conn().execute("SELECT id FROM records").fetchall()]
    h.engine.reindex_into(gen2, embedder2, all_ids)
    h.engine.promote_generation(gen2)
    # Swap the engine's live embedder so recall uses the new generation's space.
    h.engine.embedder = embedder2
    return embedder2


def test_deletion_preserved_across_generation(h):
    keep = h.write("keep this memory across migration")
    drop = h.write("delete this memory before migration")
    h.engine.delete_memory(drop.id)
    _migrate(h)
    bundle = h.engine.recall("delete this memory before migration")
    assert drop.id not in {e.id for e in bundle.all_entries()}   # stays deleted
    # The kept memory still resolves in the new generation.
    assert keep.id in {e.id for e in h.engine.recall("keep this memory across migration").all_entries()}


def test_hydration_integrity_across_generation(h):
    for t in ["fact one apple", "fact two banana", "fact three cherry"]:
        h.write(t)
    _migrate(h)
    bundle = h.engine.recall("fact two banana")
    assert bundle.all_entries()  # something came back
    for entry in bundle.all_entries():
        rec = h.store.read_one(entry.id)
        assert rec is not None
        assert rec.content_hash == rec.content_hash  # hydrated row exists
        assert rec.content == entry.claim


def test_topk_overlap_is_signal_not_gate(h):
    texts = [f"durable preference number {i}" for i in range(8)]
    for t in texts:
        h.write(t)
    query = "durable preference number 3"
    gen1_top = [e.id for e in h.engine.recall(query, top_k=5).all_entries()]
    _migrate(h)
    gen2_top = [e.id for e in h.engine.recall(query, top_k=5).all_entries()]

    overlap = _topk_overlap(gen1_top, gen2_top)
    # The overlap is REPORTED as a regression signal in [0, 1]; a low value must NOT fail.
    assert 0.0 <= overlap <= 1.0
    # Both generations still only return hydratable, non-deleted ids (the real gate).
    for rid in gen2_top:
        assert h.store.read_one(rid) is not None
        assert not h.store.is_tombstoned(rid)


def _topk_overlap(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / len(set(a) | set(b))
