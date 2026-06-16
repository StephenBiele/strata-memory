"""Step 2 unit suite: FTS5/BM25 lexical index."""

import pytest

from strata.canonical.records import MemoryRecord, Status
from strata.ids import new_operation_id


@pytest.fixture
def stores():
    from strata.canonical.store import CanonicalStore
    from strata.lexical.fts import LexicalStore
    cs = CanonicalStore(":memory:")
    lx = LexicalStore(cs)
    yield cs, lx
    cs.close()


def _add(cs, lx, content, **kw):
    rec = MemoryRecord.create(content, **kw)
    cs.write(rec)
    lx.index(rec)
    return rec


def test_secure_delete_enabled_at_creation(stores):
    cs, lx = stores
    val = cs._conn().execute(
        f"SELECT v FROM {lx.table}_config WHERE k = 'secure-delete'"
    ).fetchone()
    assert val is not None and int(val[0]) == 1


def test_bm25_ranking_order(stores):
    cs, lx = stores
    _add(cs, lx, "the quick brown fox jumps")
    _add(cs, lx, "fox fox fox everywhere a fox")
    hits = lx.search("fox", top_k=10)
    assert len(hits) == 2
    # more occurrences => stronger (more negative) bm25 => ranked first.
    assert hits[0][1] <= hits[1][1]
    assert "fox fox fox" in hits[0][0].content


def test_exact_name_and_date_lookup(stores):
    cs, lx = stores
    _add(cs, lx, "meeting with Dana on 2026-03-14 about budget")
    assert {r.id for r, _ in lx.search("Dana")} == {lx.search("Dana")[0][0].id}
    assert lx.search("2026-03-14")
    assert lx.search("budget")


def test_search_excludes_tombstoned(stores):
    cs, lx = stores
    rec = _add(cs, lx, "sensitive phrase alpha")
    assert lx.search("alpha")
    cs.tombstone(rec.id, job_id=new_operation_id())
    # Index may still hold the posting, but search must not surface a tombstoned record.
    assert lx.search("alpha") == []
    assert lx.contains(rec.id)  # defense-in-depth: filtered at query time, still in index


def test_search_excludes_nonactive_status(stores):
    cs, lx = stores
    rec = _add(cs, lx, "superseded preference gamma")
    cs.set_status(rec.id, Status.SUPERSEDED)
    assert lx.search("gamma") == []
    assert lx.search("gamma", statuses=None)  # explicit: no status filter -> visible


def test_remove_clears_postings(stores):
    cs, lx = stores
    rec = _add(cs, lx, "ephemeral token delta")
    lx.remove([rec.id])
    assert not lx.contains(rec.id)
    assert lx.search("delta", statuses=None) == []


def test_remove_is_drift_proof_after_content_change(stores):
    cs, lx = stores
    rec = _add(cs, lx, "oldword indexed here")
    # Contentful FTS holds its own copy, so removal stays exact even if canonical text changed.
    cs.write(rec.with_content("newword replaces it"))
    lx.remove([rec.id])
    assert lx.search("oldword", statuses=None) == []


def test_reindex_in_place_drops_old_postings(stores):
    cs, lx = stores
    rec = _add(cs, lx, "oldword indexed here")
    updated = rec.with_content("newword replaces it")
    cs.write(updated)
    lx.index(updated)  # re-index in place
    assert lx.search("oldword", statuses=None) == []
    assert {r.id for r, _ in lx.search("newword", statuses=None)} == {rec.id}


def test_rebuild_resyncs_from_canonical(stores):
    cs, lx = stores
    rec = MemoryRecord.create("rebuilt content epsilon")
    cs.write(rec)  # written to canonical but NOT indexed
    assert lx.search("epsilon") == []
    lx.rebuild()
    assert {r.id for r, _ in lx.search("epsilon")} == {rec.id}


def test_divergence_is_safe_and_rebuild_reconciles(stores):
    # Simulate a crash that leaves the FTS index holding text canonical no longer has.
    cs, lx = stores
    rec = _add(cs, lx, "orphan posting omega")
    cs.hard_delete(rec.id)              # canonical row gone, FTS posting NOT removed
    assert lx.contains(rec.id)         # divergence: FTS still has it
    # Query-time JOIN to canonical means the orphan can never reach active recall.
    assert lx.search("omega", statuses=None) == []
    # rebuild() is the authoritative reconciliation: it drops the orphan posting.
    lx.rebuild()
    assert not lx.contains(rec.id)
