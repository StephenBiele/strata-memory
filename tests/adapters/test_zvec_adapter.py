"""Step 7 adapter probes — pin zvec 0.5.0's verified semantics (spec §11/§12).

These are the integration tests the spec requires before relying on backend behavior. They
fail loudly if a future zvec version changes the contract Strata depends on.
"""

import math

import pytest

zvec = pytest.importorskip("zvec")
pytestmark = pytest.mark.adapter

from strata.vector.base import VectorRecord
from strata.vector.zvec_adapter import ZvecHotAdapter


def _vec(seed, dim=16):
    import random
    r = random.Random(seed)
    v = [r.gauss(0, 1) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _rec(i, gen=1):
    return VectorRecord(id=i, vector=_vec(i), content_hash=f"h{i}",
                        embedding_model_id="m", embedding_generation=gen, metadata={"status": "active"})


@pytest.fixture
def adapter(tmp_path):
    return ZvecHotAdapter(str(tmp_path / "hot"), dim=16)


def test_upsert_search_similarity_orientation(adapter):
    adapter.upsert([_rec(i) for i in range(10)])
    hits = adapter.search(_vec(3), top_k=3, generation=1)
    # similarity = 1 - distance, so the exact match ranks first with score ~1.0.
    assert hits[0].id == 3
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].content_hash == "h3"


def test_soft_delete_excludes_from_search_and_contains(adapter):
    adapter.upsert([_rec(i) for i in range(10)])
    assert adapter.contains(3)
    acks = adapter.remove([3])
    assert acks == {3: True}
    assert not adapter.contains(3)
    assert 3 not in {h.id for h in adapter.search(_vec(3), top_k=5, generation=1)}


def test_upsert_readds_after_delete(adapter):
    adapter.upsert([_rec(5)])
    adapter.remove([5])
    adapter.upsert([_rec(5)])  # adapter always upserts (never insert)
    assert adapter.contains(5)


def test_generation_filter_isolates_by_generation(adapter):
    # zvec keys by Doc.id only, so different generations of the SAME id cannot coexist in one
    # collection (a re-upsert overwrites). The generation FIELD still filters distinct ids;
    # true shadow indexing across generations uses a separate collection.
    adapter.upsert([_rec(1, gen=1), _rec(2, gen=2)])
    g1 = adapter.search(_vec(1), top_k=5, generation=1)
    g2 = adapter.search(_vec(2), top_k=5, generation=2)
    assert {h.id for h in g1} == {1}
    assert {h.id for h in g2} == {2}


def test_same_id_new_generation_overwrites(adapter):
    adapter.upsert([_rec(1, gen=1)])
    adapter.upsert([VectorRecord(id=1, vector=_vec(99), content_hash="h1g2",
                                 embedding_model_id="m2", embedding_generation=2, metadata={})])
    # Only one doc per id survives; it now carries generation 2.
    assert adapter.search(_vec(1), top_k=5, generation=1) == []
    assert {h.id for h in adapter.search(_vec(99), top_k=5, generation=2)} == {1}


def test_soft_delete_survives_compaction_purge_is_best_effort(tmp_path):
    """Flag #2: the guarantee we rely on is that soft-deleted ids stay absent from recall
    across compaction. Physical disk reclamation is best-effort (threshold-gated ~30% in the
    manual probe; see docs/PINS.md) and is NOT asserted as a hard gate — the canonical
    tombstone, not zvec, is the authoritative deletion guarantee."""
    path = str(tmp_path / "purge")
    a = ZvecHotAdapter(path, dim=16)
    a.upsert([_rec(i) for i in range(50)])
    a.remove(range(10))
    a.compact()  # below threshold: may or may not reclaim disk — must not error
    for i in range(10):
        assert not a.contains(i)
        assert i not in {h.id for h in a.search(_vec(i), top_k=50, generation=1)}
    a.remove(range(10, 30))
    a.compact()  # above threshold
    for i in range(30):
        assert not a.contains(i)
