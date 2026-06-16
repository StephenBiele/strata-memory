"""Step 8 adapter probes — pin turbovec 0.8.0's verified semantics (spec §9, §11, §12)."""

import os

import pytest

pytest.importorskip("turbovec")
pytest.importorskip("numpy")
pytestmark = pytest.mark.adapter

import numpy as np

from strata.resolver.diversity import diversify
from strata.vector.base import VectorRecord
from strata.vector.turbovec_adapter import TurboVecArchiveAdapter


def _vec(seed, dim=16):
    r = np.random.RandomState(seed)
    v = r.randn(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def _rec(i, gen=1):
    return VectorRecord(id=i, vector=_vec(i), content_hash=f"h{i}",
                        embedding_model_id="m", embedding_generation=gen, metadata={})


@pytest.fixture
def archive(tmp_path):
    return TurboVecArchiveAdapter(str(tmp_path / "arch"), dim=16)


def test_add_search_similarity_only_higher_is_better(archive):
    archive.upsert([_rec(i) for i in range(10)])
    hits = archive.search(_vec(3), top_k=3, generation=1)
    assert hits[0].id == 3                     # exact match ranks first
    assert hits[0].score >= hits[-1].score     # similarity: higher = better
    assert hits[0].content_hash == ""          # archive stores no metadata/text


def test_upsert_is_remove_then_add(archive):
    archive.upsert([_rec(5)])
    # Re-upserting the same id must not raise "id already present".
    archive.upsert([_rec(5)])
    assert archive.contains(5)


def test_remove_then_readd_same_id(archive):
    archive.upsert([_rec(7)])
    assert archive.remove([7]) == {7: True}
    assert not archive.contains(7)
    archive.upsert([_rec(7)])
    assert archive.contains(7)


def test_per_generation_indices_isolated(archive):
    archive.upsert([_rec(1, gen=1)])
    archive.upsert([_rec(1, gen=2)])  # same id, different generation -> separate index
    assert {h.embedding_generation for h in archive.search(_vec(1), generation=1)} == {1}
    assert {h.embedding_generation for h in archive.search(_vec(1), generation=2)} == {2}


def test_tvim_write_load_roundtrip(tmp_path):
    a = TurboVecArchiveAdapter(str(tmp_path / "arch"), dim=16)
    a.upsert([_rec(i) for i in range(5)])
    a.persist()
    assert os.path.exists(str(tmp_path / "arch" / "archive.gen1.tvim"))
    b = TurboVecArchiveAdapter(str(tmp_path / "arch"), dim=16)  # reload from .tvim
    assert b.contains(3)


def test_purge_generation_removes_tvim(tmp_path):
    a = TurboVecArchiveAdapter(str(tmp_path / "arch"), dim=16)
    a.upsert([_rec(i) for i in range(3)])
    a.persist()
    a.purge_generation(1)
    assert not a.contains(1)
    assert not os.path.exists(str(tmp_path / "arch" / "archive.gen1.tvim"))


def test_diversity_is_applied_on_text_not_vectors():
    # Diversity de-dupes near-identical canonical text (post-hydration), independent of vectors.
    texts = ["user loves hiking in the mountains",
             "user loves hiking in the mountains every weekend",  # near-duplicate
             "user enjoys cooking italian food"]
    keep = diversify(texts, threshold=0.6)
    kept = [texts[i] for i in keep]
    assert "user enjoys cooking italian food" in kept
    assert len(kept) == 2  # the near-duplicate hiking line is dropped
