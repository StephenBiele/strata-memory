"""Step 4 unit suite: resolver + belief bundle."""

import pytest

from strata.canonical.records import (
    MemoryRecord,
    Relation,
    RecordType,
    Status,
    Tier,
)
from strata.ids import new_operation_id
from strata.resolver.bundle import Category, DEFAULT_BUDGETS
from strata.resolver.resolver import Candidate, Resolver


@pytest.fixture
def env():
    from strata.canonical.store import CanonicalStore
    cs = CanonicalStore(":memory:")
    yield cs, Resolver(cs, now_fn=lambda: 1_000_000)
    cs.close()


def _w(cs, content, **kw):
    rec = MemoryRecord.create(content, **kw)
    cs.write(rec)
    return rec


def _cands(*recs):
    return [Candidate(r.id, source="zvec", content_hash=r.content_hash) for r in recs]


def test_dangling_id_dropped(env):
    cs, rz = env
    rec = _w(cs, "real belief")
    bundle = rz.resolve([Candidate(rec.id, content_hash=rec.content_hash), Candidate(424242)])
    ids = [e.id for e in bundle.all_entries()]
    assert ids == [rec.id]


def test_content_hash_mismatch_dropped(env):
    cs, rz = env
    rec = _w(cs, "canonical text")
    stale = Candidate(rec.id, content_hash="deadbeef")  # index drifted
    assert rz.resolve([stale]).all_entries() == []


def test_tombstoned_never_surfaced(env):
    cs, rz = env
    rec = _w(cs, "to forget")
    cs.tombstone(rec.id, job_id=new_operation_id())
    assert rz.resolve(_cands(rec)).all_entries() == []


def test_explicit_correction_overrides_repeated_old_evidence(env):
    cs, rz = env
    # Old belief reinforced many times: high confidence/salience.
    old = _w(cs, "user drinks coffee", status=Status.REINFORCED, confidence=0.95, salience=0.9)
    new = MemoryRecord.create("user drinks tea", confidence=0.6)
    cs.supersede(old.id, new)  # explicit correction -> supersedes edge, old marked superseded
    bundle = rz.resolve(_cands(old, new))
    claims = [e.claim for e in bundle.current_beliefs]
    assert "user drinks tea" in claims
    assert "user drinks coffee" not in claims


def test_scoped_claims_do_not_over_supersede(env):
    cs, rz = env
    work = _w(cs, "prefers tea for work")
    weekend = _w(cs, "prefers coffee on weekends")
    bundle = rz.resolve(_cands(work, weekend))
    claims = {e.claim for e in bundle.current_beliefs}
    assert claims == {"prefers tea for work", "prefers coffee on weekends"}


def test_unresolved_contradiction_goes_to_open_conflicts(env):
    cs, rz = env
    a = _w(cs, "meeting is monday")
    b = _w(cs, "meeting is tuesday")
    cs.add_dependency(a.id, b.id, Relation.CONTRADICTS)
    bundle = rz.resolve(_cands(a, b))
    assert bundle.current_beliefs == []
    assert {e.claim for e in bundle.open_conflicts} == {"meeting is monday", "meeting is tuesday"}
    assert all(e.status == "contradicted" for e in bundle.open_conflicts)


def test_reflection_marked_hypothesis_not_fact(env):
    cs, rz = env
    refl = _w(cs, "user might enjoy hiking", record_type=RecordType.REFLECTION,
              status=Status.CANDIDATE)
    bundle = rz.resolve(_cands(refl))
    assert bundle.current_beliefs == []
    assert len(bundle.hypotheses) == 1
    assert bundle.hypotheses[0].status == "hypothesis"


def test_superseded_status_excluded_from_current(env):
    cs, rz = env
    rec = _w(cs, "old preference", status=Status.SUPERSEDED)
    assert rz.resolve(_cands(rec)).current_beliefs == []


def test_validity_window_excludes_expired(env):
    cs, rz = env
    rec = _w(cs, "temporary status", valid_until=999_999)  # before now=1_000_000
    assert rz.resolve(_cands(rec)).all_entries() == []


def test_category_routing_and_budget(env):
    cs, rz = env
    episode = _w(cs, "worked on the API project today", tier=Tier.L2, record_type=RecordType.EPISODE)
    guidance = _w(cs, "avoid repeating greetings", record_type=RecordType.PROFILE,
                  tier=Tier.L3, record_subtype="interaction_guidance")
    facts = [_w(cs, f"fact number {i}") for i in range(12)]
    bundle = rz.resolve(_cands(episode, guidance, *facts))
    assert episode.id in [e.id for e in bundle.recent_context]
    assert guidance.id in [e.id for e in bundle.interaction_guidance]
    # current_beliefs trimmed to its max budget (8).
    assert len(bundle.current_beliefs) == DEFAULT_BUDGETS[Category.CURRENT_BELIEF][1]


def test_bundle_json_and_size(env):
    cs, rz = env
    rec = _w(cs, "serializable belief")
    bundle = rz.resolve(_cands(rec))
    import json
    d = json.loads(bundle.to_json())
    assert d["current_beliefs"][0]["claim"] == "serializable belief"
    assert d["current_beliefs"][0]["source_ids"] == [rec.id]
    assert bundle.serialized_size() > 0
