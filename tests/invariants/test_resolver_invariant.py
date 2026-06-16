"""I4 — Resolver invariant: an explicit correction overrides repeated older evidence;
non-contradictory scoped claims do not over-supersede."""

import pytest

from strata.canonical.records import MemoryRecord, RecordType, Status
from tests.invariants._harness import build_engine

pytestmark = pytest.mark.invariant


def _current(bundle):
    return {e.claim for e in bundle.current_beliefs}


def test_explicit_correction_overrides_repeated_old_evidence(h):
    # Old belief reinforced many times -> high confidence/salience.
    old = h.engine.write_memory(
        MemoryRecord.create("user commutes by car", status=Status.REINFORCED,
                            confidence=0.97, salience=0.95)
    )
    new = MemoryRecord.create("user commutes by bike", confidence=0.55)
    h.engine.correct(old.id, new)
    current = _current(h.engine.recall("user commutes"))
    assert "user commutes by bike" in current
    assert "user commutes by car" not in current


def test_active_preferred_over_superseded(h):
    old = h.write("favorite color is red")
    h.engine.correct(old.id, MemoryRecord.create("favorite color is blue"))
    current = _current(h.engine.recall("favorite color"))
    assert current == {"favorite color is blue"}


def test_scoped_claims_do_not_over_supersede(h):
    h.write("prefers tea for work")
    h.write("prefers coffee on weekends")
    current = _current(h.engine.recall("prefers"))
    assert "prefers tea for work" in current
    assert "prefers coffee on weekends" in current


def test_speculative_reflection_marked_hypothesis(h):
    h.engine.write_memory(
        MemoryRecord.create("user may be training for a marathon",
                            record_type=RecordType.REFLECTION, status=Status.CANDIDATE)
    )
    bundle = h.engine.recall("user may be training for a marathon")
    assert _current(bundle) == set()
    assert any(e.status == "hypothesis" for e in bundle.hypotheses)
