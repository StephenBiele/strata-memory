import pytest

from tests.invariants._harness import build_engine


def _available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# Every invariant suite runs against the in-memory fake (step 6 gate) AND, when installed, the
# real zvec hot adapter (step 7) and TurboVec archive adapter (step 8). The same invariants
# must hold against the real backends.
_BACKENDS = (
    ["fake"]
    + (["zvec"] if _available("zvec") else [])
    + (["turbovec"] if _available("turbovec") and _available("numpy") else [])
)


@pytest.fixture(params=_BACKENDS)
def h(request, tmp_path):
    backend = request.param
    if backend == "fake":
        harness = build_engine()
    elif backend == "zvec":
        from strata.vector.zvec_adapter import ZvecHotAdapter

        def factory(dim: int):
            return ZvecHotAdapter(str(tmp_path / "zvec_hot"), dim=dim)

        harness = build_engine(vector_factory=factory, db_path=str(tmp_path / "canon.db"))
    else:
        from strata.vector.turbovec_adapter import TurboVecArchiveAdapter

        def factory(dim: int):
            return TurboVecArchiveAdapter(str(tmp_path / "tv_arch"), dim=dim)

        harness = build_engine(vector_factory=factory, db_path=str(tmp_path / "canon.db"))
    yield harness
    harness.store.close()
