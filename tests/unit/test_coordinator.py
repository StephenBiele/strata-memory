"""Step 5 unit suite: durable Write Coordinator."""

import time

import pytest

from strata.coordinator.coordinator import WriteCoordinator
from strata.coordinator.ops import Operation, OpState, OpType
from strata.coordinator.worker import WriterWorker


class RecordingHandler:
    """Applies ops by recording order; can be told to fail specific ids a number of times."""

    def __init__(self):
        self.applied: list[str] = []
        self.fail_ids: dict[int, int] = {}  # id -> remaining failures

    def __call__(self, op: Operation) -> dict[int, bool]:
        acks = {}
        for rid in op.target_ids:
            remaining = self.fail_ids.get(rid, 0)
            if remaining > 0:
                self.fail_ids[rid] = remaining - 1
                acks[rid] = False
            else:
                acks[rid] = True
        if all(acks.values()):
            self.applied.append(op.op_id)
        return acks


@pytest.fixture
def env():
    from strata.canonical.store import CanonicalStore
    cs = CanonicalStore(":memory:")
    coord = WriteCoordinator(cs, max_attempts=3, backoff_base_ms=20)
    h = RecordingHandler()
    coord.register_handler("idx", h)
    yield cs, coord, h
    cs.close()


def _op(coord, ids, op_type=OpType.UPSERT_INDEX):
    op = Operation(op_type=op_type, target_ids=list(ids), target="idx")
    coord.enqueue(op)
    return op


def test_enqueue_apply_success(env):
    cs, coord, h = env
    op = _op(coord, [1])
    assert coord.apply_next() == op.op_id
    assert coord.op_status(op.op_id)["state"] == OpState.DONE.value
    assert h.applied == [op.op_id]


def test_idempotent_enqueue(env):
    cs, coord, h = env
    op = Operation(op_type=OpType.UPSERT_INDEX, target_ids=[1], target="idx")
    coord.enqueue(op)
    coord.enqueue(op)  # same op_id -> ignored
    assert len(coord.audit_log()) == 1


def test_fifo_per_record_id(env):
    cs, coord, h = env
    a = _op(coord, [1]); b = _op(coord, [1])
    coord.run_until_idle()
    assert h.applied == [a.op_id, b.op_id]


def test_destructive_is_global_barrier(env):
    cs, coord, h = env
    a = _op(coord, [1])
    d = _op(coord, [2], op_type=OpType.REMOVE_INDEX)  # destructive
    c = _op(coord, [3])
    coord.run_until_idle()
    # d waits for a (everything before it); c waits for d (barrier) -> strict a,d,c.
    assert h.applied == [a.op_id, d.op_id, c.op_id]


def test_no_head_of_line_block_across_ids_during_backoff(env):
    cs, coord, h = env
    a = _op(coord, [1]); b = _op(coord, [2])
    h.fail_ids = {1: 5}              # a will keep failing -> goes into backoff
    coord.run_until_idle()
    # b (different id, non-destructive) still applied despite a being stuck.
    assert b.op_id in h.applied
    assert coord.op_status(a.op_id)["state"] == OpState.PARTIAL_FAILURE.value


def test_retry_after_recovery(env):
    cs, coord, h = env
    op = _op(coord, [1])
    h.fail_ids = {1: 1}             # fail once
    coord.apply_next()             # -> partial_failure (in backoff)
    assert coord.op_status(op.op_id)["state"] == OpState.PARTIAL_FAILURE.value
    coord.retry(op.op_id)          # force retry now; handler succeeds this time
    coord.apply_next()
    assert coord.op_status(op.op_id)["state"] == OpState.DONE.value


def test_exhausted_attempts_become_failed(env):
    cs, coord, h = env
    op = _op(coord, [1])
    h.fail_ids = {1: 99}
    for _ in range(3):
        coord.retry(op.op_id)
        coord.apply_next()
    assert coord.op_status(op.op_id)["state"] == OpState.FAILED.value


def test_crash_recovery_replays_in_progress(env):
    cs, coord, h = env
    op = _op(coord, [1])
    # Simulate a crash mid-apply: force the op to in_progress and never finish.
    coord._set_state(coord.op_status(op.op_id)["seq"], OpState.IN_PROGRESS)
    assert coord.recover() == 1
    assert coord.op_status(op.op_id)["state"] == OpState.PENDING.value
    coord.apply_next()
    assert coord.op_status(op.op_id)["state"] == OpState.DONE.value


def test_per_id_acks_recorded_in_canonical(env):
    cs, coord, h = env
    op = _op(coord, [1, 2])
    coord.apply_next()
    rows = cs._conn().execute(
        "SELECT record_id, ack FROM index_acknowledgements WHERE op_id = ?", (op.op_id,)
    ).fetchall()
    assert {(r["record_id"], r["ack"]) for r in rows} == {(1, "success"), (2, "success")}


def test_background_worker_drains_queue(tmp_path):
    # Multi-threaded use is file-backed: each thread gets its own WAL connection. The shared
    # in-memory connection is a single-threaded test convenience and isn't used here.
    from strata.canonical.store import CanonicalStore
    cs = CanonicalStore(str(tmp_path / "coord.db"))
    coord = WriteCoordinator(cs, max_attempts=3, backoff_base_ms=20)
    h = RecordingHandler()
    coord.register_handler("idx", h)
    worker = WriterWorker(coord, idle_wait_s=0.01)
    worker.start()
    try:
        ops = [_op(coord, [i]) for i in range(20)]
        deadline = time.time() + 5
        while coord.pending_count() > 0 and time.time() < deadline:
            time.sleep(0.01)
        assert coord.pending_count() == 0
        assert set(h.applied) == {o.op_id for o in ops}
    finally:
        worker.stop()
        cs.close()
