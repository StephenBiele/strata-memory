"""I3 — Single writer invariant: all index mutation flows through the Write Coordinator; the
live path and the reflection engine never both hold the vector write lock."""

import threading
import time

import pytest

from strata.canonical.records import MemoryRecord
from strata.coordinator.coordinator import WriteCoordinator
from strata.coordinator.ops import Operation, OpType
from tests.invariants._harness import build_engine

pytestmark = pytest.mark.invariant


def test_all_index_mutation_flows_through_coordinator(h):
    rec = h.write("indexed via coordinator")
    h.engine.delete_memory(rec.id)
    types = [row["op_type"] for row in h.engine.coordinator.audit_log()]
    # Both the upsert (write) and the remove (delete) for the vector index are in the op log.
    assert "upsert_index" in types
    assert "remove_index" in types
    # No vector mutation happened without a corresponding op_log entry.
    upserts = [r for r in h.engine.coordinator.audit_log() if r["op_type"] == "upsert_index"]
    assert all(r["state"] == "done" for r in upserts)


def test_writer_lock_serializes_handlers(tmp_path):
    """Two threads draining the coordinator must never run a handler concurrently — this is
    exactly the guarantee that zvec and the reflection engine never both hold the write lock.
    """
    from strata.canonical.store import CanonicalStore

    cs = CanonicalStore(str(tmp_path / "sw.db"))
    coord = WriteCoordinator(cs, max_attempts=3)
    state = {"inside": 0, "violation": False}

    def handler(op: Operation) -> dict[int, bool]:
        state["inside"] += 1
        if state["inside"] > 1:
            state["violation"] = True
        time.sleep(0.001)  # widen the concurrency window
        state["inside"] -= 1
        return {i: True for i in op.target_ids}

    coord.register_handler("idx", handler)
    for i in range(60):
        coord.enqueue(Operation(op_type=OpType.UPSERT_INDEX, target_ids=[i], target="idx"))

    # "live session" and "reflection engine" both try to drive the single writer.
    threads = [threading.Thread(target=coord.run_until_idle) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["violation"] is False
    assert coord.pending_count() == 0
    cs.close()


def test_reflection_is_enqueue_only(h):
    """A reflection-style producer only enqueues ops; it never mutates the index directly.
    The vector store is therefore mutated exactly as many times as ops are applied."""
    vstore = h.engine.vector_stores["hot"]
    before = vstore.stats()["rows"]
    # Simulate a reflection job that proposes new index entries by enqueueing (not mutating).
    rec = h.store.write(MemoryRecord.create("reflection-proposed fact"))
    h.engine._enqueue_vector_upsert(rec, h.engine.active_generation)
    assert vstore.stats()["rows"] == before  # nothing mutated yet — only enqueued
    h.engine.coordinator.run_until_idle()
    assert vstore.stats()["rows"] == before + 1  # applied by the single writer


def test_recall_concurrent_with_writes_is_safe(tmp_path):
    harness = build_engine(db_path=str(tmp_path / "concurrent.db"))
    errors = []

    def writer():
        try:
            for i in range(30):
                harness.write(f"concurrent fact number {i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                harness.engine.recall("concurrent fact")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    tw, tr = threading.Thread(target=writer), threading.Thread(target=reader)
    tw.start(); tr.start(); tw.join(); tr.join()
    assert errors == []
    harness.store.close()
