"""Shared harness for the invariant suites.

``build_engine`` assembles a full MemoryEngine. The vector store is injected via a factory so
the same invariant suites run against the in-memory fake (step 6 gate) and later against the
real zvec / TurboVec adapters (steps 7, 8) by passing a different factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from strata.canonical.records import MemoryRecord, RecordType, Status, Tier
from strata.canonical.store import CanonicalStore
from strata.coordinator.coordinator import WriteCoordinator
from strata.engine import MemoryEngine
from strata.lexical.fts import LexicalStore
from strata.resolver.resolver import Resolver
from strata.vector.base import MemoryVectorStore
from strata.vector.embedder import DeterministicHashEmbedder
from strata.vector.fake import InMemoryVectorStore


@dataclass
class Harness:
    engine: MemoryEngine
    store: CanonicalStore
    embedder: DeterministicHashEmbedder

    def write(self, content: str, **kw) -> MemoryRecord:
        return self.engine.write_memory(MemoryRecord.create(content, **kw))


def build_engine(
    vector_factory: Optional[Callable[[int], MemoryVectorStore]] = None,
    *,
    db_path: str = ":memory:",
    now_fn: Callable[[], int] = lambda: 1_000_000,
) -> Harness:
    store = CanonicalStore(db_path)
    lexical = LexicalStore(store)
    coordinator = WriteCoordinator(store)
    resolver = Resolver(store, now_fn=now_fn)
    embedder = DeterministicHashEmbedder()
    vstore = vector_factory(embedder.dim) if vector_factory else InMemoryVectorStore()
    engine = MemoryEngine(
        store, lexical, {"hot": vstore}, coordinator, resolver, embedder, active_generation=1
    )
    return Harness(engine=engine, store=store, embedder=embedder)
