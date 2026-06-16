# Strata Memory

A local-first, tiered, conflict-aware memory engine for AI agents and companions.

Core thesis: **the canonical store owns truth; vector indexes are replaceable;
continuity is resolved above retrieval, not inside it.** Deletion, correction, and
supersession are first-class operations, not side effects of re-embedding.

This implements the MVP scope of the Strata Memory product spec (v0.7 §13).

## Why

Most "memory" for AI agents is a vector store with upsert/search. That makes
correction and deletion structurally hard: there's no single place that knows what
is currently true, what superseded what, or whether a "deleted" fact still hides
in an index. Strata inverts that:

- **SQLite is the canonical store.** Every fact, version, deletion, and dependency
  edge lives there. Vector and lexical indexes are *derived* and rebuildable.
- **Deletion is canonical-first.** A tombstone in SQLite is the authoritative
  deletion guarantee. Index cleanup is best-effort cleanup of a derived surface,
  verified afterward — never the other way around.
- **A resolver sits above retrieval.** Raw similarity hits are merged, hydrated
  against canonical state, filtered for status/sensitivity, and reconciled for
  recency/confidence/supersession before anything is returned to a caller.

## Architecture

```
strata/
  canonical/     SQLite schema + CanonicalStore (records, versions, dependencies,
                  tombstones, embedding generations, per-ID index acks)
  lexical/       FTS5/BM25 lexical index (contentful — see docs/PINS.md for why)
  deletion/      Canonical-first deletion/correction state machine +
                  managed-surfaces registry (each index surface implements
                  remove()/absent() so deletion can be verified per-surface)
  resolver/      Status/recency/confidence/supersession/validity resolution
                  -> ranked candidates -> BeliefBundle
  coordinator/   Durable single-writer Write Coordinator (op-log, FIFO-per-id,
                  destructive global barrier, crash recovery, idempotent retry)
  vector/        MemoryVectorStore interface + pluggable Embedder
                    - fake.py            in-memory fake (invariant gate target)
                    - zvec_adapter.py    hot adapter (zvec 0.5.0)
                    - turbovec_adapter.py archive adapter (turbovec 0.8.0, IdMapIndex)
  reflection/    L1.5 aggregation buffer, background consolidation +
                  contradiction audit, proposal review states
  policy/        Minimal sensitivity/permission policy layer
  gateway/       In-process Python API (api.py) + thin stdlib HTTP/JSON wrapper (http.py)
  cli.py         End-to-end SDK example / demo
```

Two vector backends sit behind one `MemoryVectorStore` interface:

- **zvec** — hot adapter, soft-delete with threshold-gated physical compaction.
- **TurboVec** — archive adapter, driven directly via `IdMapIndex` (no
  LangChain/Haystack/LlamaIndex wrapper), one index per embedding generation.

Both are optional; the foundation (canonical store, lexical index, deletion,
resolver, coordinator) depends on neither and is fully exercised by an in-memory
fake vector store.

See [docs/PINS.md](docs/PINS.md) for exact pinned versions, the empirical results
of every version-behavior probe, and two deliberate design notes worth reading
before changing the lexical or zvec adapters: why FTS5 is **contentful** (not
external-content) and why `model_side` defaults to `NULL`.

## Non-negotiable invariants

Five properties are enforced by dedicated test suites in `tests/invariants/`,
each run against **all three** vector backends (fake, zvec, TurboVec) via a
parametrized fixture:

1. **Deletion** — a tombstoned record is never recalled, never hydrated, even if
   a stale index entry still points to it.
2. **Hydration** — every ID a resolver returns either resolves to live canonical
   content with a matching hash, or is dropped. No dangling IDs ever surface.
3. **Single writer** — all index mutation flows through the Write Coordinator;
   reflection is enqueue-only and never holds a vector write lock.
4. **Resolver** — explicit corrections override stale repeated evidence; active
   beats superseded/stale/archived; scoped non-contradictory claims both survive.
5. **Migration** — deletion and hydration integrity hold across an embedding
   generation change; top-k overlap is reported as a regression signal, not a
   hard gate.

## Getting started

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# optional: real vector adapters (zvec + turbovec)
.venv/bin/pip install -e ".[adapters]"
```

Run the full suite:

```bash
.venv/bin/pytest
```

Run just the invariant gate against the in-memory fake:

```bash
.venv/bin/pytest tests/invariants -q
```

Run the end-to-end demo (write, correct, hard-delete + verify, reflect, recall):

```bash
.venv/bin/python -m strata.cli demo
```

## Status

All 9 MVP build steps are complete: canonical schema, lexical index, deletion/
correction flows, resolver + belief bundle, durable Write Coordinator, the
vector-store interface (proven against a fake before either real adapter was
wired in), the zvec and TurboVec adapters, and the reflection/policy/gateway
layer. 130 tests pass, including the 5 invariant suites across all 3 backends.
