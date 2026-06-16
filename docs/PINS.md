# Pinned versions & flag resolutions

Source of truth for exact dependency versions, environment, and the empirical results of the
adapter probes the spec (v0.7 §11/§12) requires. Updated as each step proves a behavior.

## Environment
- Python: **3.12.13** (homebrew `/opt/homebrew/bin/python3.12`), venv at `.venv/`.
  Spec locks 3.12; system default is 3.14.5 — not used.
- SQLite (stdlib `sqlite3`): library **3.53.1** (≥3.42 → FTS5 `'secure-delete'` supported).
- pytest: **9.1.0** (plan estimated 8.x; actual recorded here).

## Pinned runtime adapters (installed only for steps 7–8)
| Package | Version | Confirmed on PyPI | Role |
|---|---|---|---|
| zvec | 0.5.0 | yes (latest) | hot vector adapter |
| turbovec | 0.8.0 | yes (latest) | archive vector adapter (needs numpy) |

### TurboVec 0.8.0 verified semantics (step 8 probe)
- `IdMapIndex(dim)` (default bit_width 4); `add_with_ids(float32[n,dim], uint64[n])` → `prepare()`
  → `search(q, k, allowlist=)` returns `(scores, ids)`, **similarity higher = better** (no
  conversion). Incremental add + re-prepare works.
- **Duplicate `add_with_ids` of an existing id RAISES** ("id already present") → adapter upsert =
  remove-if-present then add. `remove(id)` then re-add of the same id works cleanly (no stale map).
- Stores only vectors+ids — **no metadata/text sidecar** (canonical store stays sole source of
  truth). Therefore archive hits carry no `content_hash`; hydration integrity is enforced via
  canonical hydration (dangling/tombstone), not hash matching.
- Quantized (2–4 bit) → not comparable across generations: adapter keeps **one IdMapIndex per
  generation** (`<name>.gen<N>.tvim`), matching spec §11 ".tvim generations". Diversity is applied
  post-hydration on canonical text (`resolver.diversity`), never on the quantized vectors.

## Flag resolutions (filled in as proven)
1. **Python version** — resolved: 3.12.13 venv. ✓
2. **zvec physical-purge semantics** — RESOLVED (step 7 probe, zvec 0.5.0). `delete()` is a
   soft-delete: `doc_count` drops, `fetch`/`query` immediately exclude the id, deletion is
   durable. Physical purge via `optimize()` is **threshold-gated at ~30% deleted ratio**:
   probe showed 10% deleted → disk unchanged after optimize (10.67MB→10.67MB); 45% deleted →
   disk halved (10.67MB→5.34MB). **Matches the spec assumption.** Therefore: the canonical
   tombstone is the authoritative deletion guarantee; zvec physical reclamation is
   best-effort and only fires above the threshold. Hard-delete physical-erasure for the zvec
   surface is reported as best-effort, not guaranteed by any single call.
   Note: zvec COSINE `score` is a DISTANCE (0.0 = identical); the adapter converts to
   similarity = 1 - distance so higher = better, consistent with the rest of Strata.

   **Known behavior → post-MVP compaction policy (spec §13 Growth and Compaction).** Because
   `optimize()` only reclaims below-threshold (~30% soft-deleted), a long-running companion
   with a *low delete rate* (few forgets, many adds) will accumulate soft-deleted rows that
   `optimize()` never reclaims — zvec disk grows soft-bounded by the threshold. The post-MVP
   compaction controls must do one of:
   - periodically **rebuild from canonical** (TurboVec-style: indexes are rebuildable from the
     canonical store) to force full recompaction, rather than relying on `optimize()`; or
   - **batch deletions** and trigger `optimize()` only once a batch pushes the collection over
     the ~30% threshold; or
   - explicitly **accept** the soft bound as the disk policy.
   The adapter exposes `compact()` (wraps `optimize()` + flush) for whichever policy is chosen.
   Recall is unaffected at any delete ratio — soft-deleted ids stay absent from results
   regardless of physical reclamation.
3. **FTS5 secure-delete coverage** — RESOLVED (step 2). `secure-delete=1` set at creation:
   syntax `INSERT INTO records_fts(records_fts, rank) VALUES('secure-delete', 1)` verified.

   **⚠ DO NOT revert to external-content FTS5 thinking it is "cleaner".** We deliberately use
   **contentful FTS5**. The exact failure that drove the switch (reproducible on SQLite 3.53.1):
   an external-content table (`content='records', content_rowid='id'`) populated by manual
   `INSERT INTO records_fts(rowid, content)` and then mutated with per-row `DELETE FROM
   records_fts WHERE rowid=?` raised **`sqlite3.DatabaseError: database disk image is
   malformed`** on the second incremental write. Separately, external-content `DELETE`
   recomputes postings from the *current* external row, so if `records.content` changed first
   the OLD postings leak (proven: token "oldword" still matched after the row was updated then
   deleted). That is the origin of the "FTS5-first deletion ordering" requirement — it only
   exists to work around external-content; contentful FTS5 does not need it.

   **Contentful trade-off (intentional):** the FTS5 `%_content` shadow holds a tokenized copy
   of the text, so `records.content` and the FTS index are two physical copies. This is an
   **index that needs the text to function, not a parallel truth store** — distinct from the
   TurboVec "no sidecar" principle (which forbids a second *authoritative* copy / parallel
   deletion surface). Consequences a maintainer must respect:
   - **Deletion must scrub BOTH** the canonical row and the FTS5 shadow. The FTS5 surface
     (`LexicalSurface`, registry name `fts5`) implements `remove()` (purge) and `absent()`
     (verify), so the deletion state machine cleans and verifies it per-ID like any other
     managed surface. secure-delete scrubs the shadow on delete.
   - **If the two ever diverge** (a crash between the canonical write and the FTS write), the
     index can hold text canonical no longer has. `LexicalStore.rebuild()` (clear + re-index
     from canonical) is the authoritative reconciliation and should be run on suspected
     divergence; `search()` also filters tombstoned/inactive rows at query time as
     defense-in-depth so a stale posting can never reach active recall.
   Canonical text remains solely in `records` (the FTS shadow is derived, not authoritative).
4. **zvec re-add after filter delete / IdMapIndex purge** — RESOLVED (step 7 probe, zvec
   0.5.0). After `delete_by_filter`, a re-`insert()` of the same id reports OK but `fetch`
   returns False (the doc is NOT retrievable — stale PK→doc_id map). `upsert()` of the same id
   works (`fetch` True). **Matches the spec §12 note.** The adapter therefore always uses
   `upsert()`, never `insert()`. (TurboVec IdMapIndex re-add probed at step 8.)

## HTTP API surface
- Spec §13 MVP lists "JSON/HTTP API". `gateway/api.py` is the real contract; `gateway/http.py`
  is a thin stdlib `http.server` JSON wrapper (step 9) mapping the 9 §12 endpoints 1:1.
  **Shipped** as a working single-threaded stub (roundtrip-tested). Deferred to post-MVP
  (explicit): auth, concurrency hardening, and streaming.

## Installed versions (pip freeze)
- numpy==2.4.6, pytest==9.1.0, turbovec==0.8.0, zvec==0.5.0 (Python 3.12.13).

## MVP completion status
All 9 build steps complete; 128 tests pass. The 5 non-negotiable invariants hold against the
in-memory fake AND the real zvec 0.5.0 / TurboVec 0.8.0 adapters (18 fixture-based invariant
tests × 3 backends). All four version flags resolved empirically, all matching spec assumptions.
