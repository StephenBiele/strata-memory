# Strata Memory

A local-first, tiered, conflict-aware memory engine for AI agents and companions. 

The core thesis of this framework is that memory truth belongs in a structural canonical store, vector systems are merely temporary indexes, and conversational continuity must be resolved above retrieval rather than inside it. Deletion, correction, and supersession are treated as first-class operations instead of side effects of re-embedding.

---

## 💡 The Paradigm Shift

### For the Layman: Why this feels like actual human memory
Most AI memory systems operate like an unorganized stack of digital sticky notes. When you tell them a new fact, they add a note. If you later say, *"Actually, change that, my favorite color is blue, not green,"* or *"Forget what I said about my project,"* the AI does not destroy the old note. It simply sticks a new note on top of the pile. When retrieving data, the AI frequently gets confused by the old, stale notes still lurking in its history.

Strata Memory behaves like a real brain. When you update a preference, the old fact is systematically superseded. When you tell it to forget something, the memory is permanently erased from the core. Stale, deleted data can never leak back into your conversation to disrupt your companion's continuity.

### For the Engineer: Decoupling Truth from Ephemeral Indexes
Traditional long-term agent memory patterns rely blindly on Top-$K$ semantic similarity retrieval over a vector database. This introduces structural state-synchronization problems because vector spaces lack explicit consistency boundaries, transaction controls, or native mutation workflows.

Strata Memory inverts this paradigm through a strict separation of evidence from interpretation:
* **SQLite is the Canonical Store:** A local SQLite database serves as the single source of truth for records, versions, evidence tracking, dependencies, and authoritative hard tombstones. 
* **Indexes are Ephemeral Projections:** Pluggable vector databases (`zvec`, `TurboVec`) and lexical indexes (`FTS5`) are treated strictly as read-only, disposable, and entirely rebuildable projections. They do not own the text, the metadata, or the deletion state.
* **Deterministic Resolution Layer:** Raw similarity hits from your indexes are never passed directly to the LLM. A resolver hydrates index matches against the canonical SQLite state, verifies cryptographic content hashes, drops dead tombstones, and reconciles conflicting data based on recency and explicitness before compiling a clean `BeliefBundle`.

---

## 🚀 Quickstart: Connecting to a Local LLM

Connecting Strata Memory to a local LLM client (like Ollama) takes just a few lines of code. You pass the user query to Strata first, receive a structured, conflict-resolved belief bundle, inject it straight into the system prompt context, and write the new turn back to log history.

```python
import json
import ollama
from strata.gateway import Strata

# 1. Open the local memory gateway (creates the database and parent dirs on first use)
memory = Strata.open(db_path="~/.vui/strata.db")

user_query = "What did I tell you yesterday about my current resume project?"

# 2. Recall historical context (vector + lexical + SQLite, hydrated and conflict-resolved)
belief_bundle = memory.recall(user_query, top_k=10)

# 3. Format the bundle directly into your model's system prompt context
context_injection = f"""
You are a helpful companion. Analyze the following verified user beliefs and context before responding:
{json.dumps(belief_bundle['current_beliefs'], indent=2)}
"""

messages = [
    {"role": "system", "content": context_injection},
    {"role": "user", "content": user_query}
]

# 4. Fire the prompt to your local model instance
response = ollama.chat(model="qwen3.5:4b", messages=messages)
assistant_reply = response['message']['content']
print(f"Assistant: {assistant_reply}")

# 5. Append the complete interaction turn back into L0 raw events
memory.write_event(f"user: {user_query}")
memory.write_event(f"assistant: {assistant_reply}")

memory.close()
```

---

## ⚔️ Lineage & Differentiation

Strata Memory is built for heavy local execution and high data sovereignty. While it draws inspiration from pioneering open-source memory concepts, specific architectural constraints differentiate it from its source projects.

### 1. vs. TencentDB-Agent-Memory
* **Inspiration:** Strata adopts the progressive abstraction philosophy, human-readable storage artifacts, and the concept of running a background loop to synthesize raw interactions into durable traits.
* **The Differentiation:** TencentDB-Agent-Memory utilizes a strict 4-tier pipeline where atomic facts, episodic data, and instructions are grouped together in a single layer. Strata splits atomic facts into their own tier (L1) and introduces a dedicated **L1.5 Aggregation Buffer** to cluster related records, catch emerging contradictions, and review changes before they are prematurely promoted. Furthermore, TencentDB treats deterministic guardrails as an external operational safety layer, whereas Strata integrates them directly into the core engine as an immutable memory tier (L4) enforced before and after retrieval.

### 2. vs. zvec
* **Inspiration:** Strata ships a `zvec` hot-vector adapter (opt-in today; see [Implementation Status](#-implementation-status)) chosen for its high-performance, embedded execution, and low-latency local retrieval profiles.
* **The Differentiation:** Because `zvec` is pre-1.0 and enforces single-process-exclusive write locks, running live conversational mutations alongside asynchronous background reflection threads can trigger immediate write contention. Strata resolves this by wrapping the index inside a durable, SQLite-backed **Write Coordinator**. All mutations, background optimizations, and deletion tasks are serialized through a strict single-writer queue, ensuring concurrent reads remain completely non-blocking and safe.

### 3. vs. TurboVec
* **Inspiration:** Strata utilizes `TurboVec` as a highly efficient archive index backend, leveraging its quantized format to manage long-range semantic history without bloating local storage footprints.
* **The Differentiation:** Standard `TurboVec` implementations maintain an autonomous text docstore sidecar. Strata completely bypasses these high-level integration wrappers to avoid creating a parallel, authoritative copy of memory text or a competing deletion surface. Instead, Strata drives TurboVec’s `IdMapIndex` directly using canonical uint64 records mapping straight back to the SQLite store, maintaining absolute ownership over text provenance and deletion mechanics.

---

## 🗺️ Architectural Tiers

Strata organizes information across five distinct lifecycle layers. Records move upward as they grow more stable and verified, or downward if they become contradicted or stale. The **Status** column reflects what is wired and exercised today (see [Implementation Status](#-implementation-status) below) versus what is designed and reserved in the schema but not yet driven end-to-end.

| Tier | Name | Purpose | Canonical Form | Status |
| :--- | :--- | :--- | :--- | :--- |
| **L0** | Raw Events | Preserves exact history, transcripts, tool calls, and raw session commands. | Append-only event log. | ✅ **Live** — written every turn; lexical + vector recall. |
| **L1** | Atomic Facts | Tracks small claims, individual preferences, and explicit user corrections. | Structured records with source links. | ✅ **Live** — the workhorse tier; source-linked, superseded, tombstoned. |
| **L1.5** | Aggregation Buffer | Buffers intermediate clusters to audit duplicates, catch contradictions, and stage merges. | Reviewable aggregation records. | ✅ **Live** — deterministic Jaccard clustering + reviewable merge/contradiction proposals (`run_reflection`). |
| **L2** | Episodes & Scenarios | Captures narrative chunks, ongoing projects, and multi-turn contextual blocks. | Multi-turn summaries with source spans. | ◻ **Roadmap** — episodes are currently kept at L0; no distinct L2 tier yet. |
| **L3** | Continuity Model | Synthesizes your current best understanding of a user profile and interaction guidance. | Editable profile records backed by trace evidence. | ◻ **Roadmap** — the reference app keeps the profile as a flat document, not L3 records. |
| **L4** | Deterministic Guards | Enforces strict integrity rules, hard deletion policies, and conflict invariants without an LLM. | Executable python guardrail policies. | ◐ **Partial** — guardrail records, sensitivity policy, and deterministic tombstone enforcement are live; arbitrary user policies are not. |

> **Indexing note.** The tiered `zvec` (hot) + `TurboVec` (archive) split described below is the intended vector topology. Today the engine's default vector store — and the Strata Voice reference app — run on the rebuildable in-memory adapter (`vector/fake.py`). The `zvec`/`TurboVec` adapters are implemented and tested but are opt-in via `vector_factory`, and the hot/archive aging flow is not yet wired by default. Because vectors are ephemeral projections by design, this changes performance at scale, not correctness.

---

## 🧭 Implementation Status

Strata is built in layers, and not all of them are wired end-to-end yet. This section is the honest ledger so the architecture above is read as *design intent with a status*, not a claim that every tier is running.

**Live and exercised today** (canonical spine + working tiers):
* SQLite canonical store as the single source of truth — records, versions, content hashes, dependencies (`SUPERSEDES` / `CONTRADICTS` / `DERIVED_FROM` / …).
* The deterministic **resolver → `BeliefBundle`**: vector + FTS5 lexical hits, hydrated against canonical rows, tombstones dropped, conflicts reconciled before anything reaches the model.
* First-class **deletion (hard tombstones)** and **supersession**, routed through the single **Write Coordinator**.
* **L0** raw events, **L1** atomic facts with source links, **L1.5** reflection (deterministic clustering → reviewable merge / contradiction proposals via `run_reflection`), and **L4** guardrail records + sensitivity policy.
* Pluggable embedders; **FTS5** lexical index; the **in-memory vector adapter** as the default projection.

**Built but not wired by default** (present, tested, opt-in):
* The **`zvec`** (hot) and **`TurboVec`** (archive) vector adapters — real implementations, selectable via `vector_factory`, but not the default and not yet arranged into the hot→archive aging topology.

**Designed, not yet built** (reserved in the schema / roadmap):
* **L2** episodes-and-scenarios as a distinct tier (episodes currently live at L0).
* **L3** continuity-model profile records (reference integrations keep the profile as a flat document).
* Records automatically **migrating downward** across tiers as they age or are contradicted.

The reference integration, [**Strata Voice**](https://github.com/StephenBiele/Strata-Voice), currently drives L0 / L1 / L1.5 / L4 on the in-memory vector adapter, and layers its own time-aware selection and recency logic above `recall()`.

---

## 🔒 Non-Negotiable Invariants

The following five behaviors are strictly enforced by the core testing suite across all vector backends to guarantee absolute operational data integrity:

1. **Deletion Integrity:** A tombstoned record is instantly blocked from active recall. Even if a stale, concurrent index entry accidentally points to its ID, the hydration layer drops it immediately.
2. **Hydration Sufficiency:** The resolver never returns a dangling vector ID. Every ID returned must successfully hydrate to an active canonical row with a matching cryptographic hash, or it is purged from the bundle.
3. **Single Writer Serialization:** All index mutations flow through the Write Coordinator. Async background reflection operations are strictly enqueue-only and are barred from holding active vector write locks.
4. **Resolver Supremacy:** Explicit user corrections instantly override old repeated evidence. Active states beat archived states, and speculative background reflections are strictly marked as hypotheses until explicitly verified.
5. **Migration Preservation:** When updating your underlying embedding models, deletion and hydration rules remain completely unbroken across generation transitions.

---

## 🚀 Getting Started

### Installation

Set up a local environment and install the package along with development requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To enable support for local vector index execution, pull down the real vector adapters:

```bash
pip install -e ".[adapters]"
```

### Running Tests and Demos

To execute the entire parametrization fixture suite across all backends, run:

```bash
pytest
```

To run an end-to-end operational execution trace covering fact extraction, structured correction, authoritative hard-deletion verification, and background reflection, execute the SDK demo:

```bash
python -m strata.cli demo
```
