# Phase 0 — Project skeleton & CI

* [x] Create repo structure (folders from the integration plan)
  **DoD:** Tree matches plan; `README.md` exists.
  **Verify:** `tree -L 2` shows `apps/ core/ scripts/ tests/ docs/`.

* [x] Add tooling: `ruff`, `black`, `mypy/pyright`, `pytest`, `pre-commit`
  **DoD:** `pre-commit run --all-files` passes.
  **Verify:** CI job runs lint/type/test on push.

* [x] Create base configs (`.env.example`, `pyproject.toml` or `requirements.txt`)
  **DoD:** `pip install -r requirements.txt` (or poetry/uv) works.
  **Verify:** `pytest -q` runs 0 tests successfully.

* [x] Docs seed: `docs/01_Design_Spec.md` + `docs/02_Integration_Plan.md`
  **DoD:** Both in repo; linked from `README`.

---

# Phase 1 — Config & device resolver (`core/config`)

* [x] Implement `Settings` (Pydantic) and `device_resolver()`
  **DoD:** Returns `{"llm":cuda/cpu,"reranker":cuda/cpu,"embed":cuda/cpu}` deterministically.
  **Verify:** Unit tests simulate VRAM 6 GB / 0 GB and assert routing.

* [x] Add token counter util (tiktoken or transformers length)
  **DoD:** `count_tokens(text)` stable across runs.
  **Verify:** Unit test with fixed strings.

---

# Phase 2 — Qdrant client & schema (`core/vecdb`)

* [x] Implement local Qdrant client wrapper (`VecDB`)
  **DoD:** Can open/create DB at `./qdrant_db`.

* [x] Create collections

    * `rag_multimodal`: named vectors `text_dense(1024)`, `image(512)`, `text_sparse(sparse=True)`
    * `agent_memory`: `text_dense(1024)` + `text_sparse`
      **DoD:** Idempotent creation; payload indices on `lang`, `modality`, `time`.
      **Verify:** Contract test: upsert→get→filter→delete round-trip passes.

---

# Phase 3 — Ingestion: scan & chunk (`core/ingest`)

* [x] Implement `scan()` (sha256, mtime; images add pHash)
  **DoD:** Produces a list of `IngestItem` with stable hashes.
  **Verify:** Unit test rescans and diffs = Ø.

* [x] Implement `chunk()` for text/PDF (Unstructured + PyMuPDF)
  **DoD:** Each chunk has `file://...#page`, optional `bbox`, `lang`.
  **Verify:** Integration test on `data/samples/` yields non-empty chunks.

---

# Phase 4 — Embeddings & captions (dense/sparse/image)

* [ ] Text dense embeddings (Qwen3-Embedding-0.6B)
  **DoD:** Batched inference; outputs 1024-dim arrays; cached.
  **Verify:** Deterministic vectors (tolerance); throughput logged.

* [ ] Text sparse vectors (BM25 + SPLADE++)
  **DoD:** Serialised to Qdrant sparse format.
  **Verify:** Small corpus tf-idf sanity test passes.

* [ ] Image embeddings (OpenCLIP ViT-B/32)
  **DoD:** 512-dim arrays; cached.
  **Verify:** Same image → same vector.

* [ ] Image captions (BLIP base, EN)
  **DoD:** Non-empty captions; stored in payload.
  **Verify:** 3 sample images return sensible short captions.

---

# Phase 5 — Upsert & idempotency

* [ ] Build `upsert()` with key `doc_id#chunk_id#version` + soft delete
  **DoD:** Re-ingest produces no duplicates; versioned payload updates.
  **Verify:** Contract test comparing point counts and versions.

* [ ] CLI: `ingest_cli.py scan|embed|upsert`
  **DoD:** Commands run end-to-end on samples.
  **Verify:** Prints summary (#docs, #points), exits 0.

---

# Phase 6 — Hybrid retriever (`core/retriever`)

* [ ] Implement `HybridRetriever.search()`
  **DoD:** Combines dense+sparse via Qdrant Hybrid API; supports filters.
  **Verify:** Returns `Candidate` items with `evidence(file_path, page, caption)`.

* [ ] Image-aware queries (text→image, image→text)
  **DoD:** If query has image, use image vector; else use text + caption.
  **Verify:** Query “find the flow diagram” returns the expected figure.

* [ ] Quick metrics harness (Recall\@60, nDCG\@10)
  **DoD:** Baseline metrics computed on tiny labelled set.
  **Verify:** JSON report saved under `tests/artifacts/`.

---

# Phase 7 — Reranker (`core/reranker`)

* [ ] Qwen3-Reranker-0.6B (seq-cls) batched scoring
  **DoD:** Latency within budget; returns TopR 8–12.
  **Verify:** nDCG\@10 improves ≥ target over “no rerank”.

---

# Phase 8 — Dialogue history (SQLite) (`core/history`)

* [ ] Tables `chat_messages`, `chat_summaries`; append/load API
  **DoD:** Persist messages; reload by `session_id`.
  **Verify:** Unit tests pass; token counts stored.

* [ ] Rolling compactor (summary ≤600 tokens; trigger ≥1800 tokens or 6–8 turns)
  **DoD:** Produces markdown bullets (facts/decisions/TODO).
  **Verify:** Long chat stays under history budget.

---

# Phase 9 — Long-term memory (`core/memory`)

* [ ] `agent_memory` CRUD + Hybrid retrieval (Top-5 ≤250 tokens)
  **DoD:** Retrieve relevant mem-cards for a query.
  **Verify:** Integration test: prefs/facts appear when relevant.

* [ ] `memory_gate` (rule-first + light LLM) + dedup/decay
  **DoD:** Only stable, non-dated facts are stored; merges similars.
  **Verify:** Unit tests on positive/negative examples.

---

# Phase 10 — Token budget & generator (`core/generator`)

* [ ] Budget orchestrator (summary→recent→memory→evidence; multistage compression)
  **DoD:** Never exceeds model window; only final resort trims oldest recents.
  **Verify:** Stress test with oversized packs stays under budget.

* [ ] Qwen3-1.7B wrapper (bnb 4/8-bit; `device_map="auto"`)
  **DoD:** Streaming & non-streaming; graceful OOM fallback.
  **Verify:** CPU-only run works; GPU 6 GB run faster.

* [ ] `answer_with_citations` formatter (mandatory Evidence list)
  **DoD:** Every answer emits file/ page / caption links.
  **Verify:** Contract test asserts evidence present.

---

# Phase 11 — LangGraph topology (`core/graph`)

* [ ] Wire nodes:
  `device_resolver → query_normaliser → {history, memory, retriever} → rerank → budget → generator → answer`
  **DoD:** Graph runs with mocks, then real deps.
  **Verify:** Scenario tests: zh/en; text/image; cross-lingual.

---

# Phase 12 — API & minimal UI (`apps/chat_api`, `apps/chat_ui`)

* [ ] FastAPI endpoints: `/chat` (stream), `/ingest`, `/evidence/{turn}`
  **DoD:** Returns JSON/Server-Sent Events with tokens & evidence.
  **Verify:** `smoke_cli.py chat` succeeds.

* [ ] Minimal web UI: chat pane + Evidence panel + “🧠 Memory” drawer
  **DoD:** Can toggle memory items (delete/edit).
  **Verify:** Manual E2E in browser.

---

# Phase 13 — Eval & smoke CLIs (`scripts/`)

* [ ] `eval_cli.py retrieval|rerank|end2end`
  **DoD:** Saves metrics; compares against thresholds.
  **Verify:** CI artifact uploaded.

* [ ] `smoke_cli.py chat --q ... --session ...`
  **DoD:** Returns answer with evidence links; exit 0.

---

# Phase 14 — Packaging, ops & docs

* [ ] Config templates (`config.yaml`, `.env.example`) with defaults
  **DoD:** One-line boot instructions in `README`.
  **Verify:** New machine runs skeleton in ≤10 minutes.

* [ ] Docs: `docs/api.md`, `docs/eval.md`, `docs/ops.md`
  **DoD:** Each doc has examples and troubleshooting.

* [ ] Release tag `v1.0.0-alpha` + E2E checklist
  **DoD:** All boxes in release checklist ticked; CI green.

---

## Parallelisable items (optional fast-track)

* CLIs & metrics harness (Phases 6/13) can start early with stubbed retriever.
* UI skeleton can start once API stub is up (end of Phase 0/1).
* Memory gate rules can be developed in isolation with canned inputs.

---

## Per-task PR checklist (copy into templates)

* [ ] Code + tests + docs for this task only
* [ ] `ruff/black/mypy/pytest` pass locally
* [ ] Adds/updates contracts & fixtures
* [ ] Updates `CHANGELOG.md` and `docs/02_Integration_Plan.md` status
* [ ] CI green; artefacts (metrics/logs) uploaded
