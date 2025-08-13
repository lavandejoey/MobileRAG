# 0) Repo & Conventions

**Repo layout**

```text
MobileRAG/
├── LICENSE
├── README.md
├── configs/
│  └── default.yaml
├── apps/
│  ├── chat_api/
│  │  ├── main.py
│  │  ├── routes/
│  │  │  ├── chat.py
│  │  │  ├── ingest.py
│  │  │  └── evidence.py
│  │  ├── deps.py
│  │  ├── schemas.py
│  │  └── __init__.py
│  └── chat_ui/
│     ├── public/
│     │  ├── favicon.ico
│     │  └── style.css
│     ├── src/
│     │  ├── App.jsx
│     │  └── index.jsx
│     ├── index.html
│     └── package.json
├── core/
│  ├── config/
│  │  ├── settings.py
│  │  ├── devices.py
│  │  └── __init__.py
│  ├── types.py
│  ├── graph/
│  │  ├── topology.py
│  │  ├── nodes/
│  │  │  ├── device_resolver.py
│  │  │  ├── query_normaliser.py
│  │  │  ├── history_loader.py
│  │  │  ├── history_compactor.py
│  │  │  ├── memory_retriever.py
│  │  │  ├── memory_gate.py
│  │  │  ├── retriever_hybrid.py
│  │  │  ├── reranker.py
│  │  │  ├── budget_orchestrator.py
│  │  │  └── generator.py
│  │  └── __init__.py
│  ├── retriever/
│  │  ├── hybrid.py
│  │  ├── types.py
│  │  └── __init__.py
│  ├── generator/
│  │  ├── llm.py
│  │  ├── budget.py
│  │  └── __init__.py
│  ├── memory/
│  │  ├── store.py
│  │  ├── gate.py
│  │  ├── types.py
│  │  └── __init__.py
│  ├── history/
│  │  ├── store.py
│  │  ├── compactor.py
│  │  └── __init__.py
│  ├── ingest/
│  │  ├── scan.py
│  │  ├── chunk.py
│  │  ├── embed_dense.py
│  │  ├── embed_sparse.py
│  │  ├── embed_image.py
│  │  ├── caption.py
│  │  ├── upsert.py
│  │  ├── pipeline.py
│  │  └── __init__.py
│  ├── vecdb/
│  │  ├── client.py
│  │  ├── schema.py
│  │  └── __init__.py
│  ├── sparse/
│  │  ├── fastembed.py
│  │  └── __init__.py
│  ├── clip/
│  │  ├── openclip.py
│  │  ├── blip.py
│  │  └── __init__.py
│  └── utils/
│     ├── hashing.py
│     ├── tokens.py
│     ├── langid.py
│     ├── log.py
│     ├── ids.py
│     ├── timers.py
│     └── __init__.py
├── data/
├── docs/
│  ├── 00_Todo_Impl.md
│  ├── 01_Design_Spec.md
│  └── 02_Integration_Plan.md
├── requirements.txt
├── scripts/
│  ├── ingest_cli.py
│  ├── eval_cli.py
│  └── smoke_cli.py
└── tests/
   ├── unit/
   ├── integration/
   └── contract/
```

**Engineering rules**

* Type-check (mypy/pyright), lint (ruff), format (black), tests (pytest).
* All modules expose **pure Python interfaces** with **Pydantic DTOs**; outer layers (FastAPI/LangGraph) adapt to them.
* Every task has a **Definition of Done (DoD)** + **test(s)**.

---

# 1) Modules → Tasks (low coupling, testable)

Below each module lists: **Interface → Tasks → DoD → Tests**.

## 1.1 `core/config` & Device Resolver

**Interface**

```python
class Settings(BaseSettings):
    device: Literal["auto", "cpu", "cuda:0"] = "auto"
    qdrant_path: str = "./qdrant_db"
    collection_main: str = "rag_multimodal"
    collection_mem: str = "agent_memory"
    dense_dim_text: int = 1024
    dense_dim_image: int = 512
    ...


def choose_devices(mem_required: dict) -> Dict[str, torch.device]
```

**Tasks**

1. Pydantic settings (env/.env)
2. `device_resolver` (GPU priority to reranker+LLM; fallbacks)
   **DoD**: returns stable mapping in CPU-only/GPU-6GB/oom-probe.
   **Tests**: simulate VRAM budgets; assert routing decisions.

---

## 1.2 `vecdb` (Qdrant schema & client)

**Interface**

```python
class Point(BaseModel):
    id: str;
    vectors: Dict[str, Any];
    payload: Dict[str, Any]


class VecDB:
    def create_collections(): ...

    def upsert(points: List[Point]): ...

    def hybrid_query(q_dense, q_sparse, filters, topk): ...
```

**Tasks**

1. Create `rag_multimodal` with named vectors: `text_dense(1024)`, `image(512)`, `text_sparse(sparse=True)`
2. Create `agent_memory` with `text_dense` + `text_sparse`
3. Payload filter indices (lang/modality/time)
   **DoD**: idempotent creation; upsert/read OK.
   **Tests**: contract tests spin up **local Qdrant** and roundtrip points.

---

## 1.3 `ingest` (Scan → Chunk → Embed → Caption → Upsert)

**Interface**

```python
@dataclass
class IngestItem: path: str; doc_id: str; modality: Literal["text", "image"]; meta: dict


class Ingestor:
    def scan(root) -> List[IngestItem]

        def chunk(items) -> List[Chunk]  # text/pdf with page/bbox

        def embed_dense(chunks) -> Embeds

        def embed_sparse(chunks) -> SparseEmbeds

        def caption_images(items) -> List[Caption]

        def upsert(all_vectors, payloads) -> None
```

**Tasks**

1. `scan`: sha256/mtime + pHash(images)
2. `chunk`: Unstructured + PyMuPDF; record `file://...#page=` & `bbox`
3. Dense: Qwen3-Embedding-0.6B (batch)
4. Sparse: FastEmbed BM25/SPLADE++
5. Image: OpenCLIP embeddings + BLIP caption (en)
6. Upsert with idempotent key `doc_id#chunk_id#version`
   **DoD**: given `data/samples/` produces Qdrant points with correct payloads.
   **Tests**: unit (hashing, chunking), integration (end-to-end mini corpus).

---

## 1.4 `sparse` (BM25/SPLADE++)

**Interface**

```python
def build_sparse_vectors(texts: List[str]) -> List[SparseVector]
```

**Tasks**: wrap FastEmbed; serialise to Qdrant sparse format.
**DoD**: correctness vs known tf-idf examples; determinism with seed.
**Tests**: unit.

---

## 1.5 `clip` (OpenCLIP + BLIP)

**Interface**

```python
def embed_image(paths: List[str]) -> np.ndarray


    def caption_image(paths: List[str]) -> List[str]
```

**Tasks**: load models CPU/GPU; batch inference; cache results to sqlite/lmdb.
**DoD**: same image → stable vector & non-empty caption.
**Tests**: unit with 2–3 images; checksum outputs.

---

## 1.6 `retriever` (Hybrid)

**Interface**

```python
class HybridQuery(BaseModel):
    text: Optional[str];
    image_path: Optional[str]
    filters: Optional[dict];
    topk_dense: int = 60;
    topk_sparse: int = 60


class HybridRetriever:
    def search(q: HybridQuery) -> List[Candidate]  # merges dense+sparse
```

**Tasks**:

1. text→text/image; image→text via CLIP; fuse with Qdrant Hybrid API
2. payload filtering (lang/modality)
   **DoD**: returns k candidates with scores & payload (path/page/bbox/caption).
   **Tests**: contract tests on sample corpus; asserts page anchors exist.

---

## 1.7 `reranker`

**Interface**

```python
class Reranker:
    def rank(query: str, cands: List[Candidate], topr: int = 10) -> List[Candidate]
```

**Tasks**: Qwen3-Reranker-0.6B (seq-cls) batched; CPU/GPU selectable.
**DoD**: improves nDCG\@10 on test mini-set vs no-rerank baseline.
**Tests**: metric test with tiny labelled pairs.

---

## 1.8 `history` (SQLite chat logs + compactor)

**Interface**

```python
def append_message(session_id, role, content, meta) -> MsgID


    def load_recent(session_id, n: int) -> List[Msg]


    def get_or_make_summary(session_id, upto_msg_id) -> Summary


    def maybe_compact(session_id) -> Optional[Summary]
```

**Tasks**: tables `chat_messages`, `chat_summaries`; token count; triggers (≥1800 tokens or every 6–8 turns).
**DoD**: compactor produces markdown summary ≤600 tokens.
**Tests**: unit (roll-forward summaries), integration (budget fit).

---

## 1.9 `memory` (long-term memory + gate)

**Interface**

```python
def mem_retrieve(query: str, tags: List[str]) -> List[MemCard]  # Hybrid Top-5


    def mem_gate(candidates: List[str]) -> List[MemCard]  # rule+light LLM


    def mem_upsert(cards: List[MemCard]) -> None
```

**Tasks**: `agent_memory` schema; rule-first gate; decay & merge duplicates.
**DoD**: only “long-term stable” facts get stored; retrieve ≤250 tokens total.
**Tests**: unit on rule patterns; integration with retrieval.

---

## 1.10 `generator` (LLM + budget orchestrator)

**Interface**

```python
class Pack(BaseModel): text: str; tokens: int


def build_context(budget: int, packs: Dict[str, Pack]) -> str


    def generate(prompt, ctx, stop=None) -> str
```

**Tasks**: implement **token budgeter** (summary→recent→mem→evidence; multi-stage compression before truncation);
Qwen3-1.7B wrapper (bnb 4/8-bit).
**DoD**: respects budget; never exceeds window; gracefully degrades.
**Tests**: budget stress tests; OOM guard.

---

## 1.11 `graph` (LangGraph topology)

**Nodes**: `device_resolver` → `query_normaliser` → (`history_loader/compactor`, `memory_retriever/gate`,
`retriever_hybrid`) → `rerank` → `budget_orchestrator` → `generator` → `answer_with_citations`.
**DoD**: end-to-end path runs with mocks first, then real deps.
**Tests**: scenario tests (zh/en, text/image queries).

---

## 1.12 `apps/chat_api` & `chat_ui`

**API**

* `POST /chat` JSON: `{session_id, message, options?}` → streaming tokens
* `POST /ingest` JSON: `{paths:[]}`
* `GET /evidence/:turn_id` → evidence list (for UI)

**UI**

* Minimal chat pane + “🧠 Memory” drawer + “Evidence” panel.

**DoD**: can ingest sample corpus; ask bilingual Q; returns answer + evidence links (`file://…#page=` or image path).
**Tests**: smoke/e2e with Playwright (optional) or API script.

---

## 1.13 `scripts` (CLIs)

* `ingest_cli.py scan|embed|upsert --root ./data/samples`
* `eval_cli.py retrieval|rerank|end2end`
* `smoke_cli.py chat --q "中文问题" --session s1`

**DoD**: each command exits 0 and prints brief metrics.
**Tests**: invoked by CI in ephemeral workspace.

---

# 2) Phased Integration Plan (incremental, safe)

**Phase A — Skeleton & Contracts**

1. Create repo, config, device resolver (mocks for models).
2. `vecdb.create_collections()` with contract tests.
3. Wire `apps/chat_api` with **stub retriever/generator** that echo inputs.
   **Gate**: CI green; API health check OK.

**Phase B — Ingestion Pipeline**
4\. Implement `ingest.scan` + `chunk` (text/pdf) → payload with anchors.
5\. Add **dense text embeddings** (Qwen3-Embed) and **sparse** (BM25).
6\. Upsert points; verify counts & payload.
**Gate**: `ingest_cli.py` produces non-empty points; contract tests pass.

**Phase C — Multimodal**
7\. OpenCLIP image embeddings; BLIP captions; upsert image points.
**Gate**: text→image & image→text basic retrieval works.

**Phase D — Hybrid Retrieval**
8\. Implement `retriever_hybrid` using Qdrant Hybrid queries; filters.
**Gate**: Recall\@60 and nDCG\@10 on sample reach baseline; deterministic.

**Phase E — Reranker**
9\. Qwen3-Reranker (seq-cls) batched; integrate after retriever.
**Gate**: nDCG\@10 ↑ over Phase D ≥ +10% on test set (adjustable).

**Phase F — Generator & Citations**
10\. Token budgeter; Qwen3-1.7B wrapper; `answer_with_citations`.
**Gate**: never exceeds window; answers list evidence links for every turn.

**Phase G — Dialogue History**
11\. SQLite logs; rolling summaries; near-turn selector.
**Gate**: long chats keep latency stable; context size under budget.

**Phase H — Long-term Memory**
12\. `agent_memory` + `mem_gate` + decay; integrate into context.
**Gate**: only stable prefs/facts persist; ≤250 tokens injected.

**Phase I — UI Polish & Ops**
13\. Evidence panel; Memory drawer (view/edit/delete); logs export.
14\. `eval_cli.py` end-to-end metrics; `smoke_cli.py` bilingual runs.
**Gate**: E2E checklist passes (see §4); ready for packaging.

---

# 3) Interfaces (DTOs) to reduce coupling

**Candidates**

```python
class Evidence(BaseModel):
    file_path: str
    page: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    caption: Optional[str] = None
    title: Optional[str] = None


class Candidate(BaseModel):
    id: str;
    score: float;
    text: Optional[str];
    evidence: Evidence;
    lang: str;
    modality: str
```

**Packs for budgeter**

```python
class Packs(BaseModel):
    summary: str
    recent: str
    memory: str
    evidence: str
```

> All modules depend on DTOs, not on each other’s internals.

---

# 4) Test & Validation Matrix

| Layer          | Test type        | What we assert                                  | Pass criteria          |
|----------------|------------------|-------------------------------------------------|------------------------|
| Config/Devices | unit             | device routing, OOM fallback                    | stable mapping         |
| VecDB          | contract         | create/upsert/query idempotent                  | roundtrip OK           |
| Ingest         | integration      | chunks have anchors; hashes stable              | ≥99% deterministic     |
| Sparse         | unit             | tf-idf vs toy corpus                            | tolerance ≤1e-6        |
| CLIP/BLIP      | unit             | non-empty captions; same vectors for same image | ok                     |
| Retriever      | integration      | Recall\@60, nDCG\@10                            | ≥ baseline             |
| Rerank         | metric           | nDCG\@10 uplift                                 | +10% vs no-rerank      |
| History        | unit             | compactor sizes; triggers                       | ≤600 tokens summary    |
| Memory         | unit/integration | gate rules; dedup/decay                         | only stable cards kept |
| Budgeter       | unit             | never overflow; graceful compression            | ctx ≤ budget           |
| Generator      | e2e              | answer + **Evidence links**                     | link for every answer  |
| API/UI         | smoke            | bilingual; image/text; filters                  | all green              |

**Sample data**: `data/samples/` (2 zh + 2 en PDFs; 3 images with obvious content; 1 bilingual md).
**Metrics**: Recall\@k, nDCG\@k, Faithfulness via citation presence, P95 latency (non-blocking target).

---

# 5) CI Pipeline (per PR)

1. Lint/format/type-check
2. Unit tests (fast)
3. Start ephemeral local Qdrant (or use local-mode path)
4. Ingest **tiny** sample set
5. Contract/integration tests
6. Smoke chat (non-GPU mode)
7. Artefacts: coverage, sample logs

---

# 6) Runtime Policies & Fallbacks

* **GPU policy**: allocate to reranker+LLM first; embeddings batch on CPU unless idle GPU available.
* **Memory pressure**: auto lower batch/precision, then spill to CPU.
* **Token overflow**: multi-stage compression; **only then** trim oldest recent-turns (summary retained).
* **Data integrity**: ingestion is **idempotent**; soft-delete old versions; rollback by `version`.

---

# 7) Release Checklist (E2E)

* [ ] Ingest sample corpus → points present in both collections
* [ ] zh/en Q\&A return answers **with evidence links**
* [ ] text→image and image→text queries work
* [ ] Long chat (>40 turns) stays under budget (no overflow)
* [ ] Reranker improves nDCG\@10 vs baseline
* [ ] Memory drawer shows recent inserts; delete works; no short-term noise stored
* [ ] All CLIs run and exit 0 (`ingest_cli.py`, `smoke_cli.py`, `eval_cli.py`)
* [ ] CI green

---

# 8) Risk Register & Mitigations (concise)

* **VRAM 6GB tight** → bnb 8-bit, smaller batch; move embeddings to CPU.
* **OCR/scan PDFs** → switch Unstructured to `ocr_only`; keep page anchors.
* **Caption noise** → keep captions short; also index image embeddings directly.
* **Hybrid weighting** → start λ=0.5; tune on validation queries.
* **Memory creep** → decay & merge; Top-5 cap; UI moderation.

---

# 9) Step-by-step Developer On-ramp

1. `poetry install` (or `uv pip install -r requirements.txt`)
2. `python scripts/ingest_cli.py scan --root data/samples`
3. `python scripts/ingest_cli.py embed --root data/samples`
4. `python scripts/ingest_cli.py upsert`
5. `uvicorn apps.chat_api.main:app --reload`
6. `python scripts/smoke_cli.py chat --q "给我这份PDF的要点？" --session s1`
7. Check UI Evidence & Memory drawers.

---

# 10) Documentation Map

* `docs/01_Design_Spec.md`
* `docs/02_Integration_Plan.md`
* `docs/api.md`（HTTP contract & examples）
* `docs/eval.md`（metrics与调参流程）
* `docs/ops.md`（模型/数据升级、回滚、备份）
