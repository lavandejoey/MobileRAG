# MobileRAG

MobileRAG is a self-contained Retrieval-Augmented Generation (RAG) system that pairs a FastAPI backend with a
lightweight browser UI and CLI for chat-based workflows. RAG data is indexed on disk, model responses stream over
WebSockets, and chat history is kept in a local SQLite store for fast replay and inspection.

## Architecture

- **API Server:** `src/api/server.py` exposes REST endpoints for listing chats and a WebSocket endpoint (`/v1/chat/ws`)
  that streams tokens, thinking hints, and metadata while delegating retrieval and generation to the registered
  components.
- **RAG Pipeline:** `src/rag/pipeline.py` scans the document globs defined in `configs/mobile_rag.yaml`, chunks and
  embeds every document, stores metadata in `data/rag/rag_meta.db`, and keeps vectors in `data/rag/chunks.index.faiss`.
  Retrieval reranks candidates before feeding them into the prompt slot that `build_llm_messages()` prepares for the
  LLM.
- **Persistence:** `src/storage/history_db.py` keeps every chat and assistant turn in `data/history/history.db` so the
  UI and CLI can replay multi-turn conversations with the original timestamps, thinking traces, and auxiliary metadata
  that `src/storage/persist.py` appends.
- **Clients:** The browser UI (`src/api/static/*`) connects to the WebSocket endpoint, renders Markdown + KaTeX, exposes
  a thinking drawer, and mirrors the chat list stored in the database. The CLI (`src/chat/cli.py`) is a thin WebSocket
  client that can list chats, load archives, delete sessions, and stream both thinking tokens and answers.

## Getting Started

1. **Install requirements** (any Python 3.11+ virtualenv is fine):
   ```bash
   pip install -r requirements.txt
   ```
2. **Adjust the configuration** in `configs/mobile_rag.yaml` to point at your document globs, desired model backend, and
   logging level. The defaults target a local Ollama replica with simple hashing embeddings.
3. **(Optional) Populate the RAG index** by running a quick Python script or REPL that calls
   `RagPipeline(load_config()).build_or_update_index()` so that chunks exist before the first chat. Without it, the
   pipeline will simply build on demand at first query.

## Running the system

- **Start the backend:**

  ```bash
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
  ```

  The server serves the chat UI at `/` and mounts `/static` for the supporting assets.

- **Use the browser UI:**
  Open `http://localhost:8000/` to open the built-in chat interface. Messages are sent via WebSocket to `/v1/chat/ws`,
  tokens stream back as `think_token` / `answer_token` events, and the thinking drawer captures the hidden reasoning
  trace.

- **CLI access:**
  ```bash
  python -m src.chat.cli --server http://localhost:8000
  ```
  The CLI keeps a live WebSocket session, prints assistant thinking durations, and lets you `/list`, `/load <chat_id>`,
  or `/del <chat_id>` without leaving the terminal.

## Configuration reference

| Section                    | Purpose                                                                                                                                                         |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MODEL`                    | Controls the oracle (Ollama by default), streaming behavior, temperature, and maximum output tokens for `create_chat_model()`.                                  |
| `RAG`                      | Enables/disables retrieval, tuning chunk size/overlap, embedding backend (hashing or Ollama), reranker order, and output budget for the prompt injection guard. |
| `DOCS_GLOBS` / `DOCS_EXTS` | Define where `src/rag/fs_scan.py` looks for sources when rebuilding the Faiss index.                                                                            |
| `HISTORY`                  | Path for chat persistence; the API expects this directory to exist and writes `history.db` automatically.                                                       |

## Data layout

- `data/raw/`: place source documents (text, Markdown, PDF, etc.) for chunking.
- `data/rag/`: vector index files (`chunks.index.faiss`) and metadata (`rag_meta.db`).
- `data/history/`: chat history SQLite database with assistant/user turns, reasoning traces, and meta payloads.

## Development notes

- The FastAPI app initializes `HistoryDB`, `RagPipeline`, and the model loader once at import time, so changes to
  `configs/mobile_rag.yaml` require restarting the server.
- The WebSocket handler in `src/api/server.py` buffers LLM output through `split_think_stream()` so thinking tokens can
  be surfaced independently of final answers.
- Extend `src/rag/embedder.py` to add new embedding backends, or swap in a different reranker via `src/rag/rerank.py`.

## TODO

- Improve ingestion tooling so that `RagPipeline` can be driven from a CLI command instead of relying on first-query
  laziness.
- Add automated tests for `src/rag/pipeline.py` and the persistence layer to guard regressions during refactors.
- Implement file uploads/drag-and-drop in the browser UI and emit metadata to the server when new assets are attached.
- Document deployment steps (containerized, cloud, or mobile build) and expose a health-check endpoint for readiness
  probes.
