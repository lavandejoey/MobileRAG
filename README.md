# MobileRAG

Local RAG chat over your own files.

Current stack:
- FastAPI + WebSocket chat server
- Local SQLite chat history
- Local vector index + SQLite RAG metadata
- Browser UI with route-based chats: `/` and `/<chat_id>`
- Ollama chat backend

Supported file types:
- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.html` / `.htm`
- `.csv`

## Run

Install:
```bash
pip install -r requirements.txt
```

Build or refresh the index:
```bash
python -m src.main build-index --config configs/mobile_rag.yaml
```

Start the server:
```bash
python -m src.main serve --host 127.0.0.1 --port 8000
```

Open:
```text
http://127.0.0.1:8000/
```

## Routes

- `/`: empty home view
- `/<chat_id>`: load one chat
- `GET /healthz`
- `POST /v1/index/build`
- `GET /v1/chats`
- `GET /v1/chats/{chat_id}/messages`
- `DELETE /v1/chats/{chat_id}`
- `WS /v1/chat/ws`

## Current behavior

- New chat title is generated cheaply from the first user message. No model call.
- Chat history recall for questions like “前面问你了什么” is handled server-side.
- RAG is warmed on startup.
- Retrieval no longer rebuilds the full index on every message.
- Answers append short file references when RAG snippets were used.
- Deleted source files are removed from the next index rebuild.

## Critical limits

- PDF parsing is still text-extraction only. Scanned PDFs, tables, and complex layouts are weak.
- No upload flow yet.
- No auth, no multi-user isolation, no production hardening.
- No automated tests yet.
- Current environment may still show a `requests` dependency warning if the Python env is already dirty.

## Main files

- `src/api/server.py`: app, websocket flow, SPA fallback
- `src/rag/pipeline.py`: indexing, retrieval, warmup, incremental cleanup
- `src/rag/parsers.py`: file readers
- `src/storage/history_db.py`: chat persistence and title generation
- `src/api/static/*`: browser UI
