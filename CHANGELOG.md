# Changelog

## 0.12.0 - 2025-08-15

### Added

- Implemented FastAPI backend with `/chat` (streaming), `/ingest` (placeholder), `/evidence/{turn}` (placeholder), and `/status` endpoints.
- Developed a minimal React web UI with chat pane, evidence panel, and memory drawer.
- Added API backend status indicator to the UI.
- Implemented animated thinking indicator for pending responses in the UI.
- Improved chat box width flexibility and line spacing in the UI.
- Disabled send button for empty messages in the UI.

### Changed

- Refactored `smoke_cli.py` for improved SSE parsing and reduced complexity.
- Configured React development server proxy for seamless API integration.

## 0.11.0 - 2025-08-13

### Added

- Completed Phase 0: Project skeleton & CI.
- Verified Phase 1: Config & device resolver. Fixed warnings.
- Completed Phase 2: Qdrant client and collections.
- Completed Phase 3: Ingestion (scan & chunk).
- Completed Phase 4: Embeddings & captions.
