# MobileRAG

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/lavandejoey/MobileRAG.git
cd MobileRAG
```

### 2. Install tools

```bash
sudo apt-get update && sudo apt-get install -y
sudo apt-get install build-essential python3-dev git \
    tesseract-ocr \
    libmagic1
```

## Documentation

- [Design Specification](./docs/01_Design_Spec.md)
- [Integration Plan](./docs/02_Integration_Plan.md)

## Dev Log

- **2025-08-13:** Completed Phase 0: Project skeleton & CI.
- **2025-08-13:** Verified Phase 1: Config & device resolver. Fixed warnings.
- **2025-08-13:** Completed Phase 2: Qdrant client and collections.
- **2025-08-13:** Completed Phase 3: Ingestion (scan & chunk).
- **2025-08-13:** Completed Phase 4: Embeddings & captions.
