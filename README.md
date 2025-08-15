# MobileRAG

MobileRAG is an advanced Retrieval-Augmented Generation (RAG) system designed for multimodal information processing. It leverages LangGraph to orchestrate complex workflows, integrating various components for efficient retrieval, generation, and dialogue management. The system supports multimodal RAG, allowing it to process and retrieve information from diverse sources, including text and images.

## Key Features

*   **Multimodal RAG:** Combines information from text and image sources to provide comprehensive answers.
*   **LangGraph Orchestration:** Utilizes LangGraph for flexible and scalable workflow management.
*   **FastAPI Backend:** Provides a robust and high-performance API for chat, ingestion, and evidence retrieval.
*   **React Frontend:** A minimal yet functional web interface for interactive chat.

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

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Node.js Dependencies (for UI)

```bash
cd apps/chat_ui
npm install
cd ../..
```

## How to Run

### 1. Start the Backend API

Navigate to the project root directory and run the FastAPI application:

```bash
uvicorn apps.chat_api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Frontend UI

In a new terminal, navigate to the UI directory and start the React development server:

```bash
cd apps/chat_ui
npm start
```

Once both are running, open your browser and navigate to `http://localhost:3000` to access the chat interface.

## Documentation

- [Design Specification](./docs/01_Design_Spec.md)
- [Integration Plan](./docs/02_Integration_Plan.md)
