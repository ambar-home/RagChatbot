# RagChatbot

A RAG (Retrieval-Augmented Generation) application built with **LangChain**, **LangGraph**, and **Gradio**. It supports PDF and Word documents, optional answer grounding/validation, and can be run locally or via Docker.

---

## Prerequisites

- **Python 3.11+** (for local run)
- **Docker** (optional, for containerized run)
- **OpenAI API key** (required for embeddings and chat)

---

## 1. Local setup (after cloning)

### Step 1: Clone and enter the project

```bash
git clone <your-repo-url>
cd RagChatbot
```

### Step 2: Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Step 3: Install Python packages

```bash
pip install -r requirements.txt
```

This installs LangChain, LangGraph, ChromaDB, Gradio, PyMuPDF, docx2txt, and other dependencies listed in `requirements.txt`.

### Step 4: Configure environment variables

Copy the example env file and set your keys:

```bash
cp env.example .env
```

Edit `.env` and set at least:

- `OPENAI_API_KEY` – your OpenAI API key (required)

Optional (for LangSmith tracing and grounding):

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<your-langsmith-key>`
- `LANGCHAIN_PROJECT=RagChatbot`
- `RAG_VALIDATE_GROUNDING=false` (set to `true` to enable answer validation/grounding)

### Step 5: Run the application

```bash
python app.py
```

The Gradio UI will start (default: **http://127.0.0.1:7860**). Open this URL in your browser to:

- Upload PDF/Word documents and (re-)build the index
- Ask questions and get streamed answers with optional citations

---

## 2. Docker approach

You can build and run the app in a container so you don’t need to install Python or dependencies on your host.

### Step 1: Clone and enter the project

```bash
git clone <your-repo-url>
cd RagChatbot
```

### Step 2: Create `.env` (same as local)

Ensure you have a `.env` file with at least `OPENAI_API_KEY` (and any optional vars). You can copy from `env.example` and edit:

```bash
cp env.example .env
# Edit .env and set OPENAI_API_KEY and any optional variables
```

### Step 3: Build the image

From the `RagChatbot` directory:

```bash
docker build -t rag-langgraph .
```

### Step 4: Run the container

**Basic run** (app stops when you press Ctrl+C):

```bash
docker run --rm -p 7860:7860 --env-file .env rag-langgraph
```

**Run in background (detached):**

```bash
docker run -d --name rag-langgraph -p 7860:7860 --env-file .env rag-langgraph
```

**Run with persistent data** (recommended if you upload documents and want to keep the index across restarts):

```bash
docker run --rm -p 7860:7860 --env-file .env \
  -v "$(pwd)/data:/app/data" \
  rag-langgraph
```

- `data/docs` – uploaded PDF/Word files
- `data/chroma_db` – Chroma vector store

Without the volume, documents and the index are lost when the container is removed.

**Stop a detached container:**

```bash
docker stop rag-langgraph
```

### Step 5: Use the app

Open **http://localhost:7860** in your browser (or http://127.0.0.1:7860). If the container runs on another host, use that host’s IP and port 7860.

---

## Summary

| Method   | Install                          | Run                                      |
|----------|----------------------------------|------------------------------------------|
| **Local** | `pip install -r requirements.txt` | `python app.py` → http://127.0.0.1:7860  |
| **Docker** | `docker build -t rag-langgraph .` | `docker run --rm -p 7860:7860 --env-file .env rag-langgraph` → http://localhost:7860 |

For long-term use with Docker, add `-v "$(pwd)/data:/app/data"` so documents and the vector DB persist.
