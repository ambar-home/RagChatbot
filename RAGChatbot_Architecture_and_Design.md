# RAGChatbot – Architecture & Design Document

This document explains the architecture, code flow, and key design decisions behind the RAGChatbot system.  
It describes how document ingestion, retrieval, and LLM-based answer generation are orchestrated using LangGraph, LangChain, and a clean layered design suitable for production-grade RAG applications.


## Table of Contents

- [1️⃣ Project Code Flow ](#1-project-code-flow-what-happens-when-the-app-runs)
- [2️⃣ Architecture ](#2-architecture-how-the-system-is-structured)
- [3️⃣ Design Decisions ](#3-design-decisions-why-its-built-this-way)
- [4️⃣ Reasoning ](#4-reasoning-what-problems-this-design-solves)
- [5️⃣ Trade-offs ]((#5-trade-offs))

---


# 1️⃣ Project Code Flow

---

## Step 1: App starts

- `app.py` is the starting point.
- It builds a **retrieval graph** (the logic of how questions are handled).
- It then launches a **Gradio web UI** where users can upload documents and ask questions.


## Step 2: User uploads documents

- Users upload **PDFs or Word files** from the UI.
- Files are saved locally.
- The system **only indexes new or changed files** (incremental indexing).
- Each document is broken into chunks and stored as **vectors in a database**.


## Step 3: User asks a question

The question goes through a **LangGraph pipeline**:

- Validate the question (basic guard checks)
- Retrieve the most relevant document chunks
- Build a clean **context** with citations
- If nothing relevant is found, the system retries once or politely responds with *“not found”*


## Step 4: Answer is generated

- The LLM generates an answer using **only the retrieved content**.
- The answer is **streamed live** to the UI (token by token).
- Citations like `[S1]`, `[S2]` are added at the end.
- The UI also shows **expandable source documents**.



# 2️⃣ Architecture


---

## 🧱 High-level layers

### 1. Entry Layer

- `app.py`
- Only wires things together (**no business logic**)


### 2. UI Layer

- `gradio_ui.py`

Handles:
- Chat UI
- File upload
- Streaming responses
- Source display



### 3. Orchestration Layer

- `langgraph_flow.py`

Responsibilities:
- Defines the **question → retrieval → answer** pipeline
- Uses a **state machine (LangGraph)** instead of tangled if-else logic


### 4. RAG & Infrastructure Layer

- `langchain_rag.py`

Handles:
- Vector database (Chroma)
- Document loading (PDF / Word)
- Chunking & embeddings
- RAG prompts
- Streaming & grounding
- Optional LangSmith tracing


## Why this separation matters

This separation makes the system:
- Easier to reason about
- Easier to test
- Easier to extend



# 3️⃣ Design Decisions 

---

## 🔹 1. LangGraph instead of a linear chain

**Why:**  
RAG flows are not always straight lines.

**Benefits:**
- You can add retries, guards, branching, and future steps (re-ranking, tools, agents)
- Avoids deeply nested logic
- Very interview-friendly and production-ready


## 🔹 2. Incremental indexing (not full re-index every time)

**Why:**  
Re-indexing everything is slow and expensive.

**How:**
- Each file is tracked using a hash (SHA256)
- Only new or changed files are reprocessed

**Benefits:**
- Fast
- Scales well
- Saves compute and API costs


## 🔹 3. Streaming answers

**Why:**  
Users hate waiting for long AI responses.

**Benefits:**
- Immediate feedback
- Feels responsive and modern
- Matches ChatGPT-like UX


## 🔹 4. Optional grounding validation

**Why:**  
LLMs can hallucinate even with RAG.

**Design choice:**
- Controlled by an environment variable
- When enabled:
  - First generate a draft
  - Then validate it strictly against sources
  - Stream only the grounded answer

**Benefits:**
- Safer for enterprise / compliance use cases
- Flexible for dev vs prod


## 🔹 5. Strong separation of concerns

Each file does **one job well**:
- UI ≠ AI logic
- Orchestration ≠ infrastructure
- Retrieval ≠ generation




# 4️⃣ Reasoning 

---

## ✅ Prevents hallucinations

- Answers must come from retrieved sources
- Optional validation layer enforces grounding


## ✅ Scales cleanly

- Incremental ingestion
- Persistent vector database
- Stateless query execution


## ✅ Easy to evolve

You can add later:
- Re-rankers
- Handling True compliance (HIPAA/GDPR/DPDP/PII)
- Need to handle to .ppt,.xls, and other document format.
- Multiple vector stores
- Tool calling
- Multi-agent flows
- Authentication / tenant isolation


## ✅ Production-friendly

- Clear state management
- Deterministic indexing
- Observability via LangSmith
- Config-driven behavior



# 5️⃣ Trade-offs

---

## 1️⃣ Accuracy vs Speed (Grounding validation)

- Turning `RAG_VALIDATE_GROUNDING=true` makes answers **more trustworthy** (validated against sources),
  but it becomes **slower** because the system performs a draft generation followed by a validation step before streaming the final answer.


## 2️⃣ Simplicity vs Flexibility (LangGraph pipeline)

- The guard → retrieve → context pipeline is **clean and easy to understand**,
  but it is intentionally a **basic flow** (single retry, mostly linear). Supporting advanced routing (tools, multi-hop reasoning) would add complexity.


## 3️⃣ Lower cost & faster reindex vs More engineering (Incremental indexing)

- Incremental indexing (SHA256-based) saves **time and embedding costs** by reprocessing only changed files,
  but it requires maintaining additional state (`index_state.json`) and handling document deletions and updates.


## 4️⃣ Easy local persistence vs Enterprise scalability (Chroma local DB)

- A locally persisted Chroma database is **simple to run and ideal for demos or small-scale usage**,
  but enterprise-scale, multi-user, or multi-region deployments typically require a managed vector database and stronger isolation controls.


## 5️⃣ Better user experience vs More UI complexity (Streaming in Gradio)

- Streaming responses feel **fast and ChatGPT-like**, improving perceived responsiveness,
  but they introduce **additional UI and state-management complexity**, such as partial updates and appending citations after streaming completes.
