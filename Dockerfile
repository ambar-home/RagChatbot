# RAG_LangGraph: PDF/Word/Excel RAG with LangChain + LangGraph + Gradio
#
# Build:  docker build -t rag-langgraph .
# Run:    docker run --rm -p 7860:7860 --env-file .env rag-langgraph
#         (optional: -v $(pwd)/data:/app/data to persist docs and chroma_db)
#         docker run -d --name rag-langgraph --rm -p 7860:7860 --env-file .env rag-langgraph

FROM python:3.11-slim

WORKDIR /app

# Install pip dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (data/docs and data/chroma_db are created at runtime or mount a volume)
COPY app.py gradio_ui.py langgraph_flow.py langchain_rag.py ./
RUN mkdir -p data/docs data/chroma_db

# Gradio: listen on all interfaces so the host can reach the app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

# Load .env if present; start the app
CMD ["python", "app.py"]
