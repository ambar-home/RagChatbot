from gradio_ui import create_demo
from langgraph_flow import build_retrieval_graph

def main() -> None:
    """
    Inputs:
    - None

    Outputs:
    - None

    What it does:
    - Builds the LangGraph retrieval workflow and wires it into the Gradio UI,
      then launches the web app.
    """
    graph = build_retrieval_graph()
    demo = create_demo(graph)
    demo.launch()


if __name__ == "__main__":
    main()

# -----------------------------
# (Legacy in-file implementation removed)
# The app is now split into:
# - langchain_rag.py (LangChain/RAG utilities + indexing + OpenAI streaming)
# - langgraph_flow.py (LangGraph retrieval pipeline)
# - gradio_ui.py (Gradio UI)
# - app.py (entrypoint)
