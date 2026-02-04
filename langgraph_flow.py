from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from langchain_rag import get_vectordb


class RAGState(TypedDict, total=False):
    question: str
    retrieved: List[Dict[str, Any]]
    context: str
    citations: List[str]
    why_not_found: Optional[str]
    retried: bool  # True after we've taken the context -> guard branch once (avoids infinite loop)


def node_guard(state: RAGState) -> RAGState:
    """
    Inputs:
    - state: RAGState containing at least `question`.

    Outputs:
    - Updated RAGState with trimmed question and/or `why_not_found` set.

    What it does:
    - Validates/normalizes the question so downstream nodes can assume a non-empty query.
    - If we re-entered from context (why_not_found set), sets retried so we don't loop.
    """
    if state.get("why_not_found"):
        state["retried"] = True
    q = (state["question"] or "").strip()
    if not q:
        state["why_not_found"] = "Empty question."
    state["question"] = q
    return state


def node_retrieve(state: RAGState) -> RAGState:
    """
    Inputs:
    - state: RAGState containing `question`. May include `why_not_found`.

    Outputs:
    - Updated RAGState with `retrieved` populated as serializable dicts.

    What it does:
    - Performs vector similarity retrieval against Chroma and stores results in a
      JSON-friendly shape: [{"text": ..., "meta": {...}}, ...]
    """
    if state.get("why_not_found"):
        return state

    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(state["question"])

    ser: List[Dict[str, Any]] = []
    for d in docs:
        ser.append({"text": d.page_content, "meta": dict(d.metadata or {})})
    state["retrieved"] = ser
    return state


def node_build_context(state: RAGState) -> RAGState:
    """
    Inputs:
    - state: RAGState with `retrieved` passages.

    Outputs:
    - Updated RAGState with:
      - `context`: newline-joined string of cited passages
      - `citations`: list of citation labels
      - `why_not_found`: set if nothing was retrieved

    What it does:
    - Builds a single context string that the LLM can use, with stable [S1], [S2] citations.
    """
    docs = state.get("retrieved", [])
    if not docs:
        state["why_not_found"] = "No relevant passages retrieved from PDFs."
        state["context"] = ""
        state["citations"] = []
        return state

    ctx_lines: List[str] = []
    citations: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta", {})
        source_file = meta.get("source_file", meta.get("source", "unknown.pdf"))
        page = meta.get("page", meta.get("page_number", None))
        cite = f"[S{i}] {source_file}" + (f" (page {page})" if page is not None else "")
        citations.append(cite)

        txt = (d.get("text", "") or "").strip().replace("\n", " ")
        ctx_lines.append(f"{cite}: {txt}")

    state["citations"] = citations
    state["context"] = "\n".join(ctx_lines)
    return state


def _route_after_context(state: RAGState) -> Literal["guard", "__end__"]:
    """
    Simple check: if we have no context (why_not_found) and haven't retried yet, go back to guard; else END.
    """
    if state.get("why_not_found") and not state.get("retried"):
        return "guard"
    return "__end__"


def build_retrieval_graph():
    """
    Inputs:
    - None

    Outputs:
    - A compiled LangGraph runnable that takes/returns `RAGState`.

    What it does:
    - Creates a 3-node retrieval pipeline with conditional from context:
      guard -> retrieve -> context -> (END or guard).
    - Conditional: if why_not_found and not retried, go to guard once; else END.
    """
    g = StateGraph(RAGState)
    g.add_node("guard", node_guard)
    g.add_node("retrieve", node_retrieve)
    g.add_node("context", node_build_context)
    g.set_entry_point("guard")
    g.add_edge("guard", "retrieve")
    g.add_edge("retrieve", "context")
    g.add_conditional_edges("context", _route_after_context, {"guard": "guard", "__end__": END})
    return g.compile()
