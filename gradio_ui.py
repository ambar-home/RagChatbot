import os
import uuid
from typing import Dict, List, Tuple

import gradio as gr

from langgraph_flow import RAGState
from langchain_rag import (
    ensure_index_exists,
    format_sources_panel,
    ingest_incremental,
    is_langsmith_enabled,
    save_uploaded_docs,
    stream_openai_answer,
)


def normalize_gradio_files(files) -> List[str]:
    """
    Inputs:
    - files: Gradio `gr.Files` value (varies by Gradio version; can be None, str paths, or file objects).

    Outputs:
    - List of file paths (strings). Entries are filtered to non-empty paths.

    What it does:
    - Normalizes Gradio upload payloads into a consistent list of local file paths.
    """
    filepaths: List[str] = []
    if files is None:
        return filepaths

    if isinstance(files, list):
        for f in files:
            if isinstance(f, str):
                filepaths.append(f)
            else:
                filepaths.append(getattr(f, "name", None))
    else:
        filepaths = [files] if isinstance(files, str) else [getattr(files, "name", None)]

    return [p for p in filepaths if p]


def ui_reindex(files) -> Tuple[str, str]:
    """
    Inputs:
    - files: Gradio upload payload from `gr.Files`.

    Outputs:
    - (status_text, index_health_markdown)

    What it does:
    - Saves uploaded PDF/Word into data/docs/, runs incremental ingestion, and returns UI status messages.
    """
    filepaths = normalize_gradio_files(files)
    saved_count, saved_names = save_uploaded_docs(filepaths)
    ingest_status = ingest_incremental()

    upload_msg = f"📥 Uploaded PDFs: {saved_count}\n" + (
        f"Files: {', '.join(saved_names)}\n" if saved_names else ""
    )
    return upload_msg + "\n" + ingest_status, ensure_index_exists()


def chat_stream(graph, user_message: str, messages: List[Dict[str, str]]):
    """
    Inputs:
    - graph: Compiled LangGraph runnable that accepts/returns `RAGState`.
    - user_message: User text input.
    - messages: Chat history in Gradio Chatbot's messages format.

    Outputs:
    - Yields tuples (updated_messages, sources_panel_markdown) for streaming UI updates.

    What it does:
    - Runs retrieval via LangGraph, then streams a grounded OpenAI answer while updating the UI.
    """
    messages = messages or []
    messages = messages + [{"role": "user", "content": user_message}]
    messages = messages + [{"role": "assistant", "content": ""}]

    init_state: RAGState = {
        "question": user_message,
        "retrieved": [],
        "context": "",
        "citations": [],
        "why_not_found": None,
    }
    # If LangSmith tracing is enabled, attach tags/metadata so runs are easy to find.
    # This will show up in LangSmith as a trace for the LangGraph invocation.
    run_config = None
    if is_langsmith_enabled():
        run_config = {
            "run_name": "rag_retrieval",
            "tags": ["rag_langgraph", "gradio", "retrieval"],
            "metadata": {
                "ui": "gradio",
                "request_id": str(uuid.uuid4()),
                "question_length": len(user_message or ""),
                "turn_index": max(0, (len(messages) - 2) // 2),
            },
        }

    out = graph.invoke(init_state, config=run_config) if run_config else graph.invoke(init_state)

    sources_md = format_sources_panel(out.get("retrieved", []))

    if out.get("why_not_found"):
        messages[-1]["content"] = (
            f"I couldn't answer from the PDFs. Reason: {out['why_not_found']}"
        )
        yield messages, sources_md
        return

    # Stream response only (no citations in loop); add citations once after response is done
    chat_history = messages[:-2] if len(messages) > 2 else []
    final_text = ""
    for partial in stream_openai_answer(user_message, out["context"], chat_history=chat_history):
        final_text = partial
        messages[-1]["content"] = final_text
        yield messages, sources_md

    # Append citations once after the full response (first response, then citations, only once)
    if out.get("citations"):
        messages[-1]["content"] = final_text + "\n\n---\n**Citations**\n" + "\n".join(out["citations"])
        yield messages, sources_md


def create_demo(graph) -> gr.Blocks:
    """
    Inputs:
    - graph: Compiled LangGraph runnable (retrieval pipeline).

    Outputs:
    - A configured `gr.Blocks` demo app (not launched yet).

    What it does:
    - Builds the full Gradio UI and wires up callbacks for re-indexing and chat streaming.
    """
    def _chat_stream(user_message: str, messages: List[Dict[str, str]]):
        """
        Inputs:
        - user_message: User text input.
        - messages: Chat history in Gradio Chatbot's messages format.

        Outputs:
        - Yields (updated_messages, sources_panel_markdown)

        What it does:
        - Wrapper generator function so Gradio detects streaming correctly.
        """
        yield from chat_stream(graph, user_message, messages)

    with gr.Blocks(title="Local PDF RAG (LangChain + LangGraph + OpenAI + Gradio)") as demo:
        gr.Markdown(
            "# 📄 Local RAG Chat (PDF, Word)\n"
            "Upload PDF / Word → Re-index (incremental) → Ask questions → Get answers with citations + expandable sources."
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=520, label="Chat")
                question = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g., What is the escalation procedure mentioned in the PDF?",
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("## Index Controls")
                uploads = gr.Files(label="Upload PDF / Word", file_types=[".pdf", ".docx"])
                reindex_btn = gr.Button("Re-index (incremental)", variant="secondary")
                status = gr.Textbox(label="Status", lines=10)
                index_health = gr.Markdown(ensure_index_exists())

                gr.Markdown("## Sources Panel")
                sources_panel = gr.Markdown("### Sources\n(Ask a question to see sources)")

        reindex_btn.click(fn=ui_reindex, inputs=[uploads], outputs=[status, index_health])

        send.click(
            fn=_chat_stream,
            inputs=[question, chatbot],
            outputs=[chatbot, sources_panel],
        )
        question.submit(
            fn=_chat_stream,
            inputs=[question, chatbot],
            outputs=[chatbot, sources_panel],
        )

        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: "### Sources\n(Ask a question to see sources)", None, sources_panel)
        clear.click(lambda: "", None, question)

    return demo
