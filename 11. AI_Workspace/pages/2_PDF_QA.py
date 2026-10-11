import tempfile
import time

import streamlit as st

from core.chat_store import clear_history, save_message
from core.config import MAX_FILE_SIZE_MB, PDF_DB, SESSION_TTL_MINUTES
from core.llm import describe_llm_error, stream_text
from core.prompts import PDF_PROMPT
from core.session import charge_session, get_session_id
from core.vectordb import cleanup_stale_chunks, delete_session_chunks, load_vectorstore
from utils.ui import page_header, require_provider

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
require_provider()

CHAT_DB_NAME = "pdf_chat"
session_id = get_session_id()


def _clear_chat():
    clear_history(PDF_DB, CHAT_DB_NAME, session_id=session_id)
    st.session_state.pdf_chat_history = []


if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = load_vectorstore(PDF_DB)
    cleanup_stale_chunks(st.session_state.pdf_store, SESSION_TTL_MINUTES)
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []


def _index_pdf(uploaded_file) -> None:
    if not charge_session(len(uploaded_file.getvalue())):
        return
    with st.spinner(f"Indexing {uploaded_file.name}..."):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
            chunk.metadata["session_id"] = session_id
            chunk.metadata["ts"] = time.time()

        st.session_state.pdf_store.add_documents(chunks)
        st.session_state.pdf_filename = uploaded_file.name
    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")


def _replace_pdf(new_file) -> None:
    """Wipe this session's vectors + chat history, then index the new file."""
    delete_session_chunks(st.session_state.pdf_store, session_id)
    clear_history(PDF_DB, CHAT_DB_NAME, session_id=session_id)
    st.session_state.pdf_chat_history = []
    _index_pdf(new_file)


@st.dialog("Replace uploaded PDF?")
def _confirm_replace_dialog(pending_file):
    st.warning(
        "Uploading a new PDF will clear your current chat history and indexed "
        "content. This can't be undone."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Clear and upload", type="primary", use_container_width=True):
            _replace_pdf(pending_file)
            st.rerun()


with st.sidebar:
    if st.session_state.pdf_filename:
        st.caption("📄 Uploaded file")
        st.text(st.session_state.pdf_filename)
        new_file = st.file_uploader(
            "Upload a different PDF",
            type=["pdf"],
            key="pdf_uploader",
            max_upload_size=MAX_FILE_SIZE_MB,
        )
        if new_file is not None and new_file.name != st.session_state.pdf_filename:
            _confirm_replace_dialog(new_file)
    else:
        new_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            key="pdf_uploader",
            max_upload_size=MAX_FILE_SIZE_MB,
            help=f"Max {MAX_FILE_SIZE_MB}MB per file.",
        )
        if new_file is not None:
            _index_pdf(new_file)

page_header(
    "📄 PDF Q&A",
    "Upload a PDF, then ask questions grounded strictly in its content.",
    on_clear=_clear_chat,
)

if not st.session_state.pdf_filename:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    for turn in st.session_state.pdf_chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("extra"):
                with st.expander("Source passages"):
                    for i, passage in enumerate(turn["extra"], 1):
                        st.markdown(f"**Passage {i}** — _{passage.get('source', 'unknown')}_")
                        st.info(passage.get("text", ""))

    query = st.chat_input("Ask a question about your uploaded PDF:")

    if query and charge_session(len(query.encode())):
        st.session_state.pdf_chat_history.append({"role": "user", "content": query})
        save_message(PDF_DB, CHAT_DB_NAME, "user", query, session_id=session_id)
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            docs = st.session_state.pdf_store.similarity_search(
                query, k=4, filter={"session_id": session_id}
            )
            context = "\n\n".join(d.page_content for d in docs)
            messages = [
                {"role": "system", "content": PDF_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            try:
                with st.spinner("Reading your PDF and thinking..."):
                    answer = st.write_stream(stream_text(messages)) or "(no answer)"
            except Exception as e:
                answer = describe_llm_error(e)
                st.markdown(answer)
                docs = []

            if docs:
                with st.expander("Source passages"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Passage {i}** — _{d.metadata.get('source', 'unknown')}_")
                        st.info(d.page_content)

        sources = [{"source": d.metadata.get("source", "unknown"), "text": d.page_content} for d in docs]
        charge_session(len(answer.encode()))
        st.session_state.pdf_chat_history.append(
            {"role": "assistant", "content": answer, "extra": sources}
        )
        save_message(PDF_DB, CHAT_DB_NAME, "assistant", answer, extra=sources, session_id=session_id)

