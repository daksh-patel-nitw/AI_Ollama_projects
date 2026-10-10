import tempfile

import streamlit as st

from core.config import PDF_DB
from core.llm import get_llm
from core.prompts import PDF_PROMPT
from core.vectordb import load_vectorstore
from utils.ui import page_header, require_provider

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
require_provider()

page_header("📄 PDF Q&A", "Upload a PDF, then ask questions grounded strictly in its content.")

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = load_vectorstore(PDF_DB)
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = set()

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded and uploaded.name not in st.session_state.pdf_processed:
    with st.spinner(f"Indexing {uploaded.name}..."):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = uploaded.name

        st.session_state.pdf_store.add_documents(chunks)
        st.session_state.pdf_processed.add(uploaded.name)
    st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

if st.session_state.pdf_processed:
    st.divider()
    query = st.text_input("Ask a question about your uploaded PDF(s):")

    if query:
        with st.spinner("Searching document and generating an answer..."):
            docs = st.session_state.pdf_store.similarity_search(query, k=4)
            context = "\n\n".join(d.page_content for d in docs)
            messages = [
                {"role": "system", "content": PDF_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            answer = get_llm().invoke(messages).content

        st.markdown("### Answer")
        st.write(answer)
        with st.expander("Source passages"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Passage {i}** — _{d.metadata.get('source', 'unknown')}_")
                st.info(d.page_content)
else:
    st.info("Upload a PDF to get started.")
