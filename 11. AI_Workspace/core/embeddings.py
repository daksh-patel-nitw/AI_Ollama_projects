import streamlit as st

from core.config import EMBEDDING_MODEL


@st.cache_resource(show_spinner="Loading local embedding model (first run downloads ~130MB)...")
def get_embeddings():
    """Free, local, ONNX-based embeddings (fastembed) — deliberately independent
    of the chosen chat LLM provider. See core/llm.py for why."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
