import time

from langchain_community.vectorstores import Chroma

from core.embeddings import get_embeddings


def load_vectorstore(db_path: str) -> Chroma:
    return Chroma(persist_directory=db_path, embedding_function=get_embeddings())


def delete_session_chunks(vectorstore: Chroma, session_id: str) -> None:
    """Remove only this session's indexed chunks — never touches other sessions' PDFs."""
    vectorstore.delete(where={"session_id": session_id})


def cleanup_stale_chunks(vectorstore: Chroma, ttl_minutes: int) -> None:
    """Best-effort purge of chunks from sessions idle longer than ttl_minutes —
    there's no reliable public Streamlit hook for "the user closed the tab"."""
    cutoff = time.time() - ttl_minutes * 60
    existing = vectorstore.get(include=["metadatas"])
    stale_ids = [
        id_
        for id_, meta in zip(existing["ids"], existing["metadatas"])
        if (meta or {}).get("ts", 0) < cutoff
    ]
    if stale_ids:
        vectorstore.delete(ids=stale_ids)
