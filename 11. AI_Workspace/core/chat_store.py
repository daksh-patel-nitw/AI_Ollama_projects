"""Persist chat turns in a Chroma collection so conversations survive app restarts.

Chroma is used here purely as a durable key-value log (not for similarity search),
so messages are stored with a dummy embedding and fetched back with collection.get().
"""

import json
import time
import uuid

import chromadb


def _collection(db_path: str, name: str):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name)


def load_history(db_path: str, name: str, session_id: str = "default") -> list[dict]:
    """Return persisted turns for a session, oldest first: [{"role", "content", "extra"}, ...]."""
    collection = _collection(db_path, name)
    result = collection.get(where={"session_id": session_id})
    rows = sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda pair: pair[0].get("ts", 0),
    )
    turns = []
    for meta, doc in rows:
        extra_raw = meta.get("extra")
        turns.append(
            {
                "role": meta["role"],
                "content": doc,
                "extra": json.loads(extra_raw) if extra_raw else None,
            }
        )
    return turns


def save_message(
    db_path: str,
    name: str,
    role: str,
    content: str,
    extra=None,
    session_id: str = "default",
) -> None:
    """Persist one chat turn. `extra` is any JSON-serializable extra payload (e.g. image URIs)."""
    collection = _collection(db_path, name)
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[content],
        embeddings=[[0.0]],
        metadatas=[
            {
                "session_id": session_id,
                "role": role,
                "ts": time.time(),
                "extra": json.dumps(extra) if extra is not None else "",
            }
        ],
    )


def clear_history(db_path: str, name: str, session_id: str = "default") -> None:
    collection = _collection(db_path, name)
    existing = collection.get(where={"session_id": session_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def cleanup_stale(db_path: str, name: str, ttl_minutes: int) -> None:
    """Best-effort purge of messages idle longer than ttl_minutes, across ALL
    sessions — there's no reliable public Streamlit hook for "the user left"."""
    collection = _collection(db_path, name)
    cutoff = time.time() - ttl_minutes * 60
    existing = collection.get(include=["metadatas"])
    stale_ids = [
        id_
        for id_, meta in zip(existing["ids"], existing["metadatas"])
        if meta.get("ts", 0) < cutoff
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)
