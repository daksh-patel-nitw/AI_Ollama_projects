"""Per-browser-session isolation + a soft storage cap for a shared free-tier demo deployment.

Each Streamlit session gets its own UUID, used to tag/filter persisted rows (chat
messages, PDF chunks) so concurrent visitors never see each other's data. Streamlit
has no reliable public API to detect "the user closed the tab or refreshed", so
genuinely-abandoned sessions are instead swept up lazily via a TTL check next time
any page loads, rather than cleaned up instantly.
"""

import time
import uuid

import streamlit as st

from core.chat_store import cleanup_stale
from core.config import (
    FLOWER_DB,
    PDF_DB,
    SESSION_MAX_MB,
    SESSION_TTL_MINUTES,
    VOICE_DB,
    WEB_SEARCH_DB,
)

_CHAT_SWEEP_TARGETS = [
    (WEB_SEARCH_DB, "web_search_chat"),
    (PDF_DB, "pdf_chat"),
    (FLOWER_DB, "flower_chat"),
    (VOICE_DB, "voice_chat"),
]
_SWEEP_INTERVAL_SECONDS = 300  # don't rescan more than once per 5 min, process-wide
_last_sweep_at = 0.0


def get_session_id() -> str:
    """A stable id for this browser session — resets (and so loses all data) on refresh."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.session_bytes_used = 0
    return st.session_state.session_id


def session_usage_mb() -> float:
    return st.session_state.get("session_bytes_used", 0) / (1024 * 1024)


def charge_session(num_bytes: int) -> bool:
    """Record `num_bytes` against this session's quota.

    Returns False (and shows an error) instead of charging if that would exceed
    the per-session cap — callers should skip the write in that case.
    """
    get_session_id()
    used = st.session_state.session_bytes_used
    limit = SESSION_MAX_MB * 1024 * 1024
    if used + num_bytes > limit:
        st.error(
            f"⚠️ This demo session has reached its {SESSION_MAX_MB}MB storage limit "
            "(a free-tier guardrail so one visitor can't use up all the shared space). "
            "Please refresh the page to start a fresh session."
        )
        return False
    st.session_state.session_bytes_used = used + num_bytes
    return True


def maybe_cleanup_stale_sessions() -> None:
    """Throttled, best-effort purge of chat rows from sessions idle past the TTL."""
    global _last_sweep_at
    now = time.time()
    if now - _last_sweep_at < _SWEEP_INTERVAL_SECONDS:
        return
    _last_sweep_at = now
    for db_path, name in _CHAT_SWEEP_TARGETS:
        try:
            cleanup_stale(db_path, name, SESSION_TTL_MINUTES)
        except Exception:
            pass  # best-effort housekeeping — never break a page load over this
