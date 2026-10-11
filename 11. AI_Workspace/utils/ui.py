import streamlit as st

from core.config import SESSION_MAX_MB
from core.llm import is_configured
from core.session import maybe_cleanup_stale_sessions, session_usage_mb


def page_header(title: str, description: str, on_clear=None) -> None:
    st.title(title)
    st.caption(description)
    st.divider()
    if on_clear is not None:
        # The sidebar stays put as the chat transcript grows — the native "⋮"
        # menu next to Deploy is owned by Streamlit's chrome and isn't
        # extensible by app code, so this is the closest "always visible" spot.
        with st.sidebar:
            st.divider()
            if st.button("🗑️ Clear chat history", use_container_width=True):
                on_clear()
                st.rerun()
            st.caption(f"Session storage: {session_usage_mb():.1f} / {SESSION_MAX_MB} MB")


def require_provider() -> None:
    """Guard for every tool page: stop and point back Home if no model is configured yet."""
    maybe_cleanup_stale_sessions()
    if not st.session_state.get("provider") or not is_configured():
        st.toast("⚠️ No model provider configured yet — pick one on Home first.", icon="⚠️")
        st.warning("No model provider configured yet.")
        st.page_link("app.py", label="Go to Home to choose a model", icon="🏠")
        st.stop()
