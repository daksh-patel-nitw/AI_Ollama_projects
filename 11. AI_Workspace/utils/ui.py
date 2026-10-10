import streamlit as st

from core.llm import is_configured


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)
    st.divider()


def require_provider() -> None:
    """Guard for every tool page: stop and point back Home if no model is configured yet."""
    if not st.session_state.get("provider") or not is_configured():
        st.warning("No model provider configured yet.")
        st.page_link("app.py", label="Go to Home to choose a model", icon="🏠")
        st.stop()
