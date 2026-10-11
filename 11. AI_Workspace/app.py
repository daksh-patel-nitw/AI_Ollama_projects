import streamlit as st

from core.config import (
    APP_NAME,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    GEMINI_MODELS,
    PROVIDERS,
    VERSION,
)
from core.llm import is_configured
from utils.ui import page_header

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="centered")

page_header(APP_NAME, f"v{VERSION} — pick a model provider, then open any tool below.")

st.subheader("1. Choose a model provider")

provider = st.radio(
    "Model provider",
    PROVIDERS,
    index=PROVIDERS.index(st.session_state.get("provider", "Gemini")),
    horizontal=True,
)
st.session_state["provider"] = provider

if provider == "Ollama":
    st.session_state["ollama_model"] = st.text_input(
        "Ollama model name",
        value=st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL),
    )
    st.session_state["ollama_base_url"] = st.text_input(
        "Ollama server URL",
        value=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
        help="Defaults to a locally running `ollama serve`. If you're tunneling Ollama "
        "from elsewhere (e.g. a free Google Colab GPU session via ngrok/cloudflared), "
        "paste that public URL here instead.",
    )
    st.info(
        "Use tunneling Ollama from elsewhere (e.g. a free Google Colab GPU session via ngrok/cloudflared), "
        "Paste that public URL and use.",
    )

elif provider == "OpenAI":
    st.session_state["openai_api_key"] = st.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.get("openai_api_key", ""),
    )
    st.session_state["openai_model"] = st.text_input(
        "OpenAI model", value=st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL)
    )
    st.caption("Your key is kept only in this browser session's memory — never written to disk.")

elif provider == "Gemini":
    st.session_state["gemini_api_key"] = st.text_input(
        "Google Gemini API key",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
    )

    # Dropdown of known-good API model ids — a free-text field let users paste
    # the AI Studio display name (e.g. "Gemini 2.5 Flash") which Google's API
    # rejects with a 400 "unexpected model name format" error.
    current_gemini_model = st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)
    gemini_options = GEMINI_MODELS + ["Custom…"]
    default_index = (
        gemini_options.index(current_gemini_model)
        if current_gemini_model in GEMINI_MODELS
        else len(GEMINI_MODELS)
    )
    gemini_choice = st.selectbox("Gemini model", gemini_options, index=default_index)
    if gemini_choice == "Custom…":
        st.session_state["gemini_model"] = st.text_input(
            "Custom Gemini model id (exact API id, e.g. gemini-2.5-flash)",
            value=current_gemini_model if current_gemini_model not in GEMINI_MODELS else "",
        )
    else:
        st.session_state["gemini_model"] = gemini_choice

    st.caption(
        "Google AI Studio offers a free tier for Gemini — create a key at "
        "aistudio.google.com/apikey. Kept only in this browser session's memory."
    )

st.divider()
st.subheader("2. Open a tool")

if not is_configured():
    st.warning("Finish step 1 before continuing.")
else:
    st.success(f"Using **{provider}**. Pick a tool:")
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/1_Web_Search.py", label="Web Search Agent", icon="🌐")
        st.page_link("pages/3_Flower_Search.py", label="Flower / Image Search", icon="🌸")
    with col2:
        st.page_link("pages/2_PDF_QA.py", label="PDF Q&A", icon="📄")
        st.page_link("pages/4_Voice_Assistant.py", label="Voice Assistant", icon="🎙️")
