import streamlit as st

from core.config import (
    APP_NAME,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
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
    index=PROVIDERS.index(st.session_state.get("provider", "Ollama")),
    horizontal=True,
)
st.session_state["provider"] = provider

if provider == "Ollama":
    st.session_state["ollama_model"] = st.text_input(
        "Ollama model name",
        value=st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL),
    )
    st.info(
        "Free and fully local — requires Ollama running on **this machine** "
        f"(`ollama serve`, with the model pulled via `ollama pull "
        f"{st.session_state['ollama_model']}`). This will not work once the app "
        "is deployed to a free cloud host unless you separately host and expose "
        "an Ollama server — use OpenAI or Gemini below for a hosted deployment."
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
    st.session_state["gemini_model"] = st.text_input(
        "Gemini model", value=st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)
    )
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
