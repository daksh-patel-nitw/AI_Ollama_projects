import streamlit as st

from core.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    TEMPERATURE,
)


def is_configured() -> bool:
    """Whether the user has picked a provider and supplied whatever it needs."""
    provider = st.session_state.get("provider")
    if provider == "Ollama":
        return True
    if provider == "OpenAI":
        return bool(st.session_state.get("openai_api_key"))
    if provider == "Gemini":
        return bool(st.session_state.get("gemini_api_key"))
    return False


def get_llm(temperature: float = TEMPERATURE):
    """Return a LangChain chat model for whichever provider was configured on Home.

    Provider SDKs are imported lazily inside each branch so a user who only wants
    Ollama never needs the OpenAI/Gemini packages installed, and vice versa.
    """
    provider = st.session_state.get("provider")

    if provider == "Ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL),
            temperature=temperature,
        )

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        api_key = st.session_state.get("openai_api_key")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key — set it on the Home page.")
        return ChatOpenAI(
            model=st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL),
            api_key=api_key,
            temperature=temperature,
        )

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = st.session_state.get("gemini_api_key")
        if not api_key:
            raise RuntimeError("Missing Gemini API key — set it on the Home page.")
        return ChatGoogleGenerativeAI(
            model=st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL),
            google_api_key=api_key,
            temperature=temperature,
        )

    raise RuntimeError("No model provider configured yet — go to Home first.")
