import warnings

import streamlit as st

from core.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    TEMPERATURE,
)

# Some newer Gemini models (e.g. flash-lite) ignore a custom temperature and warn
# about it every call — harmless, so silence just that one message.
warnings.filterwarnings(
    "ignore", message=".*uses fixed sampling defaults.*", category=UserWarning
)

_CONNECTION_HINTS = (
    "connect",
    "refused",
    "max retries",
    "timed out",
    "timeout",
    "failed to establish",
    "newconnectionerror",
)
_MODEL_MISSING_HINTS = ("not found", "404", "invalid", "unexpected model name", "does not exist")
_AUTH_HINTS = ("api key", "unauthorized", "401", "permission", "invalid_api_key")
_QUOTA_HINTS = ("quota", "rate limit", "429", "resource_exhausted")


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
            base_url=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
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


def describe_llm_error(exc: Exception) -> str:
    """Turn a raw provider exception into a short, actionable message for the UI."""
    provider = st.session_state.get("provider")
    msg = str(exc).strip() or exc.__class__.__name__
    lower = msg.lower()

    if not provider:
        return "No AI model provider configured yet — go to Home and pick one."

    if provider == "Ollama":
        model = st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL)
        if any(hint in lower for hint in _CONNECTION_HINTS):
            return (
                "Couldn't reach Ollama. Make sure it's installed and running "
                "(`ollama serve`) on this machine, then try again."
            )
        if "not found" in lower or "404" in lower:
            return (
                f"Model '{model}' isn't available in Ollama. Pull it first with "
                f"`ollama pull {model}`, then try again."
            )
        return f"Ollama error: {msg}"

    if provider in ("OpenAI", "Gemini"):
        if any(hint in lower for hint in _AUTH_HINTS):
            return f"{provider} rejected the request — check that your API key is valid on the Home page."
        if "model" in lower and any(hint in lower for hint in _MODEL_MISSING_HINTS):
            model_key = "gemini_model" if provider == "Gemini" else "openai_model"
            model = st.session_state.get(model_key, "")
            return (
                f"'{model}' isn't a valid {provider} model id. Pick one on the Home "
                "page using the exact API model id, not its display name."
            )
        if any(hint in lower for hint in _QUOTA_HINTS):
            return f"{provider} rate limit or quota exceeded — wait a bit and try again."
        return f"{provider} error: {msg}"

    return f"Unexpected error: {msg}"


def extract_text(content) -> str:
    """Normalize a LangChain message .content payload to plain text.

    Some providers (Gemini, Anthropic-style multimodal replies) return content as
    a list of content blocks (e.g. [{"type": "text", "text": "..."}]) instead of
    a plain string — flatten that down so the UI never renders raw dict/list reprs.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def stream_text(messages, temperature: float = TEMPERATURE):
    """Yield plain-text deltas from the configured chat model for st.write_stream."""
    for chunk in get_llm(temperature).stream(messages):
        text = extract_text(chunk.content)
        if text:
            yield text


def stream_agent_text(agent, query: str):
    """Yield plain-text deltas from a langgraph react agent's final answer only,
    skipping tool-call and tool-result chunks, for st.write_stream."""
    from langchain_core.messages import AIMessageChunk

    for message_chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": query}]}, stream_mode="messages"
    ):
        if not isinstance(message_chunk, AIMessageChunk):
            continue
        text = extract_text(message_chunk.content)
        if text:
            yield text
