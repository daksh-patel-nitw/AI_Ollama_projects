import streamlit as st

from core.config import WHISPER_MODEL_SIZE


@st.cache_resource(show_spinner="Loading local Whisper model...")
def _load_model():
    from faster_whisper import WhisperModel

    # CPU + int8 keeps this runnable on a free hosting tier with no GPU.
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_path: str) -> str:
    """Free, local speech-to-text (faster-whisper) — no API key, no cloud call."""
    model = _load_model()
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments).strip()
