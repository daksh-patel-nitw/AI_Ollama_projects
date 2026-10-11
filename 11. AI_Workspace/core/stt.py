import streamlit as st

from core.config import WHISPER_MODEL_SIZE


def _patch_av_open_for_metadata_errors() -> None:
    """faster-whisper calls av.open(..., metadata_errors="ignore"), a kwarg that
    newer PyAV releases (19+) dropped, raising a TypeError — strip it before the call."""
    import av

    if getattr(av.open, "_patched_for_metadata_errors", False):
        return

    original_open = av.open

    def patched_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return original_open(*args, **kwargs)

    patched_open._patched_for_metadata_errors = True
    av.open = patched_open


@st.cache_resource(show_spinner="Loading local Whisper model...")
def _load_model():
    _patch_av_open_for_metadata_errors()
    from faster_whisper import WhisperModel

    # CPU + int8 keeps this runnable on a free hosting tier with no GPU.
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_path: str, language: str | None = None) -> str:
    """Free, local speech-to-text (faster-whisper) — no API key, no cloud call.

    `language` forces the Whisper language code (e.g. "hi") instead of relying on
    auto-detection, which is more reliable for short recordings.
    """
    model = _load_model()
    segments, _info = model.transcribe(audio_path, language=language)
    return " ".join(segment.text.strip() for segment in segments).strip()
