import tempfile
import threading
import urllib.request
import wave
from pathlib import Path

import streamlit as st

from core.config import DEFAULT_PIPER_VOICE, PIPER_VOICE_BASE_URL, VOICE_DIR

_prefetch_lock = threading.Lock()
_prefetch_started: set[str] = set()


def _voice_url_dir(voice_key: str) -> str:
    lang_country, voice_name, quality = voice_key.split("-")
    lang = lang_country.split("_")[0]
    return f"{PIPER_VOICE_BASE_URL}/{lang}/{lang_country}/{voice_name}/{quality}"


def _ensure_voice_files(voice_key: str = DEFAULT_PIPER_VOICE) -> tuple[str, str]:
    voice_dir = Path(VOICE_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voice_dir / f"{voice_key}.onnx"
    json_path = voice_dir / f"{voice_key}.onnx.json"
    base_url = _voice_url_dir(voice_key)
    for path in (onnx_path, json_path):
        if not path.exists():
            urllib.request.urlretrieve(f"{base_url}/{path.name}", path)
    return str(onnx_path), str(json_path)


def prefetch_voice_async(voice_key: str = DEFAULT_PIPER_VOICE) -> None:
    """Download + load this voice in the background as soon as the Voice Assistant
    page loads, so _load_voice() is already a warm cache hit (no spinner) by the
    time the user actually sends a message."""
    with _prefetch_lock:
        if voice_key in _prefetch_started:
            return
        _prefetch_started.add(voice_key)
    threading.Thread(target=_load_voice, args=(voice_key,), daemon=True).start()


@st.cache_resource(show_spinner="Preparing voice...")
def _load_voice(voice_key: str = DEFAULT_PIPER_VOICE):
    from piper import PiperVoice

    onnx_path, json_path = _ensure_voice_files(voice_key)
    return PiperVoice.load(onnx_path, config_path=json_path)


def synthesize(text: str, voice_key: str = DEFAULT_PIPER_VOICE) -> str:
    """Free, fully local text-to-speech (Piper) -> path to a temp WAV file.
    No API key, no cloud call.

    NOTE: Piper's maintainer handoff (archived rhasspy/piper -> OHF-Voice/piper1-gpl,
    also a license change to GPL-3.0-or-later) shifted this method's exact signature
    across versions — the fallback below handles both call shapes, but pin a piper-tts
    version and smoke-test once before relying on this in production.
    """
    from piper import SynthesisConfig

    voice = _load_voice(voice_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        with wave.open(temp_audio.name, "wb") as wav_file:
            try:
                voice.synthesize_wav(text, wav_file, syn_config=SynthesisConfig())
            except (AttributeError, TypeError):
                voice.synthesize(text, wav_file)
        return temp_audio.name
