import tempfile
import urllib.request
import wave
from pathlib import Path

import streamlit as st

from core.config import PIPER_VOICE_BASE_URL, PIPER_VOICE_NAME, VOICE_DIR


def _ensure_voice_files() -> tuple[str, str]:
    voice_dir = Path(VOICE_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voice_dir / f"{PIPER_VOICE_NAME}.onnx"
    json_path = voice_dir / f"{PIPER_VOICE_NAME}.onnx.json"
    for path in (onnx_path, json_path):
        if not path.exists():
            urllib.request.urlretrieve(f"{PIPER_VOICE_BASE_URL}/{path.name}", path)
    return str(onnx_path), str(json_path)


@st.cache_resource(show_spinner="Loading local Piper voice (first run downloads ~60MB)...")
def _load_voice():
    from piper import PiperVoice

    onnx_path, json_path = _ensure_voice_files()
    return PiperVoice.load(onnx_path, config_path=json_path)


def synthesize(text: str) -> str:
    """Free, fully local text-to-speech (Piper) -> path to a temp WAV file.
    No API key, no cloud call.

    NOTE: Piper's maintainer handoff (archived rhasspy/piper -> OHF-Voice/piper1-gpl,
    also a license change to GPL-3.0-or-later) shifted this method's exact signature
    across versions — the fallback below handles both call shapes, but pin a piper-tts
    version and smoke-test once before relying on this in production.
    """
    from piper import SynthesisConfig

    voice = _load_voice()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        with wave.open(temp_audio.name, "wb") as wav_file:
            try:
                voice.synthesize_wav(text, wav_file, syn_config=SynthesisConfig())
            except (AttributeError, TypeError):
                voice.synthesize(text, wav_file)
        return temp_audio.name
