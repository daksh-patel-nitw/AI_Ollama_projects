import tempfile

import streamlit as st

from core.llm import get_llm
from core.prompts import VOICE_PROMPT
from core.stt import transcribe
from core.tts import synthesize
from utils.ui import page_header, require_provider

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️")
require_provider()

page_header(
    "🎙️ Voice Assistant",
    "Speak, get a transcript, a grounded reply, and a spoken answer back — "
    "Whisper (STT) + Piper (TTS), both free and fully local, no API key.",
)

if "voice_history" not in st.session_state:
    st.session_state.voice_history = []

audio = st.audio_input("Record your question")

if audio is not None:
    with st.spinner("Transcribing..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio.getvalue())
            tmp_path = tmp.name
        transcript = transcribe(tmp_path)

    st.markdown(f"**You said:** {transcript}")

    with st.spinner("Thinking..."):
        messages = [{"role": "system", "content": VOICE_PROMPT}]
        for role, text in st.session_state.voice_history[-6:]:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": transcript})
        answer = get_llm().invoke(messages).content

    st.session_state.voice_history.append(("user", transcript))
    st.session_state.voice_history.append(("assistant", answer))

    st.markdown(f"**Assistant:** {answer}")

    with st.spinner("Synthesizing speech..."):
        wav_path = synthesize(answer)
    st.audio(wav_path)

if st.session_state.voice_history:
    with st.expander("Conversation history"):
        for role, text in st.session_state.voice_history:
            st.markdown(f"**{role.title()}:** {text}")
