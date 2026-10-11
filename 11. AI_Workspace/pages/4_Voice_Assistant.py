import os
import tempfile

import streamlit as st

from core.chat_store import clear_history, save_message
from core.config import DEFAULT_VOICE_LANGUAGE, VOICE_DB, VOICE_LANGUAGES
from core.llm import describe_llm_error, extract_text, get_llm
from core.prompts import VOICE_LANGUAGE_INSTRUCTIONS, VOICE_PROMPT
from core.session import charge_session, get_session_id
from core.stt import transcribe
from core.tts import prefetch_voice_async, synthesize
from utils.ui import page_header, require_provider

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️")
require_provider()

CHAT_DB_NAME = "voice_chat"
session_id = get_session_id()


def _clear_chat():
    clear_history(VOICE_DB, CHAT_DB_NAME, session_id=session_id)
    st.session_state.voice_history = []


with st.sidebar:
    language = st.selectbox(
        "Language",
        options=list(VOICE_LANGUAGES.keys()),
        index=list(VOICE_LANGUAGES.keys()).index(
            st.session_state.get("voice_language", DEFAULT_VOICE_LANGUAGE)
        ),
    )
    st.session_state["voice_language"] = language

page_header(
    "🎙️ Voice Assistant",
    "Type or hit the mic in the chat box to record your question.\n"
    "The spoken answer plays the moment it's ready alongside the reply",
    on_clear=_clear_chat,
)

lang_config = VOICE_LANGUAGES[language]
voice_key = lang_config["voice_key"]

# Downloads the voice in the background the moment this page loads, instead of
# blocking the first reply — no-ops instantly if it's already on disk.
prefetch_voice_async(voice_key)

if "voice_history" not in st.session_state:
    st.session_state.voice_history = []

for turn in st.session_state.voice_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("extra") and os.path.exists(turn["extra"]):
            st.audio(turn["extra"])

submission = st.chat_input("Type your question, or tap the mic to record it...", accept_audio=True)

user_text = None
if submission:
    if submission.audio is not None:
        with st.spinner("Transcribing..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(submission.audio.getvalue())
                tmp_path = tmp.name
            try:
                user_text = transcribe(tmp_path, language=lang_config["stt_code"])
            except Exception as e:
                st.error(f"Couldn't transcribe your recording: {e}")
    elif submission.text:
        user_text = submission.text

if user_text and charge_session(len(user_text.encode())):
    st.session_state.voice_history.append({"role": "user", "content": user_text})
    save_message(VOICE_DB, CHAT_DB_NAME, "user", user_text, session_id=session_id)
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        system_prompt = VOICE_PROMPT + "\n" + VOICE_LANGUAGE_INSTRUCTIONS[language]
        messages = [{"role": "system", "content": system_prompt}]
        for turn in st.session_state.voice_history[-7:-1]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_text})

        # Keep the reply hidden behind a loading bar until its spoken audio is
        # ready, then reveal text + autoplaying audio together.
        progress = st.progress(0, text="Thinking...")
        wav_path = None
        try:
            answer = extract_text(get_llm().invoke(messages).content) or "(no answer)"
            progress.progress(70, text="Synthesizing speech...")
            wav_path = synthesize(answer, voice_key)
            progress.empty()
            st.markdown(answer)
            st.audio(wav_path, autoplay=True)
        except Exception as e:
            progress.empty()
            answer = describe_llm_error(e)
            st.markdown(answer)

    charge_session(len(answer.encode()))
    st.session_state.voice_history.append(
        {"role": "assistant", "content": answer, "extra": wav_path}
    )
    save_message(VOICE_DB, CHAT_DB_NAME, "assistant", answer, extra=wav_path, session_id=session_id)




