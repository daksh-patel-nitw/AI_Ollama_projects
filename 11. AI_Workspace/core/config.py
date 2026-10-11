import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "AI Workspace"
VERSION = "2.0.0"

# ---- Model providers (chosen by the user on the Home page) ----
PROVIDERS = ["Gemini","OpenAI","Ollama"]

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Gemini needs the exact API model id (not the display name shown in AI Studio,
# e.g. "gemini-2.5-flash" not "Gemini 2.5 Flash") or calls fail with a 400
# "unexpected model name format" error — offer a dropdown of known-good ids.
GEMINI_MODELS = [
    # Active Models with Free Quota Available
    "gemini-3.5-flash-lite",  # High daily limit (500 RPD)
    "gemini-3.1-flash-lite",  # High daily limit (500 RPD)
    "gemini-3.7-flash",       # (20 RPD)
    "gemini-3.6-flash",       # (20 RPD)
    "gemini-3.5-flash",       # (20 RPD)
    "gemini-2.5-flash",       # (20 RPD)
    "gemini-2.5-flash-lite",  # (20 RPD)
    "gemini-3.8-flash"        # (20 RPD)
]
DEFAULT_GEMINI_MODEL = GEMINI_MODELS[0]

TEMPERATURE = 0.3

# ---- Embeddings: always free/local, independent of the chosen chat provider,
# so PDF/image search still work even if the user picks OpenAI or Gemini for chat
# without handing over a second key just for embeddings. ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # served locally via fastembed (ONNX, CPU)

# ---- Chroma persistence paths ----
BASE_DB = "data"
PDF_DB = os.path.join(BASE_DB, "pdf_db")
FLOWER_DB = os.path.join(BASE_DB, "flower_db")
WEB_SEARCH_DB = os.path.join(BASE_DB, "web_search_db")
VOICE_DB = os.path.join(BASE_DB, "voice_db")

# ---- Uploads ----
MAX_FILE_SIZE_MB = 2
ALLOWED_PDF_EXTENSIONS = {"pdf"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}

# ---- Voice (Whisper STT + Piper TTS — both free, fully local, no API key) ----
VOICE_DIR = "voices"
PIPER_VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Spoken language -> Whisper language code (forces correct transcription script)
# and the single Piper voice offered for that language, keyed by "lang_COUNTRY-voice-quality".
VOICE_LANGUAGES = {
    "English": {
        "stt_code": "en",
        "voice_key": "en_US-lessac-medium",
        "voice_label": "Lessac",
    },
    "Hindi": {
        "stt_code": "hi",
        "voice_key": "hi_IN-pratham-medium",
        "voice_label": "Pratham",
    },
}
DEFAULT_VOICE_LANGUAGE = "English"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
WHISPER_MODEL_SIZE = "base"  # faster-whisper: tiny / base / small / medium

# ---- Web search (DuckDuckGo via ddgs — free, no API key) ----
WEB_SEARCH_MAX_RESULTS = 8

# ---- Multi-user demo deployment (e.g. a shared free Hugging Face Space) ----
# Streamlit has no reliable public "the user closed the tab" hook, so abandoned
# sessions are instead swept up lazily once they've been idle this long.
SESSION_TTL_MINUTES = 30
# Soft per-session cap so one visitor can't exhaust the shared free-tier storage;
# plenty for a few chat turns plus one modest PDF upload.
SESSION_MAX_MB = 20
