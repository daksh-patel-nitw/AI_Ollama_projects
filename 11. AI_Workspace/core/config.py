import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "AI Workspace"
VERSION = "2.0.0"

# ---- Model providers (chosen by the user on the Home page) ----
PROVIDERS = ["Ollama", "OpenAI", "Gemini"]

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

TEMPERATURE = 0.3

# ---- Embeddings: always free/local, independent of the chosen chat provider,
# so PDF/image search still work even if the user picks OpenAI or Gemini for chat
# without handing over a second key just for embeddings. ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # served locally via fastembed (ONNX, CPU)

# ---- Chroma persistence paths ----
BASE_DB = "data"
PDF_DB = os.path.join(BASE_DB, "pdf_db")
FLOWER_DB = os.path.join(BASE_DB, "flower_db")

# ---- Uploads ----
MAX_FILE_SIZE_MB = 10
ALLOWED_PDF_EXTENSIONS = {"pdf"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}

# ---- Voice (Whisper STT + Piper TTS — both free, fully local, no API key) ----
VOICE_DIR = "voices"
PIPER_VOICE_NAME = "en_US-lessac-medium"
PIPER_VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
)
WHISPER_MODEL_SIZE = "base"  # faster-whisper: tiny / base / small / medium

# ---- Web search (DuckDuckGo via ddgs — free, no API key) ----
WEB_SEARCH_MAX_RESULTS = 5
