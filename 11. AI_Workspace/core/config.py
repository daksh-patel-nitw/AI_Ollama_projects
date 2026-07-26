import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# Model Configuration
# ==========================

LLM_NAME = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"

TEMPERATURE = 0

MAX_TOKENS = 1024

# ==========================
# Chroma Paths
# ==========================

BASE_DB = "data"

WEBSITE_DB = os.path.join(BASE_DB, "website_db")

PDF_DB = os.path.join(BASE_DB, "pdf_db")

FLOWER_DB = os.path.join(BASE_DB, "flower_db")

VOICE_DB = os.path.join(BASE_DB, "voice_db")

# ==========================
# Uploads
# ==========================

UPLOAD_FOLDER = "uploads"

MAX_FILE_SIZE_MB = 10

ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

# ==========================
# App
# ==========================

APP_NAME = "AI Workspace"

VERSION = "1.0.0"