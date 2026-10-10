# AI Workspace

One Streamlit app, four tools, your choice of model:

- **Home** — pick a model provider (Ollama / OpenAI / Gemini) and enter credentials if needed.
- **Web Search Agent** — a LangGraph ReAct agent grounded in live DuckDuckGo results (free, no key).
- **PDF Q&A** — upload a PDF, ask questions answered strictly from its content.
- **Flower / Image Search** — upload images, then find them again with a text description (OpenCLIP).
- **Voice Assistant** — speak, get a transcript (Whisper), a reply, and a spoken answer back (Piper).

Embeddings for PDF/text search are always a free local ONNX model (fastembed), independent
of whichever chat provider you pick — so PDF Q&A and image search work even if you choose
OpenAI or Gemini for chat without handing over a second key just for embeddings.

## Setup

**Use Python 3.11 or 3.12** (not whatever the system default is) in a fresh virtual
environment — several deps here (`chromadb`, `onnxruntime`, `faster-whisper`, `piper-tts`,
`torch`) lag behind the newest Python releases and may not have wheels yet.

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional — only if pre-filling OpenAI/Gemini keys
streamlit run app.py
```

If you only want to use Ollama locally, you can skip installing `langchain-openai` /
`langchain-google-genai` — they're imported lazily and only needed if you pick that provider.

## Hosting for free

This app is designed to actually run once deployed, not just locally:

- **Ollama only works if you also host/expose an Ollama server** — most free hosts won't
  run it for you. For a public demo, use the OpenAI or Gemini option instead (visitors
  paste their own key into the Home page; nothing is stored server-side).
- **Hugging Face Spaces** (Streamlit SDK, free CPU tier) is the best fit here — generous
  RAM (~16GB) and persistent storage for the Piper voice model and OpenCLIP weights,
  which Streamlit Community Cloud's free tier (1GB) may struggle with.
- **Streamlit Community Cloud** is the simplest one-click deploy from a GitHub repo if
  you drop the heavier Flower Search / Voice Assistant pages, or accept slower cold starts.
- **Render** (free web service) works too, but sleeps after inactivity and cold-starts slowly.

## Known caveats

- `piper-tts`'s maintainer handoff (archived `rhasspy/piper` → `OHF-Voice/piper1-gpl`) also
  changed its license to **GPL-3.0-or-later** (was MIT) and may have shifted its exact
  Python API — `core/tts.py` has a defensive fallback, but test it once before relying on it.
- `ddgs` (formerly `duckduckgo-search`) scrapes DuckDuckGo rather than calling an official
  API — expect occasional rate-limit errors under heavy use.
- First run of Flower Search and Voice Assistant downloads model weights (OpenCLIP ~600MB,
  Piper voice ~60MB) — expect a slow first load, then cached.
