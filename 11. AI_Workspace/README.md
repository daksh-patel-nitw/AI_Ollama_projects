---
title: AI Workspace
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.65.0"
app_file: app.py
pinned: false
---

# AI Workspace

One Streamlit app, four chatbots, your choice of model:

- **Home** — pick a model provider (Ollama / OpenAI / Gemini) and enter credentials if needed.
- **Web Search Agent** — a tool-calling agent grounded in live DuckDuckGo results (free, no key).
- **PDF Q&A** — upload a PDF, ask questions answered strictly from its content.
- **Flower / Image Search** — search a built-in flower photo library by description, get bouquet ideas.
- **Voice Assistant** — speak or type, get a transcript, a reply, and a spoken answer back.

All four chatbots share the same chat UI: a scrolling message history, a chat box at the
bottom, and a **Clear chat history** button in the sidebar. Conversations are persisted to a
local Chroma collection per tool, so they survive app restarts until you clear them.

### 🌐 Web Search Agent

A tool-calling agent (`langchain.agents.create_agent`) with a single `web_search` tool backed
by DuckDuckGo (via `ddgs` — free, no API key). It's instructed to always search before
answering anything current (news, prices, scores, schedules), and to issue multiple searches
if one isn't enough to cover the question. Replies stream in token-by-token, and every answer
cites which search result(s) it drew from.

### 📄 PDF Q&A

Upload a PDF from the sidebar and it's chunked, embedded (local `fastembed` model — no API
key, works regardless of which chat provider you picked), and indexed into a dedicated Chroma
collection. Ask questions in the chat box and answers are generated strictly from the
retrieved passages (shown in an expander under each reply) — if the answer isn't in the PDF,
the assistant says so rather than guessing. Only one PDF is active at a time: uploading a
different file prompts a confirmation first, since it clears the current chat history and
indexed content.

### 🌸 Flower Image Search

A multimodal RAG pipeline: a sample of the Oxford Flowers-102 dataset (via Hugging Face
`datasets`) is embedded with OpenCLIP (fully local, no API key) and stored — embedding **and**
the image's own JPEG bytes together — in a local Chroma collection via
`python scripts/build_flower_index.py`, a one-time step meant to run **at deploy time**, not
on a visitor's first request — so nobody has to wait through a dataset download + embedding
pass, and the running app never needs to keep the source dataset around at all afterward.
Describe what you're looking for (e.g. "pink flower with yellow center") and the
closest-matching photos come back by vector similarity. The matched images are then passed to
your chosen chat provider's vision capability to generate bouquet arrangement suggestions
grounded in what was actually found (the photos aren't labeled with species names, so the
assistant describes what it sees rather than guessing a cultivar) — the retrieval step
(OpenCLIP) is independent of which model does that generation step, so this same pipeline
works with Gemini, OpenAI, or an Ollama vision model equally.

### 🎙️ Voice Assistant

Type in the chat box or tap its built-in mic to record a question — speech-to-text runs
locally via `faster-whisper`. Pick a **Language** in the sidebar (English or Hindi); the
reply is generated in that language and spoken aloud via a matching Piper voice (Lessac for
English, Pratham for Hindi), fully local and free. The voice model downloads/loads in the
background as soon as the page opens, so there's no delay when you first ask a question —
the reply text and its spoken audio appear together, autoplaying immediately.

Embeddings for PDF/text search are always a free local ONNX model (fastembed), independent
of whichever chat provider you pick — so PDF Q&A and image search work even if you choose
OpenAI or Gemini for chat without handing over a second key just for embeddings.

## Setup

### Multi-user sessions (for a shared free-tier deployment)

Each browser session gets its own id, used to isolate its chat history and any uploaded
PDF content from every other visitor sharing the same deployment — nobody can see another
visitor's data. Since Streamlit has no reliable public "the user closed the tab" hook, data
isn't wiped the instant someone leaves; instead each session has a small storage quota
(`SESSION_MAX_MB` in `core/config.py`, default 20MB — plenty for a few chat turns plus one
modest PDF) and a background sweep purges anything idle past `SESSION_TTL_MINUTES` (default
30) the next time any page loads. Refreshing the page always starts a brand new, empty
session. If a session hits its quota, a message asks the visitor to refresh to continue.

**Use Python 3.11 or 3.12** (not whatever the system default is) in a fresh virtual
environment — several deps here (`chromadb`, `onnxruntime`, `faster-whisper`, `piper-tts`,
`torch`) lag behind the newest Python releases and may not have wheels yet.

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional — only if pre-filling OpenAI/Gemini keys
python scripts/build_flower_index.py   # one-time: pre-builds Flower Search's index
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

### Running Ollama for free on Google Colab (temporary, ~1-2hr sessions)

You can't install Ollama on some machines (or want a GPU for free) — Colab works as a
throwaway Ollama server for a session. In a Colab notebook cell:

```bash
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > ollama.log 2>&1 &
!sleep 3 && ollama pull llama3.2
```

Then expose port 11434 publicly with a tunnel (pick one):

```bash
# Option A: cloudflared (no account needed)
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:11434
```

```python
# Option B: pyngrok (needs a free ngrok authtoken)
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
print(ngrok.connect(11434, "http"))
```

Either prints a public `https://...` URL — paste it into **Ollama server URL** on this
app's Home page. The Colab runtime (and tunnel) disappears when the session ends, so
this is only good for short testing windows, not a permanent deployment.

## Known caveats

- `piper-tts`'s maintainer handoff (archived `rhasspy/piper` → `OHF-Voice/piper1-gpl`) also
  changed its license to **GPL-3.0-or-later** (was MIT) and may have shifted its exact
  Python API — `core/tts.py` has a defensive fallback, but test it once before relying on it.
- `ddgs` (formerly `duckduckgo-search`) scrapes DuckDuckGo rather than calling an official
  API — expect occasional rate-limit errors under heavy use.
- First run of Flower Search and Voice Assistant downloads model weights (OpenCLIP ~600MB,
  Piper voice ~60MB) — expect a slow first load, then cached.
