import base64

import streamlit as st

from core.chat_store import clear_history, save_message
from core.config import FLOWER_DB
from core.flower_index import build_index, get_chroma_collection, get_embedder
from core.llm import describe_llm_error, stream_text
from core.prompts import FLOWER_PROMPT
from core.session import charge_session, get_session_id
from utils.ui import page_header, require_provider

st.set_page_config(page_title="Flower Image Search", page_icon="🌸")
require_provider()

CHAT_DB_NAME = "flower_chat"
session_id = get_session_id()


def _clear_chat():
    clear_history(FLOWER_DB, CHAT_DB_NAME, session_id=session_id)
    st.session_state.flower_chat_history = []


page_header(
    "🌸 Flower Image Search",
    "Search a built-in library of flower photos by plain-text description "
    "(OpenCLIP multimodal embeddings, free and fully local) — then get AI "
    "bouquet arrangement ideas grounded in the matched photos.",
    on_clear=_clear_chat,
)


@st.cache_resource(show_spinner="Loading the flower search model...")
def get_embedder_cached():
    return get_embedder()


@st.cache_resource(show_spinner="Preparing the flower photo library (first run only)...")
def get_collection_cached():
    # Normally a no-op: the index is meant to be pre-built once at deploy time
    # via `python scripts/build_flower_index.py`, so real visitors never wait
    # through a dataset download — this is just a safety-net fallback.
    build_index()
    return get_chroma_collection()


try:
    embedder = get_embedder_cached()
    collection = get_collection_cached()
except Exception as e:
    st.error(f"Couldn't prepare the flower search index: {e}")
    st.stop()

if collection.count() == 0:
    st.warning(
        "The flower photo index hasn't been built yet. Run "
        "`python scripts/build_flower_index.py` once, then reload this page."
    )
    st.stop()

if "flower_chat_history" not in st.session_state:
    st.session_state.flower_chat_history = []


def render_turn(role: str, content: str, images_b64: list[str] | None) -> None:
    with st.chat_message(role):
        if images_b64:
            cols = st.columns(len(images_b64))
            for col, img_b64 in zip(cols, images_b64):
                with col:
                    st.image(base64.b64decode(img_b64))
        st.markdown(content)


for turn in st.session_state.flower_chat_history:
    render_turn(turn["role"], turn["content"], turn.get("extra"))

query = st.chat_input(
    "Describe the flowers you're looking for (e.g. 'pink flower with yellow center', 'red rose bouquet'):"
)

if query and charge_session(len(query.encode())):
    st.session_state.flower_chat_history.append({"role": "user", "content": query})
    save_message(FLOWER_DB, CHAT_DB_NAME, "user", query, session_id=session_id)
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching the flower library..."):
        query_embedding = embedder([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=4, include=["documents"]
        )
    matched_ids = results["ids"][0]
    matched_images = results["documents"][0]

    with st.chat_message("assistant"):
        if not matched_images:
            answer = "No matching flowers found in the library."
            st.markdown(answer)
        else:
            cols = st.columns(len(matched_images))
            for col, img_b64 in zip(cols, matched_images):
                with col:
                    st.image(base64.b64decode(img_b64))

            try:
                content = [
                    {
                        "type": "text",
                        "text": (
                            "What are some good ideas for a bouquet arrangement "
                            f"using flowers like these? Query: {query}"
                        ),
                    }
                ]
                for img_b64 in matched_images[:2]:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        }
                    )

                messages = [
                    {"role": "system", "content": FLOWER_PROMPT},
                    {"role": "user", "content": content},
                ]
                with st.spinner("Dreaming up bouquet ideas..."):
                    answer = st.write_stream(stream_text(messages)) or "(no answer)"
            except Exception as e:
                answer = describe_llm_error(e)
                st.markdown(answer)

    charge_session(len(answer.encode()))
    # Keep the actual image bytes only for this run's live redraw; persist just
    # the lightweight ids to the durable chat log (never read back anyway).
    st.session_state.flower_chat_history.append(
        {"role": "assistant", "content": answer, "extra": matched_images}
    )
    save_message(FLOWER_DB, CHAT_DB_NAME, "assistant", answer, extra=matched_ids, session_id=session_id)



