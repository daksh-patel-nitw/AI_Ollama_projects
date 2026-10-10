import os
import uuid

import streamlit as st

from core.config import FLOWER_DB
from utils.ui import page_header, require_provider

st.set_page_config(page_title="Flower / Image Search", page_icon="🌸")
require_provider()

page_header(
    "🌸 Flower / Image Search",
    "Upload a few images to build your own library, then find them again with a "
    "plain-text description — powered by OpenCLIP multimodal embeddings (free, "
    "fully local, no API key).",
)


@st.cache_resource(show_spinner="Loading local multimodal (OpenCLIP) model — first run is slower...")
def get_collection():
    import chromadb
    from chromadb.utils.data_loaders import ImageLoader
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

    client = chromadb.PersistentClient(path=FLOWER_DB)
    return client.get_or_create_collection(
        "image_library",
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader(),
    )


UPLOAD_DIR = os.path.join(FLOWER_DB, "_uploaded_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)

collection = get_collection()

uploaded_images = st.file_uploader(
    "Upload images to add to your searchable library",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_images and st.button("Add to library"):
    ids, uris, metadatas = [], [], []
    for img in uploaded_images:
        img_id = str(uuid.uuid4())
        path = os.path.join(UPLOAD_DIR, f"{img_id}_{img.name}")
        with open(path, "wb") as f:
            f.write(img.getvalue())
        ids.append(img_id)
        uris.append(path)
        metadatas.append({"filename": img.name})
    collection.add(ids=ids, uris=uris, metadatas=metadatas)
    st.success(f"Added {len(uploaded_images)} image(s) to your library.")

st.divider()
query = st.text_input(
    "Describe what you're looking for (e.g. 'a red rose', 'yellow sunflower'):"
)

if query:
    if collection.count() == 0:
        st.warning("Your library is empty — upload some images above first.")
    else:
        results = collection.query(
            query_texts=[query],
            n_results=min(4, collection.count()),
            include=["metadatas", "distances", "uris"],
        )
        st.markdown("### Matches")
        matched = results["uris"][0]
        cols = st.columns(len(matched) or 1)
        for col, uri, meta, dist in zip(
            cols, matched, results["metadatas"][0], results["distances"][0]
        ):
            with col:
                st.image(uri, caption=f"{meta.get('filename', '')} (distance: {dist:.3f})")
