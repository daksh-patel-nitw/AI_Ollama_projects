"""Builds the flower photo search index: embeds each image with OpenCLIP and
stores the embedding + the image's own JPEG bytes in a local Chroma collection.

Storing the image alongside its embedding means nothing else needs to keep the
source Hugging Face dataset around at runtime — run build_index() once (e.g. via
scripts/build_flower_index.py at deploy time) and the app only ever touches the
already-built Chroma collection afterward, with no dataset download on demand.
"""

import base64
import io

from core.config import FLOWER_DB

COLLECTION_NAME = "flowers_openclip_idx_v2"
DATASET_NAME = "huggan/flowers-102-categories"
# Keep indexing time reasonable; raise this for a larger searchable library.
DATASET_SAMPLE_SIZE = 300


def get_embedder():
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

    return OpenCLIPEmbeddingFunction()


def get_chroma_collection():
    import chromadb

    client = chromadb.PersistentClient(path=FLOWER_DB)
    return client.get_or_create_collection(COLLECTION_NAME)


def build_index(sample_size: int = DATASET_SAMPLE_SIZE, embedder=None) -> int:
    """Download the dataset, embed `sample_size` images, and store each one (with
    its own JPEG bytes) in Chroma. Returns the collection's resulting row count.
    Safe to call repeatedly — a no-op once the collection already has data.
    """
    collection = get_chroma_collection()
    if collection.count() > 0:
        return collection.count()

    import numpy as np
    from datasets import load_dataset

    embedder = embedder or get_embedder()
    ds_dict = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
    ds = ds_dict[split_name]
    image_key = "image" if "image" in ds.column_names else "img"

    sample_size = min(sample_size, len(ds))
    ids, embeddings, documents, metadatas = [], [], [], []
    for i in range(sample_size):
        image = ds[i][image_key].convert("RGB")
        embedding = embedder([np.array(image)])[0]

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        ids.append(str(i))
        embeddings.append(embedding.tolist())
        documents.append(image_b64)
        metadatas.append({"index": i})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return collection.count()
