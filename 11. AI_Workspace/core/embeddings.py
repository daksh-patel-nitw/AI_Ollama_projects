from langchain_community.embeddings import OllamaEmbeddings
from core.config import EMBEDDING_MODEL


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )