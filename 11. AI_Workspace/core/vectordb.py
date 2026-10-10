from langchain_community.vectorstores import Chroma

from core.embeddings import get_embeddings


def load_vectorstore(db_path: str) -> Chroma:
    return Chroma(persist_directory=db_path, embedding_function=get_embeddings())
