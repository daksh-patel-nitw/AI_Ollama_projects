from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

name="Daksh"

emb=default_ef(name)

print(f"Embedding for '{name}': {emb}")