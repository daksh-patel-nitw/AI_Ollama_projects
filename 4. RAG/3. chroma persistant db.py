import chromadb

from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
croma_client = chromadb.PersistentClient(path="./db/chroma_persistent_db")

documents=[
  {
    "id": "doc1",
    "text": "Daksh is a studious and hardworking student who is always eager to learn new things. He is known for his dedication to his studies and his ability to grasp complex concepts quickly. Daksh is also a team player, often collaborating with his classmates on group projects and helping them understand difficult topics. His teachers appreciate his enthusiasm and commitment to his education."
  },
  {
    "id": "doc2",
    "text": "Daksh likes to play cricket in his free time and is a member of his school's cricket team. He enjoys the sport and finds it to be a great way to stay active and relieve stress. Daksh's teammates admire his sportsmanship and his ability to stay focused during matches."
  },
  {
    "id": "doc3",
    "text": "Daksh like to eat gujarati food and is particularly fond of dishes like dhokla, thepla, and khakhra. He enjoys trying out different recipes at home and often helps his family in the kitchen. Daksh's love for Gujarati cuisine has also made him a popular figure among his friends, who often seek his recommendations for the best local eateries."
  }
]

collection_name="daksh_story"

collection=croma_client.get_or_create_collection(name=collection_name, embedding_function=default_ef)

for doc in documents:
    collection.upsert(
        documents=[doc["text"]],
        ids=[doc["id"]]
    )
    
query_text="Tell me about Daksh's food"
results=collection.query(
    query_texts=[query_text],
    n_results=2
)

for idx,document in enumerate(results['documents'][0]):
    document_id = results['ids'][0][idx]
    distance = results['distances'][0][idx]
    print(f"{document_id}: Text: {document}\n(Distance: {distance} )")
    print()
