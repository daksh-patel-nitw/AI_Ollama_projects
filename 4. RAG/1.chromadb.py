import chromadb
chromadb_client = chromadb.Client()

collection_name="test"

collection =chromadb_client.get_or_create_collection(name=collection_name)

documents=[
    {
        "id": "1",
        "text": "This is the first document."
    },
    {
        "id": "2",
        "text": "This is the second document."
    },
    {
        "id": "3",
        "text": "This is the third document."
    }
]

for doc in documents:
    collection.upsert(ids=doc["id"],documents=doc["text"])

query=" first document is the one."

results = collection.query(
    query_texts=[query],
    n_results=10
)

for idx,document in enumerate(results['documents'][0]):
    document_id = results['ids'][0][idx]
    distance = results['distances'][0][idx]
    print(f"{idx+1} Distance: {distance} Text: {document}")
    print()

