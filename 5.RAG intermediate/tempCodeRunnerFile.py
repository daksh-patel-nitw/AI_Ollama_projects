import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv


# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


class EmbeddingModel:
    def __init__(self, model_type="nomic"):
        self.model_type = model_type
        if model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()


class LLMModel:
    def __init__(self, model_type="ollama"):
        self.model_type = model_type
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


def generate_csv():
    facts = [
        {
            "id": 1,
            "fact": "Gujarat is the home state of Mahatma Gandhi, the leader of India's freedom movement.",
        },
        {
            "id": 2,
            "fact": "The Statue of Unity in Gujarat is the world's tallest statue, standing at 182 meters.",
        },
        {
            "id": 3,
            "fact": "The Gir Forest National Park in Gujarat is the only natural habitat of the Asiatic lion.",
        },
        {
            "id": 4,
            "fact": "The White Rann of Kutch is one of the world's largest salt deserts and hosts the famous Rann Utsav festival.",
        },
        {
            "id": 5,
            "fact": "Lothal, an ancient city of the Indus Valley Civilization, is located in Gujarat and is known for its dockyard.",
        },
        {
            "id": 6,
            "fact": "Gujarat has the longest coastline of any Indian state, stretching for about 1,600 kilometers.",
        },
        {
            "id": 7,
            "fact": "Dwarka in Gujarat is one of the four sacred Char Dham pilgrimage sites of Hinduism.",
        },
        {
            "id": 8,
            "fact": "The Somnath Temple in Gujarat is one of the twelve Jyotirlingas dedicated to Lord Shiva.",
        },
        {
            "id": 9,
            "fact": "Ahmedabad became India's first UNESCO World Heritage City in 2017.",
        },
        {
            "id": 10,
            "fact": "Navratri celebrations in Gujarat are famous worldwide for traditional Garba and Dandiya dances.",
        },
        {
            "id": 11,
            "fact": "Gandhinagar is capital of gujarat and is known for its well-planned architecture and the Akshardham Temple.",
        },
    ]
    with open("gujarat_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)

    return facts


def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        client.delete_collection("gujarat_facts")
    except:
        pass

    collection = client.create_collection(
        name="gujarat_facts", embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    return collection


def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)
    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            ),
        )
    )


def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def rag_pipeline(query, collection, llm_model, top_k=2):
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about Gujarat but only answers questions that are directly related to the sources/documents given.",
            },
            {"role": "user", "content": augmented_prompt},
        ]
    )

    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt


def streamlit_app():
    st.set_page_config(page_title="Gujarat Facts RAG", layout="wide")
    st.title("Gujarat Facts RAG System")

    st.sidebar.title("Model Configuration")

    llm_type = "ollama"

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["chroma", "nomic"],
        format_func=lambda x: {
            "nomic": "Nomic Embed Text (Ollama)",
            "chroma": "Chroma Default",
        }[x],
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.facts = generate_csv()

        # Initialize models
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        # Setup ChromaDB
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # If models changed, reinitialize
    if (
        st.session_state.llm_model.model_type != llm_type
        or st.session_state.embedding_model.model_type != embedding_type
    ):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )

    # Display available facts
    with st.expander(" Available Gujarat Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact['fact']}")

    # Query input
    query = st.text_input(
        "Enter your question about Gujarat:",
        placeholder="e.g., What is the capital of Gujarat?",
    )

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Response")
                st.write(response)

            with col2:
                st.markdown("### 📖 References Used")
                for ref in references:
                    st.write(f"- {ref}")

            # Show technical details in expander
            with st.expander("🔍 Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")


if __name__ == "__main__":
    streamlit_app()
