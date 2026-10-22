
from .document_loader import load_documents
from .text_splitter import split_documents
from .EmbeddingManager import EmbeddingManager
from .VectorStore import VectorStore
from .Generator import Generator


def initialize_rag():
    """Load the existing vector store and initialize the RAG components."""

    print("Initializing embedding manager...")
    embedding_manager = EmbeddingManager()

    print("Loading existing vector store...")
    vector_store = VectorStore(embedding_manager)

    count = vector_store.collection.count()
    print(f"Existing chunks in ChromaDB: {count}")

    if count == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Index your documents first."
        )

    print("Initializing generator...")
    generator = Generator()

    print("RAG pipeline initialized successfully.")

    return vector_store, generator



def answer_question(query, vector_store, generator):
    """Retrieve relevant chunks and generate an answer."""

    results = vector_store.search(query)

    retrieved_chunks = (
        results["documents"][0]
        if results.get("documents")
        else []
    )

    if not retrieved_chunks:
        return "No relevant chunks found for this question."

    answer = generator.generate(query, retrieved_chunks)
    return answer
