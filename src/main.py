
from document_loader import load_documents
from text_splitter import split_documents
from EmbeddingManager import EmbeddingManager
from VectorStore import VectorStore
from Generator import Generator


def initialize_rag():
    """Load documents and initialize the RAG components."""

    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents=documents)
    print(f"Number of chunks: {len(chunks)}")

    texts = [chunk.page_content for chunk in chunks]

    print("Generating embeddings...")
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.generate_embeddings(texts)

    print("Initializing vector store...")
    vector_store = VectorStore(embedding_manager)
    vector_store.add_documents(texts, embeddings)

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
