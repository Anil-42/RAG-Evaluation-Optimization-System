from src.chunking.chunking import chunk_text
from src.retrieval.embedding import get_embeddings
from src.retrieval.retrieve import retrieve


def fixed_vector_retrieve(text, question_text):
    # Fixed-size chunking
    chunks = chunk_text(text)

    # Create embeddings for chunks
    embeddings = get_embeddings(chunks)

    # Vector retrieval
    retrieved_chunks, retrieved_scores = retrieve(
        question_text,
        chunks,
        embeddings
    )

    return retrieved_chunks, retrieved_scores