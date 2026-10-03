from src.chunking.semantic_chunking import split_sentences
from src.chunking.semantic_chunking import semantic_chunking
from src.retrieval.embedding import get_embeddings
from src.retrieval.retrieve import retrieve


def create_semantic_chunks(text):

    # Split document into sentences
    sentences = split_sentences(text)

    # Create sentence embeddings
    sentence_embeddings = get_embeddings(sentences)

    # Our tested semantic threshold
    threshold = 0.222

    # Create semantic chunks
    chunks = semantic_chunking(
        sentences,
        sentence_embeddings,
        threshold
    )

    return chunks


def semantic_vector_retrieve(text, question_text):

    chunks = create_semantic_chunks(text)

    # Create embeddings for semantic chunks
    embeddings = get_embeddings(chunks)

    # Vector retrieval
    retrieved_chunks, retrieved_scores = retrieve(
        question_text,
        chunks,
        embeddings
    )
    
    return retrieved_chunks, retrieved_scores