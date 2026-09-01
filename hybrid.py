import numpy as np

from chunking import chunk_text
from embedding import get_embeddings
from retrieve import retrieve
from bm25_scores import bm25_scores
from hybrid_retrieve import normalize_scores


def hybrid_retrieve_chunks(text, question_text):

    # Fixed-size chunking
    chunks = chunk_text(text)

    # Create embeddings
    embeddings = get_embeddings(chunks)

    # ------------------------------------------------
    # BM25 scores
    # ------------------------------------------------

    bm25_score_values = bm25_scores(
        question_text,
        chunks
    )

    # ------------------------------------------------
    # Vector scores
    # ------------------------------------------------

    vector_score_values = retrieve(
        question_text,
        chunks,
        embeddings
    )

    # ------------------------------------------------
    # Normalize scores
    # ------------------------------------------------

    normalized_bm25 = normalize_scores(
        bm25_score_values
    )

    normalized_vector = normalize_scores(
        vector_score_values
    )

    # ------------------------------------------------
    # Hybrid scoring
    # ------------------------------------------------

    alpha = 0.7

    hybrid_scores = [
        alpha * vector_score + (1 - alpha) * bm25_score
        for vector_score, bm25_score
        in zip(
            normalized_vector,
            normalized_bm25
        )
    ]

    # ------------------------------------------------
    # Select top 10 candidates
    # ------------------------------------------------

    candidate_k = 10

    candidate_indices = np.argsort(
        hybrid_scores
    )[-candidate_k:][::-1]

    hybrid_chunks = [
        chunks[i]
        for i in candidate_indices
    ]

    hybrid_scores = [
        hybrid_scores[i]
        for i in candidate_indices
    ]

    return hybrid_chunks, hybrid_scores