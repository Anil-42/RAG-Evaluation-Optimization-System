import numpy as np

from chunking import chunk_text
from embedding import get_embeddings
from retrieve import retrieve
from bm25_scores import bm25_scores
from normalize_scores import normalize_scores


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

    hybrid_score = [
        hybrid_scores[i]
        for i in candidate_indices
    ]

    hybrid_indices = [
    int(i)
    for i in candidate_indices
]


# -------------------------------testing--------------------------
    # print(question_text)
    # print("Hybrid retrieval top 10 candidates:")
    # for rank, index in enumerate(candidate_indices[:10], start=1):
    #     print("Rank:", rank)
    #     print("chunk index:", index)
    #     # print("hybrid score:", hybrid_scores[index])
    #     print("-"*50)
    # print("\nHybrid Reranking retrieval top 10 candidates:")


    # chunks_index = [chunks.index(chunks[i]) for i in candidate_indices]

    # return hybrid_chunks, hybrid_score, chunks_index, chunks
# ----------------------------------------------------------------

    return hybrid_chunks, hybrid_score, hybrid_indices, chunks
    # return hybrid_chunks, hybrid_score