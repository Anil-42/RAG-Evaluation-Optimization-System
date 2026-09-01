from hybrid import hybrid_retrieve_chunks
from reranker import rerank


def hybrid_rerank_retrieve(text, question_text):

    # ------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------

    hybrid_chunks, hybrid_scores = hybrid_retrieve_chunks(
        text,
        question_text
    )

    # ------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------

    reranked = rerank(
        question_text,
        hybrid_chunks,
        k=3
    )

    # ------------------------------------------------
    # Final top 3 chunks
    # ------------------------------------------------

    final_chunks = [
        chunk
        for chunk, score in reranked
    ]

    final_scores = [
        score
        for chunk, score in reranked
    ]

    return final_chunks, final_scores