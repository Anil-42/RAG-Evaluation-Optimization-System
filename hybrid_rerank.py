from hybrid import hybrid_retrieve_chunks
from reranker import rerank


def hybrid_rerank_retrieve(text, question_text):

    # ------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------

    # add chunk_index, chunks for texting the original chunks after reranking, if needed.
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

# ---------------------------------testing--------------------------
    # for rank,index in enumerate(chunks_index, start=1):
    #     print("Rank:", rank)
    #     print("chunk index:", index)
    #     print("chunk:", chunks[index])
    #     print("-"*50)

    # reranked = reranked[:10]
# ------------------------------------------------------------------

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