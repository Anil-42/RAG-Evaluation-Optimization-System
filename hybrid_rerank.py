from hybrid import hybrid_retrieve_chunks
from reranker import rerank

from chunking import chunk_text

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
        k=10
    )

# -----------------------gpt testing---------------------
    # print("\nQUESTION:", question_text)
    # print("\nRERANKER RESULTS:")

    # for rank, (chunk, score) in enumerate(reranked, start=1):
    #     print("Rank:", rank)
    #     print("Score:", score)
    #     print("Chunk index:", hybrid_chunks.index(chunk))
    #     print("-" * 50)

    # for rank, (chunk, score) in enumerate(reranked, start=1):
    #     print(
    #         "Reranker rank:", rank,
    #         "| Hybrid candidate position:", hybrid_chunks.index(chunk),
    #         "| Score:", score
    #     )

    # chunks = chunk_text(text)

    # for index in [15, 23, 181, 22]:
    #     print(f"\n========== CHUNK {index} ==========")
    #     print(chunks[index])
# -------------------------------------------------------

# ---------------------------------testing--------------------------
    # for rank,index in enumerate(chunks_index, start=1):
    #     print("Rank:", rank)
    #     print("chunk index:", index)
    #     # print("chunk:", chunks[index])
    #     print("-"*50)

    # reranked = reranked[:10]
# ------------------------------------------------------------------

    # ------------------------------------------------
    # Final top 10 chunks
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