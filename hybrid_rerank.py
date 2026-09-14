from hybrid import hybrid_retrieve_chunks
from reranker import rerank
from neighbor_merge import merge_neighbors


def hybrid_rerank_retrieve(text, question_text):

    hybrid_chunks, hybrid_scores, hybrid_indices, chunks = hybrid_retrieve_chunks(
        text,
        question_text
    )

    # ---------------------------------------------------
    print("question:",question_text)
    print("\nHYBRID CANDIDATES:")

    for rank, (index, score) in enumerate(
        zip(hybrid_indices, hybrid_scores), start=1
    ):
        print("\nHybrid Rank:", rank)
        print("Original chunk index:", index)
        print("Hybrid score:", score)
        print("Text:")
        print(chunks[index][:500])
    print("-"*50)
    # ---------------------------------------------------

    # Merge neighboring chunks
    merged_candidates = merge_neighbors(
        hybrid_indices,
        chunks
    )

    # Extract only the merged text
    merged_chunks = [
        item[2]
        for item in merged_candidates
    ]



    # print("\nQUESTION:", question_text)
    # print("Hybrid indices:", hybrid_indices)
    # print("Merged candidate count:", len(merged_chunks))

    # Rerank merged contexts
    reranked = rerank(
        question_text,
        merged_chunks,
        k=len(merged_chunks)
    )

# --------------------------------------------------------------------
    # print("question:",question_text)
    # print("\nRERANKED MERGED RESULTS:")

    # for rank, (chunk, score) in enumerate(reranked, start=1):
    #     print("\nRank:", rank)
    #     print("Reranker score:", score)
    #     print("Text:")
    #     print(chunk[:700])
    # print("-"*50)

    # print("\nMERGED CANDIDATES:")
    # for i, item in enumerate(merged_candidates):

    #     start, end, chunk = item

    #     print("\nCandidate:", i)
    #     print("Original chunks:", start, "-", end)
    #     print("Preview:", chunk[:300])
# ---------------------------------------------------------------------

    final_chunks = [
        chunk
        for chunk, score in reranked
    ]

    final_scores = [
        score
        for chunk, score in reranked
    ]

    return final_chunks, final_scores

# ========================================================================================================================

# from hybrid import hybrid_retrieve_chunks
# from reranker import rerank
# from neighbor_expansion import expand_neighbors

# def hybrid_rerank_retrieve(text, question_text):
#     # ------------------------------------------------
#     # Hybrid retrieval
#     # ------------------------------------------------

#     # add chunk_index, chunks for texting the original chunks after reranking, if needed.
#     hybrid_chunks, hybrid_scores, hybrid_indices, chunks = hybrid_retrieve_chunks(
#         text,
#         question_text
#     )

#     # ------------------------------------------------
#     # Neighbor expansion
#     # ------------------------------------------------
#     expanded_indices = expand_neighbors(hybrid_indices,len(chunks))

#     expanded_chunks = [chunks[i] for i in expanded_indices]


#     # # -------------   --------------testing ------------------------------------
#     # print("\nQUESTION:", question_text)
#     # print("Hybrid indices:", hybrid_indices)
#     # print("Expanded indices:", expanded_indices)
#     # print("Hybrid candidate count:", len(hybrid_chunks))
#     # print("Expanded candidate count:", len(expanded_chunks))
#     # # --------------------------------------------------------------------------
    
#     # ------------------------------------------------
#     # Cross-encoder reranking
#     # ------------------------------------------------

#     reranked = rerank(
#         question_text,
#         expanded_chunks,
#         k=len(expanded_chunks)
#     )

# # # -------------------------testing---------------------
# #     print("\nRERANKED RESULTS:")
# #     for rank, (chunk, score) in enumerate(reranked, start=1):

# #         original_index = chunks.index(chunk)

# #         print(
# #             "Rank:", rank,
# #             "| Original chunk:", original_index,
# #             "| Score:", score
# #         )
# #     # --------------------------------------------------------

# # -----------------------gpt testing---------------------
#     # print("\nQUESTION:", question_text  )
#     # print("\nRERANKER RESULTS:")

#     # for rank, (chunk, score) in enumerate(reranked, start=1):
#     #     print("Rank:", rank)
#     #     print("Score:", score)
#     #     print("Chunk index:", hybrid_chunks.index(chunk))
#     #     print("-" * 50)

#     # for rank, (chunk, score) in enumerate(reranked, start=1):
#     #     print(
#     #         "Reranker rank:", rank,
#     #         "| Hybrid candidate position:", hybrid_chunks.index(chunk),
#     #         "| Score:", score
#     #     )

#     # chunks = chunk_text(text)

#     # for index in [15, 23, 181, 22]:
#     #     print(f"\n========== CHUNK {index} ==========")
#     #     print(chunks[index])
# # -------------------------------------------------------

# # ---------------------------------testing--------------------------
#     # for rank,index in enumerate(hybrid_index, start=1):
#     #     print("Rank:", rank)
#     #     print("chunk index:", index)
#         # print("chunk:", chunks[index])
#         # print("-"*50)

#     # reranked = reranked[:10]
# # ------------------------------------------------------------------

#     # ------------------------------------------------
#     # Final top 10 chunks
#     # ------------------------------------------------

#     final_chunks = [
#         chunk
#         for chunk, score in reranked
#     ]

#     final_scores = [
#         score
#         for chunk, score in reranked
#     ]

#     return final_chunks, final_scores

