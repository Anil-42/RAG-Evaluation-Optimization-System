from hybrid import hybrid_retrieve_chunks
from reranker import rerank
from neighbor_merge import merge_neighbors


def hybrid_rerank_retrieve(text, question_text):

    hybrid_chunks, hybrid_scores, hybrid_indices, chunks = hybrid_retrieve_chunks(
        text,
        question_text
    )
# -------------------------------------------------------------
    # ---------------------------------------------------
    # Merging before reranking
    # ---------------------------------------------------

    # # Merge neighboring chunks
    # merged_candidates = merge_neighbors(
    #     hybrid_indices,
    #     chunks
    # )

    # # Extract only the merged text
    # merged_chunks = [
    #     item[2]
    #     for item in merged_candidates
    # ]
# -------------------------------------------------------------


    # Rerank contexts
    reranked = rerank(
        question_text,
        hybrid_chunks,
        hybrid_indices,
        k=3
    )


    # --------------------------------------------
    # Merge to 3 after reranking
    # --------------------------------------------

    # Get the original chunk indices of FINAL TOP 3
    final_indices = [index for chunk, index, score in reranked]

    # Now expand/merge their neighbors
    merged_candidates = merge_neighbors(
        final_indices,
        chunks
    )


    final_chunks = [
        merged_text
        for start, end, merged_text in merged_candidates
    ]

    final_scores = [
        score
        for chunk, index, score in reranked
    ]

    # --------------------------testing----------------------------
    # print("\nTop 3 reranked chunk indices:",hybrid_indices)
    # print("\nMerged neighbor chunk final indices:",final_indices)
    # print("\nMerged text:",final_chunks)
    # print("-"*50)
    # -------------------------------------------------------------

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

