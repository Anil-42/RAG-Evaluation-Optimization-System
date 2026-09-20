from hybrid import hybrid_retrieve_chunks
from reranker import rerank
from neighbor_merge import merge_neighbors

from evaluation import evidence_coverage


def hybrid_rerank_retrieve(text, question_text, evidence_points):

    hybrid_chunks, hybrid_scores, hybrid_indices, chunks = hybrid_retrieve_chunks(
        text,
        question_text,
    )




    # Rerank contexts
    reranked = rerank(
        question_text,
        hybrid_chunks,
        hybrid_indices,
        k=3
    )


    # --------------------------------------------
    # Merge top 3 after reranking
    # --------------------------------------------

    # # Get the original chunk indices of FINAL TOP 3
    # final_indices = [index for chunk, index, score in reranked]

    # # Now expand/merge their neighbors
    # merged_candidates = merge_neighbors(
    #     final_indices,
    #     chunks
    # )


    # final_chunks = [
    #     merged_text
    #     for start, end, merged_text in merged_candidates
    # ]

    # final_scores = [
    #     score
    #     for chunk, index, score in reranked
    # ]

# ----------------------------Hybrid Reranking----------------------------
    final_chunks = [
        chunk
        for chunk, index, sores in reranked
    ]

    final_scores = [
        score
        for chunk, index, score in reranked
    ]

    final_indices = [
        index
        for chunk, index, score in reranked
    ]

# ------------------------------------------------------------------------



    return final_chunks, final_scores, hybrid_indices, final_indices


