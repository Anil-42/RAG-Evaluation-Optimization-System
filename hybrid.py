import numpy as np

from chunking import chunk_text
from embedding import get_embeddings
from retrieve import retrieve
from bm25_scores import bm25_scores
from normalize_scores import normalize_scores
from semantic_vector import create_semantic_chunks

from normalize import normalize


def hybrid_retrieve_chunks(text, question_text, evidence_points=None, alpha=0.2):



    # Fixed-size chunking
    # chunks = chunk_text(text)

    # Create semantic chunks
    chunks = create_semantic_chunks(text)


    


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

    # vector_score_values = retrieve(
    #     question_text,
    #     chunks,
    #     embeddings
    # )

    # ------------------------------------------------
    # Semantic scores
    # ------------------------------------------------

    # for semantic chunking
    semantic_score_values = retrieve(
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

    ## for vector chunking
    # normalized_vector = normalize_scores(
    #     vector_score_values
    # )

    # for semantic chunksing
    normalized_semantic = normalize_scores(
        semantic_score_values
    )

    # ------------------------------------------------
    # Hybrid scoring
    # ------------------------------------------------


    ## for vector chunking
    # hybrid_scores = [
    #     alpha * vector_score + (1 - alpha) * bm25_score
    #     for vector_score, bm25_score
    #     in zip(
    #         normalized_vector,
    #         normalized_bm25
    #     )
    # ]

    # for semantic chunking
    hybrid_scores = [
        alpha * semantic_score
        + (1 - alpha) * bm25_score
        for semantic_score, bm25_score
        in zip(
            normalized_semantic,
            normalized_bm25
        )
    ]

    
    # ------------------------------------------------
    # Select top 10 candidates
    # ------------------------------------------------

    candidate_k = 20

    candidate_indices = np.argsort(
        hybrid_scores
    )[-candidate_k:][::-1]

# -------------------------------------------------------------------------------------------
    # if evidence_points is not None:

    #     semantic_ranks = np.argsort(normalized_semantic)[::-1]
    #     bm25_ranks = np.argsort(normalized_bm25)[::-1]
    #     hybrid_ranks = np.argsort(hybrid_scores)[::-1]

    #     print("\n========== RETRIEVAL RANK DIAGNOSTIC ==========")

    #     for point in evidence_points:

    #         normalized_point = normalize(point)

    #         evidence_index = None

    #         # Find which original chunk contains this evidence
    #         for index, chunk in enumerate(chunks):

    #             if normalized_point in normalize(chunk):

    #                 evidence_index = index
    #                 break

    #         # Evidence not found in semantic chunks
    #         if evidence_index is None:

    #             print("\nEvidence:", point)
    #             print("Evidence chunk: NOT FOUND")
    #             print("Failure: ORIGINAL CHUNKING")
    #             continue

    #         # Find ranks
    #         semantic_rank = (
    #             np.where(semantic_ranks == evidence_index)[0][0] + 1
    #         )

    #         bm25_rank = (
    #             np.where(bm25_ranks == evidence_index)[0][0] + 1
    #         )

    #         hybrid_rank = (
    #             np.where(hybrid_ranks == evidence_index)[0][0] + 1
    #         )

    #         semantic_score = normalized_semantic[evidence_index]
    #         bm25_score = normalized_bm25[evidence_index]
    #         hybrid_score_value = hybrid_scores[evidence_index]

    #         print("\nEvidence:", point)

    #         print("Original chunk index:", evidence_index)

    #         print("Semantic rank:", semantic_rank)
    #         print("Semantic score:", f"{semantic_score:.4f}")

    #         print("BM25 rank:", bm25_rank)
    #         print("BM25 score:", f"{bm25_score:.4f}")

    #         print("Hybrid rank:", hybrid_rank)
    #         print("Hybrid score:", f"{hybrid_score_value:.4f}")

    #         # Diagnose only evidence that is outside Hybrid Top-10
    #         if hybrid_rank > 10:

    #             print("Failure: HYBRID_RETRIEVAL")

    #             if semantic_rank > 10 and bm25_rank > 10:

    #                 print(
    #                     "Reason: Both Semantic and BM25 ranked "
    #                     "this evidence outside Top-10."
    #                 )

    #             elif semantic_rank > 10:

    #                 print(
    #                     "Reason: Semantic retrieval missed it, "
    #                     "but BM25 found it."
    #                 )

    #             elif bm25_rank > 10:

    #                 print(
    #                     "Reason: BM25 retrieval missed it, "
    #                     "but Semantic retrieval found it."
    #                 )

    #             else:

    #                 print(
    #                     "Reason: Both methods found it, "
    #                     "but Hybrid scoring pushed it below Top-10."
    #                 )

    #         else:

    #             print("Status: IN HYBRID TOP-10")
# -------------------------------------------------------------------------------------------

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

    return hybrid_chunks, hybrid_score, hybrid_indices, chunks
    # return hybrid_chunks, hybrid_score