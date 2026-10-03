import numpy as np

from src.chunking.chunking import chunk_text
from src.retrieval.embedding import get_embeddings
from src.retrieval.retrieve import retrieve
from src.retrieval.bm25_scores import bm25_scores
from src.retrieval.normalize_scores import normalize_scores
from src.retrieval.semantic_vector import create_semantic_chunks

from src.retrieval.normalize import normalize


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

    # for semantic chunking
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
