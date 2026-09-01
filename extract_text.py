import pymupdf

def extract_text(pdf_path):

    pdf = pymupdf.open(pdf_path)

    text=""
    for page in pdf:
        text+=page.get_text()

    return text


# import json
# import numpy as np
# from extract_text import extract_text
# from chunking import chunk_text
# from embedding import get_embeddings
# from retrieve import retrieve
# from ollama_test import ollama_query
# from evaluation import evidence_coverage
# from bm25_scores import bm25_scores
# from hybrid_retrieve import normalize_scores
# from reranker import rerank


# def main():
#     # ============================================================
#     # 1. Document Extraction & Preprocessing
#     # ============================================================

#     pdf_path = "documents/sample.pdf"
#     text = extract_text(pdf_path)


#     # ============================================================
#     # 2. FIXED-SIZE CHUNKING
#     # ============================================================
#     # This is our original baseline chunking.
#     #
#     #   chunk_size = 100 words
#     #   overlap = 20 words
#     #
#     # Keep this as the default when testing our original
#     # fixed-size/vector retrieval baseline.

#     #    chunks = chunk_text(text)


#     # ============================================================
#     # 3. SEMANTIC CHUNKING - OPTIONAL
#     # ============================================================
#     # IMPORTANT:
#     # Do NOT run this together with the fixed-size chunking above.
#     #
#     # To test SEMANTIC retrieval:
#     #
#     # 1. Uncomment the imports below.
#     # 2. Comment out:
#     #       chunks = chunk_text(text)
#     # 3. Uncomment the semantic chunking code below.
#     #
#     # Our semantic chunking process:
#     #
#     # PDF text
#     #     ↓
#     # Split into sentences
#     #     ↓
#     # Generate sentence embeddings
#     #     ↓
#     # Calculate similarity between consecutive sentences
#     #     ↓
#     # Split when similarity < threshold
#     #     ↓
#     # Apply minimum chunk size
#     #
#     # Our tested semantic configuration:
#     # threshold = 0.222
#     # minimum chunk size = 20 words
#     #
#     # Previous result:
#     # Average Coverage = 68%
#     # Pass Rate = 60%
#     #
#     # ------------------------------------------------------------
#     # UNCOMMENT WHEN TESTING SEMANTIC CHUNKING
#     # ------------------------------------------------------------

#     # from semantic_chunking import split_sentences
#     # from semantic_chunking import semantic_chunking

#     # sentences = split_sentences(text)

#     # sentence_embeddings = get_embeddings(sentences)

#     # threshold = 0.222

#     # chunks = semantic_chunking(
#     #     sentences,
#     #     sentence_embeddings,
#     #     threshold
#     # )

#     # ------------------------------------------------------------


#     # ============================================================
#     # 4. VECTOR EMBEDDINGS
#     # ============================================================
#     # Whichever chunking method is active above will be embedded.
#     #
#     # Fixed-size:
#     chunks = chunk_text(text)
#     #
#     # OR semantic:
#     #     chunks = semantic_chunking(...)
#     #
#     embeddings = get_embeddings(chunks)


#     # ============================================================
#     # 5. LOAD EVALUATION DATASET
#     # ============================================================

#     with open("questions.json", "r", encoding="utf-8") as file:
#         questions = json.load(file)


#     results = []


#     # ============================================================
#     # 6. HYBRID SEARCH CONFIGURATION
#     # ============================================================
#     #
#     # alpha = 0.7 means:
#     #
#     # Vector similarity = 70%
#     # BM25              = 30%
#     #
#     # DO NOT CHANGE THIS SECTION unless we are intentionally
#     # running a different hybrid-weight experiment.

#     alpha = 0.7


#     # ============================================================
#     # 7. RAG PIPELINE & EVALUATION LOOP
#     # ============================================================

#     for question in questions:

#         question_text = question["question"]
#         evidence_points = question["evidence_points"]


#         # ========================================================
#         # A. FIXED-SIZE + VECTOR RETRIEVAL
#         # ========================================================
#         #
#         # This is our original baseline retrieval.
#         #
#         # For the baseline experiment:
#         #
#         #     chunks = fixed-size chunks
#         #     embeddings = chunk embeddings
#         #     retrieve() = vector similarity
#         #     top_k = 3
#         #
#         # --------------------------------------------------------
#         # To run this experiment:
#         #
#         # 1. Keep:
#         #       chunks = chunk_text(text)
#         #
#         # 2. Comment out semantic chunking.
#         #
#         # 3. Replace the hybrid retrieval section below with:
#         #
#         vector_retrieved_chunks, vector_retrieved_scores = retrieve(
#             question_text,
#             chunks,
#             embeddings,
#         )
        
#         found, total, coverage = evidence_coverage(
#             vector_retrieved_chunks,
#             evidence_points
#         )
        
#         final_chunks = vector_retrieved_chunks
#         final_scores = vector_retrieved_scores
#         #
#         # This gives us the original Fixed-size + Vector baseline.
#         #
#         # Previous 5-question result:
#         #
#         # Average Coverage = 92%
#         # Pass Rate = 100%
#         #
#         # --------------------------------------------------------


#         # ========================================================
#         # B. SEMANTIC CHUNKING + VECTOR RETRIEVAL
#         # ========================================================
#         #
#         # If semantic chunking is enabled above, the SAME vector
#         # retrieval function can be used.
#         #
#         # The important difference is the chunks themselves:
#         #
#         # Fixed:
#         #       chunks = chunk_text(text)
#         #
#         # Semantic:
#         #       chunks = semantic_chunking(...)
#         #
#         # --------------------------------------------------------
#         # To run semantic retrieval:
#         #
#         # vector_retrieved_chunks, vector_retrieved_scores = retrieve(
#         #     question_text,
#         #     chunks,
#         #     embeddings,
#         #     k=3
#         # )
#         #
#         # found, total, coverage = evidence_coverage(
#         #     vector_retrieved_chunks,
#         #     evidence_points
#         # )
#         #
#         # final_chunks = vector_retrieved_chunks
#         # final_scores = vector_retrieved_scores
#         #
#         # Previous semantic result:
#         #
#         # Threshold = 0.222
#         # Minimum chunk size = 20 words
#         # Average Coverage = 68%
#         # Pass Rate = 60%
#         #
#         # --------------------------------------------------------


#         # ========================================================
#         # C. HYBRID RETRIEVAL
#         # ========================================================
#         #
#         # DO NOT CHANGE THIS LOGIC.
#         #
#         # BM25 + Vector
#         #     ↓
#         # Normalize scores
#         #     ↓
#         # Weighted combination
#         #     ↓
#         # Select top 10 candidates
#         #
#         # ========================================================


#         # Step C1: BM25 retrieval scores

#         # bm25_score_values = bm25_scores(
#         #     question_text,
#         #     chunks
#         # )


#         # Step C2: Vector retrieval scores

#         # vector_score_values = retrieve(
#         #     question_text,
#         #     chunks,
#         #     embeddings
#         # )


#         # Step C3: Normalize scores

#         # normalized_bm25 = normalize_scores(
#         #     bm25_score_values
#         # )

#         # normalized_vector = normalize_scores(
#         #     vector_score_values
#         # )


#         # Step C4: Hybrid scoring
#         #
#         # alpha = vector weight
#         # 1-alpha = BM25 weight

#         # hybrid_scores = [
#         #     alpha * vector_score + (1 - alpha) * bm25_score
#         #     for vector_score, bm25_score
#         #     in zip(
#         #         normalized_vector,
#         #         normalized_bm25
#         #     )
#         # ]


#         # Step C5: Select top candidate chunks

#         # candidate_k = 10

#         # candidate_indices = np.argsort(
#         #     hybrid_scores
#         # )[-candidate_k:][::-1]


#         # hybrid_chunks = [
#         #     chunks[i]
#         #     for i in candidate_indices
#         # ]

#         # hybrid_scores = [
#         #     hybrid_scores[i]
#         #     for i in candidate_indices
#         # ]


#         # ========================================================
#         # D. CROSS-ENCODER RE-RANKING
#         # ========================================================
#         #
#         # Take the best 10 hybrid candidates.
#         #
#         # Cross-encoder scores the question + chunk together.
#         #
#         # Then select the final 3 chunks.
#         #
#         # Candidate K = 10
#         # Final K = 3

#         # reranked = rerank(
#         #     question_text,
#         #     hybrid_chunks,
#         #     k=3
#         # )


#         # final_chunks = [
#         #     chunk
#         #     for chunk, score in reranked
#         # ]

#         # final_scores = [
#         #     score
#         #     for chunk, score in reranked
#         # ]


#         # ========================================================
#         # E. EVALUATION
#         # ========================================================

#         # found, total, coverage = evidence_coverage(
#         #     final_chunks,
#         #     evidence_points
#         # )


#         # ========================================================
#         # F. CREATE LLM CONTEXT
#         # ========================================================

#         context = "\n".join(final_chunks)


#         prompt = f"""
#         Context: {context}

#         Question: {question_text}

#         Answer the question using only the information provided in the context.
#         Do not add information that is not present in the context.
#         If the answer cannot be found in the context, say that the information is not available in the provided context.
#         """


#         # ========================================================
#         # G. LLM GENERATION
#         # ========================================================

#         response = ollama_query(prompt)


#         # ========================================================
#         # H. SAVE RESULT
#         # ========================================================

#         results.append({
#             "question": question_text,
#             "ground_truth": question["answer"],
#             "answer": response,
#             "retrieved_chunks": final_chunks,
#             "retrieved_scores": final_scores,
#             "evidence_found": found,
#             "evidence_total": total,
#             "evidence_coverage": coverage
#         })


#     # ============================================================
#     # 8. PRINT RESULTS
#     # ============================================================

#     print("alpha:", alpha)


#     # ============================================================
#     # 9. AVERAGE EVIDENCE COVERAGE
#     # ============================================================

#     total_coverage = sum(
#         result["evidence_coverage"]
#         for result in results
#     )

#     average_coverage = (
#         total_coverage / len(results)
#     )

#     print(
#         f"Average evidence coverage: "
#         f"{average_coverage:.2%}"
#     )


#     # ============================================================
#     # 10. RETRIEVAL PASS RATE
#     # ============================================================

#     passed = sum(
#         1
#         for result in results
#         if result["evidence_coverage"] >= 0.5
#     )

#     pass_rate = passed / len(results)

#     print(
#         f"Retrieval pass rate: "
#         f"{pass_rate:.2%}"
#     )


#     # ============================================================
#     # 11. OPTIONAL: SAVE RESULTS
#     # ============================================================

#     # with open("result.json", "w", encoding="utf-8") as file:
#     #     json.dump(
#     #         results,
#     #         file,
#     #         indent=4,
#     #         ensure_ascii=False
#     #     )


# if __name__ == "__main__":
#     main()

# alpha: 0.7
# Average evidence coverage: 11.52%
# Retrieval pass rate: 13.04%