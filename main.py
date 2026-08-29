import json
import numpy as np
from extract_text import extract_text
from chunking import chunk_text
from embedding import get_embeddings # Removed unused 'get_similarities'
from retrieve import retrieve
from ollama_test import ollama_query
from evaluation import evidence_coverage
# Removed unused 'retrieval_check' import
from bm25_scores import bm25_scores
from hybrid_retrieve import normalize_scores
from reranker import rerank

def main():
    # 1. Document Extraction & Preprocessing
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)

    # 2. Text Chunking
    # Break down the full document into smaller, manageable pieces
    chunks = chunk_text(text)

    # 3. Vector Embeddings
    # Convert chunks into numerical vectors for semantic search
    embeddings = get_embeddings(chunks)

    # 4. Load Evaluation Dataset
    # Load questions and their expected ground-truth evidence points
    with open("questions.json", "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []
    
    # Weighting factor for hybrid search (0.7 = equal weight to BM25 and Vector search)
    alpha = 0.7

    # 5. RAG Pipeline & Evaluation Loop
    for question in questions:
        question_text = question["question"]
        evidence_points = question["evidence_points"]

        # Step A: Sparse Retrieval (Keyword-based search)
        bm25_score_values = bm25_scores(question_text, chunks)

        # Step B: Dense Retrieval (Semantic vector search)
        vector_score_values = retrieve(question_text, chunks, embeddings)

        # Step C: Score Normalization
        # Normalize both score arrays to the same scale so they can be combined
        normalized_bm25 = normalize_scores(bm25_score_values)
        normalized_vector = normalize_scores(vector_score_values)

        # Step D: Hybrid Scoring
        # Combine normalized scores using the alpha weight
        hybrid_scores = [
            alpha * vector_score + (1 - alpha) * bm25_score
            for vector_score, bm25_score
            in zip(normalized_vector, normalized_bm25)
        ]

        # Step E: Initial Top-K Selection
        # Get the indices of the top 10 scoring chunks
        candidate_k = 10
        candidate_indices = np.argsort(hybrid_scores)[-candidate_k:][::-1]
        
        # Extract the actual chunks and scores for the top candidates
        hybrid_chunks = [chunks[i] for i in candidate_indices]
        hybrid_scores = [hybrid_scores[i] for i in candidate_indices]

        # Step F: Reranking
        # Pass the top 10 chunks through a reranker model to select the best 3
        reranked = rerank(question_text, hybrid_chunks, k=3)

        final_chunks = [chunk for chunk, score in reranked]
        final_scores = [score for chunk, score in reranked]

        # Step G: Evaluation (Retrieval Performance)
        # Check if the retrieved chunks contain the required ground-truth evidence
        found, total, coverage = evidence_coverage(final_chunks, evidence_points) 

        # Step H: Prompt Construction
        # Combine the final chunks into a single context string
        context = "\n".join(final_chunks)

        prompt = f"""
        Context: {context}

        Question: {question_text}

        Answer the question using only the information provided in the context.
        Do not add information that is not present in the context.
        If the answer cannot be found in the context, say that the information is not available in the provided context.
        """

        # Step I: LLM Generation
        # Send the context and question to Ollama to generate an answer
        response = ollama_query(prompt)
        
        # Step J: Log Results
        results.append({
             "question" : question_text,
             "ground_truth" : question["answer"],
             "answer" : response,
             "retrieved_chunks" : final_chunks,
             "retrieved_scores" : final_scores,
             "evidence_found" : found,
             "evidence_total" : total,
             "evidence_coverage" : coverage
        })

    print("alpha:", alpha)

    # 6. Calculate Final Pipeline Metrics
    # ---------------------Average coverage---------------------------
    total_coverage = sum(
         result["evidence_coverage"] for result in results
    )
    average_coverage = total_coverage / len(results)
    print(f"Average evidence coverge: {average_coverage:.2%}")

    # -----------------------Pass Rate---------------------------------
    # A query is considered "passed" if it retrieves at least 50% of the needed evidence
    passed = sum(
        1 for result in results
        if result["evidence_coverage"] >= 0.5        
    )
    pass_rate = passed / len(results)
    print(f"Retrieval pass rate: {pass_rate:.2%}")
    # -----------------------------------------------------------------

    # Optional: Save results to disk
    # with open("result.json", "w", encoding="utf-8") as file:
    #     json.dump(results, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()