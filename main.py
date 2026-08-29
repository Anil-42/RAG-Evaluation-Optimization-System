import json
import numpy as np
from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings,get_similarities;
from retrieve import retrieve;
from ollama_test import ollama_query
from evaluation import evidence_coverage
from retrieval_analysis import retrieval_check


from bm25_scores import bm25_scores
from hybrid_retrieve import normalize_scores
from reranker import rerank


def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)

    # chunking
    chunks = chunk_text(text)


    # 3. Embed chunks for retrieval
    embeddings = get_embeddings(chunks)

    # 4. Load evaluation dataset
    with open("questions.json", "r", encoding="utf-8") as file:
            questions = json.load(file)


    results = []
    alpha = 0.7

    # 5. RAG Pipeline & Evaluation Loop
    for question in questions:
        question_text = question["question"]
        evidence_points = question["evidence_points"]

        # Retrieve scores from BM25
        bm25_score_values = bm25_scores(question_text, chunks)


        # retrieve chunk scores from vector
        vector_score_values = retrieve(question_text, chunks, embeddings)

        normalized_bm25 = normalize_scores(vector_score_values)
        normalized_vector = normalize_scores(bm25_score_values)

        

        hybrid_scores = [
            alpha * vector_score + (1 - alpha) * bm25_score
            for vector_score, bm25_score
            in zip(normalized_vector, normalized_bm25)
        ]

        candidate_k = 10

        candidate_indices = np.argsort(hybrid_scores)[-candidate_k:][::-1]
        # 3. Extract chunks and scores
        hybrid_chunks = [chunks[i] for i in candidate_indices]
        hybrid_scores = [hybrid_scores[i] for i in candidate_indices]


        reranked = rerank(question_text, hybrid_chunks, k=3)


        final_chunks = [chunk for chunk, score in reranked]
        final_scores = [score for chunk, score in reranked]

        # Evaluate evidence coverage
        found, total, coverage = evidence_coverage(final_chunks,evidence_points) 

        context = "\n".join(final_chunks)

        prompt = f"""
        Context: {context}

        Question: {question_text}

        Answer the question using only the information provided in the context.
        Do not add information that is not present in the context.
        If the answer cannot be found in the context, say that the information is not available in the provided context.
        """


        response = ollama_query(prompt)
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

    print("alpha:",alpha)

# ---------------------Average coverage---------------------------
    total_coverage = sum(
         result["evidence_coverage"] for result in results
    )
    average_coverage = total_coverage/len(results)
    print(f"Average evidence coverge: {average_coverage:.2%}")

# -----------------------Pass Rate---------------------------------
    passed = sum(
        1 for result in results
        if result["evidence_coverage"] >=0.5        
    )
    pass_rate = passed/len(results)
    print(f"Retrieval pass rate: {pass_rate:.2%}")
# -----------------------------------------------------------------

    # Optional: Save results to disk
    # with open("result.json", "w", encoding="utf-8") as file:
    #     json.dump(results, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()