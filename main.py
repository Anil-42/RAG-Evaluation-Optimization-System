import json
import numpy as np
from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings,get_similarities;
from retrieve import retrieve;
from ollama_test import ollama_query
from evaluation import evidence_coverage
from retrieval_analysis import retrieval_check

# from semantic_chunking import split_sentences
# from semantic_chunking import semantic_chunking

from bm25_scores import bm25_scores
from hybrid_retrieve import normalize_scores


def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)

    # chunking
    chunks = chunk_text(text)
# #--------------------------------semantic chunking---------------------------------
    # 1. Split text into sentences and embed them
    # sentences = split_sentences(text)
    # sentence_embeddings = get_embeddings(sentences)
    # 2. Semantic Chunking (using static threshold or dynamic percentile)
    # threshold = 0.474
    # semantic_chunks = semantic_chunking(sentences, sentence_embeddings, threshold)
# #-----------------------------------------------------------------------------------

    # 3. Embed chunks for retrieval
    embeddings = get_embeddings(chunks)

    # 4. Load evaluation dataset
    with open("questions.json", "r", encoding="utf-8") as file:
            questions = json.load(file)


    results = []

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

        alpha = 0.7

        hybrid_scores = [
            alpha * vector_score + (1 - alpha) * bm25_score
            for vector_score, bm25_score
            in zip(normalized_vector, normalized_bm25)
        ]

        k = 3

        top_indices = np.argsort(hybrid_scores)[-k:][::-1]

        # 3. Extract chunks and scores
        retrieved_chunks = [chunks[i] for i in top_indices]
        retrieved_scores = [hybrid_scores[i] for i in top_indices]

        # print("Top K chunks:")

        # for rank, (index, chunk, score) in enumerate(
        #     zip(top_indices, retrieved_chunks, retrieved_scores), start=1
        # ):
        #     print(f"Rank {rank} | Chunk {index} | Score {score}")
        #     print(chunk)
        #     print("-" * 80)

        # Evaluate evidence coverage
        found, total, coverage = evidence_coverage(retrieved_chunks,evidence_points) 

        context = "\n".join(retrieved_chunks)

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
             "retrieved_chunks" : retrieved_chunks,
             "retrieved_scores" : retrieved_scores,
             "evidence_found" : found,
             "evidence_total" : total,
             "evidence_coverage" : coverage
        })


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