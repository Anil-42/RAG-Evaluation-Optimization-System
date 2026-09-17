import json

from extract_text import extract_text
from ollama_test import ollama_query
from evaluation import evidence_coverage

from fixed_vector import fixed_vector_retrieve
from semantic_vector import semantic_vector_retrieve
from hybrid import hybrid_retrieve_chunks
from hybrid_rerank import hybrid_rerank_retrieve

from semantic_vector import create_semantic_chunks
from chunking import chunk_text
from normalize import normalize

def main():

    # ============================================================
    # SELECT EXPERIMENT
    # ============================================================

    method = "hybrid_rerank"  # Change this to select the retrieval method

    # Available methods:
    # "fixed_vector"
    # "semantic_vector"
    # "hybrid"
    # "hybrid_rerank"

    # Number of top chunks to retrieve for hybrid_rerank method
    k=3

    # ============================================================
    # 1. DOCUMENT EXTRACTION
    # ============================================================

    pdf_path = "documents/sample.pdf"

    text = extract_text(pdf_path)

    # ============================================================
    # 2. LOAD EVALUATION DATASET
    # ============================================================

    with open(
        "questions.json",
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)


    results = []

    # ============================================================
    # 3. RAG EVALUATION LOOP
    # ============================================================

    for question in questions:

        question_text = question["question"]

        evidence_points = question["evidence_points"]

        


        # ========================================================
        # RETRIEVAL METHOD
        # ========================================================

        if method == "fixed_vector":

            retrieved_chunks, retrieved_scores = (
                fixed_vector_retrieve(
                    text,
                    question_text
                )
            )


        elif method == "semantic_vector":

            retrieved_chunks, retrieved_scores = (
                semantic_vector_retrieve(
                    text,
                    question_text
                )
            )


        elif method == "hybrid":

            retreved_chunks, retreved_scores = (
                hybrid_retrieve_chunks(
                    text,
                    question_text
                )
            )

            # Hybrid returns top 10 candidates.
            #
            # For pure hybrid experiment,
            # select top 3 before evaluation.
            
            retrieved_chunks = retreved_chunks[:5]

            retrieved_scores = retreved_scores[:5]
    

        elif method == "hybrid_rerank":

            retrieved_chunks, retrieved_scores = (
                hybrid_rerank_retrieve(
                    text,
                    question_text,
                    
                )
            )
            # k=3
            # retrieved_chunks = retrieved_chunks[:k]
            # retrieved_scores = retrieved_scores[:k]

        else:

            raise ValueError(
                "Unknown retrieval method"
            )

    #  ---------------------------testing--------------------------------

    # -------------------------------------------------------------------
        
    

        # ========================================================
        # EVALUATION
        # ========================================================

        found, total, coverage = evidence_coverage(
            retrieved_chunks,
            evidence_points
        )


        # ========================================================
        # LLM CONTEXT
        # ========================================================

        context = "\n".join(
            retrieved_chunks
        )


        prompt = f"""
        Context: {context}

        Question: {question_text}

        Answer the question using only the information provided in the context.
        Do not add information that is not present in the context.
        If the answer cannot be found in the context, say that the information is not available in the provided context.
        """


        # ========================================================
        # LLM GENERATION
        # ========================================================

        response = ollama_query(prompt)


        # ========================================================
        # SAVE RESULT
        # ========================================================

        results.append({

            "question": question_text,

            "ground_truth": question["answer"],

            "answer": response,

            "retrieved_chunks": retrieved_chunks,

            "retrieved_scores": retrieved_scores,

            "evidence_found": found,

            "evidence_total": total,

            "evidence_coverage": coverage

        })
# -----------------------------------testing-------------------------------------   
        # if(coverage < 0.5):
            # print("\nQUESTION:", question_text)
            # print("Evidence Coverage:", coverage)
            # chunks = chunk_text(text)
            # chunk_indices = [chunks.index(chunk) for chunk in retrieved_chunks]
            # print("\nTop 3 chunk Indices:", chunk_indices)
            # print("\nEvidence points:", evidence_points)
            # for index in chunk_indices:
            #     print("\nChunk ",index,":")
            #     print(chunks[index])
            # print("-"*50)
#           print("10 retrieved chunks:", retrieved_chunks)
#           print("10 chunk Scores:", retrieved_scores)

#         print("top1 chunk index:", chunks.index(retrieved_chunks[0]))
#         print("top1 chunk:", retrieved_chunks[0])
#         print("previous chunk index:", chunks.index(retrieved_chunks[0])-1)
#         print("previous chunk:", chunks[chunks.index(retrieved_chunks[0])-1])
#         print("next chunk index:", chunks.index(retrieved_chunks[0])+1)
#         print("next chunk:", chunks[chunks.index(retrieved_chunks[0])+1])
#         print("-" * 50)

# -------------------------------------------------------------------------------

    

    # ============================================================
    # 4. PRINT INDIVIDUAL RESULTS
    # ============================================================

    # for result in results:

    #     print(result["question"])

    #     print(
    #         f"Evidence coverage: "
    #         f"{result['evidence_coverage']:.2%}"
    #     )

    #     print("-" * 50)


    # ============================================================
    # 5. AVERAGE EVIDENCE COVERAGE
    # ============================================================

    total_coverage = sum(
        result["evidence_coverage"]
        for result in results
    )

    average_coverage = (
        total_coverage / len(results)
    )

    print(
        f"\nMethod: {method}"
    )

    print(
        f"Average evidence coverage: "
        f"{average_coverage:.2%}"
    )


    # ============================================================
    # 6. RETRIEVAL PASS RATE
    # ============================================================

    passed = sum(

        1

        for result in results

        if result["evidence_coverage"] >= 0.5

    )

    pass_rate = passed / len(results)

    print(
        f"Retrieval pass rate: "
        f"{pass_rate:.2%}"
    )



if __name__ == "__main__":
    main()