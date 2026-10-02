import json
from evaluation import print_evaluation_summary
from extract_text import extract_text
from ollama_test import ollama_query

from evidence_coverage import evidence_coverage

from fixed_vector import fixed_vector_retrieve
from semantic_vector import semantic_vector_retrieve
from hybrid import hybrid_retrieve_chunks
from hybrid_rerank import hybrid_rerank_retrieve



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
    # k=3

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
    alpha = 0.2


    # ============================================================
    # 3. RAG EVALUATION LOOP
    # ============================================================
   
    # reranking_improved = 0
    # reranking_hurt = 0
    # reranking_same = 0
 

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

            hybrid_chunks, hybrid_scores, hybrid_indices, chunks = (
                hybrid_retrieve_chunks(
                    text,
                    question_text,
                    evidence_points,
                    alpha
                )
            )

            # Hybrid returns top 10 candidates.
            #
            # For pure hybrid experiment,
            # select top 7 before evaluation.
            
            retrieved_chunks = hybrid_chunks[:7]

            retrieved_scores = hybrid_scores[:7]

           

        elif method == "hybrid_rerank":

            retrieved_chunks, retrieved_scores, hybrid_chunks, hybrid_indices, chunks = (
                hybrid_rerank_retrieve(
                    text,
                    question_text,
                    evidence_points,
                    alpha
                )
            )
            

        else:

            raise ValueError(
                "Unknown retrieval method"
            )

    

        # ========================================================
        # EVALUATION
        # ========================================================

        found, total, coverage = evidence_coverage(
            retrieved_chunks,
            evidence_points
        )


# ----------------------------------------------------------------
        # if coverage < 1.0:
        #     print("\n" + "=" * 60)
        #     print("FAILED / PARTIAL RETRIEVAL")
        #     print("=" * 60)
        #     print("Question:", question_text)
        #     print("Evidence found:", found)
        #     print("Evidence total:", total)
        #     print("Coverage:", coverage)

        #     print("\nEvidence points:")
        #     for point in evidence_points:
        #         print("-", point)
# ----------------------------------------------------------------


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

            "retrieved_scores": [float(score) for score in retrieved_scores],

            "evidence_found": found,

            "evidence_total": total,

            "evidence_coverage": coverage

        })

    print_evaluation_summary(results, method)



    # Optional: Save results to disk
    with open("result.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


    

if __name__ == "__main__":
    main()