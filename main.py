import json

from extract_text import extract_text
from ollama_test import ollama_query
from evaluation import evidence_coverage

from fixed_vector import fixed_vector_retrieve
from semantic_vector import semantic_vector_retrieve
from hybrid import hybrid_retrieve_chunks
from hybrid_rerank import hybrid_rerank_retrieve


def main():

    # ============================================================
    # SELECT EXPERIMENT
    # ============================================================

    method = "hybrid"  # Change this to select the retrieval method

    # Available methods:
    #
    # "fixed_vector"
    # "semantic_vector"
    # "hybrid"
    # "hybrid_rerank"


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
            # # --------------------------------------------
            # print("\nQUESTION:", question_text)

            # for i, (chunk, score) in enumerate(
            #     zip(retrieved_chunks, retrieved_scores)
            # ):
            #     print("\nRank:", i + 1)
            #     print("Score:", score)
            #     print("Chunk:", chunk)
            # # --------------------------------------------

        elif method == "semantic_vector":

            retrieved_chunks, retrieved_scores = (
                semantic_vector_retrieve(
                    text,
                    question_text
                )
            )


        elif method == "hybrid":

            retrieved_chunks, retrieved_scores = (
                hybrid_retrieve_chunks(
                    text,
                    question_text
                )
            )

            # Hybrid returns top 10 candidates.
            #
            # For pure hybrid experiment,
            # select top 3 before evaluation.

            retrieved_chunks = retrieved_chunks[:3]

            retrieved_scores = retrieved_scores[:3]


        elif method == "hybrid_rerank":

            retrieved_chunks, retrieved_scores = (
                hybrid_rerank_retrieve(
                    text,
                    question_text
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


    # ============================================================
    # 4. PRINT INDIVIDUAL RESULTS
    # ============================================================

    for result in results:

        print(result["question"])

        print(
            f"Evidence coverage: "
            f"{result['evidence_coverage']:.2%}"
        )

        print("-" * 50)


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