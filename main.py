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

    method = "hybrid"  # Change this to select the retrieval method
    
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
    alpha = 0.4


    # ============================================================
    # 3. RAG EVALUATION LOOP
    # ============================================================
   
    reranking_improved = 0
    reranking_hurt = 0
    reranking_same = 0
 

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

            retreved_chunks, retreved_scores, hybrid_indices, chunks = (
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
            # select top 3 before evaluation.
            
            retrieved_chunks = retreved_chunks[:3]

            retrieved_scores = retreved_scores[:3]

            # if len(retrieved_chunks) < 3:
            #     pass

            # else:
            #     normalized_top3 = " ".join(
            #         normalize(chunk)
            #         for chunk in retrieved_chunks
            #     )

            #     for point in evidence_points:

            #         normalized_point = normalize(point)

            #         if normalized_point not in normalized_top3:

            #             print("\n--------------------------------")
            #             print("Question:", question_text)
            #             print("Missing evidence:", point)

            #             # Check whether the evidence exists
            #             # anywhere in the semantic chunks
            #             found_index = None

            #             for index, chunk in enumerate(chunks):
            #                 if normalized_point in normalize(chunk):
            #                     found_index = index
            #                     break

            #             if found_index is None:
            #                 print("Evidence NOT found in semantic chunks.")
            #                 continue

            #             print("Original semantic chunk:", found_index)

            #             # Find its position inside the Hybrid Top-10
            #             if found_index in hybrid_indices:
            #                 hybrid_rank = hybrid_indices.index(found_index) + 1
            #                 print("Hybrid rank:", hybrid_rank)
            #             else:
            #                 print("Hybrid rank: NOT IN TOP-10")

            #             print("--------------------------------")


        elif method == "hybrid_rerank":

            retrieved_chunks, retrieved_scores, hybrid_indices, reranked_indices = (
                hybrid_rerank_retrieve(
                    text,
                    question_text,
                    evidence_points
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

    # -------------------------------------------------------------------------------------------------
        # if coverage < 1.0:

        #     print("Question:", question_text)
        #     print("Evidence points:", evidence_points)
        #     print("Top-3 chunks:", retrieved_chunks)

        #     normalized_chunks = " ".join(
        #         normalize(t) for t in retrieved_chunks
        #     )

        #     for point in evidence_points:

        #         # Check whether this evidence point is missing
        #         if normalize(point) not in normalized_chunks:

        #             print("Missing Evidence point:", point)

        #             # Search for the missing evidence in all Hybrid candidates
        #             for rank, (index, chunk) in enumerate(
        #                 zip(hybrid_indices, retreved_chunks),
        #                 start=1
        #             ):

        #                 if normalize(point) in normalize(chunk):

        #                     print(
        #                         "Hybrid rank of the chunk containing "
        #                         "this missing evidence:",
        #                         rank
        #                     )

        #                     print("Chunk index:", index)

        #                     break

        #             else:
        #                 for index, chunk in enumerate(chunks):

        #                     if normalize(point) in normalize(chunk):

        #                         print("Found in original document chunk:", index)
        #                         break
        #                 else:
        #                     print("Evidence not found in original document chunks.")

        #     print("-" * 50)
    
    # -------------------------------------------------------------------------------------------------


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

    # for result in results:

    #     print(result["question"])

    #     print(
    #         f"Evidence coverage: "
    #         f"{result['evidence_coverage']:.2%}"
    #     )

    #     print("-" * 50)

    # print("Reranking improved:", reranking_improved)
    # print("Reranking hurt:", reranking_hurt)
    # print("Reranking unchanged:", reranking_same,"\n")

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