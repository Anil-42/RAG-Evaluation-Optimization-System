import json
import sys
from pathlib import Path

# Add project root (d:\other\rag-project) to Python's path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.evaluation import print_evaluation_summary
from evaluation.evidence_coverage import evidence_coverage

from extraction.extract_text import extract_text
from generation.generator import ollama_query

from retrieval.fixed_vector import fixed_vector_retrieve
from retrieval.semantic_vector import semantic_vector_retrieve
from retrieval.hybrid import hybrid_retrieve_chunks
from retrieval.hybrid_rerank import hybrid_rerank_retrieve



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
    # ===========================================================

    # This creates an absolute path to evaluation/questions.json
    questions_path = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"


    with open(
        questions_path,
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
    results_path = Path(__file__).resolve().parent.parent / "results" / "result.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


    

if __name__ == "__main__":
    main()