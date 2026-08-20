import json
from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings,get_similarities;
from retrieve import retrieve;
from ollama_test import ollama_query
from evaluation import evidence_coverage
from retrieval_analysis import retrieval_check


def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    # print(len(chunks))
    # print(len(embeddings))
    # print(len(embeddings[0]))

    # question = "what is the purpose of plain english?"
    # question = input("Enter your question: ")

    with open("questions.json", "r", encoding="utf-8") as file:
            questions = json.load(file)

    # check if evidence_points load properly # print(questions[0]["evidence_points"])
    results = []

    for question in questions:
        question_text = question["question"]

        retrieved_chunks, retrieved_scores = retrieve(question_text, chunks, embeddings, k=3)
        # print(retrieved_chunks)
        # print(retrieved_scores)

        evidence_points = question["evidence_points"]
        
        evaluation_result = evidence_coverage(retrieved_chunks,evidence_points)
        print(evaluation_result)

        # evidence = question["evidence"]
        # retrieved_analysis = retrieval_check(retrieved_chunks,evidence)
        # check retrieved_analysis # print(retrieved_analysis) 

# --------------------------------------------
            # print("EVIDENCE:")
            # print(evidence)

            # print("\nRETRIEVED_CHUNKS")
            # for chunk in retrieved_chunks:
            #     print(chunk)
            #     print("-"*50)
# --------------------------------------------

    #     context = "\n".join(retrieved_chunks)

    #     prompt = f"""
    #     Context: {context}

    #     Question: {question_text}

    #     Answer the question using only the information provided in the context.
    #     Do not add information that is not present in the context.
    #     If the answer cannot be found in the context, say that the information is not available in the provided context.
    #     """


    #     response = ollama_query(prompt)
    #     results.append({
    #          "question" : question_text,
    #          "ground_truth" : question["answer"],
    #          "answer" : response,
    #          "retrieved_chunks" : retrieved_chunks,
    #          "retrieved_scores" : retrieved_scores
    #     })

    # # print(results[0])

    # with open("result.json", "w", encoding="utf-8") as file:
    #     json.dump(results, file, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    main()