import json
from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings,get_similarities;
from retrieve import retrieve;
from ollama_test import ollama_query
from evaluation import evidence_coverage
from retrieval_analysis import retrieval_check

from semantic_chunking import split_sentences
from semantic_chunking import semantic_chunking


def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)
    # chunks = chunk_text(text)

    # 1. Split text into sentences and embed them
    sentences = split_sentences(text)
    sentence_embeddings = get_embeddings(sentences)

    # 2. Semantic Chunking (using static threshold or dynamic percentile)
    threshold = 0.474
    semantic_chunks = semantic_chunking(sentences, sentence_embeddings, threshold)

# ------------------------------testing---------------------------------------------
    # chunk_sizes = [
    #     len(chunk.split())
    #     for chunk in semantic_chunks
    # ]

    # print("threshold:",threshold)
    # print("chunks:",len(semantic_chunks))
    # print("min words:",min(chunk_sizes))
    # print("max words:",max(chunk_sizes))
    # print("average word:",sum(chunk_sizes)/len(chunk_sizes))

    # for i, chunk in enumerate(semantic_chunks):
    #     word_count = len(chunk.split())

    #     if word_count < 20 and "a plain english handbook" not in chunk.lower():
    #         print("\nChunk:", i)
    #         print("Words:", word_count)
    #         print(chunk)
    #         print("-" * 80)
#-----------------------------------------------------------------------------------

    # 3. Embed chunks for retrieval
    embeddings = get_embeddings(semantic_chunks)

    # 4. Load evaluation dataset
    with open("questions.json", "r", encoding="utf-8") as file:
            questions = json.load(file)

    results = []

    # 5. RAG Pipeline & Evaluation Loop
    for question in questions:
        question_text = question["question"]
        evidence_points = question["evidence_points"]

        # Retrieve top-k chunks
        retrieved_chunks, retrieved_scores = retrieve(question_text, semantic_chunks, embeddings, k=3)

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

# ---------------------------------testing-------------------------------------
    for i,result in enumerate(results):
         print("question:",i)
         print("evidence coverage:",result["evidence_coverage"])
         if result["evidence_coverage"]<1.0:
              print("total evidence:",result["evidence_total"])
              print("evidence found:",result["evidence_found"])
              print("retrieved chunks:",result["retrieved_chunks"])
         print("-"*80)
#------------------------------------------------------------------------------


# # ---------------------Average coverage---------------------------
#     total_coverage = sum(
#          result["evidence_coverage"] for result in results
#     )
#     average_coverage = total_coverage/len(results)
#     print(f"Average evidence coverge: {average_coverage:.2%}")

# # -----------------------Pass Rate---------------------------------
#     passed = sum(
#         1 for result in results
#         if result["evidence_coverage"] >=0.5        
#     )
#     pass_rate = passed/len(results)
#     print(f"Retrieval pass rate: {pass_rate:.2%}")
# # -----------------------------------------------------------------

    # Optional: Save results to disk
    # with open("result.json", "w", encoding="utf-8") as file:
    #     json.dump(results, file, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    main()