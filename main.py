# import torch
from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings,get_similarities;
from retrieve import retrieve;
from ollama_test import ollama_query

def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    # print(len(chunks))
    # print(len(embeddings))
    # print(len(embeddings[0]))

    question = "what is the purpose of plain english?"
    # question = input("Enter your question: ")

    retrieved_chunks, retrieved_scores = retrieve(question, chunks, embeddings, k=3)
    # print(retrieved_chunks)
    # print(retrieved_scores)

    context = "\n".join(retrieved_chunks)

    prompt = f"""Context: {context}
    Question: {question}
    Answer the question using only the information provided in the context.
    Do not add information that is not present in the context.
    If the answer cannot be found in the context, say that the information is not available in the provided context."""

    response = ollama_query(prompt)
    print(response)




if __name__ == "__main__":
    main()