import torch
from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings,get_similarities;

def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    print(len(chunks))
    print(len(embeddings))
    print(len(embeddings[0]))

    question = "what is the purpose of plain english?"
    question_embedding = get_embeddings(question)
    similarity = get_similarities(question_embedding, embeddings)
    print(similarity.shape)

    values, indices = torch.topk(similarity, k=3)    
    print(values)
    print(indices)

    for index, score in zip(indices[0], values[0]):
        print("score:",score.item())
        print("index:",index.item())
        print("chunk:",chunks[index])
        print("-" * 50)

if __name__ == "__main__":
    main()