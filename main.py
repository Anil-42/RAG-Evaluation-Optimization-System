from extract_text import extract_text;
from chunking import chunk_text;
from embedding import get_embeddings;

def main():
    pdf_path = "documents/sample.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    print(len(chunks))
    print(len(embeddings))
    print(len(embeddings[0]))


if __name__ == "__main__":
    main()