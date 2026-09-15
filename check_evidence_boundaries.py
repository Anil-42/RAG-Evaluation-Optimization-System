import json
from chunking import chunk_text
from extract_text import extract_text
from normalize import normalize



def check():
    pdf_path = "documents/sample.pdf"

    # Extract PDF text
    text = extract_text(pdf_path)

    # Load evaluation dataset
    with open("questions.json", "r", encoding="utf-8") as file:
        questions = json.load(file)

    # Create the same chunks used by the RAG system
    chunks = chunk_text(text)

    total_questions = len(questions)

    normal = 0
    boundary_split = 0
    not_found = 0

    for question in questions:

        question_text = question["question"]
        evidence_points = question["evidence_points"]

        for point in evidence_points:
            normalized_point=normalize(point)

            point_found = False

            # Check every chunk
            for i in range(len(chunks)):
                normalized_chunk1 = normalize(chunks[i])

                # Case 1: Evidence completely exists in one chunk
                if normalized_point in normalized_chunk1:

                    normal += 1

                    print(f"\nQuestion: {question_text}")
                    print(f"Evidence: {point}")
                    print(f"Chunk: {i}")
                    print("Status: NORMAL")

                    point_found = True
                    break

                # Case 2: Evidence is split across two neighboring chunks
                if (i + 1 < len(chunks)):
                    normalized_chunks2 = normalize(chunks[i+1])
                    if(normalized_point in normalized_chunk1 + normalized_chunks2):
                        boundary_split += 1

                        print(f"\nQuestion: {question_text}")
                        print(f"Evidence: {point}")
                        print(f"Chunks: {i} + {i + 1}")
                        print("Status: BOUNDARY_SPLIT")

                        point_found = True
                        break

            # Case 3: Evidence wasn't found
            if not point_found:

                not_found += 1

                print(f"\nQuestion: {question_text}")
                print(f"Evidence: {point}")
                print("Status: NOT_FOUND")
        print("-"*100)

    # Final summary
    total_evidence_points = normal + boundary_split + not_found

    print("\n" + "-" * 60)
    print("EVIDENCE BOUNDARY ANALYSIS")
    print("-" * 60)

    print(f"Total Questions: {total_questions}")
    print(f"Total Evidence Points: {total_evidence_points}")
    print(f"Normal: {normal}")
    print(f"Boundary Split: {boundary_split}")
    print(f"Not Found: {not_found}")


if __name__ == "__main__":
    check()