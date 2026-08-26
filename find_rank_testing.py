from embedding import get_embeddings, get_similarities

def rank(question_text,semantic_embeddings):
    question_embedding = get_embeddings(question_text)

    scores=[]
    for i, chunk_embedding in enumerate(semantic_embeddings):
        score=get_similarities(question_embedding,chunk_embedding).item()
        scores.append((i, score))

    scores.sort(key=lambda x:x[1], reverse=True)

    for rank, (chunk_index,score) in enumerate(scores[:10], start=1):
        print(
        "Rank:", rank,
        "Chunk:", chunk_index,
        "Score:", score
        )

    for rank, (chunk_index, score) in enumerate(scores, start=1):
        if chunk_index == 39:
            print("Chunk 39 rank:", rank)
            print("Chunk 39 score:", score)
