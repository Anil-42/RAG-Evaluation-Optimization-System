from sentence_transformers import CrossEncoder


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, hybrid_chunks, hybrid_indices, k):
    pairs = [[question, chunk] for chunk in hybrid_chunks]

    scores = model.predict(pairs)

    # ------------------------------------------------------
    print("\n Question:",question)
    print("\n hybrid top-10:",hybrid_indices)
    for index, score in zip(hybrid_indices, scores):
        print(f"Chunk {index} → Reranker score: {score:.4f}")
    # ------------------------------------------------------

    results = list(zip(hybrid_chunks, hybrid_indices, scores))
    results.sort(key=lambda x: x[2], reverse=True)


    return results[:k]

