from sentence_transformers import CrossEncoder


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, hybrid_chunks, hybrid_indices, k):
    pairs = [[question, chunk] for chunk in hybrid_chunks]

    scores = model.predict(pairs)

    results = list(zip(hybrid_chunks, hybrid_indices, scores))
    results.sort(key=lambda x: x[2], reverse=True)


    return results[:k]

