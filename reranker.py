from sentence_transformers import CrossEncoder


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, candidate_chunks, k=3):
    pairs = [[question, chunk] for chunk in candidate_chunks]

    scores = model.predict(pairs)

    results = list(zip(candidate_chunks, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:k]


