from normalize import normalize

def retrieval_check(retrieved_chunks,evidence):
    normalized_chunks=[normalize(t) for t in retrieved_chunks]
    normalized_evidence=normalize(evidence)

    flag=False
    for chunk in normalized_chunks:
        if normalized_evidence in chunk:
            flag = True
            break

    return flag
