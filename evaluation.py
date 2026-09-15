from normalize import normalize

def evidence_coverage(retrieved_chunks,evidence_points):
    normalized_chunks = " ".join(normalize(t) for t in retrieved_chunks)

    count=0
    total=len(evidence_points)
    for point in evidence_points:
        if normalize(point) in normalized_chunks:
            count+=1          
     
    return (count, total, count/total)