

def merge_neighbors(candidate_indices, chunks):
    
    merged = []
    used = set()

    for index in candidate_indices:

        if index in used:
            continue

        start = index
        end = index

        # Include previous chunk
        if index - 1 >= 0:
            start = index - 1

        # Include next chunk
        if index + 1 < len(chunks):
            end = index + 1

        merged_text = " ".join(
            chunks[i] for i in range(start, end + 1)
        )

        merged.append((start, end, merged_text))

        for i in range(start, end + 1):
            used.add(i)

    return merged

