

def expand_neighbors(candidate_indices, total_chunks):

    expanded_indices = []

    for index in candidate_indices:

        if index-1 >=0:
            expanded_indices.append(index-1)

        expanded_indices.append(index)

        if index+1<total_chunks:
            expanded_indices.append(index+1)

        expanded_indices = list(dict.fromkeys(expanded_indices))

    return expanded_indices


   
