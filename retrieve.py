import torch
from embedding import get_embeddings,get_similarities;

def retrieve(question, chunks, embeddings):

    question_embedding = get_embeddings(question)
    similarity = get_similarities(question_embedding, embeddings)
    # print(similarity.shape)


    # change according to retrieval method
    
    # fixedsize retrieval
    # k=min(3,len(chunks))

    # hybrid retrieval
    k=len(chunks)

    values, indices = torch.topk(similarity,k)

#[[0.8952, 0.7413, 0.6120]], [[18, 191, 19]] this is how values will be stored and we indices[o],values[0] to get elements of first row.
#zip pairs indices and values. index, score is unpacking.
#.item() function converts a single-element PyTorch tensor into a normal Python number.

    # for fixedsize retrieval
    retrieved_chunks = []

    retrieved_scores = []
    for index, score in zip(indices[0],values[0]): 
        # for fixedsize and semantic retrieval
        # retrieved_chunks.append(chunks[index.item()])
        
        retrieved_scores.append(score.item())

    # for fixedsize and semantic retrieval
    # return retrieved_chunks, retrieved_scores

    # for hybrid retrieval
    return retrieved_scores


