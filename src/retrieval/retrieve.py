import torch
from embedding import get_embeddings,get_similarities;

def retrieve(question, chunks, embeddings):

    question_embedding = get_embeddings(question)
    similarity = get_similarities(question_embedding, embeddings)
    # print(similarity.shape)


    # ----------------change according to retrieval method-------------------

    # fixedsize retrieval
    # k=min(7,len(chunks))
    
    # hybrid retrieval
    k=len(chunks)
    # -----------------------------------------------------------------------

    values, indices = torch.topk(similarity,k)

    # gives values, indices in sorted order. for hybridretrieval it must be in the same order as chunks.
    # so for hybrid we sort based on indices to restore original order of chunks as coded below.

    # --------------------for hybrid retrieval------------------------------
    # 1. Zip the indices and values together as native Python types
    pairs = list(zip(indices[0].tolist(), values[0].tolist()))
    # 2. Sort the pairs based on the index (x[0]) to restore original order
    pairs.sort(key=lambda x: x[0])
    # --------------------------------------------------------------------

    # for fixedsize retrieval
    retrieved_chunks = []

    retrieved_scores = []

    #------------------------ for fixedsize and semantic retrieval--------
    # for index, score in zip(indices[0],values[0]): 
    #     retrieved_chunks.append(chunks[index.item()])
    #     retrieved_scores.append(score.item())
    # --------------------------------------------------------------------

    # ------------------------ for hybrid retrieval---------------------- 
    for index, score in pairs:
        retrieved_scores.append(score)
    # --------------------------------------------------------------------

    # for fixed size and semantic retrieval
    # return retrieved_chunks, retrieved_scores


    # for hybrid retrieval
    return retrieved_scores


