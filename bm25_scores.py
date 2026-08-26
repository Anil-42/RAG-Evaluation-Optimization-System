from rank_bm25 import BM25Okapi
import numpy as np

def bm25_scores(question, chunks):
    # 1. Tokenize query and chunks
    tokenized_query = question.split()
    tokenized_chunks = [chunk.split() for chunk in chunks]

    # 2. Initialize BM25 and get scores
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)

    return scores

