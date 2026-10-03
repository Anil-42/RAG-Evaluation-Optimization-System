import re
import numpy as np
from src.retrieval.embedding import get_embeddings
from src.retrieval.embedding import get_similarities    

def split_sentences(text):
    # Split by sentence-ending punctuation followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip()) # [.!?] and \s+ looks for sentence ending with .!? and a space after it. 
    # Normalize whitespaces inside sentences
    sentences = [re.sub(r'\s+',' ',s).strip() for s in raw_sentences if s.strip()] 
    return sentences


def semantic_chunking(sentences, sentence_embeddings, threshold):
    chunks=[]
    current_chunk = [sentences[0]]

    for i in range(len(sentences)-1):
        similarity = get_similarities(
            sentence_embeddings[i],
            sentence_embeddings[i+1]
        ).item()
        min_words=20
        max_words=200
        current_chunk_size = len(" ".join(current_chunk).split())

        if similarity<threshold and current_chunk_size>=min_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        elif current_chunk_size>=max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
            

    chunks.append(" ".join(current_chunk))

    return chunks
