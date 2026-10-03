from sentence_transformers import SentenceTransformer,util


model=SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_input):
    return model.encode(text_input)


def get_similarities(embedding_a,embedding_b):
    return util.cos_sim(embedding_a,embedding_b)
