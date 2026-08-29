from sentence_transformers import CrossEncoder


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, candidate_chunks, k=3):
    pairs = [[question, chunk] for chunk in candidate_chunks]

    scores = model.predict(pairs)

    results = list(zip(candidate_chunks, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:k]




# question = "What is the purpose of this handbook?"

# chunks = [
#     "This handbook reflects their substantial contributions and those of highly regarded experts in the field.",
    
#     "This handbook gives you practical tips on how to create plain English documents.",
    
#     "The benefits of plain English abound. Investors will be more likely to understand what they are buying.",
#     "Using plain English assures the orderly and clear presentation of complex information so that investors have the best possible chance of understanding it. Plain English means analyzing and deciding what information investors need to make informed decisions, before words, sentences, or paragraphs are considered. A plain English document uses words economically and at a level the audience can understand. Its sentence structure is tight. Its tone is welcoming and direct. Its design is visually appealing. A plain English document is easy to read and looks like it’s meant to be read. a plain english handbook 5 This handbook’s purpose This",
#     "easy to read and looks like it’s meant to be read. a plain english handbook 5 This handbook’s purpose This handbook gives you practical tips on how to create plain English documents. All of these were born of experience. They come from experts and those who have already written or rewritten their docu­ ments in plain English. As with all the advice in this handbook, feel free to tailor these tips to your schedule, your document, and your budget. Not all of the tips will apply to everyone or to every document. Pick and choose the ones that make sense",
#     "once it is at the printer, making text changes can be tedious and expensive. If you don’t have a design professional, fear not. You can apply many of the simple concepts discussed in this chapter to produce a readable, visually appealing document. While the field of design extends broadly, this chapter covers five basic design elements and how they contribute to creating a plain English document: • hierarchy or distinguishing levels of information • typography • layout • graphics • color 38 a plain english handbook Hierarchy Much like an outline, a document’s hierarchy shows how you’ve orga­ nized the"
     
# ]

# results = rerank(question, chunks)
# for rank, (chunk, score) in enumerate(results, start=1):
#     print(f"Rank {rank}: Score {score}")
#     print(chunk)
#     print("-" * 80)
