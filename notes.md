Baseline RAG observations:

1.  Vector retrieval can find directly relevant chunks.

2.  Top-K retrieval can include weakly relevant or unrelated chunks.

3.  Vector retrieval can completely miss the chunk containing
    the answer.

4.  Similarity score does not always indicate exact relevance
    to the question.

5.  The LLM can produce a grounded answer when relevant context
    is available.

6.  The LLM can correctly refuse to answer when the retrieved
    context doesn't contain the required information.

7.  Even when relevant context is retrieved, the LLM may not
    use all relevant information optimally.

---

EVALUATE CORRECTNESS OF ANSWER USING EVIDENCE:

when we use an evidence to evaluate correctness of answer we just compare if evidence exist in retrieved chunk or not.
But, the evidence may exist in multiple chunks.
ex:
chunk : As with all the advice in this handbook, feel free to tailor these tips to your schedule, your document, and your budget. Not all of the tips will apply to everyone or to every document. Pick and choose the ones that make sense

            evidence: As with all the advice in this handbook, feel free to tailor these tips to your schedule, your document, and your budget. Not all of the tips will apply to everyone or to every document. Pick and choose the ones that make sense for you.

in this example last par ehich says ..that makes sense "for you." is missing in that particular chunk so the answer returned was false, which is not right.

so instead of giving entire evidence we break it into points
"evidence_points": [
"Investors will be more likely to understand what they are buying",
"Brokers and investment advisers can make better recommendations",
"Companies form stronger relationships with their investors"
]
convert top 3 chunsk into single txt and check how many evidence_points are present in txt.

normalize evidence_points and chunk using regex expression to remove all sepecial characters.
then compare and find count of found, total, coverage=found/total

using this calsulate average_coverage, and pass_rate

<!-- -------------------------------------------------------------------------------------------------- -->

SEMANTIC CHUNKING:

    In our normal chunking we use a fixed size(100 in our case), This method does not care about meaning.
        for example: in our case we had "...dealing with confused and sometimes" int one chunk and its remaining "angry investors." was missing, which led to incomplete sentence and meaning.

    To over come this we use SEMANTIC CHUNKING. in semantic chunking we "keep related sentence together and create a new chunk when topic/meaning changes.
    A simple approach to do this is:
            1.Split the document into sentences.
            2.Embed each sentence.
            3.Compare neighboring sentence embeddings.
            4.If similarity drops below a threshold → create a new chunk.
            5.Otherwise → continue adding sentences.
    we are using this instead of advanced method directly to first check if this improve our retrieval metric compared with 100-word fixed chunks.

    Semantic chunking has a disavantage here,
    we improved semantic chunking from basic chunking to moifying it as "min chunk size to 20 and limiting max chunk size to 200"
    this made our result a little better than normal semantic chunking, but its still not better than batch chunking.
    Therefore we go with hybrid chunking.

<!-- ------------------------------------------------------------------------------------------------------- -->

HYBRID Retrieval:

    in hybrid retrieval we will combine two signals:
        1.Semantic similarity — finds text with similar meaning.
        2.Keyword matching — finds text containing important words from the question.

    1.Why can keyword/BM25 search find something that vector search misses?
    ans:It checks whether the query's important words occur in the documents/chunks.
        example:
            What is the purpose of this handbook?
            Chunk A:
            This handbook reflects their substantial contributions...
            Chunk B:
            This handbook's purpose is to provide practical tips...
            Keyword/BM25 can recognize that "handbook" and "purpose" are strong query terms and that Chunk B contains them.
        So, BM25 focuses on matching important words/terms between the query and the chunk.

    2.Why can vector search find something that keyword search misses?
    ans:Vector search works on similarity of the meaning represented by the embeddings.
        example:
            Question:What are the advantages of plain English?
            Chunk:
            Investors are more likely to understand what they are buying and make informed judgments...
            The word "advantages" might not appear anywhere in the chunk.
            But the embedding model can recognize that.
        Vector search is good at finding relevant information even when the exact query words aren't present.

    3.Why would combining them potentially be better than either one alone?
    ans:Combining both would give us a better result than just using vectpor chunking as it checks for both vector similarity
        and keywords.
        example:
            Question:What are the benefits of plain English?
            BM25 might find:
            "The benefits of plain English abound..."
            because benefits and plain English match directly.

            Vector search might find:
            "Investors are more likely to understand what they are buying..."
            because benefits is semantically related to the outcomes described there.
        Hybrid search gets signals from both.

    "Bm25 does not need embeddings, instead it tokenizes them."
        Vector retrieval:question → embedding → similarity → top K
        BM25:question → tokenize → BM25 scores → top K

    $ calculate bm25_score and vector_score to find hybrid_score
    $ normalize these scores so it will be in the range of [0-1]: (score - minval) / (maxval - minval)
    $ find hybrid_score:
            hybrid_scores = [
            alpha * vector_score + (1 - alpha) * bm25_score
            for vector_score, bm25_score
            in zip(normalized_vector, normalized_bm25)
            ]
        --------------reranking-----------------
    vector + BM25 -> 78% coverage which is lower than batch chunking.
    so now we intoduce "reranking"
            Question
            ↓
            Initial Retrieval
            (BM25 + Vector / Hybrid)
            ↓
            Retrieve more candidates
            (e.g. Top-10)
            ↓
            Re-ranker
            ↓
            Re-ranked candidates
            ↓
            Final Top-3
            ↓
            LLM

    to rerank we use cross-encoder.
    difference between hybrid similaity scorer and cross encoder is that,
        $ similarity wiil embedd questions and chunks then checks for similarity and gives a score.
            Question → vector
            Chunk    → vector
                ↓
            similarity score
        $ but, cross-encoder gets both question and chunk together cross encodes it and gives a score.
            Question + Chunk
                ↓
            Cross-Encoder
                ↓
            Relevance score

    ----------------conclusion of retrieval------------------
        1. Fixed-size + Vector is already very strong on this dataset.
        Your baseline achieved 92% / 100%.

        2. Semantic chunking did not perform better.
        Your tested thresholds produced lower retrieval performance, so there's no reason to choose semantic chunking simply because it sounds more advanced.

        3. Hybrid retrieval alone was worse than the vector baseline.
        Your best hybrid configuration was α = 0.7, giving 78.67% / 80%.

        4. Hybrid + re-ranking recovered the baseline performance.
        With α = 0.7, Top-10 candidates and final Top-3 re-ranked chunks, you reached 92% / 100%.

    "On the current 5-question evaluation dataset, hybrid retrieval followed by cross-encoder re-ranking achieved 92% average evidence coverage and a 100% retrieval pass rate, matching the fixed-size vector-search baseline."
