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

HYBRID CHUNKING:

    in hybrid retrieval we will combine two signals:
        1.Semantic similarity — finds text with similar meaning.
        2.Keyword matching — finds text containing important words from the question.

    1.Why can keyword/BM25 search find something that vector search misses?
    2.Why can vector search find something that keyword search misses?
    3.Why would combining them potentially be better than either one alone?
