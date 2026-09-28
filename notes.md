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

<!-- ------------------------------------------------------ -->

fixedsize and hybrid retrieval worked well for 5 questions, but when data set size became large(40-50) it failed in most cases. Therefore we must evaluate that first

We are using data sets for testing rag than blindly writing the code without evaluating it and implementing it directly.

after modifying the data set it worked much better.
Method: fixed_vector
Average evidence coverage: 69.20%
Retrieval pass rate: 73.91%

---

Method: hybrid_rerank
Average evidence coverage: 73.55%
Retrieval pass rate: 78.26%
But there are few cases where vector has 100% but in hybrid/reranking 0%, and
in few cases vector 0% and hybrid 100%

difference between vector and hybrid reranking is:
73.55 − 69.20 = 4.35 percentage points

Don't say:"Hybrid + reranking is better because 73.55% > 69.20%."
That's partially true, but incomplete.

TORCH.TOPK() gives values, indices in DECSENDING order. for hybridretrieval it must be in the same order as chunks, As thats how BM25 retrieves
so for hybrid we sort based on indices to restore original order of chunks after torch.topk().

<!-- After solvibg above problem -->

    Method: hybrid
    Average evidence coverage: 82.25%
    Retrieval pass rate: 86.96%

    Method: hybrid_rerank
    Average evidence coverage: 84.78%
    Retrieval pass rate: 89.13%

hybrid -> hybrid reranking
Evidence coverage: +2.53 percentage points
Pass rate: +2.17 percentage points
Failures: 6 → 5

Even after reanking there might be a possibility where the ans chunk is in top 10 but the reranker fails to get it to top 3.
five failure cases show that reranking isn't universally better. For example, it can push a relevant chunk downward, as we saw with chunk 22.

- in one case the chunk is in top 1 in case of both hybrid and reranking but its evidence coverage is 0.
  this happens because, while normalizing the evidence we remove every space and special characters,
  Evidence:
  The longer and more complex a sentence
  normalized:
  thelongerandmorecomplexasentence

      this is not wrong but it makes substring matching somewhat fragile.

- and one more thing is
  if normalize(point) in normalized_chunks:
  here we have a case in which a point contiunes from one chunk to next chunk
  But it requires the entire evidence point to exist inside one retrieved chunk.
  - There are two possible approaches to rectify this:
    - Change your evaluation dataset's evidence_points so that each evidence point corresponds to text that can actually occur within a chunk.

    but this is not efficient way as we cannot do it for al the questions manually.
    - Make evidence_coverage() capable of recognizing evidence that is split across chunks.

    for this we introduce another design decision: how much of an evidence point must be retrieved to count as covered?
    That's something we should define carefully because this metric is the foundation of your experiment.

---

HYBRID + RERANKING + NEIGHBOUR EXPANSION

The problem we have is if chunk 79 is present in top 10 but the evidence point continues in chunk 80 its neighbour then the retrieval fails.

Our next optimization hypothesis

We now have a measurable hypothesis:
If we add neighboring chunks around retrieved candidates before reranking, Top-3 evidence coverage should improve.
This is called neighbor-aware retrieval / context expansion.

Current:

Hybrid
↓
Top 10 chunks
↓
Reranker
↓
Top 3

Proposed experiment:

Hybrid
↓
Top 10 chunks
↓
Add neighboring chunks
↓
Expanded candidate pool
↓
Reranker all expanded candidates
↓
Take Top 3

As the above experiment didnt produce much of difference than hybrid + reranking.

instead of merging and then reranking we rerank top 3 first and then we merge their neighbors.

PDF
↓
100-word chunks + 20 overlap
↓
Vector + BM25
↓
Hybrid
↓
Top 10 candidates
↓
Cross-encoder reranking
↓
Top 3
↓
Add immediate neighbors
↓
Evidence Coverage

this produced 95% evidence coverage

=============================================================================================================================

Hybrid retrieval substantially improves evidence retrieval compared with vector-only retrieval, while cross-encoder reranking further improves the ordering of relevant chunks, particularly when only a small number of chunks are used. Neighbor-aware evaluation addresses evidence that naturally spans adjacent chunks without changing the retrieval ranking itself.

=============================================================================================================================

Testing the same 46 questions with "semantic chunking" instead of vector chunksing got similar result that is 91% evidence coverage.

1. Hybrid retrieval is failing on 3 questions

These are genuine candidate-retrieval failures:

Who can work on a plain English project? → 0/3
What is the first principle of good document organization? → 0/1
What should writers do when a conditional statement contains multiple ifs and thens? → 1/2

2. Reranking is losing evidence on 5 questions

These are the important ones:

## Question Hybrid Reranked Lost

Investors will read a document 1/1 0/1 Yes
Common problems in disclosure documents 9/9 0/9 Yes — huge
Active voice + strong verbs 3/3 2/3 Yes
Abstract → concrete terms 2/2 0/2 Yes
Creating new acronyms 2/2 0/2 Yes

After evaluating a lot we say that,
The reranker is not consistently bad. It is making a different judgment about which chunks are relevant.

    And because your evaluation metric is evidence coverage, sometimes that judgment agrees with your benchmark and sometimes it doesn't.

we can say that The cross-encoder often identifies relevant chunks but ranks evidence-bearing chunks too low for a Top-3 cutoff.
as evaluation of,
k=1 -> 0
k=3 -> 53.33
k=5,7,10 -> 100

therefore we can conclude that the cross-encoder ranks chunks low for top-3 cutoff

k=10 result reaching 100% on the five diagnostic questions makes sense because your candidate pool itself contains the required evidence.

Reranker diagnosis: On the five manually investigated failure cases, increasing reranker output k from 3 to 10 recovered evidence that was already present in the Hybrid Top-10 candidate pool. This suggests that the current reranker problem is substantially a ranking/cutoff problem, rather than purely a candidate-retrieval problem.

Your current best pipeline

Based on everything you've tested so far:

                 PDF
                  ↓
            Text Extraction
                  ↓
           Semantic Chunking
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓

Vector Search BM25
↓ ↓
└─────────┬─────────┘
↓
Hybrid Retrieval
α = 0.2
↓
Top 10 Candidates
↓
Cross-Encoder Reranker
↓
Top 7 Chunks
↓
LLM / Answer

"On the 46-question evaluation benchmark, k=7 was the smallest tested reranking cutoff that achieved the maximum observed performance of 93.84% evidence coverage and 95.65% retrieval pass rate. Increasing k from 7 to 10 produced no additional improvement."

after neighbor merging only one question failed.
it failed as non of its evidence point chunks were retrieved.
but these evidence are present in the original document

============================================================
FAILED / PARTIAL RETRIEVAL
============================================================
Question: Who can work on a plain English project according to the handbook?
Evidence found: 0
Evidence total: 3
Coverage: 0.0

Evidence points:

- Many of you routinely select a team to think and talk about how to write a document from scratch or rewrite an existing document.
- Or you may do it on your own.
- In that case, rest assured that one person can do it alone.

============================================================
EVALUATION RESULTS
============================================================
Method: hybrid_rerank
Questions evaluated: 46
Average evidence cover: 97.83%
Retrieval pass rate: 97.83%
Full evidence retrieval: 97.83%
============================================================

so now lets test it by increasing candidate_k from 10 to 20.
