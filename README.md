# RAG Evaluation & Optimization System

An experimental Retrieval-Augmented Generation (RAG) system designed to **evaluate, compare, diagnose, and optimize retrieval strategies** rather than simply build a "chat with PDF" application.

The project uses the **SEC's A Plain English Handbook** as the source document and evaluates different retrieval approaches against a manually constructed benchmark of **46 questions and approximately 95 evidence points**.

The primary goal is to experimentally determine:

> **Which retrieval strategy retrieves the information required to answer a question, and what optimizations improve retrieval reliability?**

---

## 1. Project Overview

Retrieval-Augmented Generation systems depend heavily on the quality of the information retrieved before an LLM generates an answer.

A system can produce an incorrect answer for two fundamentally different reasons:

1. **Retrieval failure**
   The required information was not retrieved.

2. **Generation failure**
   The required information was retrieved, but the LLM failed to use it correctly.

This project investigates both problems separately.

The project follows an experimental progression:

```text
Basic RAG
   ↓
Evaluation Dataset
   ↓
Retrieval Benchmark
   ↓
Retrieval Strategy Comparison
   ↓
Failure Diagnosis
   ↓
Targeted Optimization
   ↓
Generation Evaluation
```

Instead of assuming that a more advanced retrieval technique is automatically better, each technique is evaluated using the same benchmark.

---

# 2. Problem Statement

A basic RAG pipeline may retrieve semantically similar chunks, but semantic similarity alone does not guarantee that the exact evidence required to answer a question will be retrieved.

For example:

- An important phrase may contain specific keywords that semantic search ranks poorly.
- A keyword-heavy query may be better handled by lexical search.
- Important information may be split across neighboring chunks.
- A relevant chunk may be retrieved, but the surrounding context needed to understand it may be missing.
- Even when the correct evidence is retrieved, an LLM may incorrectly claim that the answer is unavailable.

Therefore, the project investigates both:

**Retrieval quality**

and

**Grounded answer generation quality.**

---

# 3. Objectives

The main objectives are:

- Build a working baseline RAG pipeline.
- Construct a verified evaluation dataset.
- Compare multiple retrieval strategies experimentally.
- Measure evidence coverage rather than relying only on similarity scores.
- Diagnose why retrieval failures occur.
- Test targeted retrieval optimizations.
- Evaluate the generated answers separately from retrieval.
- Preserve reproducible results for future experimentation.

---

# 4. Source Document

The experiments use:

**A Plain English Handbook**

The handbook contains guidance related to:

- Plain English
- Disclosure documents
- Writing style
- Audience analysis
- Organization
- Readability
- Sentence construction
- Active voice
- Headers and subheaders
- Financial concepts
- Investor communication
- Document evaluation

The document is converted from PDF into text before being processed by the RAG pipeline.

---

# 5. System Architecture

The current experimental pipeline is:

```text
                    PDF Document
                         │
                         ▼
                  Text Extraction
                         │
                         ▼
                    Chunking
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Fixed Chunking         Semantic Chunking
             │                       │
             ▼                       ▼
         Embeddings             Embeddings
             │                       │
             ▼                       ▼
       Vector Retrieval       Semantic Retrieval
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Hybrid Retrieval
                  Vector + BM25
                         │
                         ▼
                  Candidate Chunks
                         │
                         ▼
                 Cross-Encoder
                    Reranking
                         │
                         ▼
               Neighbor Reconstruction
                         │
                         ▼
                 Retrieved Context
                         │
                         ▼
                    LLM Generation
                         │
                         ▼
                    Final Answer
                         │
                         ▼
                 Answer Evaluation
```

Not every stage is necessarily enabled simultaneously. The project treats each retrieval strategy as an experiment so that its effect can be measured independently.

---

# 6. Basic RAG Pipeline

The initial RAG pipeline follows:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Retrieval
 ↓
Retrieved Context
 ↓
LLM
 ↓
Answer
```

The initial fixed chunking configuration used:

- Chunk size: **100 words**
- Overlap: **20 words**

This provided the baseline against which later retrieval strategies were compared.

---

# 7. Retrieval Strategies Compared

The project experimentally compares five configurations.

## 7.1 Fixed Vector Retrieval

The baseline approach.

The document is divided into fixed-size chunks and embedded.

The query is embedded and compared against the chunk embeddings.

```text
Question
   ↓
Question Embedding
   ↓
Vector Similarity
   ↓
Top-k Chunks
```

This establishes the baseline retrieval performance.

---

## 7.2 Semantic Vector Retrieval

Instead of relying only on fixed-size boundaries, semantic chunking is used to create more meaningful chunks.

The goal is to keep related content together and reduce cases where important information is split across arbitrary chunk boundaries.

---

## 7.3 Hybrid Retrieval

Hybrid retrieval combines:

- Semantic/vector similarity
- BM25 lexical retrieval

The two scores are normalized before being combined.

The hybrid score is:

```text
Hybrid Score =
    α × Semantic Score
    +
    (1 - α) × BM25 Score
```

The experiments found that:

```text
α = 0.2
```

performed best among the tested alpha values.

The tested alpha values were:

```text
0.1
0.2  ← best tested value
0.3
0.4
0.5
0.7
0.9
```

At α = 0.2, the semantic component contributes 20% and the BM25 component contributes 80%.

This result suggests that lexical matching was particularly valuable for this benchmark, while semantic retrieval still contributed useful information.

---

## 7.4 Hybrid + Cross-Encoder Reranking

Hybrid retrieval first produces a candidate set.

A cross-encoder then scores each:

```text
(question, candidate_chunk)
```

pair directly.

The project uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The purpose of reranking is to improve the ordering of the retrieved candidates according to question-chunk relevance.

However, an important experimental result was observed:

> **Reranking did not improve the aggregate evidence-retrieval metrics in the final benchmark.**

Therefore, the project does **not** claim that reranking automatically improves this RAG system.

---

## 7.5 Hybrid + Neighbor Merging

The final optimization addressed a different problem.

Some evidence was located close to a retrieved chunk but was not contained in the selected chunk itself.

Instead of treating every chunk independently, neighboring chunks can be reconstructed into a larger context:

```text
Previous Chunk
      +
Retrieved Chunk
      +
Next Chunk
```

This preserves local document continuity and helps recover evidence that was split across chunk boundaries.

This optimization produced the strongest retrieval result in the final benchmark.

---

# 8. Evaluation Dataset

A major part of the project is the evaluation dataset.

The benchmark contains:

- **46 questions**
- Approximately **95 evidence points**
- Ground-truth answers
- Source information
- Supporting evidence

Each evaluation item contains information such as:

```json
{
    "question": "...",
    "answer": "...",
    "source": "...",
    "evidence": "...",
    "evidence_points": [...]
}
```

The `evidence_points` provide granular pieces of information that should be recoverable from the source document.

This allows retrieval to be evaluated based on whether the **required evidence was actually retrieved**, rather than simply whether the retrieved chunk had a high similarity score.

---

# 9. Retrieval Evaluation Methodology

Three retrieval metrics are used.

## Evidence Coverage

Measures how much of the required evidence points were found in the retrieved context.

```text
Evidence Coverage =
retrieved evidence points
/
total required evidence points
```

---

## Retrieval Pass

A question passes retrieval when the required retrieval condition defined by the evaluation system is satisfied.

This measures question-level retrieval success rather than only individual evidence points.

---

## Full Retrieval

Measures the percentage of questions for which **all required evidence points** were successfully retrieved.

This is especially useful because partial retrieval can still result in an incomplete answer.

---

# 10. Final Retrieval Results

The final comparison uses:

- 46 evaluation questions
- `k = 7`
- The same benchmark across retrieval strategies
- Hybrid α = 0.2
- Hybrid candidate set = 10
- Hybrid + reranking candidate set = 20
- Neighbor merging evaluated as a separate optimization

### Results

| Method                    |   k | Evidence Coverage | Retrieval Pass | Full Retrieval |
| ------------------------- | --: | ----------------: | -------------: | -------------: |
| Fixed Vector              |   7 |            80.07% |         84.78% |         73.91% |
| Semantic Vector           |   7 |            82.97% |         84.78% |         80.43% |
| Hybrid                    |   7 |        **91.67%** |     **93.48%** |     **89.13%** |
| Hybrid + Reranking        |   7 |        **91.67%** |     **93.48%** |     **89.13%** |
| Hybrid + Neighbor Merging |   7 |       **100.00%** |    **100.00%** |    **100.00%** |

---

# 11. Results Analysis

## Fixed Vector → Semantic Vector

Semantic chunking improved:

```text
Evidence Coverage:
80.07% → 82.97%

Full Retrieval:
73.91% → 80.43%
```

This indicates that more meaningful chunk boundaries helped preserve relevant information.

However, semantic chunking alone did not solve all retrieval problems.

---

## Semantic Vector → Hybrid

Hybrid retrieval produced a substantial improvement:

```text
Evidence Coverage:
82.97% → 91.67%

Retrieval Pass:
84.78% → 93.48%

Full Retrieval:
80.43% → 89.13%
```

This was one of the clearest improvements in the experiments.

The result demonstrates the value of combining:

```text
Semantic similarity
+
Lexical matching
```

rather than relying exclusively on vector similarity.

---

## Hybrid → Hybrid + Reranking

The aggregate benchmark produced:

```text
Evidence Coverage:
91.67% → 91.67%

Retrieval Pass:
93.48% → 93.48%

Full Retrieval:
89.13% → 89.13%
```

Therefore:

> **Cross-encoder reranking did not improve the aggregate retrieval metrics in this experiment.**

This is an important finding rather than a failure of the project.

It demonstrates why retrieval components should be evaluated experimentally instead of assuming that adding a more advanced model will automatically improve the system.

---

## Hybrid + Neighbor Merging

Neighbor-aware context reconstruction increased the final retrieval performance to:

```text
Evidence Coverage: 100%
Retrieval Pass:    100%
Full Retrieval:    100%
```

All **46 evaluation questions** achieved complete evidence retrieval after neighbor merging.

The improvement occurred because some retrieval failures were caused by **document continuity and chunk boundaries**, rather than simply poor semantic or lexical matching.

---

# 12. Retrieval Failure Diagnosis

The project did not stop at measuring the scores.

Individual failures were inspected to determine why evidence was missed.

For example, one question had important evidence in chunk 21.

The chunk's ranks were approximately:

```text
Semantic rank: 29
BM25 rank:     27
Hybrid rank:   24
```

With a candidate cutoff of 10, the chunk was not selected.

Other failures showed similar behavior.

Some evidence had:

- poor semantic ranking,
- poor lexical ranking,
- a relevant neighboring chunk already retrieved,
- or evidence split across chunk boundaries.

This led to the important distinction:

```text
Retrieval ranking problem
        vs.
Context reconstruction problem
```

The neighbor-merging optimization addressed the second category.

---

# 13. Alpha Experiment

The hybrid retrieval weight was also experimentally evaluated.

|       α | Evidence Coverage | Retrieval Pass |
| ------: | ----------------: | -------------: |
|     0.1 |            83.33% |         86.96% |
| **0.2** |        **88.41%** |     **91.30%** |
|     0.3 |            86.23% |         89.13% |
|     0.4 |            87.32% |         89.13% |
|     0.5 |            85.14% |         86.96% |
|     0.7 |            85.14% |         86.96% |
|     0.9 |            76.09% |         78.26% |

α = 0.2 was selected because it produced the best result among the tested configurations.

This should not be interpreted as a universal optimal value. It is the best value **for this benchmark and the tested configurations**.

---

# 14. Reranking Cutoff Experiment

The reranking stage was also tested with different final `k` values.

|     k | Evidence Coverage | Retrieval Pass |
| ----: | ----------------: | -------------: |
|     1 |            68.84% |         71.74% |
|     3 |            84.42% |         86.96% |
|     5 |            89.49% |         91.30% |
| **7** |        **93.84%** |     **95.65%** |
|    10 |            93.84% |         95.65% |

For the tested configurations, `k = 7` was the smallest cutoff that reached the maximum observed performance.

The final comparison standardized the benchmark around `k = 7`.

---

# 15. Generation Evaluation

Retrieval quality alone does not guarantee a good final answer.

Therefore, the project also evaluates generated answers using an LLM-based evaluator.

Three independent metrics are used.

## Correctness

Does the generated answer actually answer the question correctly?

```text
3 = Fully correct
2 = Mostly correct
1 = Partially correct with an important issue
0 = Incorrect / does not answer
```

---

## Faithfulness

Are the claims made by the generated answer supported by the retrieved context?

```text
3 = Fully supported
2 = Mostly supported
1 = Some important unsupported content
0 = Largely unsupported
```

---

## Completeness

Does the answer include the important information contained in the ground-truth answer?

```text
3 = Complete
2 = Minor information missing
1 = Substantial information missing
0 = Essentially missing
```

The metrics are evaluated independently.

In particular:

> Missing information should primarily reduce **Completeness**, rather than automatically reducing Correctness or Faithfulness.

Faithfulness is evaluated against the **retrieved context**, while Completeness is evaluated against the **ground truth**.

---

# 16. Generation Evaluation Results

The final 46-question generation evaluation produced:

| Metric       |     Score | Percentage |
| ------------ | --------: | ---------: |
| Correctness  | 131 / 138 | **94.93%** |
| Faithfulness | 135 / 138 | **97.83%** |
| Completeness | 128 / 138 | **92.75%** |

The results show that the system generally produces grounded answers when relevant context is available.

However, four questions received non-perfect evaluations.

---

# 17. Important Generation Failures

The generation evaluation revealed an important pattern.

Some questions had the correct evidence **already present in the retrieved context**, but the LLM still failed to use it correctly.

For example:

### Descriptive headers and subheaders

The retrieved context explicitly explained that descriptive headers and subheaders:

- break documents into manageable sections
- tell readers what upcoming sections will cover

However, the generated answer incorrectly stated that the context did not provide a clear answer.

This is a **generation/context-use failure**, not a retrieval failure.

---

### Active voice and strong verbs

The retrieved context explained that active voice and strong verbs can make sentences:

- shorter
- easier to understand
- less confusing

The generated answer captured the main idea but did not fully express all of the important ground-truth information.

This is primarily a **partial generation/completeness issue**.

---

### Why should writers use short sentences?

The retrieved context explicitly stated that longer and more complex sentences are harder for readers to understand.

The generated answer still claimed that the context did not provide a specific reason.

Again, the required evidence was retrieved.

This demonstrates:

```text
Evidence available
       ↓
LLM fails to use evidence
       ↓
Generation failure
```

---

### Multiple "if" and "then" statements

The retrieved context contained the specific recommendation:

- break the information into multiple sentences,
- clarify which `if` applies to which `then`,
- and consider using a table if the information remains unclear.

The generated answer incorrectly said that the context did not provide the required answer.

This is a clear generation failure.

---

# 18. Key Finding: Retrieval vs Generation

One of the most important outcomes of this project is the distinction between:

```text
Retrieval Failure
```

and

```text
Generation Failure
```

### Retrieval Failure

```text
Question
   ↓
Required evidence
   ↓
Evidence NOT retrieved
   ↓
LLM cannot reliably answer
```

### Generation Failure

```text
Question
   ↓
Required evidence
   ↓
Evidence successfully retrieved
   ↓
LLM fails to use / interpret it
   ↓
Incorrect or incomplete answer
```

This distinction is important because the solution is different.

Improving retrieval will not necessarily fix an LLM that ignores information already present in its context.

---

# 19. Project Structure

```text
RAG-Evaluation-Optimization-System/
│
├── documents/
│   └── sample.pdf
│
├── src/
│   ├── main.py
│   │
│   ├── chunking/
│   │   ├── chunking.py
│   │   └── semantic_chunking.py
│   │
│   ├── extraction/
│   │   └── extract_text.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   └── retrieval/
│       ├── bm25_scores.py
│       ├── embedding.py
│       ├── fixed_vector.py
│       ├── hybrid.py
│       ├── hybrid_rerank.py
│       ├── neighbor_merge.py
│       ├── normalize.py
│       ├── normalize_scores.py
│       ├── reranker.py
│       ├── retrieve.py
│       └── semantic_vector.py
│
├── evaluation/
│   ├── evaluation.py
│   ├── evidence_coverage.py
│   ├── llm_evaluator.py
│   └── questions.json
│
├── results/
│   ├── evaluation_result.json
│   └── result.json
│
├── notes/
│   └── (ignored development files)
│
├── backups/
│   └── (ignored development files)
│
├── requirements.txt
├── README.md
└── .gitignore
```

The `notes/` and `backups/` directories are development-only and are excluded from version control.

---

# 20. Installation

## 20.1 Clone the repository

```bash
git clone https://github.com/Anil-42/RAG-Evaluation-Optimization-System
cd RAG-Evaluation-Optimization-System
```

---

## 20.2 Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

## 20.3 Install Python dependencies

```powershell
pip install -r requirements.txt
```

The main dependencies are:

```text
ollama
numpy
pymupdf
rank-bm25
sentence-transformers
torch
```

---

# 21. Ollama Setup

The project uses Ollama for local LLM generation and evaluation.

Install Ollama separately and make sure it is available from the terminal.

Then download the model:

```powershell
ollama pull llama3.2
```

Verify that it works:

```powershell
ollama run llama3.2
```

Example:

```text
>>> 2+2
4
```

The project uses:

```text
llama3.2
```

for:

- RAG answer generation
- LLM-based answer evaluation

The Ollama model is not included in `requirements.txt` because it is an external runtime/model rather than a Python package.

---

# 22. Running the Project

The main experimental pipeline is started through:

```powershell
python src/main.py
```

The evaluation dataset is loaded from:

```text
evaluation/questions.json
```

Generated results are saved to:

```text
results/result.json
```

LLM-based evaluation results are saved to:

```text
results/evaluation_result.json
```

The project uses paths relative to the project structure so that execution does not depend on the current working directory.

---

# 23. Reproducibility

The project records the evaluation dataset and generated results so that retrieval experiments can be compared consistently.

Important experimental settings should remain explicit, including:

```text
Number of evaluation questions: 46
Final retrieval k:               7
Hybrid alpha:                    0.2
Hybrid candidate_k:              10
Reranking candidate_k:           20
Cross-encoder:                   ms-marco-MiniLM-L-6-v2
LLM:                              llama3.2
```

The benchmark is intended to compare retrieval strategies under controlled conditions rather than changing multiple components simultaneously.

---

# 24. What the Experiments Demonstrated

The experiments produced several important conclusions.

### 1. Semantic chunking helped

Semantic chunking improved full retrieval compared with the fixed-size baseline.

```text
73.91% → 80.43%
```

---

### 2. Hybrid retrieval produced a substantial improvement

Combining BM25 and semantic retrieval increased full retrieval:

```text
80.43% → 89.13%
```

---

### 3. Reranking was not automatically beneficial

Cross-encoder reranking produced the same aggregate final retrieval metrics as hybrid retrieval in the final comparison.

Therefore, the experiment demonstrates that:

> Adding a more advanced retrieval component does not guarantee better performance on a particular benchmark.

---

### 4. Chunk boundaries were a significant source of failure

Some missed evidence was located immediately before or after retrieved chunks.

Neighbor-aware context reconstruction solved these cases.

---

### 5. Neighbor merging produced complete benchmark retrieval

The optimized retrieval configuration achieved:

```text
100% Evidence Coverage
100% Retrieval Pass
100% Full Retrieval
```

across all 46 questions.

---

### 6. Retrieval quality and generation quality are different problems

The generation evaluation demonstrated that an LLM can fail even when the required evidence is already present in the retrieved context.

This means future improvements should target both:

```text
Retrieval
```

and

```text
Generation / Context Utilization
```

rather than treating every incorrect answer as a retrieval problem.

---

# 25. Limitations

This project is an experimental evaluation system rather than a production-ready general-purpose RAG application.

Current limitations include:

- Evaluation is based on one primary source document.
- The benchmark contains 46 questions.
- Retrieval metrics are specific to the constructed evidence-point dataset.
- The best hybrid alpha was determined only among tested values.
- The 100% retrieval result depends on the current benchmark and neighbor reconstruction strategy.
- Cross-encoder reranking was not beneficial on the final aggregate benchmark.
- Generation evaluation still contains some evaluator-vs-human judgment differences.
- The current system is not yet designed as a fully general arbitrary-document RAG application.

These limitations are intentionally preserved because they define the next stages of experimentation.

---

# 26. Future Work

The current project establishes a strong experimental foundation.

The next version can evolve from a benchmark-focused RAG system into a more advanced general-purpose architecture.

Planned direction:

```text
                         User Question
                              │
                              ▼
                  ┌──────────────────────┐
                  │   Pre-Retrieval      │
                  │                      │
                  │ Query Understanding  │
                  │ Multi-Query Expansion│
                  │ Route Planning       │
                  │ Entity Extraction    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Structural Retrieval │
                  │                      │
                  │ Vector Database      │
                  │        +             │
                  │ Neo4j Knowledge Graph│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Mid-Retrieval      │
                  │                      │
                  │ Candidate Retrieval  │
                  │ Cross-Encoder        │
                  │ Reranking            │
                  │ Information Density  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Post-Retrieval      │
                  │                      │
                  │ Answer Grading       │
                  │ Retrieval Grading    │
                  │ Corrective Search    │
                  │ Query Reformulation  │
                  └──────────┬───────────┘
                             │
                             ▼
                            LLM
                             │
                             ▼
                    Grounded Final Answer
```

Potential future stages include:

### Pre-Retrieval

- Multi-query expansion
- Query decomposition
- Route planning
- Named-entity extraction

### Structural Retrieval

- Vector database retrieval
- Knowledge graph retrieval
- Vector + graph hybrid retrieval using Neo4j

### Mid-Retrieval

- Cross-encoder reranking
- Information-density filtering
- Better context selection

### Post-Retrieval

- Retrieval grading
- Answer grading
- Corrective retrieval
- Query reformulation
- Self-correcting RAG loops

### Evaluation

- Larger evaluation datasets
- Automated regression testing
- Retrieval quality thresholds
- Generation quality thresholds
- GitHub Actions that fail when evaluation quality falls below an accepted threshold

The important principle is to introduce these components **one at a time and measure their effect**, rather than building a complex RAG system without knowing which components actually improve performance.

---

# 27. Development Philosophy

This project follows an experimental approach:

```text
Build
 ↓
Measure
 ↓
Diagnose
 ↓
Change one component
 ↓
Measure again
 ↓
Keep or reject the change
```

A more complex method is not automatically considered better.

Every optimization should answer:

> **What problem does this component solve, and can the evaluation demonstrate that it actually solved it?**

This makes the project an evaluation and optimization system rather than simply an implementation of existing RAG techniques.

---

# 28. Conclusion

This project began as a basic RAG pipeline and evolved into an experimental framework for understanding where RAG systems succeed and fail.

The final retrieval experiments showed:

```text
Fixed Vector              73.91% Full Retrieval
        ↓
Semantic Vector           80.43%
        ↓
Hybrid                    89.13%
        ↓
Hybrid + Reranking        89.13%
        ↓
Hybrid + Neighbor Merge  100.00%
```

The most important lesson is not simply that the final system achieved 100%.

The experiments showed **why** performance changed:

- semantic chunking improved document representation,
- hybrid retrieval improved lexical + semantic matching,
- reranking did not improve this particular benchmark,
- chunk continuity caused several remaining retrieval failures,
- neighbor-aware reconstruction recovered the missing evidence,
- and generation evaluation revealed that retrieved evidence can still be ignored or misused by the LLM.

The project therefore provides a foundation for moving toward a more advanced, corrective, and structurally aware RAG system while keeping each improvement experimentally measurable.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
