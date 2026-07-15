# RetrievalBench Design and Experiment Specification

## 1. Purpose

RetrievalBench studies how document segmentation and embedding-model selection interact in cross-domain retrieval evaluation.

The central research question is:

> When a corpus must be segmented before indexing, how much does chunking strategy change retrieval quality relative to embedding-model choice, and does that relationship generalize across domains?

This document describes the experiment that has actually been completed and reported. It replaces the earlier project proposal, which included planned latency, cost, structure-aware chunking, synthetic-query, and end-to-end generation experiments that were not completed at the same evidentiary level.

## 2. Evidence Boundary

### 2.1 Core completed study

The paper's central evidence consists of:

- 12 retrieval domains
- 4 operational chunking strategies
- 3 dense embedding models
- BM25 as a sparse reference
- NDCG@10 with bootstrap confidence intervals
- per-domain CSI/ESI variance decomposition

Supported conclusions:

1. Changing only the chunking strategy changes NDCG@10 by up to 47.1% relative in the observed matrix.
2. The best chunking strategy varies across domains and retrieval models.
3. Chunking sensitivity exceeds embedding sensitivity in 5 of 12 domains under the evaluated candidate pool.
4. Embedding sensitivity exceeds chunking sensitivity in 7 of 12 domains.

### 2.2 Exploratory capabilities

The repository also implements or archives:

- cross-encoder reranking
- latency-adjusted NDCG
- OpenAI token and API-cost metering
- hybrid fusion
- structure-aware legal, financial, and medical chunkers
- ranked-list saturation analysis
- synthetic hardware-aware scheduling utilities

These capabilities are not treated as equally validated headline findings in the current paper. In particular, archived latency measurements do not share a fully production-complete timing boundary, some reranker runs produced unchanged rankings, and structure-aware chunkers were not evaluated across the core matrix.

## 3. Datasets

### 3.1 Core domains

| Domain | Dataset key | Source type | Documents | Queries |
|---|---|---|---:|---:|
| Finance | `fiqa` | BEIR | 2,000 | 500 |
| Legal | `auslegalqa` | Open Australian Legal QA | 2,099 | 2,124 |
| Biomedical | `trec-covid` | BEIR | 17,537 | 50 |
| Argumentation | `arguana` | BEIR | 1,999 | 500 |
| Open-domain QA | `nq` | BEIR | 2,000 | 500 |
| Medical | `nfcorpus` | BEIR | 3,128 | 323 |
| Scientific | `scifact` | BEIR | 2,000 | 300 |
| Technical | `scidocs` | BEIR | 2,180 | 500 |
| Community | `quora` | BEIR | 2,000 | 500 |
| Fact checking | `fever` | BEIR | 2,000 | 500 |
| Multi-hop QA | `hotpotqa` | BEIR | 2,000 | 500 |
| Entity retrieval | `dbpedia-entity` | BEIR | 14,877 | 400 |

The legal collection contains deduplicated court-decision snippets paired with questions. Its question text is LLM-synthesized over real case-law material, and its qrels reflect source provenance. This differs from the standard BEIR collections and should be disclosed when interpreting the legal result.

### 3.2 Sampling protocol

The matrix uses a shared seeded sampling policy:

1. Load standard queries and positive qrels.
2. Retain every document referenced by the selected qrels.
3. Add randomly selected non-relevant fill documents until reaching a nominal corpus target of 2,000.
4. Retain at most 500 queries when a collection contains more.
5. Use seed 42 for corpus fill and query sampling.

If the relevant-document set alone exceeds 2,000 documents, all relevant documents are retained and the realized corpus exceeds the nominal cap. This explains TREC-COVID and DBpedia-Entity.

Subsampling reduces the distractor pool and can inflate absolute effectiveness. Valid interpretation is restricted to comparisons among configurations evaluated on the same sampled domain. Matrix scores are not full-corpus leaderboard estimates.

## 4. Chunking Specification

The matrix evaluates four concrete policies.

### 4.1 Fixed-512

- Unit: whitespace-delimited words, not model-tokenizer tokens
- Maximum window: 512 words
- Overlap: 50 words
- Chunk ID: derived from parent document ID and starting word offset

### 4.2 Sentence

- Sentence segmentation: NLTK
- Window: three consecutive sentences
- Overlap: none

The label `sentence` is retained for result compatibility, but it represents three-sentence windows rather than one sentence per chunk.

### 4.3 Recursive

1. Split documents on blank-line paragraph boundaries.
2. Keep paragraphs of at most 512 whitespace-delimited words intact.
3. Split longer paragraphs into NLTK sentences.

This is a lightweight paragraph-then-sentence policy, not a general recursive-character splitter. A single sentence longer than the nominal maximum is not subdivided further in the historical runner.

### 4.4 Semantic

1. Segment documents into sentences with NLTK.
2. Embed every sentence with OpenAI `text-embedding-3-small`.
3. L2-normalize sentence embeddings.
4. Greedily merge adjacent sentences when cosine similarity is at least 0.75.

Semantic boundaries are computed once with the OpenAI model and reused for all downstream dense retrievers. This controls the chunk boundaries across embedders but may align the segmentation with the model that generated them.

## 5. Retrieval Systems

### 5.1 Sparse reference

BM25 is run over chunk text using lowercased whitespace tokenization.

### 5.2 Dense embedders

| Result label | Model | Dimensions | Prefix behavior |
|---|---|---:|---|
| `openai` | `text-embedding-3-small` | 1,536 | no explicit prefix |
| `bge-small` | `BAAI/bge-small-en-v1.5` | 384 | BGE retrieval instruction for queries |
| `e5-small` | `intfloat/e5-small-v2` | 384 | `query:` / `passage:` prefixes |

Embeddings are converted to float32, L2-normalized, and scored by inner product.

### 5.3 Chunk-to-document aggregation

For each query:

1. Retrieve the top 50 chunks.
2. Map each chunk to its parent document.
3. Assign each document the maximum score among its retrieved chunks.
4. Sort documents by the max-pooled score.
5. Evaluate the top 10 unique documents.

Document-level qrels are used for evaluation. The parent mapping establishes that a retrieved chunk belongs to a relevant document; it does not establish that every chunk from that document contains the judged evidence.

## 6. Metrics

### 6.1 Primary and secondary metrics

- Primary: NDCG@10
- Secondary: Recall@10, MRR, MAP
- Uncertainty: 95% bootstrap confidence interval over per-query NDCG, using 1,000 seeded resamples
- Available paired analyses: permutation test and Cohen's d when aligned per-query values are retained

### 6.2 Chunking and Embedding Sensitivity Indices

For a fixed domain, define:

```text
M[c, e] = NDCG@10 for dense retrieval with chunker c and embedder e
```

Then:

```text
CSI = mean over embedders of variance across chunkers / variance of all cells
ESI = mean over chunkers of variance across embedders / variance of all cells
```

CSI > ESI indicates that, within the declared candidate pool, changing chunkers moves effectiveness more than changing embedders on average.

CSI and ESI are not a complete orthogonal ANOVA and are not required to sum to one. Interactions and the use of conditional variances can produce a sum above or below one. They should be reported with the evaluated candidate set.

## 7. Result Artifacts

### 7.1 Aggregate schema

Core artifacts are stored at:

```text
results/<dataset>/chunking_results.json
```

Historical matrix records commonly contain:

```text
domain
dataset
chunking
system
embedder
ndcg@10
ndcg@10_ci_lower
ndcg@10_ci_upper
recall@10
mrr
map
n_queries
latency_ms
embed_tokens           # present for applicable OpenAI dense rows
embed_cost_usd         # present for applicable OpenAI dense rows
```

Per-domain metadata contains the realized document/query counts and aggregate OpenAI token usage.

CSI/ESI output is stored in:

```text
results/csi_esi_summary.json
```

### 7.2 Artifact limitations

Complete per-query ranked-list sidecars were not retained for every historical run. `ranked_lists.json` exists only for a subset of experiments. Therefore:

- aggregate matrix values are released for all twelve domains;
- CSI/ESI can be recomputed from aggregate values;
- not every confidence interval or paired comparison can be reconstructed from released ranked lists alone.

Earlier exploratory runs are retained under `results/_fullcorpus_archive/`. They use heterogeneous schemas and experimental paths and must not be silently mixed with the core matrix.

## 8. Code Architecture and Provenance

### 8.1 Historical matrix runner

`scripts/run_chunking_pipeline.py` generated the reported 12-domain matrix. It contains the exact historical implementations of:

- dataset loading and sampling
- four matrix chunkers
- BGE/E5 prefix handling
- BM25 and dense retrieval
- metric aggregation
- token metering

This script remains the provenance implementation for the paper results.

### 8.2 Modular pipeline

`src/retrievalbench/pipeline/` is a newer configuration-driven implementation with separate loaders, chunkers, embedders, retrievers, rerankers, metrics, and result writers.

The modular path is the preferred direction for future development, but it must pass a cell-by-cell regression comparison before replacing the historical runner as the reproduction path. Known protocol differences include default chunker selection, embedder prefix handling, result schemas, and ranked-list persistence.

### 8.3 Synthetic utilities

The top-level `core.py`, `data.py`, `scheduling.py`, `cost.py`, and `scripts/run_demo.py` implement an earlier synthetic benchmark and systems simulation. They remain useful for unit tests and examples but do not generate the paper's main matrix.

## 9. Reproduction

Example matrix run:

```bash
RB_MAX_CORPUS=2000 RB_MAX_QUERIES=500 \
python3 scripts/run_chunking_pipeline.py \
  --dataset fiqa \
  --embedders openai bge-small e5-small
```

Recompute CSI/ESI:

```bash
python3 scripts/analyze_csi_esi_matrix.py
```

Run tests:

```bash
python3 -m pytest -q
```

Network access, dataset downloads, model downloads, and an OpenAI API key are required only for configurations that use them. Recomputing CSI/ESI from released JSON artifacts does not require model or API access.

## 10. Interpretation Rules

The following claims are supported by the completed study:

- chunking can materially change retrieval effectiveness;
- the best chunker varies by domain and system;
- chunking versus embedding sensitivity is domain-dependent;
- the 47.1% maximum observed relative swing and 5/12 CSI result describe this matrix.

The following claims must not be made from the current evidence without new experiments:

- BEIR rankings are generally poor predictors of production RAG rankings;
- latency-adjusted rankings universally reorder systems;
- the archived dense latency values represent end-to-end online latency;
- structure-aware chunking improves legal or financial retrieval by a stated margin;
- retrieval rank position predicts downstream long-context generation quality;
- one model or chunker is universally optimal;
- metered OpenAI spend equals total deployment cost.

## 11. Known Limitations and Future Work

### Current limitations

1. One seeded sample per domain.
2. Document-level rather than chunk-level relevance judgments.
3. A mixed-capacity embedder pool.
4. One concrete parameterization per chunker family.
5. Semantic boundaries generated by one of the evaluated model providers.
6. Incomplete per-query artifacts for historical runs.
7. Two experiment implementations that have not yet been proven numerically equivalent.

### High-value future work

1. Add two or more corpus/query sampling seeds for representative domains.
2. Annotate chunk-level evidence on a targeted subset.
3. Repeat CSI/ESI with a same-capacity embedder pool.
4. Add chunk-size, overlap, and semantic-threshold sweeps.
5. Create an end-to-end, per-query latency harness with explicit online boundaries and p50/p95/p99 reporting.
6. Validate task-appropriate rerankers with finite-score and degeneracy diagnostics.
7. Run structure-aware chunkers as a separate, pre-registered extension rather than mixing them into the existing matrix.
8. Complete a regression suite that compares the modular pipeline with historical matrix cells.

## 12. Project Outputs

- Paper: `paper/main.tex` and `paper/main.pdf`
- Research blog: `blog/retrievalbench.md`
- Blog figures: `blog/images/`
- Core aggregate results: `results/<dataset>/chunking_results.json`
- CSI/ESI summary: `results/csi_esi_summary.json`

These outputs should remain aligned. Any future change to the reported matrix protocol must update the runner, result metadata, paper, blog, README, and this design specification together.
