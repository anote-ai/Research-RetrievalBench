# RetrievalBench

**RetrievalBench evaluates document chunking and embedding-model choice as interacting variables in cross-domain retrieval.**

Most retrieval benchmarks start from a fixed corpus of atomic passages. RetrievalBench instead asks what happens when the same source documents are indexed under different segmentation policies and retrieved with different embedding models.

The main experiment crosses:

- **12 domains**
- **4 chunking strategies**
- **3 dense embedding models**
- **BM25** as a sparse reference

The largest observed within-system change is a **47.1% relative difference in NDCG@10** caused by changing only the chunking strategy. A variance decomposition further shows that chunking sensitivity exceeds embedding-model sensitivity in 5 of 12 domains; embedding choice dominates in the other 7.

## Paper and Blog

- [Paper source](paper/main.tex)
- [Compiled paper](paper/main.pdf)
- [Research blog](blog/retrievalbench.md)

The paper focuses on the completed chunking × embedding × domain sensitivity study. Latency, reranking, cost metering, structure-aware chunking, and synthetic scheduling utilities exist in the repository, but they are not treated as equally established headline findings.

## Main Findings

1. **Chunking is part of the retrieval system.** Holding the corpus and retriever fixed, chunking changes NDCG@10 by up to 47.1% relative.
2. **There is no universally best chunker.** The preferred strategy varies across domains and sometimes across retrieval models within the same domain.
3. **“Chunking matters more than the embedder” is not universal.** CSI exceeds ESI in 5 of 12 domains; ESI exceeds CSI in 7.
4. **Short, nearly atomic documents are a useful control.** On Quora, all four chunkers produce almost identical scores.

## Experimental Matrix

### Domains

| Domain | Dataset | Matrix documents | Matrix queries |
|---|---|---:|---:|
| Finance | FiQA | 2,000 | 500 |
| Legal | Open Australian Legal QA | 2,099 | 2,124 |
| Biomedical | TREC-COVID | 17,537 | 50 |
| Argumentation | ArguAna | 1,999 | 500 |
| Open-domain QA | Natural Questions | 2,000 | 500 |
| Medical | NFCorpus | 3,128 | 323 |
| Scientific | SciFact | 2,000 | 300 |
| Technical | SciDocs | 2,180 | 500 |
| Community | Quora | 2,000 | 500 |
| Fact checking | FEVER | 2,000 | 500 |
| Multi-hop QA | HotpotQA | 2,000 | 500 |
| Entity retrieval | DBpedia-Entity | 14,877 | 400 |

Each matrix corpus contains 2,000 seeded fill documents plus every qrels-relevant document. A corpus may therefore exceed 2,000 documents when its relevant set is larger. Up to 500 queries are retained using seed 42. Because subsampling reduces the distractor pool, matrix scores support comparisons among configurations on the same sampled corpus, not direct comparison with full-corpus leaderboards.

### Chunking strategies

The reported names refer to these concrete implementations:

- **`fixed_512`** — windows of up to 512 whitespace-delimited words with 50-word overlap.
- **`sentence`** — non-overlapping windows of three NLTK-segmented sentences.
- **`recursive`** — split first on blank-line paragraph boundaries; paragraphs longer than 512 whitespace-delimited words are split into sentences.
- **`semantic`** — greedily merge adjacent sentences when cosine similarity is at least 0.75. Boundaries are generated with OpenAI `text-embedding-3-small` and reused across downstream embedders.

### Retrievers

- BM25
- OpenAI `text-embedding-3-small`
- `BAAI/bge-small-en-v1.5`
- `intfloat/e5-small-v2`

BGE and E5 use their model-specific query/passage prefixes in the runner that generated the reported matrix. Dense embeddings are L2-normalized and scored by inner product. Retrieval selects the top 50 chunks, max-pools chunk scores to parent documents, and evaluates the top 10 unique documents.

## CSI and ESI

For a domain, let `M[chunker, embedder]` be dense-retrieval NDCG@10.

```text
CSI = mean_embedder Var_chunker(M) / Var(M)
ESI = mean_chunker Var_embedder(M) / Var(M)
```

CSI > ESI means that chunker selection moved quality more than embedder selection within the evaluated candidate pool. The indices are diagnostics rather than immutable corpus constants: changing the candidate models or chunker parameterizations can change both values.

The released summary is in [`results/csi_esi_summary.json`](results/csi_esi_summary.json).

## Repository Layout

```text
src/retrievalbench/
├── evaluate.py                 # IR metrics, bootstrap CIs, paired tests
├── pipeline/
│   ├── chunkers/               # Generic and experimental structure-aware chunkers
│   ├── retrievers/             # BM25 and dense retrieval backends
│   ├── config.py               # Experiment specifications and registries
│   ├── loaders.py              # Dataset loading and seeded subsampling
│   ├── metrics.py              # Per-config evaluation and CSI/ESI
│   ├── rerankers.py            # Optional cross-encoder reranking
│   └── run.py                  # Modular experiment executor
scripts/
├── run_chunking_pipeline.py    # Runner used for the reported 12-domain matrix
├── run_experiment.py           # Newer modular experiment runner
└── analyze_csi_esi_matrix.py   # CSI/ESI aggregation
results/                        # Released aggregate experiment artifacts
paper/                          # Current paper source and PDF
blog/                           # Research blog and figures
tests/                          # Unit tests
```

The repository currently contains two experiment paths. `scripts/run_chunking_pipeline.py` is the provenance path for the published matrix. `src/retrievalbench/pipeline/` is the newer modular implementation. They should not be assumed to be numerically interchangeable without a regression comparison; see [`DESIGN_DOC.md`](DESIGN_DOC.md).

## Installation

Python 3.10 or newer is required.

Install the package and development tools:

```bash
python3 -m pip install -e ".[dev]"
```

Real retrieval experiments additionally require the backends used by the selected configuration, for example:

```bash
python3 -m pip install \
  datasets nltk openai python-dotenv rank-bm25 \
  sentence-transformers
```

Some environments may also require NLTK tokenizer data:

```bash
python3 -m nltk.downloader punkt punkt_tab
```

Set `OPENAI_API_KEY` only when running OpenAI or semantic-chunking configurations.

## Reproducing the Matrix

The historical runner used for the reported results accepts one dataset at a time:

```bash
RB_MAX_CORPUS=2000 RB_MAX_QUERIES=500 \
python3 scripts/run_chunking_pipeline.py \
  --dataset fiqa \
  --embedders openai bge-small e5-small
```

Replace `fiqa` with another supported matrix dataset. The sampling procedure always retains qrels-relevant documents, so the realized corpus may exceed `RB_MAX_CORPUS`.

Recompute CSI/ESI from released aggregate results:

```bash
python3 scripts/analyze_csi_esi_matrix.py
```

Run the unit tests:

```bash
python3 -m pytest -q
```

## Result Artifacts

The main aggregate artifacts live at:

```text
results/<dataset>/chunking_results.json
results/csi_esi_summary.json
```

Earlier full-corpus and exploratory runs are retained in:

```text
results/_fullcorpus_archive/
```

Complete per-query ranked-list sidecars were not retained for every historical matrix run. Aggregate results are available for all twelve core domains, but not every aggregate can be reconstructed from ranked lists alone.

## Scope and Limitations

- One seeded corpus/query sample is used per domain.
- Relevance judgments are document-level and are inherited through parent-document identity; the benchmark does not contain chunk-level relevance annotation.
- The dense pool mixes two small open-source models with an API model from a different training and capacity regime.
- The four chunkers are specific implementations, not exhaustive representatives of their strategy families.
- Semantic boundaries are generated using the OpenAI model for all downstream embedders.
- Archived latency and reranking runs are exploratory and should not be interpreted as universal deployment comparisons.
- Metered OpenAI token spend is not total serving cost; local compute, hardware, energy, and reranker inference are not monetized.

## Citation

```bibtex
@article{retrievalbench2026,
  title   = {RetrievalBench: Chunking and Embedding Choice as First-Class
             Variables in Cross-Domain Retrieval Evaluation},
  author  = {Anonymous},
  year    = {2026},
  note    = {Manuscript under review}
}
```
