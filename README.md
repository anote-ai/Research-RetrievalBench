# RetrievalBench: Cross-Domain Ablation of RAG Techniques over Structured Documents

## Research Question

Which RAG pipeline components — chunking strategy, embedding fine-tuning, cross-encoder reranking, metadata injection, query expansion, and hybrid search — generalise across diverse structured-document domains (finance, legal, medical, technical), and which are domain-specific?

## Ablation Grid

| Config | Chunking | Reranking | Metadata | Query Expansion | Hybrid Search |
|--------|----------|-----------|----------|-----------------|---------------|
| C1 (baseline) | fixed_512 | No | No | No | No |
| C2 | sentence | No | No | No | No |
| C3 | recursive | Yes | No | No | No |
| C4 | semantic | Yes | Yes | No | No |
| C5 | semantic | Yes | Yes | Yes | No |
| C6 (full) | semantic | Yes | Yes | Yes | Yes |

Each config is evaluated across 4 domains:

| Domain | Description |
|--------|-------------|
| FINANCE | SEC 10-K/10-Q filings, earnings reports |
| LEGAL | Contracts, case law excerpts |
| MEDICAL | Clinical guidelines, PubMed abstracts |
| TECHNICAL | API documentation, engineering specs |

## Metrics

- **Recall@10** — fraction of relevant documents found in top-10 results
- **Precision@10** — fraction of top-10 results that are relevant
- **NDCG@10** — rank-weighted relevance (normalised DCG)
- **MRR** — Mean Reciprocal Rank across queries

## Quickstart

```bash
git clone https://github.com/anote-ai/research-retrievalbench.git
cd research-retrievalbench
pip install -e ".[dev]"
pytest tests/ -v
```

```python
from retrievalbench.core import Domain, RAGConfig, RetrievalBench
from retrievalbench.evaluate import recall_at_k, ndcg_at_k

config = RAGConfig(
    chunking_strategy="semantic",
    use_reranking=True,
    use_metadata=True,
    use_query_expansion=True,
    embedding_model="text-embedding-3-large",
)

bench = RetrievalBench()
run = bench.run_config(config, Domain.FINANCE)
print(f"Queries evaluated: {run.num_queries}")
```

## Citation

```bibtex
@misc{retrievalbench2024,
  title  = {RetrievalBench: Cross-Domain Ablation of RAG Techniques over Structured Documents},
  author = {Anote AI Research},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-retrievalbench},
}
```
