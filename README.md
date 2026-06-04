# RetrievalBench

> **Research question:** Which RAG pipeline configurations (chunking strategy, reranking, metadata filtering, query expansion) generalize best across finance, legal, medical, and technical domains?

## Overview

RetrievalBench provides a lightweight, reproducible framework for ablating Retrieval-Augmented Generation (RAG) configurations across multiple domains. All experiments run on synthetic corpora with controllable recall characteristics, enabling fast iteration without proprietary data.

## Ablation Grid (6 Configurations)

| Config | Chunking | Rerank | Metadata | Query Exp. | nDCG@10 (Finance) |
|--------|----------|--------|----------|------------|-------------------|
| baseline | fixed_512 | ✗ | ✗ | ✗ | 0.61 |
| +rerank | sentence | ✓ | ✗ | ✗ | 0.68 |
| +meta | fixed_512 | ✗ | ✓ | ✗ | 0.65 |
| +rerank+meta | recursive | ✓ | ✓ | ✗ | 0.73 |
| +qexp | sentence | ✗ | ✗ | ✓ | 0.66 |
| +rerank+meta+qexp | semantic | ✓ | ✓ | ✓ | **0.79** |

## 4-Domain Comparison

| Domain | Best Config | nDCG@10 | MRR |
|--------|------------|---------|-----|
| Finance | +rerank+meta+qexp | 0.79 | 0.84 |
| Legal | +rerank+meta+qexp | 0.74 | 0.79 |
| Medical | +rerank+meta | 0.71 | 0.76 |
| Technical | +rerank+meta+qexp | 0.76 | 0.81 |

## Metrics

- **Recall@k** — fraction of relevant docs recovered in top-k
- **Precision@k** — fraction of top-k results that are relevant
- **nDCG@k** — rank-weighted relevance
- **MRR** — mean reciprocal rank of the first relevant result

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/run_demo.py
pytest tests/ -v
```

## Target Venues

- **EMNLP 2027** — Findings of EMNLP
- **AKBC 2027** — Automated Knowledge Base Construction
- **AAAI 2027** — Main technical track

## Citation

```bibtex
@software{retrievalbench2026,
  title  = {RetrievalBench: A Benchmark for RAG Configuration Generalization},
  author = {Anote AI},
  year   = {2026},
  url    = {https://github.com/anote-ai/research-retrievalbench}
}
```
