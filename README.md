# RetrievalBench

> **Research question:** Which RAG pipeline configurations (chunking strategy, reranking, metadata filtering, query expansion) generalize best across finance, legal, medical, and technical domains?

## Overview

RetrievalBench provides a lightweight, reproducible framework for ablating Retrieval-Augmented Generation (RAG) configurations across multiple domains. All experiments run on synthetic corpora with controllable recall characteristics, enabling fast iteration without proprietary data.

## Methodology

The benchmark is synthetic and deterministic. It does not download external
datasets, call an LLM, compute embeddings, or build a vector database. Instead,
it generates domain-specific corpora, queries, relevance labels, and retrieval
results with fixed seeds so the demo runs quickly and is reproducible.

- **Domains:** finance, legal, medical, and technical
- **Corpus:** 240 synthetic documents sampled from domain-specific vocabulary
- **Queries:** 30 synthetic queries per domain, 120 queries total
- **Configurations:** 6 RAG pipeline variants over chunking, reranking,
  metadata filtering, and query expansion
- **Relevance labels:** generated per query from same-domain documents
- **Retrieval simulation:** relevant and non-relevant document IDs are sampled
  into the top-10 result set using config/domain recall assumptions
- **Ranking simulation:** reranking, metadata, semantic chunking, and query
  expansion modify a relevance-bias term that affects document ordering
- **Metrics:** Recall@10, Precision@10, nDCG@10, MRR, MAP, and R-Precision
- **Reproducibility:** run `python3 scripts/run_demo.py` from the repo root

These results are useful for validating the benchmark harness and comparing
controlled synthetic assumptions. They should not be interpreted as performance
on real finance, legal, medical, or technical corpora.

## Ablation Grid (6 Configurations)

| Config | Chunking | Rerank | Metadata | Query Exp. | nDCG@10 (Finance) |
|--------|----------|--------|----------|------------|-------------------|
| baseline | fixed_512 | ✗ | ✗ | ✗ | 0.41 |
| +rerank | sentence | ✓ | ✗ | ✗ | 0.60 |
| +meta | fixed_512 | ✗ | ✓ | ✗ | 0.50 |
| +rerank+meta | recursive | ✓ | ✓ | ✗ | 0.64 |
| +qexp | sentence | ✗ | ✗ | ✓ | 0.47 |
| +rerank+meta+qexp | semantic | ✓ | ✓ | ✓ | **0.82** |

## Full 4-Domain Results

| Domain | Config | Recall@10 | Precision@10 | nDCG@10 | MRR | MAP | R-Precision |
|--------|--------|-----------|--------------|---------|-----|-----|-------------|
| finance | fixed_512 | 0.7667 | 0.1400 | 0.4134 | 0.3816 | 0.2419 | 0.1056 |
| finance | sentence+rerank | 0.7667 | 0.1400 | 0.5952 | 0.6708 | 0.4678 | 0.4111 |
| finance | fixed_512+meta | 0.7667 | 0.1400 | 0.4990 | 0.5098 | 0.3517 | 0.2389 |
| finance | recursive+rerank+meta | 0.8667 | 0.1600 | 0.6390 | 0.6895 | 0.4908 | 0.3500 |
| finance | sentence+qexp | 0.7667 | 0.1400 | 0.4716 | 0.4860 | 0.3137 | 0.2333 |
| finance | semantic+rerank+meta+qexp | 1.0000 | 0.2000 | 0.8211 | 0.8044 | 0.7299 | 0.5778 |
| legal | fixed_512 | 0.6778 | 0.1267 | 0.4133 | 0.4365 | 0.2656 | 0.1889 |
| legal | sentence+rerank | 0.6778 | 0.1267 | 0.5782 | 0.6993 | 0.4610 | 0.3944 |
| legal | fixed_512+meta | 0.6778 | 0.1267 | 0.3475 | 0.3183 | 0.1884 | 0.0833 |
| legal | recursive+rerank+meta | 0.9111 | 0.1733 | 0.6603 | 0.6539 | 0.5236 | 0.3778 |
| legal | sentence+qexp | 0.6778 | 0.1267 | 0.3868 | 0.3648 | 0.2391 | 0.1722 |
| legal | semantic+rerank+meta+qexp | 1.0000 | 0.2000 | 0.8856 | 0.9056 | 0.8070 | 0.6833 |
| medical | fixed_512 | 0.7000 | 0.1300 | 0.3826 | 0.3451 | 0.2266 | 0.1056 |
| medical | sentence+rerank | 0.7000 | 0.1300 | 0.5308 | 0.6215 | 0.3927 | 0.3111 |
| medical | fixed_512+meta | 0.7000 | 0.1300 | 0.4272 | 0.4501 | 0.2705 | 0.1722 |
| medical | recursive+rerank+meta | 0.9000 | 0.1700 | 0.6976 | 0.7136 | 0.5770 | 0.4778 |
| medical | sentence+qexp | 0.6000 | 0.1000 | 0.3306 | 0.3068 | 0.2063 | 0.1667 |
| medical | semantic+rerank+meta+qexp | 0.7000 | 0.1300 | 0.5943 | 0.7214 | 0.4739 | 0.4056 |
| technical | fixed_512 | 0.7611 | 0.1367 | 0.3923 | 0.3192 | 0.2320 | 0.1278 |
| technical | sentence+rerank | 0.7611 | 0.1367 | 0.5093 | 0.5306 | 0.3632 | 0.2611 |
| technical | fixed_512+meta | 0.7611 | 0.1367 | 0.4400 | 0.4333 | 0.2770 | 0.1722 |
| technical | recursive+rerank+meta | 0.8778 | 0.1600 | 0.7018 | 0.7512 | 0.5793 | 0.4667 |
| technical | sentence+qexp | 0.7611 | 0.1367 | 0.4277 | 0.3827 | 0.2705 | 0.1500 |
| technical | semantic+rerank+meta+qexp | 1.0000 | 0.1967 | 0.8632 | 0.8770 | 0.7797 | 0.6389 |

## 4-Domain Comparison

| Domain | Best Config | nDCG@10 | MRR |
|--------|------------|---------|-----|
| Finance | +rerank+meta+qexp | 0.82 | 0.80 |
| Legal | +rerank+meta+qexp | 0.89 | 0.91 |
| Medical | +rerank+meta | 0.70 | 0.71 |
| Technical | +rerank+meta+qexp | 0.86 | 0.88 |

## Metrics

- **Recall@k** — fraction of relevant docs recovered in top-k
- **Precision@k** — fraction of top-k results that are relevant
- **nDCG@k** — rank-weighted relevance
- **MRR** — mean reciprocal rank of the first relevant result

## Hardware-Aware Adaptive Scheduling

The codebase also includes a synthetic systems layer for studying
energy-efficient RAG retrieval scheduling:

- Fixed-stride retrieval baselines, such as retrieving every 8 or 16 generated tokens
- Adaptive retrieval based on uncertainty, semantic drift, retrieval-score decay, and
  document overlap
- Dynamic retrieval depth via top-k adjustment
- Hardware power/frequency modes (`eco`, `balanced`, `turbo`) for edge GPU, server GPU,
  and CPU-only profiles
- Simulated TTFT, TBT, end-to-end latency, quality, retrieval calls, and energy in joules

Run `python3 scripts/run_demo.py` to print both the RAG ablation table and the
hardware-aware scheduling comparison.

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/run_demo.py
python3 -m pytest tests/ -v
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
