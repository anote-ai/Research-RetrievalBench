# RetrievalBench — Research Design Document

## Goal

Establish RetrievalBench as the definitive evaluation benchmark for production RAG retrieval systems — going beyond academic IR metrics to capture the dimensions that determine real-world RAG quality: chunking strategies, context window effects, latency constraints, and cost-quality tradeoffs.

## Objective

1. Evaluate retrieval systems on tasks that reflect actual enterprise RAG deployments (multi-domain, multi-format, variable document length)
2. Measure dimensions that BEIR and MTEB do not: chunking sensitivity, latency-adjusted NDCG, cost-per-query, context window saturation
3. Produce a leaderboard that becomes the community standard for comparing retrieval systems in production RAG contexts

## Background / Motivation

BEIR (2021) is the current standard for zero-shot retrieval evaluation. It covers 18 datasets across diverse domains and is widely cited. However, BEIR was designed for academic IR — not production RAG. It does not capture:
- How chunking strategy affects retrieval quality
- How retrieval quality degrades as context windows fill with irrelevant passages
- The latency-accuracy tradeoff that determines whether a retrieval system is deployable
- Cost per query (critical for production budget planning)

As enterprise RAG deployment has grown 10x since 2022, the gap between what BEIR measures and what practitioners need has widened significantly.

## Experimental Design

### Baseline Experiment

**Replicate BEIR evaluation on 5 leading retrieval systems (BM25, E5-large, BGE-m3, Cohere Embed v3, Voyage-large-2)**

- Metric: NDCG@10 on BEIR's 18 datasets
- Purpose: establish that our evaluation infrastructure is correct (numbers should match published BEIR results within ±0.5%)
- Runtime: ~2 days on a single GPU node
- Expected result: all systems reproduce published NDCG@10 within ±0.5 points

### Test Experiment 1: Chunking Sensitivity

Take 5 BEIR datasets. For each, create 4 chunking variants of the corpus:
- Sentence-level chunking (avg 30 tokens)
- Paragraph-level chunking (avg 200 tokens)
- Fixed 512-token chunks with 64-token overlap
- Structure-aware chunking (headers, sections, tables preserved)

Evaluate all 5 retrieval systems on all 4 chunking variants.

**Metrics:** NDCG@10 variance across chunking strategies per retrieval system; optimal chunking strategy per retrieval system and domain type

**Expected result:** retrieval quality varies by up to 15% NDCG@10 across chunking strategies; structure-aware chunking outperforms fixed-size chunking by 8–12% on document-heavy domains

### Test Experiment 2: Latency-Adjusted NDCG

For each retrieval system, measure:
- Retrieval latency at p50, p95, p99 (ms per query)
- NDCG@10

Compute composite metric: `NDCG@10 × min(latency_budget / actual_latency, 1.0)`

**Expected result:** some academic SOTA models score below BM25 on latency-adjusted NDCG — a finding directly useful for practitioners with <100ms SLA requirements

### Test Experiment 3: Cost-Quality Tradeoff

For embedding-API-based retrieval systems, measure NDCG@10 and cost per 1M queries at current vendor pricing. Plot the cost-quality Pareto frontier.

**Expected result:** a clear Pareto frontier identifying which systems offer best NDCG per dollar at each budget level

## Expected Results

1. A benchmark dataset of 10+ domains with chunking variants, ground-truth relevance judgments, and latency measurements
2. A published leaderboard at `retrievalbench.anote.ai`
3. **Key finding:** "BEIR rankings do not predict latency-adjusted production RAG rankings"
4. **Key finding:** "Chunking strategy accounts for more NDCG variance than embedding model choice for document-heavy domains"

## Why This Matters / Why People Would Care

- **ML researchers:** BEIR has 2,000+ citations; a benchmark that improves on it for production settings will be highly cited
- **Practitioners:** no principled way to choose chunking strategy or compare retrieval systems on deployment constraints — this benchmark gives them that
- **Vendors** (Cohere, Voyage, OpenAI, Pinecone): a credible third-party benchmark is how they demonstrate production value
- **RAG ecosystem:** retrieval is the #1 failure mode in RAG systems; improving retrieval evaluation directly improves RAG reliability

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | Benchmark construction (datasets, chunking variants, relevance judgments) |
| 3 | Baseline replication + latency measurement experiments |
| 4 | Test experiments + analysis |
| 5 | Paper writing + leaderboard launch |
| 6 | Submission to SIGIR 2027 (deadline ~January 2027) |

## Related Issues

- Design doc GitHub issue: #20
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
