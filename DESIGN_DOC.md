# RetrievalBench: Production-Realistic Evaluation for Enterprise RAG Retrieval Systems

## Vision Statement

BEIR (2021) redefined retrieval evaluation for the neural era. Five years later, production RAG has introduced an entirely new set of failure modes — chunking artifacts, context-window saturation, latency SLAs, per-query cost budgets — that BEIR was never designed to measure. **RetrievalBench is the evaluation framework that bridges the gap between academic IR and production RAG.**

We argue that a retrieval system's *academic* NDCG@10 and its *production* performance are only weakly correlated, and that the dimensions BEIR ignores account for the majority of variance in real enterprise deployments. This paper proves that claim empirically and provides the community with tools to measure and optimize for what actually matters.

---

## 1. Problem Statement and Novelty

### What BEIR Gets Right
BEIR established zero-shot cross-domain retrieval as the right paradigm for evaluating retrieval generalization. Its 18 datasets, diverse domains, and standardized evaluation protocol made it the universal leaderboard for embedding models and sparse retrieval systems.

### What BEIR Gets Wrong for Production RAG
Production RAG introduces four dimensions BEIR systematically ignores:

**1. Chunking as a first-class variable.** In BEIR, documents arrive pre-chunked (or are treated as atomic units). In production, a 40-page 10-K filing must be split into retrievable chunks — and the chunking strategy *is* a retrieval design decision. We show empirically that chunking strategy explains more NDCG@10 variance than embedding model choice for document-heavy domains.

**2. Context-window saturation.** When top-k retrieved passages are fed to a generator LLM, passages beyond a certain rank contribute noise rather than signal. BEIR measures retrieval in isolation; we measure retrieval *as a component of RAG*, where the effective recall cutoff is determined by the generator's context window and attention patterns.

**3. Latency-adjusted quality.** A retrieval system with NDCG@10 = 0.52 and p99 latency = 800ms is worse than one with NDCG@10 = 0.48 and p99 latency = 50ms for most production applications. We introduce **LA-NDCG** (Latency-Adjusted NDCG), the first composite metric that captures this tradeoff.

**4. Cost-quality Pareto optimality.** Dense retrieval via API embeddings (Cohere, Voyage, OpenAI) has per-query costs that matter enormously at scale. We characterize the cost-quality Pareto frontier, enabling practitioners to select the system that meets their accuracy requirement at minimum cost.

### Novel Contributions Summary
- **RetrievalBench dataset**: 12 domains × 4 chunking strategies × 3 document length distributions = 144 evaluation conditions (vs. BEIR's 18 fixed conditions)
- **LA-NDCG metric**: first latency-aware composite retrieval metric with theoretical grounding
- **Cost-quality frontier**: first systematic cost characterization for embedding API retrieval at production scale
- **Chunking taxonomy**: first empirical taxonomy of how chunking strategies interact with retrieval system type and document domain
- **Context saturation analysis**: first study of how retrieval rank correlates with downstream generator performance across context window sizes

---

## 2. Research Objectives

1. **Demonstrate** that BEIR rankings are poor predictors of production RAG rankings (primary claim)
2. **Quantify** how much chunking strategy, latency budget, and cost constraints alter system rankings relative to BEIR
3. **Provide** a public leaderboard and evaluation toolkit that practitioners can use to select retrieval systems for their specific deployment constraints
4. **Release** a reproducible evaluation framework so the community can add new systems, domains, and evaluation conditions

---

## 3. Dataset Construction

### 3.1 Domain Selection (12 domains)
We select 12 domains covering the range of enterprise RAG use cases, stratified by document length distribution and domain specificity:

| Domain | Avg Doc Length | Domain Specificity | Source |
|---|---|---|---|
| News articles | Short (300 tokens) | Low | BEIR/TREC-NEWS |
| Wikipedia paragraphs | Medium (150 tokens) | Low | BEIR/NQ |
| Legal contracts | Long (2,000+ tokens) | High | CUAD (public) |
| SEC 10-K filings | Very long (5,000+ tokens) | High | EDGAR (public) |
| Medical literature | Medium (250 tokens) | Very high | PubMed (public) |
| Scientific papers | Long (1,500 tokens) | High | S2ORC subset |
| Software documentation | Variable (100-2,000 tokens) | Medium | GitHub Wikis |
| Customer support tickets | Short (200 tokens) | Medium | Publicly available datasets |
| HR policy documents | Medium (500 tokens) | High | Synthetic (GPT-4o generated) |
| Financial analyst reports | Long (3,000+ tokens) | High | SEC (public filings) |
| Product manuals | Variable (500-5,000 tokens) | Medium | Publicly available |
| Academic textbooks | Very long (chapter-level) | High | Open textbooks |

### 3.2 Query Construction
For each domain, we construct 3 query types:
- **Factoid queries**: single-passage answer (BEIR-style)
- **Multi-evidence queries**: answer requires synthesizing 2-3 passages (novel)
- **Negative queries**: answer is NOT in the corpus (tests false positive rate) (novel)

Total: ~5,000 queries across all domains and query types.

### 3.3 Relevance Judgments
Relevance judgments collected via:
- Automated pooling (top-20 from 5 retrieval systems)
- Human annotation for ambiguous relevance grades (crowdworkers + domain expert review)
- 3-level relevance: highly relevant (2), partially relevant (1), not relevant (0)

### 3.4 Chunking Variants
For each domain corpus, we create 4 chunking variants:
- **Sentence-level**: spaCy sentence segmentation, avg ~30 tokens
- **Fixed-size**: 512 tokens with 64-token overlap (standard practice)
- **Paragraph-level**: double-newline splitting, avg ~200 tokens
- **Structure-aware**: domain-specific parser that preserves semantic units (sections, table rows, list items). We implement structure-aware chunkers for legal (clause detection), financial (section detection + table preservation), and medical (section detection via standard section headers).

---

## 4. Systems Under Evaluation

### Sparse Retrieval
- BM25 (Pyserini implementation, k1=0.9, b=0.4)
- BM25 + query expansion via RM3
- SPLADE-v3

### Dense Retrieval (Open-Source)
- E5-large-v2
- BGE-m3
- GTE-large
- Nomic-embed-text-v1.5

### Dense Retrieval (API)
- OpenAI text-embedding-3-large
- Cohere Embed v3 English
- Voyage-large-2-instruct

### Hybrid
- BM25 + E5-large (RRF fusion)
- BM25 + Cohere (RRF fusion)

---

## 5. Experimental Design

### 5.1 Baseline Experiment: BEIR Replication

**Purpose**: Validate evaluation infrastructure and establish the BEIR baseline that we will show is insufficient.

**Protocol**:
1. Index all BEIR 18 datasets using each retrieval system
2. Run standard NDCG@10 evaluation
3. Verify numbers match published results within ±0.5 NDCG points for open-source systems
4. Publish baseline numbers as the "BEIR table" in the paper

**Expected results**: All systems reproduce published BEIR numbers within tolerance. BM25 ≈ 0.43, E5-large ≈ 0.54, BGE-m3 ≈ 0.57, Cohere ≈ 0.56 (average across all BEIR domains).

**Why this matters for the paper**: We need this table to establish that our evaluation infrastructure is correct before making novel claims.

---

### 5.2 Experiment 1: Chunking Sensitivity Analysis

**Hypothesis**: Chunking strategy is a higher-variance factor than embedding model choice for long-document-heavy domains.

**Protocol**:
1. For each of the 12 RetrievalBench domains and 4 chunking variants, index the corpus
2. Evaluate all 11 retrieval systems on all 48 domain×chunking conditions
3. For each retrieval system, compute the **Chunking Sensitivity Index (CSI)**: CSI = (max NDCG across chunking variants - min NDCG across chunking variants) / mean NDCG across chunking variants
4. For each domain, compute the **Embedding Sensitivity Index (ESI)**: ESI = std(NDCG across systems) / mean(NDCG across systems)
5. Test hypothesis: CSI > ESI for domains with avg doc length > 500 tokens

**Expected results**:
- CSI > ESI for 8/12 domains (domains with long documents)
- Structure-aware chunking outperforms fixed 512-token chunking by 10-18 NDCG points for legal and financial domains
- Sentence-level chunking hurts dense retrieval but helps BM25 (different granularity optima)
- The optimal chunking strategy varies by retrieval system type (sparse vs. dense) *and* by domain

**Novel contribution**: First empirical demonstration that the chunking-retrieval interaction effect is larger than the embedding model effect for enterprise document types. This reframes the practitioner's decision hierarchy: choose chunking strategy before choosing embedding model.

---

### 5.3 Experiment 2: Latency-Adjusted NDCG (LA-NDCG)

**Metric definition**:

```
LA-NDCG(system, budget_ms) = NDCG@10(system) × min(budget_ms / P95_latency(system), 1.0)
```

Where `P95_latency` is the 95th percentile query latency measured on a standard benchmark machine (we use a c5.2xlarge AWS instance, 8 vCPU, no GPU, to simulate cost-conscious production deployments).

**Protocol**:
1. For each system, run 1,000 queries and measure P50, P95, P99 latency
2. Compute LA-NDCG at latency budgets: {50ms, 100ms, 200ms, 500ms, 1000ms}
3. Rank systems by LA-NDCG at each budget
4. Compare to BEIR ranking (Kendall's τ between BEIR rank and LA-NDCG rank at each budget)

**Expected results**:
- At 100ms budget: BM25 and SPLADE rank higher than dense API models (API embedding latency exceeds 100ms)
- At 500ms budget: Rankings converge toward BEIR rankings but don't fully match
- Kendall's τ between BEIR rank and LA-NDCG rank at 100ms ≈ 0.3 (weak correlation — ranking significantly reshuffled)
- Kendall's τ at 1000ms ≈ 0.75 (strong correlation — enough time for all systems)

**Novel contribution**: LA-NDCG as a principled, parameterized metric that practitioners can set to their actual SLA budget. The first retrieval metric that directly encodes deployment constraints.

---

### 5.4 Experiment 3: Cost-Quality Pareto Frontier

**Protocol**:
1. For API-based systems, record vendor-published pricing ($/1M tokens as of submission date)
2. Compute cost per 1M queries = (avg tokens per query × $/1M tokens)
3. Plot cost vs. NDCG@10 for all systems on a 2D Pareto chart
4. Identify the Pareto-dominant systems at each cost tier: {<$0.01/1M queries, $0.01–$0.10, $0.10–$1.00, >$1.00}
5. For open-source systems: compute cost as (GPU instance cost × inference time)

**Expected results**:
- BM25 is Pareto-dominant at the lowest cost tier
- Nomic-embed (open-source, self-hosted) achieves near-API quality at 5-10x lower cost at scale
- Cohere and Voyage are Pareto-dominant among API-only options (higher quality than OpenAI at same or lower price)
- No single system is Pareto-dominant across all conditions — the tradeoff is real and practitioners must choose based on their constraints

---

### 5.5 Experiment 4: Context-Window Saturation

**Protocol**:
1. For each domain, run E2E RAG evaluation: retrieve top-k passages → generate answers with GPT-4o (4k, 8k, 32k, 128k context) → score answers against gold
2. Vary k ∈ {1, 3, 5, 10, 20, 50}
3. Measure: answer quality (ROUGE-L, BERTScore, exact match) as a function of k and context window size
4. Identify the **effective recall depth**: the value of k beyond which answer quality stops improving (or starts declining)

**Expected results**:
- Effective recall depth = 5-10 for 4k context models (additional passages beyond rank 10 are truncated or dilute attention)
- Effective recall depth = 20-30 for 32k context models
- For long-document domains, effective recall depth is *lower* because retrieved passages are longer and fill the context window faster
- **Key finding**: NDCG@100 is a poor proxy for RAG quality; NDCG@5 and NDCG@10 at the correct granularity are more predictive

---

## 6. Expected Results and Claims

### Primary Claims (Main Paper)
1. **Claim 1**: BEIR system rankings have Kendall's τ ≤ 0.4 with LA-NDCG rankings at 100ms latency budget — the systems BEIR recommends are not the systems production deployments should use.
2. **Claim 2**: Chunking strategy explains more NDCG variance than embedding model choice for documents longer than 500 tokens (CSI > ESI for 8/12 domains).
3. **Claim 3**: Structure-aware chunking outperforms best fixed-size chunking by 10-18 NDCG points on legal and financial documents.
4. **Claim 4**: The cost-quality Pareto frontier has a clear elbow: self-hosted open-source models at GPU cost achieve >95% of best API quality at <20% of API cost at scale.

### Secondary Claims (Appendix / Extended Version)
- Effective recall depth correlates strongly with context window size (r > 0.85)
- For negative queries (answer not in corpus), all systems have false positive rates of 15-35% (retrieving irrelevant passages with high confidence)
- Domain-specific chunkers reduce false positive rate by 8-12% vs. general-purpose chunkers

---

## 7. Why This Matters / Why People Would Care

### Immediate Practitioner Impact
Every organization deploying RAG in 2025-2026 faces the question: "Which retrieval system should I use?" BEIR gives them an answer that optimizes the wrong thing. RetrievalBench gives them an answer tuned to their actual deployment constraints (latency budget, cost budget, document type).

### Research Community Impact
- LA-NDCG will be adopted by future retrieval papers that target production deployment
- The chunking-retrieval interaction effect is a genuinely surprising finding that will motivate new work on adaptive chunking and retrieval-aware document preprocessing
- The cost-quality frontier provides a reference for efficiency-focused retrieval research

### Industry Impact
- Embedding model vendors (Cohere, Voyage, OpenAI) will integrate RetrievalBench as a third-party evaluation to differentiate on production-relevant dimensions
- RAG framework vendors (LlamaIndex, LangChain) will use the chunking findings to improve default configurations

### Longer-Term Impact
- Establishes the paradigm shift: retrieval evaluation must include deployment constraints, not just academic metrics
- BEIR has 2,000+ citations since 2021; a paper that significantly advances BEIR's methodology is positioned for similar long-term impact

---

## 8. Implementation Plan

### Codebase Structure
```
retrievalbench/
├── data/                    # Dataset construction scripts
│   ├── chunking/            # 4 chunking strategy implementations
│   ├── domains/             # Domain-specific corpus processors
│   └── queries/             # Query construction and annotation
├── systems/                 # Retrieval system adapters
│   ├── sparse/              # BM25, SPLADE
│   ├── dense/               # E5, BGE, GTE, Nomic
│   └── api/                 # Cohere, Voyage, OpenAI
├── eval/                    # Evaluation metrics
│   ├── ndcg.py              # Standard NDCG
│   ├── la_ndcg.py           # Latency-Adjusted NDCG (novel)
│   ├── cost_quality.py      # Cost-quality Pareto analysis
│   └── context_sat.py       # Context saturation analysis
├── latency/                 # Latency benchmarking harness
└── leaderboard/             # Leaderboard website code
```

### Reproducibility Commitment
- All code open-sourced on GitHub under Apache 2.0
- All datasets on HuggingFace under CC-BY 4.0
- Docker container reproducing all paper results
- Leaderboard hosted at `retrievalbench.anote.ai` accepting community submissions

---

## 9. Timeline

| Month | Milestone | Owner |
|---|---|---|
| 1 | Domain corpus collection and preprocessing | Data team |
| 1 | Chunking strategy implementation (4 variants × 12 domains) | Engineering |
| 2 | Query construction and relevance judgment collection | Research + annotation |
| 2 | Retrieval system adapters (all 11 systems) | Engineering |
| 3 | Baseline BEIR replication experiment | Research |
| 3 | Latency benchmarking infrastructure | Engineering |
| 4 | Chunking sensitivity + LA-NDCG experiments | Research |
| 4 | Cost-quality frontier analysis | Research |
| 5 | Context saturation experiment | Research |
| 5 | Paper writing (first full draft) | All |
| 6 | Internal mock review + revision | All |
| 6 | Submit to SIGIR 2027 (deadline ~January 2027) | Lead author |

---

## 10. Open Questions and Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| API pricing changes before paper submission | Medium | Record prices at time of experiment; note in paper that pricing is time-sensitive |
| BEIR rankings don't actually change under production conditions | Low | Pre-registration of hypothesis; if rankings are stable, that's also a finding |
| Structure-aware chunker quality varies by domain | Medium | Human evaluation of chunking quality; ablation with and without domain-specific chunker |
| Latency measurements are hardware-dependent | High | Standardize on AWS c5.2xlarge; provide normalization formula; report raw latency numbers |

---

## 11. Related Issues

- GitHub issue #20: Design doc
- GitHub issues labeled `conference-prep`: conference targeting, reproducibility, ethics, mock review
- GitHub issue #15: Reproducibility & artifact release
- GitHub issue #16: Statistical rigor
- GitHub issue #17: Ethics & broader impact
- GitHub issue #18: Related work & novelty audit
- GitHub issue #19: Internal mock peer review
