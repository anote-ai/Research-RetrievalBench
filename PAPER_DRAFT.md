# RetrievalBench: Production-Realistic Evaluation for Enterprise RAG Retrieval Systems

**Status: early draft / skeleton.** This document maps the claims in
[`DESIGN_DOC.md`](DESIGN_DOC.md) to their current implementation and evidence
status. Anything not explicitly labeled "measured" below is a projection or
hypothesis from the design doc, not a result obtained by running code in this
repository. This distinction is load-bearing: do not cite numbers from this
draft as experimental results until they are replaced with measured values
and the corresponding run artifacts are checked into `results/`.

## Abstract (draft)

BEIR established zero-shot cross-domain retrieval as the standard paradigm
for evaluating retrieval generalization, but it does not capture several
dimensions that dominate production RAG deployments: chunking strategy,
latency budgets, per-query cost, and context-window saturation. We propose
RetrievalBench, an evaluation framework and (eventually) a 12-domain, 11-system
benchmark that measures retrieval quality jointly with these production
constraints. *(Projected, pending full experiment run: we expect to show that
BEIR rankings correlate weakly with production-constrained rankings, and that
chunking strategy explains more ranking variance than embedding model choice
for long-document domains.)* This draft documents the evaluation
infrastructure built so far and the gap remaining before the full claims can
be substantiated.

## 1. Introduction

See `DESIGN_DOC.md` Section 1 for the full motivation. In short: production
RAG systems are constrained by latency SLAs, per-query cost budgets, and
chunking decisions that BEIR's protocol does not model. This project's
contribution, once complete, would be (a) a benchmark dataset spanning 12
domains and 4 chunking variants, (b) a latency-adjusted retrieval metric
(LA-NDCG), (c) a cost-quality Pareto characterization, and (d) an empirical
chunking-sensitivity taxonomy.

## 2. What Is Implemented Today (measured / verifiable by running code)

| Component | File | Status |
|---|---|---|
| Recall@k, Precision@k, F1@k, nDCG@k, MRR, MAP, R-Precision | `src/retrievalbench/evaluate.py` | Implemented, unit-tested |
| Latency-adjusted nDCG (exponential-penalty variant) | `src/retrievalbench/evaluate.py::latency_adjusted_ndcg` | Implemented; simpler functional form than the `min(budget/p95, 1.0)` formula proposed in DESIGN_DOC.md Section 5.3 — reconciliation needed |
| Kendall's tau (ranking comparison) | `src/retrievalbench/evaluate.py::kendall_tau` | Implemented, used for BEIR-vs-production-style ranking comparisons |
| Bootstrap confidence intervals, permutation significance test, Bonferroni correction, Cohen's d, multi-seed variance | `src/retrievalbench/evaluate.py` | Implemented, unit-tested (addresses DESIGN_DOC.md / issue #16 statistical rigor concerns) |
| RAG config model (chunking x rerank x metadata x query expansion), hybrid fusion (RRF / linear), latency SLA checking | `src/retrievalbench/core.py` | Implemented, unit-tested |
| Synthetic corpus/query/qrel generation for 4 domains | `src/retrievalbench/data.py` | Implemented — synthetic, not real corpora |
| Hardware-aware adaptive retrieval scheduling simulator (fixed-stride vs. adaptive, edge/server GPU/CPU power profiles, TTFT/TBT/energy estimates) | `src/retrievalbench/scheduling.py` | Implemented, unit-tested — simulated, not measured on physical hardware |
| End-to-end demo producing an ablation table across 4 domains x 6 configs | `scripts/run_demo.py` | Runnable; output is the basis for the numbers currently published in `README.md`'s ablation tables (these are real outputs of the synthetic simulation, not projections, but they describe synthetic data, not production RAG performance) |

## 3. What Is Not Yet Implemented (gap vs. DESIGN_DOC.md)

| Design doc component | Status |
|---|---|
| 12-domain real corpora (CUAD, EDGAR, PubMed, S2ORC, etc.) | Not started — only 4 synthetic domains exist |
| 4 real chunking strategy implementations operating on real documents (sentence/fixed/paragraph/structure-aware parsers) | Not started — `core.py` models chunking as a string label, not an actual chunker |
| 11 retrieval system adapters (BM25/Pyserini, SPLADE, E5, BGE, GTE, Nomic, OpenAI, Cohere, Voyage, hybrid RRF variants) | Not started — no real retrieval system is called anywhere in the codebase |
| Real query construction + human relevance judgments | Not started |
| Real latency benchmarking harness (P50/P95/P99 on a standard EC2 instance) | Not started — `scheduling.py` simulates latency, does not measure it |
| Cost-quality Pareto analysis using vendor pricing | Not started — no `cost_quality.py` module exists |
| Context-window saturation experiment (E2E RAG with GPT-4o at varying context sizes) | Not started |
| Public leaderboard website | Not started |

## 4. Primary Claims — Evidence Status

1. **Claim 1** (BEIR vs. LA-NDCG ranking divergence, Kendall's tau <= 0.4 at 100ms budget): *(projected, pending full experiment run)*. The `kendall_tau` and `latency_adjusted_ndcg` functions needed to test this exist and are unit-tested, but have never been run against real system latencies.
2. **Claim 2** (chunking variance > embedding variance for long documents, CSI > ESI for 8/12 domains): *(projected, pending full experiment run)*. No CSI/ESI computation exists yet; would be a straightforward addition on top of `evaluate.py` once real chunking variants and real systems exist.
3. **Claim 3** (structure-aware chunking beats fixed-size by 10-18 NDCG points on legal/financial): *(projected, pending full experiment run)*. No structure-aware chunker is implemented.
4. **Claim 4** (self-hosted open-source models reach >95% of API quality at <20% of cost): *(projected, pending full experiment run)*. No cost model or real embedding system calls exist.

## 5. Methods (for the infrastructure that does exist)

The statistical testing utilities follow standard IR evaluation practice:
bootstrap percentile confidence intervals (1000 resamples) for mean metric
estimates, paired permutation tests (1000 resamples) for comparing two
configurations' per-query scores (preferred over a t-test given the
non-normality of bounded IR metrics), Bonferroni correction for multiple
comparisons across configs/domains, and Cohen's d for effect size reporting.
These directly address the statistical-rigor concerns raised in the repo's
issue tracker (see DESIGN_DOC.md Section 11, issue #16) and are ready to be
applied once real experimental data is available.

## 6. Next Steps to De-Risk This Paper

In priority order (also see the companion GitHub issue for the full
improvement plan):

1. Stand up at least one real retrieval system adapter (e.g., BM25 via
   `rank_bm25` or Pyserini) against one small real public dataset (e.g., a
   BEIR subset) to replace synthetic data with a first real measurement.
2. Implement the CSI/ESI chunking-sensitivity computation in `evaluate.py`
   and validate it against a toy case with known ground truth.
3. Add a minimal real latency-measurement harness (wall-clock timing of
   actual retrieval calls) to replace the simulated `scheduling.py` figures
   for at least one system.
4. Replace this draft's projected claims with measured ones, one at a time,
   checking the raw run output into `results/`.
