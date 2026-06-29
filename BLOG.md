# Why Your RAG Benchmark Numbers Are Lying to You (a Little)

*A plain-English summary of the RetrievalBench project.*

## The short version

If you've ever picked an embedding model for a retrieval-augmented generation
(RAG) system because it topped a leaderboard like BEIR, you may have optimized
for the wrong thing. BEIR — the standard academic benchmark for retrieval —
tests how well a model finds relevant documents. It does **not** tell you:

- How you should split your documents into chunks before indexing them.
- Whether your retrieval system is fast enough to meet a latency budget.
- How much it will cost you to query an API-based embedding model at scale.
- Whether feeding more retrieved passages to your LLM actually helps once you
  account for how much context the model can pay attention to.

RetrievalBench is a project that asks: what would retrieval evaluation look
like if it were designed around how people actually run RAG in production,
instead of how academic IR papers measure it?

## The core idea

Picture two retrieval setups. One scores higher on the standard nDCG@10
metric but takes 800ms per query. The other scores slightly lower but
responds in 50ms. In a chatbot serving thousands of users, the second system
is very likely the better choice — but a leaderboard sorted purely by nDCG@10
would never tell you that.

The same logic applies to chunking strategy (how you cut a 40-page legal
contract into retrievable pieces) and to per-query API costs. None of these
show up in BEIR's leaderboard, yet practitioners report that they dominate
real-world deployment decisions.

## What's been built so far

This is an early-stage research project, and it's important to be upfront
about where things stand:

- **A working metrics and statistics library.** The codebase implements
  standard IR metrics (Recall@k, Precision@k, nDCG@k, MRR, MAP,
  R-Precision) plus research-oriented extras: a latency-adjusted nDCG
  variant, Kendall's tau for comparing rankings, bootstrap confidence
  intervals, permutation significance tests, Bonferroni correction, and
  Cohen's d effect sizes. These are tested and runnable today
  (`src/retrievalbench/evaluate.py`).
- **A synthetic ablation harness.** The current demo (`scripts/run_demo.py`)
  generates synthetic corpora and queries across four domains (finance,
  legal, medical, technical) and six RAG pipeline configurations (chunking
  x reranking x metadata x query expansion), then scores them with the
  metrics above. This is useful for validating the evaluation code itself,
  but the numbers come from a controlled simulation, not from running real
  embedding models against real documents.
- **A hardware-aware scheduling simulator.** A separate module
  (`src/retrievalbench/scheduling.py`) models adaptive vs. fixed-stride
  retrieval scheduling under different power profiles (edge GPU, server
  GPU, CPU-only), estimating latency and energy tradeoffs — again,
  simulated rather than measured on real hardware.

## What's still missing

The full research vision (see `DESIGN_DOC.md`) calls for a 12-domain,
11-system benchmark with real document corpora (SEC filings, legal
contracts, PubMed abstracts, etc.), real embedding models (BM25, E5, BGE,
Cohere, Voyage, OpenAI), and a genuine latency/cost characterization
measured on real hardware. None of that exists in the codebase yet — what
exists is the *evaluation infrastructure* (the metrics, the statistics, the
ablation harness shape) that the real experiments would plug into.

In short: the measurement tools are real and tested. The headline claims
about chunking sensitivity, cost-quality frontiers, and BEIR-vs-production
ranking divergence are still projections from the design doc, not results
from running real systems on real data.

## Why this matters anyway

Even at this stage, the project is asking a question that matters: academic
retrieval benchmarks and production RAG deployments are evaluated on
different axes, and nobody has systematically quantified the gap. If the
full experiment suite bears out the design doc's hypotheses — that chunking
strategy can matter more than embedding model choice, or that BEIR rankings
reshuffle substantially under realistic latency budgets — that would be a
genuinely useful, citable result for anyone building or buying RAG
infrastructure.

## Where to look in the repo

- `DESIGN_DOC.md` — the full research plan, hypotheses, and claims.
- `src/retrievalbench/evaluate.py` — the metrics and statistical-testing code.
- `src/retrievalbench/core.py` — config and benchmark-run data model.
- `src/retrievalbench/scheduling.py` — the hardware-aware scheduling simulator.
- `scripts/run_demo.py` — runnable end-to-end synthetic demo.
- `PAPER_DRAFT.md` — an early paper skeleton with sections mapped to the
  design doc's claims (clearly marked where numbers are real vs. projected).
- `results/README.md` — where benchmark output will be checked in once real
  experiments run.
