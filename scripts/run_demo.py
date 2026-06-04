#!/usr/bin/env python3
"""Demo script: synthetic RAG ablation benchmark."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievalbench.core import (
    Domain,
    RAGConfig,
    BenchmarkRun,
    RetrievalBench,
)
from retrievalbench.data import make_corpus, make_queries, make_retrieval_result
from retrievalbench.evaluate import ablation_table

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None  # type: ignore


def main() -> None:
    print("=== RetrievalBench Demo ===")

    corpus = make_corpus(n_docs=200, seed=0)
    queries, qrels = make_queries(n=30, corpus=corpus, seed=0)

    configs = [
        RAGConfig("fixed_512"),
        RAGConfig("sentence", use_reranking=True),
        RAGConfig("recursive", use_reranking=True, use_metadata=True),
        RAGConfig("semantic", use_reranking=True, use_metadata=True, use_query_expansion=True),
        RAGConfig("fixed_512", use_metadata=True),
        RAGConfig("sentence", use_query_expansion=True),
    ]

    bench = RetrievalBench()
    for cfg in configs:
        run = BenchmarkRun(config=cfg, domain=Domain.FINANCE)
        recall_boost = 0.1 * (cfg.use_reranking + cfg.use_metadata + cfg.use_query_expansion)
        for q in queries:
            rel = qrels.get(q["query_id"], set())
            res = make_retrieval_result(
                q["query_id"], corpus, rel, recall=min(0.95, 0.6 + recall_boost)
            )
            run.results.append(res)
        bench.add_run(run)

    df = ablation_table(bench.runs, qrels, k=10)
    print("\n--- Ablation Table (sorted by nDCG@10) ---")
    print(df.to_string(index=False))

    best_ndcg = df.iloc[0]["ndcg@k"]
    best_cfg = df.iloc[0]["config"]
    print(f"\nBest config: {best_cfg}  (nDCG@10={best_ndcg:.4f})")


if __name__ == "__main__":
    main()
