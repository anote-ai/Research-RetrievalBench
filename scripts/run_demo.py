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
from retrievalbench.evaluate import evaluate_run
from retrievalbench.scheduling import (
    AdaptiveRetrievalScheduler,
    CPU_ONLY,
    EDGE_GPU,
    FixedStrideScheduler,
    SERVER_GPU,
    make_generation_signals,
    simulate_scheduled_generation,
)

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None  # type: ignore


def _format_table(rows: list[dict], columns: list[str]) -> str:
    formatted_rows = []
    for row in rows:
        formatted = {}
        for column in columns:
            value = row[column]
            formatted[column] = f"{value:.4f}" if isinstance(value, float) else str(value)
        formatted_rows.append(formatted)

    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(row[column].ljust(widths[column]) for column in columns)
        for row in formatted_rows
    ]
    return "\n".join([header, divider, *body])


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

    rows = [evaluate_run(run, qrels, k=10) for run in bench.runs]
    rows.sort(key=lambda row: row["ndcg@k"], reverse=True)
    print("\n--- Ablation Table (sorted by nDCG@10) ---")
    print(
        _format_table(
            rows,
            [
                "config",
                "domain",
                "recall@k",
                "precision@k",
                "ndcg@k",
                "mrr",
                "map",
                "r_precision",
                "n_queries",
            ],
        )
    )

    best_ndcg = rows[0]["ndcg@k"]
    best_cfg = rows[0]["config"]
    print(f"\nBest config: {best_cfg}  (nDCG@10={best_ndcg:.4f})")

    print("\n--- Hardware-Aware Adaptive Retrieval Scheduling ---")
    signals = make_generation_signals(n_tokens=96, seed=7, difficulty="medium")
    schedulers = [
        FixedStrideScheduler(stride=8, top_k=8, power_mode="balanced"),
        FixedStrideScheduler(stride=16, top_k=8, power_mode="balanced"),
        AdaptiveRetrievalScheduler(
            min_interval=8,
            max_interval=24,
            base_top_k=3,
            max_top_k=8,
        ),
    ]
    schedule_rows = []
    for hardware in (EDGE_GPU, SERVER_GPU, CPU_ONLY):
        for scheduler in schedulers:
            scheduled = simulate_scheduled_generation(
                signals=signals,
                scheduler=scheduler,
                hardware=hardware,
                generation_power_mode="balanced",
            )
            schedule_rows.append(scheduled.as_dict())

    print(
        _format_table(
            schedule_rows,
            [
                "scheduler",
                "hardware",
                "retrieval_calls",
                "avg_top_k",
                "ttft_ms",
                "mean_tbt_ms",
                "latency_ms",
                "energy_j",
                "quality",
            ],
        )
    )

    edge_fixed = next(
        row
        for row in schedule_rows
        if row["hardware"] == "edge_gpu" and row["scheduler"] == "fixed_stride_8_k8_balanced"
    )
    edge_adaptive = next(
        row
        for row in schedule_rows
        if row["hardware"] == "edge_gpu" and row["scheduler"] == "adaptive_k3-8"
    )
    energy_savings = (
        100.0 * (edge_fixed["energy_j"] - edge_adaptive["energy_j"]) / edge_fixed["energy_j"]
    )
    quality_delta = edge_adaptive["quality"] - edge_fixed["quality"]
    print(
        "\nEdge GPU adaptive vs fixed stride-8: "
        f"{energy_savings:.1f}% energy savings, quality delta={quality_delta:+.4f}"
    )


if __name__ == "__main__":
    main()
