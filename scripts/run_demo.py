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
from retrievalbench.cost import build_leaderboard, leaderboard_rows
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


BENCHMARK_DOMAINS = (
    Domain.FINANCE,
    Domain.LEGAL,
    Domain.MEDICAL,
    Domain.TECHNICAL,
)

CONFIGS = (
    RAGConfig("fixed_512"),
    RAGConfig("sentence", use_reranking=True),
    RAGConfig("fixed_512", use_metadata=True),
    RAGConfig("recursive", use_reranking=True, use_metadata=True),
    RAGConfig("sentence", use_query_expansion=True),
    RAGConfig("semantic", use_reranking=True, use_metadata=True, use_query_expansion=True),
)


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


def _config_recall(cfg: RAGConfig, domain: Domain) -> float:
    domain_base = {
        Domain.FINANCE: 0.58,
        Domain.LEGAL: 0.54,
        Domain.MEDICAL: 0.52,
        Domain.TECHNICAL: 0.56,
    }[domain]
    chunk_bonus = {
        "fixed_512": 0.00,
        "sentence": 0.02,
        "recursive": 0.03,
        "semantic": 0.04,
    }[cfg.chunking_strategy]
    metadata_bonus = {
        Domain.FINANCE: 0.08,
        Domain.LEGAL: 0.10,
        Domain.MEDICAL: 0.12,
        Domain.TECHNICAL: 0.07,
    }[domain]
    query_expansion_bonus = {
        Domain.FINANCE: 0.10,
        Domain.LEGAL: 0.08,
        Domain.MEDICAL: -0.05,
        Domain.TECHNICAL: 0.09,
    }[domain]

    recall = domain_base + chunk_bonus
    if cfg.use_reranking:
        recall += 0.11
    if cfg.use_metadata:
        recall += metadata_bonus
    if cfg.use_query_expansion:
        recall += query_expansion_bonus
    return max(0.05, min(0.95, recall))


def _config_relevance_bias(cfg: RAGConfig, domain: Domain) -> float:
    query_expansion_bonus = {
        Domain.FINANCE: 0.04,
        Domain.LEGAL: 0.03,
        Domain.MEDICAL: -0.03,
        Domain.TECHNICAL: 0.04,
    }[domain]

    bias = 0.04
    if cfg.chunking_strategy in {"recursive", "semantic"}:
        bias += 0.03
    if cfg.use_reranking:
        bias += 0.24
    if cfg.use_metadata:
        bias += 0.05
    if cfg.use_query_expansion:
        bias += query_expansion_bonus
    return max(0.0, min(0.50, bias))


def _run_domain_ablation() -> tuple[RetrievalBench, dict[str, set[str]]]:
    corpus = make_corpus(n_docs=240, seed=0, domains=BENCHMARK_DOMAINS)
    bench = RetrievalBench()
    all_qrels: dict[str, set[str]] = {}

    for domain_index, domain in enumerate(BENCHMARK_DOMAINS):
        queries, qrels = make_queries(
            n=30,
            corpus=corpus,
            seed=domain_index,
            domain=domain,
            query_id_prefix=f"{domain.value}_q",
        )
        all_qrels.update(qrels)

        for cfg in CONFIGS:
            run = BenchmarkRun(config=cfg, domain=domain)
            recall = _config_recall(cfg, domain)
            for query in queries:
                relevant_ids = qrels.get(query["query_id"], set())
                result = make_retrieval_result(
                    query["query_id"],
                    corpus,
                    relevant_ids,
                    recall=recall,
                    seed_salt=f"{domain.value}:{cfg.name()}",
                    relevance_bias=_config_relevance_bias(cfg, domain),
                )
                run.results.append(result)
            bench.add_run(run)

    return bench, all_qrels


def _domain_summary_rows(rows: list[dict]) -> list[dict]:
    summary = []
    for domain in BENCHMARK_DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain.value]
        best = max(domain_rows, key=lambda row: row["ndcg@k"])
        summary.append(
            {
                "domain": domain.value,
                "best_config": best["config"],
                "ndcg@10": best["ndcg@k"],
                "mrr": best["mrr"],
                "n_queries": best["n_queries"],
            }
        )
    return summary


def _domain_config_rows(rows: list[dict]) -> list[dict]:
    domain_order = {domain.value: index for index, domain in enumerate(BENCHMARK_DOMAINS)}
    config_order = {cfg.name(): index for index, cfg in enumerate(CONFIGS)}
    return sorted(
        rows,
        key=lambda row: (
            domain_order[row["domain"]],
            config_order[row["config"]],
        ),
    )


def main() -> None:
    print("=== RetrievalBench Demo ===")

    bench, qrels = _run_domain_ablation()
    rows = [evaluate_run(run, qrels, k=10) for run in bench.runs]
    rows.sort(key=lambda row: row["ndcg@k"], reverse=True)
    finance_rows = [row for row in rows if row["domain"] == Domain.FINANCE.value]
    print(f"\nEvaluated {len(CONFIGS)} configs across {len(BENCHMARK_DOMAINS)} domains.")
    print("\n--- Finance Ablation Table (sorted by nDCG@10) ---")
    print(
        _format_table(
            finance_rows,
            [
                "config",
                "domain",
                "recall@k",
                "precision@k",
                "f1@k",
                "ndcg@k",
                "ndcg@k_ci_lower",
                "ndcg@k_ci_upper",
                "mrr",
                "map",
                "r_precision",
                "n_queries",
            ],
        )
    )

    best_ndcg = finance_rows[0]["ndcg@k"]
    best_cfg = finance_rows[0]["config"]
    print(f"\nBest config: {best_cfg}  (nDCG@10={best_ndcg:.4f})")

    print("\n--- Full 4-Domain Results (6 configs x 4 domains) ---")
    print(
        _format_table(
            _domain_config_rows(rows),
            [
                "domain",
                "config",
                "recall@k",
                "precision@k",
                "f1@k",
                "ndcg@k",
                "ndcg@k_ci_lower",
                "ndcg@k_ci_upper",
                "mrr",
                "map",
                "r_precision",
                "n_queries",
            ],
        )
    )

    print("\n--- 4-Domain Comparison ---")
    print(
        _format_table(
            _domain_summary_rows(rows),
            ["domain", "best_config", "ndcg@10", "mrr", "n_queries"],
        )
    )

    print("\n--- Cost-Latency-Quality Leaderboard (Finance, averaged across configs) ---")
    finance_rows = [row for row in rows if row["domain"] == Domain.FINANCE.value]
    leaderboard = build_leaderboard(finance_rows)
    print(
        _format_table(
            leaderboard_rows(leaderboard),
            ["system", "ndcg@10", "latency_ms", "cost_per_1k_usd", "pareto_optimal"],
        )
    )
    pareto_systems = [p.system for p in leaderboard if p.pareto_optimal]
    print(f"\nPareto-optimal systems: {', '.join(pareto_systems)}")

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
