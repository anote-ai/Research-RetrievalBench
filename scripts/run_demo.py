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
from retrievalbench.evaluate import (
    bridge_recall_at_k,
    bridge_recall_strict_at_k,
    evaluate_run,
    nugget_recall_at_k,
    position_bias_audit,
)
from retrievalbench.data import make_nuggets
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


def _position_penalty_for_chunking(chunking_strategy: str) -> float:
    """Fixed-size chunking has the strongest position bias; semantic the least."""
    return {
        "fixed_512": 0.30,
        "sentence": 0.18,
        "recursive": 0.12,
        "semantic": 0.05,
    }[chunking_strategy]


def _run_domain_ablation() -> tuple[RetrievalBench, dict[str, set[str]], list[dict]]:
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
            pos_penalty = _position_penalty_for_chunking(cfg.chunking_strategy)
            for query in queries:
                relevant_ids = qrels.get(query["query_id"], set())
                result = make_retrieval_result(
                    query["query_id"],
                    corpus,
                    relevant_ids,
                    recall=recall,
                    seed_salt=f"{domain.value}:{cfg.name()}",
                    relevance_bias=_config_relevance_bias(cfg, domain),
                    position_penalty_scale=pos_penalty,
                )
                run.results.append(result)
            bench.add_run(run)

    return bench, all_qrels, corpus


def _run_query_type_ablation() -> dict[str, tuple[RetrievalBench, dict[str, set[str]]]]:
    """Run the ablation separately for each query type.

    Returns a dict mapping query_type -> (bench, qrels) so that per-type
    nDCG and bridge-recall can be computed and compared.
    """
    corpus = make_corpus(n_docs=240, seed=0, domains=BENCHMARK_DOMAINS)
    results: dict[str, tuple[RetrievalBench, dict[str, set[str]]]] = {}

    for query_type in ("single_hop", "multi_hop", "temporal"):
        bench = RetrievalBench()
        all_qrels: dict[str, set[str]] = {}

        for domain_index, domain in enumerate(BENCHMARK_DOMAINS):
            queries, qrels = make_queries(
                n=10,
                corpus=corpus,
                seed=domain_index + 10,
                domain=domain,
                query_id_prefix=f"{domain.value}_{query_type}_q",
                query_type=query_type,
            )
            all_qrels.update(qrels)

            for cfg in CONFIGS:
                run = BenchmarkRun(config=cfg, domain=domain)
                recall = _config_recall(cfg, domain)
                # multi_hop is harder — apply a recall penalty to model
                # the difficulty of retrieving all bridge documents
                if query_type == "multi_hop":
                    recall *= 0.75
                for query in queries:
                    relevant_ids = qrels.get(query["query_id"], set())
                    result = make_retrieval_result(
                        query["query_id"],
                        corpus,
                        relevant_ids,
                        recall=recall,
                        seed_salt=f"{domain.value}:{cfg.name()}:{query_type}",
                        relevance_bias=_config_relevance_bias(cfg, domain),
                    )
                    run.results.append(result)
                bench.add_run(run)

        results[query_type] = (bench, all_qrels)

    return results


def _degradation_rows(
    type_results: dict[str, tuple[RetrievalBench, dict[str, set[str]]]],
) -> list[dict]:
    """Build a per-config degradation table: single_hop nDCG vs multi_hop nDCG.

    Also computes bridge-recall@10 for multi_hop queries (both soft and strict).
    """
    single_bench, single_qrels = type_results["single_hop"]
    multi_bench, multi_qrels = type_results["multi_hop"]
    temporal_bench, temporal_qrels = type_results["temporal"]

    def avg_ndcg(bench: RetrievalBench, qrels: dict) -> dict[str, float]:
        per_cfg: dict[str, list[float]] = {}
        for run in bench.runs:
            metrics = evaluate_run(run, qrels, k=10)
            per_cfg.setdefault(metrics["config"], []).append(metrics["ndcg@k"])
        return {cfg: sum(v) / len(v) for cfg, v in per_cfg.items()}

    def avg_bridge_recall(bench: RetrievalBench, qrels: dict) -> dict[str, tuple[float, float]]:
        """Returns (soft_bridge_recall, strict_bridge_recall) per config."""
        per_cfg: dict[str, list[tuple[float, float]]] = {}
        for run in bench.runs:
            for result in run.results:
                bridge_ids = qrels.get(result.query_id, set())
                soft = bridge_recall_at_k(result.retrieved_ids, bridge_ids, k=10)
                strict = bridge_recall_strict_at_k(result.retrieved_ids, bridge_ids, k=10)
                cfg_name = run.config.name()
                per_cfg.setdefault(cfg_name, []).append((soft, strict))
        return {
            cfg: (
                sum(s for s, _ in pairs) / len(pairs),
                sum(st for _, st in pairs) / len(pairs),
            )
            for cfg, pairs in per_cfg.items()
        }

    single_ndcg = avg_ndcg(single_bench, single_qrels)
    multi_ndcg = avg_ndcg(multi_bench, multi_qrels)
    temporal_ndcg = avg_ndcg(temporal_bench, temporal_qrels)
    bridge_recalls = avg_bridge_recall(multi_bench, multi_qrels)

    rows = []
    for cfg in single_ndcg:
        s = single_ndcg[cfg]
        m = multi_ndcg.get(cfg, 0.0)
        t = temporal_ndcg.get(cfg, 0.0)
        soft_br, strict_br = bridge_recalls.get(cfg, (0.0, 0.0))
        rows.append({
            "config": cfg,
            "ndcg_single": s,
            "ndcg_multi": m,
            "ndcg_temporal": t,
            "multi_hop_delta": m - s,
            "bridge_recall": soft_br,
            "bridge_recall_strict": strict_br,
        })
    rows.sort(key=lambda r: r["multi_hop_delta"])
    return rows


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

    bench, qrels, corpus = _run_domain_ablation()
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

    print("\n--- Nugget Recall vs NDCG Divergence (Finance) ---")
    print("(gap = NDCG - nugget_recall; positive gap = retriever ranks peripheral chunks highly)")
    nuggets = make_nuggets(qrels, seed=99)
    nugget_rows = []
    for run in bench.runs:
        if run.domain.value != "finance":
            continue
        from retrievalbench.evaluate import ndcg_at_k
        ndcg_scores, nugget_scores = [], []
        for result in run.results:
            rel = qrels.get(result.query_id, set())
            nug = nuggets.get(result.query_id, set())
            ndcg_scores.append(ndcg_at_k(result.retrieved_ids, rel, k=10))
            nugget_scores.append(nugget_recall_at_k(result.retrieved_ids, nug, k=10))
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        avg_nugget = sum(nugget_scores) / len(nugget_scores) if nugget_scores else 0.0
        nugget_rows.append({
            "config": run.config.name(),
            "ndcg@10": avg_ndcg,
            "nugget_recall@10": avg_nugget,
            "gap": avg_ndcg - avg_nugget,
        })
    nugget_rows.sort(key=lambda r: r["gap"], reverse=True)
    print(_format_table(nugget_rows, ["config", "ndcg@10", "nugget_recall@10", "gap"]))
    worst_gap = nugget_rows[0]
    print(
        f"\nLargest gap: {worst_gap['config']} "
        f"(NDCG={worst_gap['ndcg@10']:.4f}, nugget_recall={worst_gap['nugget_recall@10']:.4f}, gap={worst_gap['gap']:+.4f})"
    )

    print("\n--- Position Bias Audit by Chunking Strategy ---")
    print("(bias_gap = early_recall - late_recall; larger gap = stronger position bias)")
    bias_results = position_bias_audit(bench.runs, corpus, qrels, k=10)
    bias_rows = [
        {"config": cfg, **tiers}
        for cfg, tiers in sorted(bias_results.items(), key=lambda x: -x[1]["bias_gap"])
    ]
    print(_format_table(bias_rows, ["config", "early", "mid", "late", "bias_gap"]))
    most_biased = bias_rows[0]
    least_biased = bias_rows[-1]
    print(f"\nMost position-biased:  {most_biased['config']} (gap={most_biased['bias_gap']:+.4f})")
    print(f"Least position-biased: {least_biased['config']} (gap={least_biased['bias_gap']:+.4f})")

    print("\n--- Multi-Hop & Temporal Degradation Leaderboard ---")
    print("(sorted by multi_hop_delta ascending = worst degradation first)")
    type_results = _run_query_type_ablation()
    deg_rows = _degradation_rows(type_results)
    print(
        _format_table(
            deg_rows,
            [
                "config",
                "ndcg_single",
                "ndcg_multi",
                "ndcg_temporal",
                "multi_hop_delta",
                "bridge_recall",
                "bridge_recall_strict",
            ],
        )
    )
    worst = deg_rows[0]
    best = deg_rows[-1]
    print(
        f"\nMost degraded on multi-hop: {worst['config']} "
        f"(delta={worst['multi_hop_delta']:+.4f}, bridge_recall={worst['bridge_recall']:.4f})"
    )
    print(
        f"Most robust on multi-hop:   {best['config']} "
        f"(delta={best['multi_hop_delta']:+.4f}, bridge_recall={best['bridge_recall']:.4f})"
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
