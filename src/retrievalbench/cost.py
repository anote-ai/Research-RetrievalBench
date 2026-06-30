"""Cost-latency-quality trade-off analysis for RAG retrieval pipelines.

Models cost-per-query and latency for different retrieval operating points,
computes Pareto frontiers, and produces a leaderboard table.

Reference: arXiv:2511.09545 — Practical RAG Evaluation: Cost-Latency-Quality Trade-offs
"""
from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

# Simulated per-query costs in USD (approximate production estimates)
_EMBEDDING_COST_USD = 0.0001    # text-embedding-3-small per query
_ANN_COST_USD = 0.00005         # vector DB lookup per query
_RERANKER_COST_USD = 0.0003     # cross-encoder reranker per query
_QUERY_EXPANSION_COST_USD = 0.0002  # LLM call for query expansion


@dataclass
class OperatingPoint:
    """A single retrieval system operating point with quality, latency, and cost."""
    system: str
    ndcg: float
    latency_ms: float
    cost_per_query_usd: float
    pareto_optimal: bool = False

    @property
    def cost_per_1k_usd(self) -> float:
        return self.cost_per_query_usd * 1000


def estimate_cost(
    use_reranking: bool = False,
    use_query_expansion: bool = False,
    n_retrieved: int = 10,
) -> float:
    """Estimate cost per query in USD based on pipeline configuration."""
    cost = _EMBEDDING_COST_USD + _ANN_COST_USD
    if use_reranking:
        cost += _RERANKER_COST_USD * n_retrieved
    if use_query_expansion:
        cost += _QUERY_EXPANSION_COST_USD
    return cost


def estimate_latency(
    base_latency_ms: float,
    use_reranking: bool = False,
    use_query_expansion: bool = False,
    use_metadata: bool = False,
) -> float:
    """Estimate end-to-end retrieval latency in ms."""
    latency = base_latency_ms
    if use_reranking:
        latency += 120.0   # cross-encoder reranking overhead
    if use_query_expansion:
        latency += 80.0    # LLM expansion call overhead
    if use_metadata:
        latency += 15.0    # metadata filter overhead
    return latency


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(points: list[OperatingPoint]) -> list[OperatingPoint]:
    """Mark Pareto-optimal points in the quality vs cost-latency space.

    A point is Pareto-optimal if no other point is strictly better on ALL of:
    - higher ndcg
    - lower latency_ms
    - lower cost_per_query_usd
    """
    for p in points:
        p.pareto_optimal = True
        for other in points:
            if other is p:
                continue
            if (
                other.ndcg >= p.ndcg
                and other.latency_ms <= p.latency_ms
                and other.cost_per_query_usd <= p.cost_per_query_usd
                and (
                    other.ndcg > p.ndcg
                    or other.latency_ms < p.latency_ms
                    or other.cost_per_query_usd < p.cost_per_query_usd
                )
            ):
                p.pareto_optimal = False
                break
    return points


# ---------------------------------------------------------------------------
# Leaderboard builder
# ---------------------------------------------------------------------------

def build_leaderboard(
    eval_rows: list[dict],
    base_latency_ms: float = 50.0,
) -> list[OperatingPoint]:
    """Build operating points from evaluate_run output rows.

    Args:
        eval_rows: List of dicts returned by evaluate_run(), one per config/domain.
        base_latency_ms: Base ANN lookup latency before pipeline overhead.

    Returns:
        List of OperatingPoint sorted by ndcg descending, with pareto flags set.
    """
    points = []
    seen = set()
    for row in eval_rows:
        config_name = row["config"]
        if config_name in seen:
            continue
        seen.add(config_name)

        use_reranking = "rerank" in config_name
        use_qexp = "qexp" in config_name
        use_metadata = "meta" in config_name

        latency = estimate_latency(
            base_latency_ms,
            use_reranking=use_reranking,
            use_query_expansion=use_qexp,
            use_metadata=use_metadata,
        )
        cost = estimate_cost(
            use_reranking=use_reranking,
            use_query_expansion=use_qexp,
        )

        points.append(OperatingPoint(
            system=config_name,
            ndcg=row["ndcg@k"],
            latency_ms=latency,
            cost_per_query_usd=cost,
        ))

    points = pareto_frontier(points)
    points.sort(key=lambda p: p.ndcg, reverse=True)
    return points


def leaderboard_rows(points: list[OperatingPoint]) -> list[dict]:
    """Convert operating points to plain dicts for tabular display."""
    return [
        {
            "system": p.system,
            "ndcg@10": round(p.ndcg, 4),
            "latency_ms": round(p.latency_ms, 1),
            "cost_per_1k_usd": round(p.cost_per_1k_usd, 4),
            "pareto_optimal": "✓" if p.pareto_optimal else "",
        }
        for p in points
    ]