from __future__ import annotations
import pytest
from retrievalbench.cost import (
    estimate_cost,
    estimate_latency,
    OperatingPoint,
    pareto_frontier,
    build_leaderboard,
    leaderboard_rows,
)


def test_estimate_cost_baseline() -> None:
    cost = estimate_cost(use_reranking=False, use_query_expansion=False)
    assert cost == pytest.approx(0.00015)


def test_estimate_cost_with_reranking() -> None:
    cost_base = estimate_cost(use_reranking=False)
    cost_rerank = estimate_cost(use_reranking=True)
    assert cost_rerank > cost_base


def test_estimate_cost_with_qexp() -> None:
    cost_base = estimate_cost(use_query_expansion=False)
    cost_qexp = estimate_cost(use_query_expansion=True)
    assert cost_qexp > cost_base


def test_estimate_latency_baseline() -> None:
    latency = estimate_latency(50.0)
    assert latency == pytest.approx(50.0)


def test_estimate_latency_with_reranking() -> None:
    base = estimate_latency(50.0)
    with_rerank = estimate_latency(50.0, use_reranking=True)
    assert with_rerank > base


def test_estimate_latency_all_features() -> None:
    latency = estimate_latency(50.0, use_reranking=True, use_query_expansion=True, use_metadata=True)
    assert latency > 50.0 + 120.0 + 80.0 + 15.0 - 0.01


def test_pareto_frontier_basic() -> None:
    points = [
        OperatingPoint("A", ndcg=0.8, latency_ms=200, cost_per_query_usd=0.001),
        OperatingPoint("B", ndcg=0.6, latency_ms=100, cost_per_query_usd=0.0005),
        OperatingPoint("C", ndcg=0.5, latency_ms=300, cost_per_query_usd=0.002),
    ]
    result = pareto_frontier(points)
    pareto = {p.system: p.pareto_optimal for p in result}
    assert pareto["A"] is True
    assert pareto["B"] is True
    assert pareto["C"] is False


def test_pareto_frontier_dominated() -> None:
    points = [
        OperatingPoint("A", ndcg=0.8, latency_ms=100, cost_per_query_usd=0.001),
        OperatingPoint("B", ndcg=0.6, latency_ms=200, cost_per_query_usd=0.002),
    ]
    result = pareto_frontier(points)
    pareto = {p.system: p.pareto_optimal for p in result}
    assert pareto["A"] is True
    assert pareto["B"] is False


def test_build_leaderboard_sorted_by_ndcg() -> None:
    rows = [
        {"config": "fixed_512", "ndcg@k": 0.41},
        {"config": "sentence+rerank", "ndcg@k": 0.60},
        {"config": "semantic+rerank+meta+qexp", "ndcg@k": 0.82},
    ]
    leaderboard = build_leaderboard(rows)
    ndcgs = [p.ndcg for p in leaderboard]
    assert ndcgs == sorted(ndcgs, reverse=True)


def test_build_leaderboard_pareto_flags_set() -> None:
    rows = [
        {"config": "fixed_512", "ndcg@k": 0.41},
        {"config": "semantic+rerank+meta+qexp", "ndcg@k": 0.82},
    ]
    leaderboard = build_leaderboard(rows)
    assert any(p.pareto_optimal for p in leaderboard)


def test_leaderboard_rows_keys() -> None:
    rows = [{"config": "fixed_512", "ndcg@k": 0.41}]
    leaderboard = build_leaderboard(rows)
    result = leaderboard_rows(leaderboard)
    assert "system" in result[0]
    assert "ndcg@10" in result[0]
    assert "latency_ms" in result[0]
    assert "cost_per_1k_usd" in result[0]
    assert "pareto_optimal" in result[0]


def test_cost_per_1k_property() -> None:
    p = OperatingPoint("X", ndcg=0.5, latency_ms=100, cost_per_query_usd=0.001)
    assert p.cost_per_1k_usd == pytest.approx(1.0)