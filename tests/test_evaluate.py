from __future__ import annotations
import pytest
from retrievalbench.evaluate import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    r_precision,
    latency_adjusted_ndcg,
    query_difficulty_tier,
)


def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == pytest.approx(0.5)


def test_precision_perfect():
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_precision_zero():
    assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)


def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_mrr_first_hit():
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_hit():
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)


def test_average_precision_perfect():
    assert average_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)


def test_map_single_query():
    result = mean_average_precision([(["a", "b", "c"], {"a", "b"})])
    assert 0.0 < result <= 1.0


def test_r_precision_perfect():
    assert r_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)


def test_latency_adjusted_ndcg_within_budget():
    score = latency_adjusted_ndcg(["a", "b"], {"a", "b"}, latency_ms=100.0, k=2, latency_budget_ms=500.0)
    assert score == pytest.approx(1.0)


def test_latency_adjusted_ndcg_over_budget():
    score_penalised = latency_adjusted_ndcg(["a", "b"], {"a", "b"}, latency_ms=1500.0, k=2, latency_budget_ms=500.0)
    score_base = ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
    assert score_penalised < score_base


def test_query_difficulty_easy():
    assert query_difficulty_tier({"a", "b", "c", "d", "e", "f"}, corpus_size=100) == "easy"


def test_query_difficulty_hard():
    assert query_difficulty_tier({"a"}, corpus_size=1000) == "hard"


def test_query_difficulty_medium():
    assert query_difficulty_tier({"a", "b", "c"}, corpus_size=100) == "medium"
