from __future__ import annotations
import pytest
from retrievalbench.evaluate import (
    recall_at_k,
    precision_at_k,
    f1_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    r_precision,
    evaluate_run,
    compare_domains,
    latency_adjusted_ndcg,
    query_difficulty_tier,
)
from retrievalbench.core import RAGConfig, BenchmarkRun, RetrievalResult, Domain


def test_recall_at_k_perfect() -> None:
    rel = {"a", "b", "c"}
    assert recall_at_k(["a", "b", "c", "d"], rel, 3) == pytest.approx(1.0)


def test_recall_at_k_zero() -> None:
    assert recall_at_k(["x", "y"], {"a"}, 2) == pytest.approx(0.0)


def test_recall_at_k_no_relevant() -> None:
    assert recall_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)


def test_precision_at_k_basic() -> None:
    rel = {"a", "b"}
    assert precision_at_k(["a", "x", "b", "y"], rel, 2) == pytest.approx(0.5)


def test_precision_at_k_zero_k() -> None:
    assert precision_at_k(["a"], {"a"}, 0) == pytest.approx(0.0)


def test_f1_at_k_perfect() -> None:
    rel = {"a", "b"}
    assert f1_at_k(["a", "b", "c"], rel, 2) == pytest.approx(1.0)


def test_f1_at_k_zero() -> None:
    assert f1_at_k(["x", "y"], {"a"}, 2) == pytest.approx(0.0)


def test_f1_at_k_no_relevant() -> None:
    assert f1_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)


def test_f1_at_k_zero_k() -> None:
    assert f1_at_k(["a"], {"a"}, 0) == pytest.approx(0.0)


def test_f1_at_k_partial() -> None:
    rel = {"a", "b", "c"}
    # top-2: ["a", "x"] → precision@2=0.5, recall@2=1/3
    prec, rec = 0.5, 1 / 3
    expected = 2 * prec * rec / (prec + rec)
    assert f1_at_k(["a", "x", "b"], rel, 2) == pytest.approx(expected)


def test_ndcg_perfect_ranking() -> None:
    rel = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)


def test_mrr_first_hit() -> None:
    assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_average_precision_perfect() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert average_precision(retrieved, relevant) == pytest.approx(1.0)


def test_average_precision_no_relevant() -> None:
    assert average_precision(["a", "b"], set()) == pytest.approx(0.0)


def test_average_precision_no_hits() -> None:
    assert average_precision(["x", "y"], {"a", "b"}) == pytest.approx(0.0)


def test_average_precision_partial() -> None:
    retrieved = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    expected = (1.0 / 1 + 2.0 / 3) / 2
    assert average_precision(retrieved, relevant) == pytest.approx(expected)


def test_map_empty() -> None:
    assert mean_average_precision([]) == pytest.approx(0.0)


def test_map_single_perfect() -> None:
    assert mean_average_precision([([ "a", "b"], {"a", "b"})]) == pytest.approx(1.0)


def test_map_multiple() -> None:
    results = [
        (["a", "b", "c"], {"a", "c"}),
        (["x", "y"], {"x"}),
    ]
    ap1 = (1.0 + 2.0 / 3) / 2
    ap2 = 1.0
    expected = (ap1 + ap2) / 2
    assert mean_average_precision(results) == pytest.approx(expected)


def test_r_precision_perfect() -> None:
    rel = {"a", "b", "c"}
    assert r_precision(["a", "b", "c", "d"], rel) == pytest.approx(1.0)


def test_r_precision_zero_relevant() -> None:
    assert r_precision(["a", "b"], set()) == pytest.approx(0.0)


def test_r_precision_none_in_top_r() -> None:
    rel = {"a", "b"}
    assert r_precision(["x", "y", "a"], rel) == pytest.approx(0.0)


def test_r_precision_partial() -> None:
    rel = {"a", "b"}
    assert r_precision(["a", "x", "b"], rel) == pytest.approx(0.5)


def test_r_precision_fewer_than_r_retrieved() -> None:
    rel = {"a", "b", "c"}
    assert r_precision(["a"], rel) == pytest.approx(1 / 3)


def test_evaluate_run_keys() -> None:
    config = RAGConfig("fixed_512")
    run = BenchmarkRun(config=config, domain=Domain.FINANCE)
    result = RetrievalResult(
        query_id="q_0000",
        retrieved_ids=["doc_0000", "doc_0001"],
        scores=[0.9, 0.7],
    )
    run.results.append(result)
    qrels = {"q_0000": {"doc_0000"}}
    metrics = evaluate_run(run, qrels, k=5)
    assert "map" in metrics
    assert "r_precision" in metrics
    assert "f1@k" in metrics
    assert 0.0 <= metrics["map"] <= 1.0
    assert 0.0 <= metrics["r_precision"] <= 1.0
    assert 0.0 <= metrics["f1@k"] <= 1.0


def test_compare_domains_returns_domain_keys() -> None:
    config = RAGConfig("sentence")
    run_fin = BenchmarkRun(config=config, domain=Domain.FINANCE)
    run_fin.results.append(
        RetrievalResult("q_0", ["d0", "d1"], [0.9, 0.5])
    )
    run_leg = BenchmarkRun(config=config, domain=Domain.LEGAL)
    run_leg.results.append(
        RetrievalResult("q_1", ["d2", "d3"], [0.8, 0.6])
    )
    qrels = {"q_0": {"d0"}, "q_1": {"d2"}}
    report = compare_domains([run_fin, run_leg], qrels)
    assert "finance" in report
    assert "legal" in report
    for domain_metrics in report.values():
        assert "map" in domain_metrics
        assert "r_precision" in domain_metrics
        assert "f1@k" in domain_metrics


def test_latency_adjusted_ndcg_within_budget() -> None:
    score = latency_adjusted_ndcg(["a", "b"], {"a", "b"}, latency_ms=100.0, k=2, latency_budget_ms=500.0)
    assert score == pytest.approx(1.0)


def test_latency_adjusted_ndcg_over_budget() -> None:
    score_penalised = latency_adjusted_ndcg(["a", "b"], {"a", "b"}, latency_ms=1500.0, k=2, latency_budget_ms=500.0)
    assert score_penalised < 1.0


def test_query_difficulty_easy() -> None:
    assert query_difficulty_tier({"a", "b", "c", "d", "e", "f"}, corpus_size=100) == "easy"


def test_query_difficulty_hard() -> None:
    assert query_difficulty_tier({"a"}, corpus_size=1000) == "hard"


def test_query_difficulty_medium() -> None:
    assert query_difficulty_tier({"a", "b", "c"}, corpus_size=100) == "medium"
