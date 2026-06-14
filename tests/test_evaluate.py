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
    bootstrap_ci,
    score_variance,
    permutation_test,
    bonferroni_correct,
    cohens_d,
    kendall_tau,
    multi_seed_variance,
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


# --- statistical significance tests ---

def test_bootstrap_ci_contains_mean() -> None:
    values = [0.6, 0.7, 0.8, 0.65, 0.75]
    lo, hi = bootstrap_ci(values, n_resamples=1000)
    mean = sum(values) / len(values)
    assert lo < mean < hi


def test_bootstrap_ci_single_value() -> None:
    lo, hi = bootstrap_ci([0.5])
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_wider_at_99() -> None:
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    lo_95, hi_95 = bootstrap_ci(values, confidence=0.95, n_resamples=1000)
    lo_99, hi_99 = bootstrap_ci(values, confidence=0.99, n_resamples=1000)
    assert (hi_99 - lo_99) >= (hi_95 - lo_95)


def test_score_variance_identical() -> None:
    assert score_variance([0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_score_variance_single() -> None:
    assert score_variance([0.8]) == pytest.approx(0.0)


def test_score_variance_positive() -> None:
    assert score_variance([0.2, 0.8]) > 0.0


def test_permutation_test_identical() -> None:
    result = permutation_test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert not result["significant"]


def test_permutation_test_clearly_different() -> None:
    a = [0.9] * 30
    b = [0.1] * 30
    result = permutation_test(a, b, n_resamples=1000)
    assert result["significant"]


def test_permutation_test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        permutation_test([0.5, 0.6], [0.5])


def test_bonferroni_correct() -> None:
    corrected = bonferroni_correct([0.01, 0.03, 0.05])
    assert corrected[0] == pytest.approx(0.03)
    assert corrected[1] == pytest.approx(0.09)
    assert corrected[2] == pytest.approx(0.15)


def test_bonferroni_caps_at_1() -> None:
    corrected = bonferroni_correct([0.5, 0.5])
    assert all(p <= 1.0 for p in corrected)


def test_cohens_d_identical() -> None:
    assert cohens_d([0.5, 0.6, 0.7], [0.5, 0.6, 0.7]) == pytest.approx(0.0)


def test_cohens_d_large_effect() -> None:
    a = [0.85, 0.90, 0.88, 0.92, 0.87]
    b = [0.12, 0.10, 0.15, 0.11, 0.13]
    assert abs(cohens_d(a, b)) > 1.0


def test_kendall_tau_identical() -> None:
    configs = ["a", "b", "c"]
    assert kendall_tau(configs, configs) == pytest.approx(1.0)


def test_kendall_tau_reversed() -> None:
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)


def test_multi_seed_variance_three_seeds() -> None:
    seeds = [[0.7, 0.8, 0.75], [0.72, 0.78, 0.74], [0.68, 0.82, 0.76]]
    result = multi_seed_variance(seeds)
    assert "mean" in result
    assert "std" in result
    assert len(result["seed_means"]) == 3
    assert result["std"] >= 0.0


def test_evaluate_run_has_ci_keys() -> None:
    config = RAGConfig("fixed_512")
    run = BenchmarkRun(config=config, domain=Domain.FINANCE)
    for i in range(5):
        run.results.append(
            RetrievalResult(f"q_{i:04d}", ["doc_0000", "doc_0001"], [0.9, 0.7])
        )
    qrels = {f"q_{i:04d}": {"doc_0000"} for i in range(5)}
    metrics = evaluate_run(run, qrels, k=5)
    assert "ndcg@k_ci_lower" in metrics
    assert "ndcg@k_ci_upper" in metrics
    assert "ndcg@k_variance" in metrics
    assert "mrr_ci_lower" in metrics
    assert "mrr_ci_upper" in metrics
    assert metrics["ndcg@k_ci_lower"] <= metrics["ndcg@k"] <= metrics["ndcg@k_ci_upper"]
