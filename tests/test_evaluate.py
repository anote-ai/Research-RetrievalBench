from __future__ import annotations
import pytest
from retrievalbench.evaluate import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    evaluate_run,
    ablation_table,
)
from retrievalbench.core import RAGConfig, BenchmarkRun, RetrievalResult, Domain


# --- recall@k ---

def test_recall_at_k_perfect() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_recall_at_k_zero() -> None:
    assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_recall_at_k_partial() -> None:
    r = recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=2)
    assert 0.0 < r < 1.0


# --- precision@k ---

def test_precision_at_k_perfect() -> None:
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_precision_at_k_zero_k() -> None:
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


# --- ndcg@k ---

def test_ndcg_at_k_perfect() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_zero() -> None:
    assert ndcg_at_k(["x", "y"], {"a", "b"}, k=5) == pytest.approx(0.0)


# --- mrr ---

def test_mrr_first_position() -> None:
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_position() -> None:
    assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_not_found() -> None:
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


# --- evaluate_run ---

def _make_run() -> tuple[BenchmarkRun, dict[str, set[str]]]:
    run = BenchmarkRun(config=RAGConfig("fixed_512"), domain=Domain.FINANCE)
    run.results = [
        RetrievalResult("q1", ["a", "b", "c"], [0.9, 0.8, 0.7]),
        RetrievalResult("q2", ["d", "e"], [0.6, 0.5]),
    ]
    qrels = {"q1": {"a"}, "q2": {"d"}}
    return run, qrels


def test_evaluate_run_keys() -> None:
    run, qrels = _make_run()
    result = evaluate_run(run, qrels)
    for key in ["config", "domain", "recall@k", "precision@k", "ndcg@k", "mrr", "n_queries"]:
        assert key in result


def test_evaluate_run_n_queries() -> None:
    run, qrels = _make_run()
    result = evaluate_run(run, qrels)
    assert result["n_queries"] == 2


def test_ablation_table_returns_df() -> None:
    import pandas as pd
    run, qrels = _make_run()
    df = ablation_table([run], qrels)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
