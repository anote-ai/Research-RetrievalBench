from __future__ import annotations
import pytest
from retrievalbench.core import (
    Domain,
    RAGConfig,
    RetrievalResult,
    BenchmarkRun,
    RetrievalBench,
)


def test_ragconfig_valid_strategy() -> None:
    cfg = RAGConfig("fixed_512")
    assert cfg.chunking_strategy == "fixed_512"


def test_ragconfig_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError):
        RAGConfig("unknown_strategy")


def test_ragconfig_name_baseline() -> None:
    assert RAGConfig("fixed_512").name() == "fixed_512"


def test_ragconfig_name_with_flags() -> None:
    cfg = RAGConfig("sentence", use_reranking=True, use_metadata=True)
    assert cfg.name() == "sentence+rerank+meta"


def test_ragconfig_name_all_flags() -> None:
    cfg = RAGConfig("recursive", use_reranking=True, use_metadata=True, use_query_expansion=True)
    assert "qexp" in cfg.name()


def test_retrieval_result_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        RetrievalResult(query_id="q1", retrieved_ids=["a", "b"], scores=[0.9])


def test_retrieval_result_valid() -> None:
    r = RetrievalResult(query_id="q1", retrieved_ids=["a"], scores=[0.9])
    assert r.query_id == "q1"


def test_benchmark_run_avg_latency_empty() -> None:
    run = BenchmarkRun()
    assert run.avg_latency_ms() == 0.0


def test_benchmark_run_avg_latency() -> None:
    run = BenchmarkRun()
    run.results = [
        RetrievalResult("q1", ["a"], [0.5], latency_ms=100.0),
        RetrievalResult("q2", ["b"], [0.4], latency_ms=200.0),
    ]
    assert run.avg_latency_ms() == pytest.approx(150.0)


def test_retrieval_bench_filter_by_domain() -> None:
    bench = RetrievalBench()
    run_fin = BenchmarkRun(domain=Domain.FINANCE)
    run_leg = BenchmarkRun(domain=Domain.LEGAL)
    bench.add_run(run_fin)
    bench.add_run(run_leg)
    assert bench.filter_by_domain(Domain.FINANCE) == [run_fin]
    assert bench.filter_by_domain(Domain.LEGAL) == [run_leg]


def test_retrieval_bench_best_config_by_ndcg() -> None:
    bench = RetrievalBench()
    ndcg_map = {"cfg_a": 0.5, "cfg_b": 0.8, "cfg_c": 0.3}
    assert bench.best_config_by_ndcg(ndcg_map) == "cfg_b"
