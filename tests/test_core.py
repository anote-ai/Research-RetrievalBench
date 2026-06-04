"""Tests for retrievalbench.core."""

import pytest

from retrievalbench.core import (
    BenchmarkRun,
    CHUNKING_STRATEGIES,
    Domain,
    RAGConfig,
    RetrievalBench,
    RetrievalResult,
)


def _make_config(**overrides):
    defaults = dict(
        chunking_strategy="fixed_512",
        use_reranking=False,
        use_metadata=True,
        use_query_expansion=False,
        embedding_model="text-embedding-3-small",
    )
    defaults.update(overrides)
    return RAGConfig(**defaults)


# --- Domain enum ---

def test_domain_enum_values():
    assert Domain.FINANCE.value == "finance"
    assert Domain.LEGAL.value == "legal"
    assert Domain.MEDICAL.value == "medical"
    assert Domain.TECHNICAL.value == "technical"


def test_domain_enum_count():
    assert len(list(Domain)) == 4


# --- CHUNKING_STRATEGIES ---

def test_chunking_strategies_is_list():
    assert isinstance(CHUNKING_STRATEGIES, list)


def test_chunking_strategies_content():
    assert "fixed_512" in CHUNKING_STRATEGIES
    assert "sentence" in CHUNKING_STRATEGIES
    assert len(CHUNKING_STRATEGIES) == 4


# --- RAGConfig ---

def test_ragconfig_creation():
    cfg = _make_config()
    assert cfg.chunking_strategy == "fixed_512"
    assert cfg.use_reranking is False
    assert cfg.use_metadata is True
    assert cfg.embedding_model == "text-embedding-3-small"


def test_ragconfig_invalid_strategy():
    with pytest.raises(ValueError, match="chunking_strategy"):
        _make_config(chunking_strategy="invalid_strategy")


def test_ragconfig_all_strategies_valid():
    for strategy in CHUNKING_STRATEGIES:
        cfg = _make_config(chunking_strategy=strategy)
        assert cfg.chunking_strategy == strategy


# --- RetrievalResult ---

def test_retrieval_result_construction():
    rr = RetrievalResult(
        query_id="q1",
        retrieved_ids=["d1", "d2", "d3"],
        scores=[0.9, 0.8, 0.7],
        latency_ms=42.5,
    )
    assert rr.query_id == "q1"
    assert len(rr.retrieved_ids) == 3
    assert rr.latency_ms == 42.5


def test_retrieval_result_mismatched_lengths():
    with pytest.raises(ValueError):
        RetrievalResult(
            query_id="q1",
            retrieved_ids=["d1", "d2"],
            scores=[0.9],
            latency_ms=10.0,
        )


# --- BenchmarkRun ---

def test_benchmark_run_construction():
    cfg = _make_config()
    run = BenchmarkRun(config=cfg, domain=Domain.FINANCE)
    assert run.domain == Domain.FINANCE
    assert run.num_queries == 0
    assert run.mean_latency_ms == 0.0


def test_benchmark_run_with_results():
    cfg = _make_config()
    results = [
        RetrievalResult("q1", ["d1"], [0.9], 10.0),
        RetrievalResult("q2", ["d2"], [0.8], 20.0),
    ]
    run = BenchmarkRun(config=cfg, domain=Domain.LEGAL, results=results)
    assert run.num_queries == 2
    assert run.mean_latency_ms == 15.0


# --- RetrievalBench stubs ---

def test_retrieval_bench_load_domain_stub():
    bench = RetrievalBench()
    docs = bench.load_domain(Domain.MEDICAL)
    assert isinstance(docs, list)


def test_retrieval_bench_run_config_stub():
    bench = RetrievalBench()
    cfg = _make_config()
    run = bench.run_config(cfg, Domain.TECHNICAL)
    assert isinstance(run, BenchmarkRun)
    assert run.domain == Domain.TECHNICAL
