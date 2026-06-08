from __future__ import annotations
import pytest
from retrievalbench.data import make_corpus, make_queries, make_retrieval_result
from retrievalbench.core import RetrievalResult, Domain


def test_make_corpus_length() -> None:
    corpus = make_corpus(n_docs=50)
    assert len(corpus) == 50


def test_make_corpus_keys() -> None:
    corpus = make_corpus(n_docs=10)
    for doc in corpus:
        assert "doc_id" in doc
        assert "text" in doc
        assert "domain" in doc


def test_make_corpus_can_target_domains() -> None:
    corpus = make_corpus(n_docs=20, domains=[Domain.LEGAL, Domain.MEDICAL])
    assert {doc["domain"] for doc in corpus} <= {Domain.LEGAL.value, Domain.MEDICAL.value}


def test_make_queries_length() -> None:
    corpus = make_corpus(n_docs=100)
    queries, qrels = make_queries(n=15, corpus=corpus)
    assert len(queries) == 15
    assert len(qrels) == 15


def test_make_queries_can_target_domain() -> None:
    corpus = make_corpus(n_docs=30, domains=[Domain.TECHNICAL], seed=0)
    queries, qrels = make_queries(
        n=8,
        corpus=corpus,
        domain=Domain.TECHNICAL,
        query_id_prefix="technical_q",
    )
    doc_domains = {doc["doc_id"]: doc["domain"] for doc in corpus}

    assert all(query["domain"] == Domain.TECHNICAL.value for query in queries)
    assert all(query["query_id"].startswith("technical_q_") for query in queries)
    for relevant_ids in qrels.values():
        assert all(doc_domains[doc_id] == Domain.TECHNICAL.value for doc_id in relevant_ids)


def test_make_queries_rejects_missing_domain() -> None:
    corpus = make_corpus(n_docs=10, domains=[Domain.FINANCE], seed=0)
    with pytest.raises(ValueError):
        make_queries(n=1, corpus=corpus, domain=Domain.LEGAL)


def test_make_retrieval_result_type() -> None:
    corpus = make_corpus(n_docs=50)
    relevant = {corpus[0]["doc_id"], corpus[1]["doc_id"]}
    result = make_retrieval_result("q_test", corpus, relevant, recall=0.8)
    assert isinstance(result, RetrievalResult)
    assert result.query_id == "q_test"
