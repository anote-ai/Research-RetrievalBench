from __future__ import annotations
from retrievalbench.data import make_corpus, make_queries, make_retrieval_result
from retrievalbench.core import RetrievalResult


def test_make_corpus_length() -> None:
    corpus = make_corpus(n_docs=50)
    assert len(corpus) == 50


def test_make_corpus_keys() -> None:
    corpus = make_corpus(n_docs=10)
    for doc in corpus:
        assert "doc_id" in doc
        assert "text" in doc
        assert "domain" in doc


def test_make_queries_length() -> None:
    corpus = make_corpus(n_docs=100)
    queries, qrels = make_queries(n=15, corpus=corpus)
    assert len(queries) == 15
    assert len(qrels) == 15


def test_make_retrieval_result_type() -> None:
    corpus = make_corpus(n_docs=50)
    relevant = {corpus[0]["doc_id"], corpus[1]["doc_id"]}
    result = make_retrieval_result("q_test", corpus, relevant, recall=0.8)
    assert isinstance(result, RetrievalResult)
    assert result.query_id == "q_test"
