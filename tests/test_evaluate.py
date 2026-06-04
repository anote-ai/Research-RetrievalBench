"""Tests for retrievalbench.evaluate."""

import math

import pytest

from retrievalbench.evaluate import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Shared fixtures
RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c"}


# --- recall_at_k ---

def test_recall_at_k_full_recall():
    assert recall_at_k(RETRIEVED, RELEVANT, k=5) == 1.0


def test_recall_at_k_partial():
    # only "a" is in top-2; "c" is at position 3
    result = recall_at_k(RETRIEVED, RELEVANT, k=2)
    assert result == pytest.approx(0.5)


def test_recall_at_k_zero():
    assert recall_at_k(["x", "y"], RELEVANT, k=2) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(RETRIEVED, set(), k=3) == 0.0


# --- precision_at_k ---

def test_precision_at_k_basic():
    # top-2: ["a", "b"] -> 1 hit
    assert precision_at_k(RETRIEVED, RELEVANT, k=2) == pytest.approx(0.5)


def test_precision_at_k_perfect():
    assert precision_at_k(["a", "c"], RELEVANT, k=2) == pytest.approx(1.0)


def test_precision_at_k_zero_k():
    assert precision_at_k(RETRIEVED, RELEVANT, k=0) == 0.0


# --- ndcg_at_k ---

def test_ndcg_at_k_perfect():
    # ideal order matches retrieved
    ideal = ["a", "c", "b"]
    assert ndcg_at_k(ideal, {"a", "c"}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_zero_k():
    assert ndcg_at_k(RETRIEVED, RELEVANT, k=0) == 0.0


def test_ndcg_at_k_partial():
    score = ndcg_at_k(RETRIEVED, RELEVANT, k=5)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_no_hits():
    assert ndcg_at_k(["x", "y"], RELEVANT, k=2) == 0.0


# --- mean_reciprocal_rank ---

def test_mrr_first_hit():
    assert mean_reciprocal_rank(["a", "b", "c"], RELEVANT) == pytest.approx(1.0)


def test_mrr_second_hit():
    assert mean_reciprocal_rank(["x", "a", "c"], RELEVANT) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert mean_reciprocal_rank(["x", "y", "z"], RELEVANT) == 0.0


def test_mrr_third_hit():
    assert mean_reciprocal_rank(["x", "y", "c"], RELEVANT) == pytest.approx(1 / 3)
