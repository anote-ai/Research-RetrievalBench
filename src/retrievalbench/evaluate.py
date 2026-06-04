"""Evaluation metrics for RetrievalBench."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from retrievalbench.core import BenchmarkRun


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    if not relevant_ids or k == 0:
        return 0.0

    def dcg(ids: list[str], limit: int) -> float:
        score = 0.0
        for rank, doc_id in enumerate(ids[:limit], start=1):
            if doc_id in relevant_ids:
                score += 1.0 / math.log2(rank + 1)
        return score

    ideal_ids = list(relevant_ids)[:k]
    idcg = dcg(ideal_ids, k)
    if idcg == 0.0:
        return 0.0
    return dcg(retrieved_ids, k) / idcg


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank (MRR) for a single query."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ablation_table(runs: list[BenchmarkRun], relevant_map: dict) -> "pd.DataFrame":
    """Build a summary DataFrame of recall@10 across all (config, domain) pairs.

    Args:
        runs: List of BenchmarkRun instances.
        relevant_map: Mapping of query_id -> set[str] of relevant doc IDs.

    Returns:
        pd.DataFrame with columns [domain, chunking_strategy, use_reranking,
        use_metadata, use_query_expansion, embedding_model, recall@10, ndcg@10, mrr].
    """
    import pandas as pd  # lazy import — not required at module level

    rows = []
    for run in runs:
        recalls, ndcgs, mrrs = [], [], []
        for result in run.results:
            relevant = relevant_map.get(result.query_id, set())
            recalls.append(recall_at_k(result.retrieved_ids, relevant, k=10))
            ndcgs.append(ndcg_at_k(result.retrieved_ids, relevant, k=10))
            mrrs.append(mean_reciprocal_rank(result.retrieved_ids, relevant))
        rows.append(
            {
                "domain": run.domain.value,
                "chunking_strategy": run.config.chunking_strategy,
                "use_reranking": run.config.use_reranking,
                "use_metadata": run.config.use_metadata,
                "use_query_expansion": run.config.use_query_expansion,
                "embedding_model": run.config.embedding_model,
                "recall@10": sum(recalls) / len(recalls) if recalls else float("nan"),
                "ndcg@10": sum(ndcgs) / len(ndcgs) if ndcgs else float("nan"),
                "mrr": sum(mrrs) / len(mrrs) if mrrs else float("nan"),
            }
        )
    return pd.DataFrame(rows)
