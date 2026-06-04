from __future__ import annotations
import math
from .core import RetrievalResult, BenchmarkRun, Domain


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs retrieved in top-k."""
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for r in retrieved_k if r in relevant_ids)
    return hits / k


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""

    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 if ids[i] in relevant_ids else 0.0) / math.log2(i + 2)
            for i in range(min(k, len(ids)))
        )

    actual_dcg = dcg(retrieved_ids)
    # Ideal ranking: relevant docs first
    ideal_ids = list(relevant_ids) + [x for x in retrieved_ids if x not in relevant_ids]
    ideal_dcg = dcg(ideal_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(
    run: BenchmarkRun, qrels: dict[str, set[str]], k: int = 10
) -> dict:
    """Compute aggregate metrics for a benchmark run.

    Args:
        run: BenchmarkRun with retrieval results.
        qrels: Mapping query_id -> set of relevant doc ids.
        k: Cutoff for rank-based metrics.

    Returns:
        Dict with recall@k, precision@k, ndcg@k, mrr, n_queries.
    """
    recalls, precisions, ndcgs, mrrs = [], [], [], []
    for result in run.results:
        rel = qrels.get(result.query_id, set())
        recalls.append(recall_at_k(result.retrieved_ids, rel, k))
        precisions.append(precision_at_k(result.retrieved_ids, rel, k))
        ndcgs.append(ndcg_at_k(result.retrieved_ids, rel, k))
        mrrs.append(mean_reciprocal_rank(result.retrieved_ids, rel))

    def mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "config": run.config.name(),
        "domain": run.domain.value,
        "recall@k": mean(recalls),
        "precision@k": mean(precisions),
        "ndcg@k": mean(ndcgs),
        "mrr": mean(mrrs),
        "n_queries": len(run.results),
    }


def ablation_table(
    runs: list[BenchmarkRun], qrels: dict[str, set[str]], k: int = 10
):
    """Return a pandas DataFrame with one row per run, sorted by ndcg@k desc."""
    import pandas as pd

    rows = [evaluate_run(r, qrels, k) for r in runs]
    return pd.DataFrame(rows).sort_values("ndcg@k", ascending=False)
