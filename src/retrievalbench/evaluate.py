from __future__ import annotations
import math
from .core import BenchmarkRun, Domain


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
    ideal_ids = list(relevant_ids) + [x for x in retrieved_ids if x not in relevant_ids]
    ideal_dcg = dcg(ideal_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> float:
    """Average Precision (AP) for a single query.

    AP is the mean of precision@k values computed at each rank position
    where a relevant document is found.  Returns 0.0 if there are no
    relevant documents.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / (i + 1)
    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def mean_average_precision(
    results: list[tuple[list[str], set[str]]]
) -> float:
    """Mean Average Precision (MAP) over a list of (retrieved, relevant) pairs."""
    if not results:
        return 0.0
    return sum(
        average_precision(retrieved, relevant)
        for retrieved, relevant in results
    ) / len(results)


def r_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """R-Precision: precision at rank R, where R = |relevant_ids|.

    R-Precision evaluates the top-R retrieved documents, where R is the
    total number of known-relevant documents for the query.  If R is 0
    the metric is undefined and 0.0 is returned.
    """
    r = len(relevant_ids)
    if r == 0:
        return 0.0
    top_r = retrieved_ids[:r]
    hits = sum(1 for doc_id in top_r if doc_id in relevant_ids)
    return hits / r


def evaluate_run(
    run: BenchmarkRun, qrels: dict[str, set[str]], k: int = 10
) -> dict:
    """Compute aggregate metrics for a benchmark run."""
    recalls, precisions, ndcgs, mrrs, aps, r_precs = [], [], [], [], [], []
    for result in run.results:
        rel = qrels.get(result.query_id, set())
        recalls.append(recall_at_k(result.retrieved_ids, rel, k))
        precisions.append(precision_at_k(result.retrieved_ids, rel, k))
        ndcgs.append(ndcg_at_k(result.retrieved_ids, rel, k))
        mrrs.append(mean_reciprocal_rank(result.retrieved_ids, rel))
        aps.append(average_precision(result.retrieved_ids, rel))
        r_precs.append(r_precision(result.retrieved_ids, rel))

    def mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "config": run.config.name(),
        "domain": run.domain.value,
        "recall@k": mean(recalls),
        "precision@k": mean(precisions),
        "ndcg@k": mean(ndcgs),
        "mrr": mean(mrrs),
        "map": mean(aps),
        "r_precision": mean(r_precs),
        "n_queries": len(run.results),
    }


def compare_domains(
    runs: list[BenchmarkRun],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Compare metric averages across domains.

    Returns a dict mapping domain name to aggregate metric dict (mean over
    all runs in that domain).
    """
    from collections import defaultdict

    domain_metrics: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        metrics = evaluate_run(run, qrels, k)
        domain_metrics[run.domain.value].append(metrics)

    result: dict[str, dict[str, float]] = {}
    scalar_keys = ["recall@k", "precision@k", "ndcg@k", "mrr", "map", "r_precision"]
    for domain, metrics_list in domain_metrics.items():
        result[domain] = {
            key: sum(m[key] for m in metrics_list) / len(metrics_list)
            for key in scalar_keys
        }
    return result


def ablation_table(
    runs: list[BenchmarkRun], qrels: dict[str, set[str]], k: int = 10
):
    """Return a pandas DataFrame with one row per run, sorted by ndcg@k desc."""
    import pandas as pd

    rows = [evaluate_run(r, qrels, k) for r in runs]
    return pd.DataFrame(rows).sort_values("ndcg@k", ascending=False)
