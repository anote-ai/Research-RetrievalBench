from __future__ import annotations
import math
import statistics
import random
from .core import BenchmarkRun


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


def f1_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k."""
    prec = precision_at_k(retrieved_ids, relevant_ids, k)
    rec = recall_at_k(retrieved_ids, relevant_ids, k)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


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
    """Average Precision (AP) for a single query."""
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
    """R-Precision: precision at rank R, where R = |relevant_ids|."""
    r = len(relevant_ids)
    if r == 0:
        return 0.0
    top_r = retrieved_ids[:r]
    hits = sum(1 for doc_id in top_r if doc_id in relevant_ids)
    return hits / r


def latency_adjusted_ndcg(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    latency_ms: float,
    k: int = 10,
    latency_budget_ms: float = 500.0,
) -> float:
    """NDCG@k penalised by retrieval latency relative to a budget.

    score = ndcg * exp(-max(0, latency - budget) / budget)
    """
    base = ndcg_at_k(retrieved_ids, relevant_ids, k)
    overage = max(0.0, latency_ms - latency_budget_ms)
    penalty = math.exp(-overage / max(latency_budget_ms, 1e-9))
    return base * penalty


def query_difficulty_tier(relevant_ids: set[str], corpus_size: int) -> str:
    """Classify query difficulty by relevance density.

    Tiers: 'easy' (>=5 % relevant), 'medium' (1-5 %), 'hard' (<1 %).
    """
    if corpus_size <= 0:
        return "hard"
    density = len(relevant_ids) / corpus_size
    if density >= 0.05:
        return "easy"
    if density >= 0.01:
        return "medium"
    return "hard"


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean (percentile method).

    Uses 1000 resamples by default, as required for conference submissions.
    Returns (lower, upper) bounds.
    """
    n = len(values)
    if n < 2:
        m = values[0] if values else 0.0
        return (m, m)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1 - confidence
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return (boot_means[lo_idx], boot_means[hi_idx])


def score_variance(values: list[float]) -> float:
    """Sample variance of per-query metric scores."""
    return statistics.variance(values) if len(values) >= 2 else 0.0


def permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired permutation test comparing two configs' per-query scores.

    More appropriate than t-test for IR metrics (non-normal distributions).
    Returns p-value (two-tailed) and significance at alpha=0.05.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("score lists must have equal length")
    n = len(scores_a)
    if n < 2:
        return {"observed_diff": 0.0, "p_value": 1.0, "significant": False, "n": n}
    observed_diff = abs(
        sum(scores_a) / n - sum(scores_b) / n
    )
    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(n_resamples):
        perm_a, perm_b = [], []
        for a, b in zip(scores_a, scores_b):
            if rng.random() < 0.5:
                perm_a.append(a)
                perm_b.append(b)
            else:
                perm_a.append(b)
                perm_b.append(a)
        diff = abs(sum(perm_a) / n - sum(perm_b) / n)
        if diff >= observed_diff:
            count_extreme += 1
    p_value = count_extreme / n_resamples
    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n": n,
    }


def bonferroni_correct(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction to a list of p-values."""
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Cohen's d effect size between two sets of per-query scores."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        return 0.0
    mean_a = statistics.mean(scores_a)
    mean_b = statistics.mean(scores_b)
    pooled_std = math.sqrt(
        (statistics.variance(scores_a) + statistics.variance(scores_b)) / 2
    )
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Kendall's tau rank correlation between two config orderings.

    Measures how stable the ranking of configs is across two metrics or domains.
    Returns value in [-1, 1]; 1 = identical ordering, -1 = reversed.
    """
    n = len(ranking_a)
    if n < 2:
        return 1.0
    pos_a = {item: i for i, item in enumerate(ranking_a)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = pos_a[ranking_a[i]] - pos_a[ranking_a[j]]
            b_i = ranking_b.index(ranking_a[i]) if ranking_a[i] in ranking_b else -1
            b_j = ranking_b.index(ranking_a[j]) if ranking_a[j] in ranking_b else -1
            if b_i == -1 or b_j == -1:
                continue
            b_order = b_i - b_j
            if a_order * b_order > 0:
                concordant += 1
            elif a_order * b_order < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


def multi_seed_variance(
    score_lists: list[list[float]],
) -> dict:
    """Compute mean ± std across multiple random seeds.

    Args:
        score_lists: One list of per-query scores per seed (at least 3 seeds).
    Returns:
        dict with mean, std, and per-seed means.
    """
    if not score_lists:
        return {"mean": 0.0, "std": 0.0, "seed_means": []}
    seed_means = [sum(s) / len(s) for s in score_lists if s]
    mean = sum(seed_means) / len(seed_means)
    std = statistics.stdev(seed_means) if len(seed_means) >= 2 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "seed_means": seed_means}


def evaluate_run(
    run: BenchmarkRun, qrels: dict[str, set[str]], k: int = 10
) -> dict:
    """Compute aggregate metrics for a benchmark run."""
    recalls, precisions, f1s, ndcgs, mrrs, aps, r_precs = [], [], [], [], [], [], []
    for result in run.results:
        rel = qrels.get(result.query_id, set())
        recalls.append(recall_at_k(result.retrieved_ids, rel, k))
        precisions.append(precision_at_k(result.retrieved_ids, rel, k))
        f1s.append(f1_at_k(result.retrieved_ids, rel, k))
        ndcgs.append(ndcg_at_k(result.retrieved_ids, rel, k))
        mrrs.append(mean_reciprocal_rank(result.retrieved_ids, rel))
        aps.append(average_precision(result.retrieved_ids, rel))
        r_precs.append(r_precision(result.retrieved_ids, rel))

    def mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    ndcg_ci = bootstrap_ci(ndcgs) if ndcgs else (0.0, 0.0)
    mrr_ci = bootstrap_ci(mrrs) if mrrs else (0.0, 0.0)

    return {
        "config": run.config.name(),
        "domain": run.domain.value,
        "recall@k": mean(recalls),
        "precision@k": mean(precisions),
        "f1@k": mean(f1s),
        "ndcg@k": mean(ndcgs),
        "ndcg@k_ci_lower": ndcg_ci[0],
        "ndcg@k_ci_upper": ndcg_ci[1],
        "ndcg@k_variance": score_variance(ndcgs),
        "mrr": mean(mrrs),
        "mrr_ci_lower": mrr_ci[0],
        "mrr_ci_upper": mrr_ci[1],
        "mrr_variance": score_variance(mrrs),
        "map": mean(aps),
        "r_precision": mean(r_precs),
        "n_queries": len(run.results),
    }


def compare_domains(
    runs: list[BenchmarkRun],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Compare metric averages across domains."""
    from collections import defaultdict

    domain_metrics: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        metrics = evaluate_run(run, qrels, k)
        domain_metrics[run.domain.value].append(metrics)

    result: dict[str, dict[str, float]] = {}
    scalar_keys = ["recall@k", "precision@k", "f1@k", "ndcg@k", "mrr", "map", "r_precision"]
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
