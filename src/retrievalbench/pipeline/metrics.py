"""Per-config evaluation + cross-config sensitivity indices (CSI/ESI).

Wraps the metric functions already in retrievalbench.evaluate so the pipeline
doesn't reimplement them, and adds:
  - evaluate_config(): aggregate metrics for one (domain, chunking, system) cell,
    INCLUDING the per-query ranked list (B5) so saturation/CSI can recompute.
  - csi_esi(): Chunking Sensitivity Index vs Embedding Sensitivity Index (B4).

CSI/ESI formalization (Claim 2: chunking matters more than embedding model):
  For a fixed domain, build a matrix M[chunker, embedder] = mean nDCG@10.
  total_var = variance of all cells.
  CSI = mean over embedders of var_chunking(embedder) / total_var
  ESI = mean over chunkers of var_embedder(chunker) / total_var
  Claim 2 <=> CSI > ESI across domains.
"""
from __future__ import annotations

import math
from collections import defaultdict


def _ndcg_at_k(retrieved_ids, relevant_ids, k):
    from retrievalbench.evaluate import ndcg_at_k
    return ndcg_at_k(list(retrieved_ids), set(relevant_ids), k)


def _recall_at_k(retrieved_ids, relevant_ids, k):
    from retrievalbench.evaluate import recall_at_k
    return recall_at_k(list(retrieved_ids), set(relevant_ids), k)


def _mrr(retrieved_ids, relevant_ids):
    from retrievalbench.evaluate import mean_reciprocal_rank
    return mean_reciprocal_rank(list(retrieved_ids), set(relevant_ids))


def _ap(retrieved_ids, relevant_ids):
    from retrievalbench.evaluate import average_precision
    return average_precision(list(retrieved_ids), set(relevant_ids))


def _la_ndcg(retrieved_ids, relevant_ids, latency_ms, k=10):
    from retrievalbench.evaluate import latency_adjusted_ndcg
    return latency_adjusted_ndcg(list(retrieved_ids), set(relevant_ids), latency_ms, k)


def _bootstrap_ci(values, confidence=0.95, n_resamples=1000, seed=42):
    from retrievalbench.evaluate import bootstrap_ci
    return bootstrap_ci(values, confidence, n_resamples, seed)


def evaluate_config(
    ranked_docs: dict[str, list[str]],
    qrels: dict[str, set[str]],
    latency_ms: float = 0.0,
    k: int = 10,
    ranked_chunks: dict[str, list[str]] | None = None,
) -> dict:
    """Aggregate metrics for one cell.

    `ranked_docs` (the top-k doc ids per query) is stored verbatim under
    _ranked_docs so downstream context-saturation can recompute recall/nDCG
    at arbitrary k without re-running retrieval (B5). `ranked_chunks` is the
    chunk-level top-k_chunks, kept for chunk-level analysis.
    """
    ndcgs, recalls, mrrs, aps, la_ndcgs = [], [], [], [], []
    for qid, ret in ranked_docs.items():
        rel = qrels.get(qid, set())
        ndcgs.append(_ndcg_at_k(ret, rel, k))
        recalls.append(_recall_at_k(ret, rel, k))
        mrrs.append(_mrr(ret, rel))
        aps.append(_ap(ret, rel))
        la_ndcgs.append(_la_ndcg(ret, rel, latency_ms, k))

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    ci = _bootstrap_ci(ndcgs) if len(ndcgs) >= 2 else (mean(ndcgs), mean(ndcgs))
    out = {
        "ndcg@10": round(mean(ndcgs), 4),
        "ndcg@10_ci_lower": round(ci[0], 4),
        "ndcg@10_ci_upper": round(ci[1], 4),
        "la_ndcg@10": round(mean(la_ndcgs), 4),
        "recall@10": round(mean(recalls), 4),
        "mrr": round(mean(mrrs), 4),
        "map": round(mean(aps), 4),
        "n_queries": len(ranked_docs),
        "latency_ms": round(latency_ms, 2),
        "_ndcg_per_query": ndcgs,
    }
    if ranked_chunks is not None:
        # Stored separately (large); run.py writes it into a side file or a
        # dedicated ranked_lists key to keep the summary JSON readable.
        out["_ranked_chunks"] = ranked_chunks
    out["_ranked_docs"] = ranked_docs
    return out


# ---------------------------------------------------------------------------
# CSI / ESI  (B4)
# ---------------------------------------------------------------------------

def _variance(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return sum((v - m) ** 2 for v in values) / (n - 1)


def csi_esi(records: list[dict], k: int = 10) -> dict:
    """Compute CSI vs ESI per domain from result records.

    Each record must have: domain, chunking, embedder (or system parsed to
    embedder), ndcg@10. Builds M[chunker, embedder] per domain and returns
    {domain: {csi, esi, claim2_holds (csi>esi), matrix}}.
    """
    # group by domain -> (chunker, embedder) -> ndcg
    by_domain: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for r in records:
        dom = r.get("domain", "unknown")
        chk = r["chunking"]
        emb = r.get("embedder", "openai-3-small")
        by_domain[dom][(chk, emb)] = r["ndcg@10"]

    out: dict[str, dict] = {}
    for dom, cell_map in by_domain.items():
        chunkers = sorted({c for c, _ in cell_map})
        embedders = sorted({e for _, e in cell_map})
        if len(chunkers) < 2 or len(embedders) < 2:
            out[dom] = {"csi": None, "esi": None, "claim2_holds": None,
                        "note": "need >=2 chunkers and >=2 embedders"}
            continue

        # M[chunker][embedder] = ndcg
        M = {c: {e: cell_map.get((c, e)) for e in embedders} for c in chunkers}
        all_vals = [v for row in M.values() for v in row.values() if v is not None]
        total_var = _variance(all_vals)
        if total_var == 0:
            out[dom] = {"csi": None, "esi": None, "claim2_holds": None,
                        "note": "zero total variance"}
            continue

        # var across chunkers, per embedder
        csi_terms = []
        for e in embedders:
            vals = [M[c][e] for c in chunkers if M[c][e] is not None]
            if len(vals) >= 2:
                csi_terms.append(_variance(vals))
        # var across embedders, per chunker
        esi_terms = []
        for c in chunkers:
            vals = [M[c][e] for e in embedders if M[c][e] is not None]
            if len(vals) >= 2:
                esi_terms.append(_variance(vals))

        csi = (sum(csi_terms) / len(csi_terms) / total_var) if csi_terms else 0.0
        esi = (sum(esi_terms) / len(esi_terms) / total_var) if esi_terms else 0.0
        out[dom] = {
            "csi": round(csi, 4),
            "esi": round(esi, 4),
            "claim2_holds": csi > esi,
            "n_chunkers": len(chunkers),
            "n_embedders": len(embedders),
        }
    return out
