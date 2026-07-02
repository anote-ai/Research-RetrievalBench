"""Real retrieval pipeline: BM25 + Dense (BGE-small) + Hybrid RRF on BEIR scifact.

Usage:
    python scripts/run_real_retrieval.py

Downloads scifact from HuggingFace (~5K docs), runs BM25, dense, and hybrid
retrieval, evaluates with RetrievalBench metrics, saves results to results/scifact/.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "scifact")


# ---------------------------------------------------------------------------
# 1. Load BEIR scifact dataset
# ---------------------------------------------------------------------------

def load_scifact() -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    from datasets import load_dataset

    print("Loading scifact corpus from HuggingFace...")
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    corpus = [
        {"doc_id": str(row["_id"]), "text": row["title"] + " " + row["text"]}
        for row in corpus_ds
    ]
    queries = [
        {"query_id": str(row["_id"]), "text": row["text"]}
        for row in queries_ds
    ]
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in qrels_ds:
        if int(row["score"]) > 0:
            qrels[str(row["query-id"])].add(str(row["corpus-id"]))
    qrels = dict(qrels)
    queries = [q for q in queries if q["query_id"] in qrels]

    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)}")
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# 2. BM25
# ---------------------------------------------------------------------------

def build_bm25(corpus: list[dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [doc["text"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


def retrieve_bm25(
    bm25, corpus: list[dict], queries: list[dict], k: int = 10
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
    """Returns (top-k ids, raw scores dict, avg latency ms)."""
    doc_ids = [doc["doc_id"] for doc in corpus]
    results: dict[str, list[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    latencies: list[float] = []

    for query in queries:
        t0 = time.perf_counter()
        scores = bm25.get_scores(query["text"].lower().split())
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        elapsed = (time.perf_counter() - t0) * 1000
        results[query["query_id"]] = [doc_ids[i] for i in top_k_idx]
        raw_scores[query["query_id"]] = {doc_ids[i]: float(scores[i]) for i in top_k_idx}
        latencies.append(elapsed)

    return results, raw_scores, sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# 3. Dense (BGE-small-en)
# ---------------------------------------------------------------------------

def build_dense_index(corpus: list[dict], model_name: str = "BAAI/bge-small-en-v1.5"):
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"  Encoding {len(corpus)} docs with {model_name}...")
    model = SentenceTransformer(model_name)
    t0 = time.perf_counter()
    embeddings = model.encode(
        [doc["text"] for doc in corpus],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True,
    )
    print(f"  Encoded in {time.perf_counter() - t0:.1f}s")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return model, index


def retrieve_dense(
    model, index, corpus: list[dict], queries: list[dict], k: int = 10
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
    doc_ids = [doc["doc_id"] for doc in corpus]
    results: dict[str, list[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    latencies: list[float] = []

    for query in queries:
        t0 = time.perf_counter()
        q_emb = model.encode([query["text"]], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_emb, k)
        elapsed = (time.perf_counter() - t0) * 1000
        results[query["query_id"]] = [doc_ids[i] for i in indices[0]]
        raw_scores[query["query_id"]] = {doc_ids[i]: float(scores[0][j]) for j, i in enumerate(indices[0])}
        latencies.append(elapsed)

    return results, raw_scores, sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# 4. Hybrid RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def retrieve_hybrid_rrf(
    bm25_scores: dict[str, dict[str, float]],
    dense_scores: dict[str, dict[str, float]],
    queries: list[dict],
    k: int = 10,
    rrf_k: int = 60,
) -> tuple[dict[str, list[str]], float]:
    """Fuse BM25 and dense rankings using Reciprocal Rank Fusion."""
    results: dict[str, list[str]] = {}
    latencies: list[float] = []

    for query in queries:
        qid = query["query_id"]
        t0 = time.perf_counter()

        # Rank each list
        bm25_ranked = sorted(bm25_scores.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        dense_ranked = sorted(dense_scores.get(qid, {}).items(), key=lambda x: x[1], reverse=True)

        # RRF score = sum of 1/(rrf_k + rank) across both lists
        rrf: dict[str, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(bm25_ranked):
            rrf[doc_id] += 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(dense_ranked):
            rrf[doc_id] += 1.0 / (rrf_k + rank + 1)

        top_k = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        elapsed = (time.perf_counter() - t0) * 1000
        results[qid] = [doc_id for doc_id, _ in top_k]
        latencies.append(elapsed)

    return results, sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    retrieved: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict:
    from retrievalbench.evaluate import (
        ndcg_at_k, recall_at_k, mean_reciprocal_rank,
        average_precision, bootstrap_ci, permutation_test,
    )

    ndcgs, recalls, mrrs, aps = [], [], [], []
    for qid, ret_ids in retrieved.items():
        rel = qrels.get(qid, set())
        ndcgs.append(ndcg_at_k(ret_ids, rel, k))
        recalls.append(recall_at_k(ret_ids, rel, k))
        mrrs.append(mean_reciprocal_rank(ret_ids, rel))
        aps.append(average_precision(ret_ids, rel))

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    ci = bootstrap_ci(ndcgs)

    return {
        "n_queries": len(retrieved),
        f"ndcg@{k}": round(mean(ndcgs), 4),
        f"ndcg@{k}_ci_lower": round(ci[0], 4),
        f"ndcg@{k}_ci_upper": round(ci[1], 4),
        f"recall@{k}": round(mean(recalls), 4),
        "mrr": round(mean(mrrs), 4),
        "map": round(mean(aps), 4),
        "_ndcg_per_query": ndcgs,  # kept for significance tests
    }


# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------

def save_results(all_results: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Strip internal per-query lists before saving
    saveable = []
    for r in all_results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        saveable.append(row)

    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": "BeIR/scifact",
            "date": datetime.now().isoformat(timespec="seconds"),
            "results": saveable,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    from retrievalbench.evaluate import permutation_test, bonferroni_correct

    corpus, queries, qrels = load_scifact()

    # BM25
    print("\nBuilding BM25 index...")
    bm25 = build_bm25(corpus)
    print("Running BM25 retrieval...")
    bm25_results, bm25_raw, bm25_latency = retrieve_bm25(bm25, corpus, queries)
    bm25_metrics = evaluate(bm25_results, qrels)

    # Dense BGE-small
    print("\nBuilding dense index (BGE-small-en)...")
    model, index = build_dense_index(corpus)
    print("Running dense retrieval...")
    dense_results, dense_raw, dense_latency = retrieve_dense(model, index, corpus, queries)
    dense_metrics = evaluate(dense_results, qrels)

    # Hybrid RRF
    print("\nRunning Hybrid RRF...")
    # Use top-100 from each for fusion
    _, bm25_raw100, _ = retrieve_bm25(bm25, corpus, queries, k=100)
    _, dense_raw100, _ = retrieve_dense(model, index, corpus, queries, k=100)
    hybrid_results, hybrid_latency = retrieve_hybrid_rrf(bm25_raw100, dense_raw100, queries)
    hybrid_metrics = evaluate(hybrid_results, qrels)

    # Significance tests (BM25 as baseline)
    sig_dense = permutation_test(dense_metrics["_ndcg_per_query"], bm25_metrics["_ndcg_per_query"])
    sig_hybrid = permutation_test(hybrid_metrics["_ndcg_per_query"], bm25_metrics["_ndcg_per_query"])
    corrected = bonferroni_correct([sig_dense["p_value"], sig_hybrid["p_value"]])

    # Print table
    print("\n" + "=" * 78)
    print("REAL RETRIEVAL RESULTS — BEIR scifact (300 queries, 5183 docs)")
    print("=" * 78)
    header = f"{'System':<20} {'nDCG@10':>8} {'95% CI':>22} {'Recall@10':>10} {'MRR':>7} {'MAP':>7} {'Lat(ms)':>9}"
    print(header)
    print("-" * 78)

    def row(name, m, lat, sig_marker=""):
        return (
            f"{name+sig_marker:<20} "
            f"{m['ndcg@10']:>8.4f} "
            f"[{m['ndcg@10_ci_lower']:.4f},{m['ndcg@10_ci_upper']:.4f}] "
            f"{m['recall@10']:>10.4f} "
            f"{m['mrr']:>7.4f} "
            f"{m['map']:>7.4f} "
            f"{lat:>9.1f}"
        )

    print(row("BM25", bm25_metrics, bm25_latency))
    dense_sig = "†" if corrected[0] < 0.05 else ""
    hybrid_sig = "†" if corrected[1] < 0.05 else ""
    print(row("Dense-BGE-small", dense_metrics, dense_latency, dense_sig))
    print(row("Hybrid-RRF", hybrid_metrics, hybrid_latency, hybrid_sig))

    print("\n† = significant vs BM25 (permutation test, Bonferroni corrected p<0.05)")
    print(f"\nSignificance vs BM25:")
    print(f"  Dense:  p={corrected[0]:.4f} (corrected), observed_diff={sig_dense['observed_diff']:.4f}")
    print(f"  Hybrid: p={corrected[1]:.4f} (corrected), observed_diff={sig_hybrid['observed_diff']:.4f}")

    print(f"\nnDCG@10 deltas vs BM25:")
    print(f"  Dense  : {dense_metrics['ndcg@10'] - bm25_metrics['ndcg@10']:+.4f}")
    print(f"  Hybrid : {hybrid_metrics['ndcg@10'] - bm25_metrics['ndcg@10']:+.4f}")
    print(f"\nLatency (ms): BM25={bm25_latency:.1f} | Dense={dense_latency:.1f} | Hybrid={hybrid_latency:.1f}")

    # Save
    records = [
        {"system": "BM25", **{k: v for k, v in bm25_metrics.items() if not k.startswith("_")}, "latency_ms": round(bm25_latency, 2)},
        {"system": "Dense-BGE-small", **{k: v for k, v in dense_metrics.items() if not k.startswith("_")}, "latency_ms": round(dense_latency, 2)},
        {"system": "Hybrid-RRF", **{k: v for k, v in hybrid_metrics.items() if not k.startswith("_")}, "latency_ms": round(hybrid_latency, 2)},
    ]
    save_results(records, RESULTS_DIR)


if __name__ == "__main__":
    main()
