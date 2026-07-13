"""Retrievers: BM25 (sparse) and dense (cosine), both operating on chunks
and aggregating to doc level via max-pooling.

Both return a RetrievalOutput holding per-query ranked doc ids (length k_docs)
AND the full chunk-level ranking (top k_chunks chunk ids) so downstream
context-saturation analysis can recompute metrics at arbitrary k without
re-running retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass
class RetrievalOutput:
    """Result of retrieving over a chunked corpus for a set of queries.

    `ranked_docs[qid]`: top-k doc ids (aggregated from chunks).
    `ranked_chunks[qid]`: top-k_chunks chunk ids (for saturation / re-k).
    `doc_scores[qid]`: doc-level max-pooled score per ranked doc.
    `latency_ms`: mean per-query latency.
    """
    ranked_docs: dict[str, list[str]] = field(default_factory=dict)
    ranked_chunks: dict[str, list[str]] = field(default_factory=dict)
    doc_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: float = 0.0


def _aggregate_to_docs(
    chunk_ids: list[str],
    scores: np.ndarray,
    chunk_to_doc: dict[str, str],
    k_docs: int,
) -> tuple[list[str], dict[str, float]]:
    """Max-pool chunk scores to doc level, return top-k docs + scores."""
    doc_scores: dict[str, float] = {}
    for cid, s in zip(chunk_ids, scores):
        did = chunk_to_doc[cid]
        if s > doc_scores.get(did, -np.inf):
            doc_scores[did] = float(s)
    top = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)[:k_docs]
    return top, {d: doc_scores[d] for d in top}


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def retrieve_bm25(
    chunk_corpus: list[dict],
    queries: list[dict],
    k_chunks: int = 50,
    k_docs: int = 10,
) -> RetrievalOutput:
    from rank_bm25 import BM25Okapi

    tokenized = [c["text"].lower().split() for c in chunk_corpus]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [c["chunk_id"] for c in chunk_corpus]
    chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunk_corpus}

    out = RetrievalOutput()
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        scores = bm25.get_scores(q["text"].lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_chunks]
        top_chunk_ids = [chunk_ids[i] for i in top_idx]
        top_scores = scores[top_idx]
        top_docs, doc_sc = _aggregate_to_docs(top_chunk_ids, top_scores, chunk_to_doc, k_docs)
        out.ranked_docs[q["query_id"]] = top_docs
        out.ranked_chunks[q["query_id"]] = top_chunk_ids
        out.doc_scores[q["query_id"]] = doc_sc
        latencies.append((time.perf_counter() - t0) * 1000)
    out.latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return out


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------

def retrieve_dense(
    chunk_corpus: list[dict],
    queries: list[dict],
    embedder,  # Embedder instance
    k_chunks: int = 50,
    k_docs: int = 10,
) -> RetrievalOutput:
    chunk_ids = [c["chunk_id"] for c in chunk_corpus]
    chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunk_corpus}

    chunk_embs = embedder.embed([c["text"] for c in chunk_corpus])
    query_embs = embedder.embed([q["text"] for q in queries])
    # both already L2-normalized in embedder
    all_scores = np.dot(query_embs, chunk_embs.T)

    out = RetrievalOutput()
    latencies = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        row = all_scores[i]
        top_idx = np.argpartition(row, -k_chunks)[-k_chunks:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        top_chunk_ids = [chunk_ids[j] for j in top_idx]
        top_scores = row[top_idx]
        top_docs, doc_sc = _aggregate_to_docs(top_chunk_ids, top_scores, chunk_to_doc, k_docs)
        out.ranked_docs[q["query_id"]] = top_docs
        out.ranked_chunks[q["query_id"]] = top_chunk_ids
        out.doc_scores[q["query_id"]] = doc_sc
        latencies.append((time.perf_counter() - t0) * 1000)
    out.latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return out
