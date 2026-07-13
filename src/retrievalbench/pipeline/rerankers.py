"""Cross-encoder rerankers.

Multiple rerankers matter: A1 found that cross-encoder/ms-marco-MiniLM-L-6-v2
is a no-op on quora (question-duplicate task, OOD for a passage-trained model
-> near-constant scores + Python stable sort = unchanged order). A
domain-appropriate reranker (quora-distilroberta) restores signal.

The rerank() function now BREAKS TIES BY INPUT ORDER only when scores actually
differ; if all scores are identical (the no-op case) it records a flag so the
result JSON can surface the degenerate case instead of silently passing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import RerankerSpec


_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()


def _get_reranker(model_id: str):
    with _LOCK:
        if model_id not in _CACHE:
            from sentence_transformers import CrossEncoder
            _CACHE[model_id] = CrossEncoder(model_id)
        return _CACHE[model_id]


@dataclass
class RerankResult:
    ranked_docs: dict[str, list[str]]
    latency_ms: float
    # per query: True if reranker scores were (near-)constant -> no-op
    degenerate: dict[str, bool]


def rerank(
    retrieved: dict[str, list[str]],
    queries: list[dict],
    corpus: list[dict],
    spec: RerankerSpec,
    k: int = 10,
    degenerate_threshold: float = 1e-6,
) -> RerankResult:
    """Rerank top-k docs per query with a cross-encoder.

    `degenerate_threshold`: if the stdev of reranker scores for a query's
    candidates is below this, the query is flagged degenerate (no-op).
    """
    model = _get_reranker(spec.model)
    doc_text = {d["doc_id"]: d["text"] for d in corpus}

    ranked: dict[str, list[str]] = {}
    degenerate: dict[str, bool] = {}
    latencies = []
    for q in queries:
        qid = q["query_id"]
        doc_ids = retrieved.get(qid, [])
        if not doc_ids:
            ranked[qid] = []
            continue
        t0 = time.perf_counter()
        pairs = [[q["text"], doc_text.get(d, "")] for d in doc_ids]
        scores = np.asarray(model.predict(pairs), dtype="float64")
        degenerate[qid] = bool(scores.std() < degenerate_threshold)
        order = np.argsort(-scores)  # stable descending
        ranked[qid] = [doc_ids[i] for i in order[:k]]
        latencies.append((time.perf_counter() - t0) * 1000)
    return RerankResult(
        ranked_docs=ranked,
        latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        degenerate=degenerate,
    )


# ---------------------------------------------------------------------------
# Registry — domain-aware defaults + cross-reranker ablation pool.
# ---------------------------------------------------------------------------

RERANKER_REGISTRY: dict[str, RerankerSpec] = {
    "msmarco-minilm": RerankerSpec(
        name="msmarco-minilm",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        suited_domains=("scientific", "medical", "finance", "technical",
                        "encyclopedic", "web", "multi_hop"),
    ),
    "quora-distilroberta": RerankerSpec(
        name="quora-distilroberta",
        model="cross-encoder/quora-distilroberta-base",
        suited_domains=("community", "argumentation"),
    ),
    "bge-reranker": RerankerSpec(
        name="bge-reranker",
        model="BAAI/bge-reranker-base",
        suited_domains=None,  # general, strong all-rounder
    ),
}


def default_reranker_for(domain: str) -> str:
    """Pick the registry key of the reranker best suited to a domain."""
    for key, spec in RERANKER_REGISTRY.items():
        if spec.suited_domains and domain in spec.suited_domains:
            return key
    return "bge-reranker"  # general fallback
