"""Experiment executor: run one ExperimentConfig cell end-to-end.

For each cell it executes BOTH bm25 and dense retrievers (dense uses the
cell's embedder), then the reranker arm if spec.reranker is set. Results are
written to results/{dataset}/ with per-query ranked lists so saturation and
CSI/ESI can be recomputed post-hoc without re-running retrieval (B5).

This is the engine scripts/run_experiment.py drives over an ExperimentGrid.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from .config import ExperimentConfig, DATASET_DOMAINS
from .loaders import load_dataset
from .chunkers import build_chunk_corpus
from .retrievers import make_embedder, retrieve_bm25, retrieve_dense
from .rerankers import rerank, default_reranker_for, RERANKER_REGISTRY
from .metrics import evaluate_config


def results_dir(dataset: str) -> str:
    here = os.path.dirname(__file__)
    # src/retrievalbench/pipeline/run.py -> repo root / results (3 levels up)
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "results", dataset)


def _system_label(retriever: str, embedder_name: str, reranker_name: str | None) -> str:
    if retriever == "bm25":
        base = "BM25"
    else:
        base = f"Dense-{embedder_name}"
    if reranker_name:
        return f"{base}+Rerank-{reranker_name}"
    return base


def run_cell(config: ExperimentConfig, corpus, queries, qrels,
             reranker_spec=None) -> list[dict]:
    """Run bm25 + dense (+rerank) for one cell. Returns result records.

    `reranker_spec`: RerankerSpec or None. If None, only no-rerank arms run.
    """
    domain = config.domain
    embedder = make_embedder(config.embedder)
    embedder.tokens_used = 0  # reset per cell

    # Build chunk corpus (semantic chunking needs the embedder)
    chunk_corpus = build_chunk_corpus(corpus, config.chunker, embedder=embedder)

    records: list[dict] = []

    # --- BM25 ---
    bm25_out = retrieve_bm25(chunk_corpus, queries, config.k_chunks, config.k_docs)
    bm25_m = evaluate_config(bm25_out.ranked_docs, qrels, bm25_out.latency_ms,
                             k=config.k_docs, ranked_chunks=bm25_out.ranked_chunks)
    records.append(_record(domain, config, "bm25", None, bm25_m,
                           embedder.tokens_used, config.embedder))

    if reranker_spec is not None:
        rr_bm25 = rerank(bm25_out.ranked_docs, queries, corpus, reranker_spec, k=config.k_docs)
        lat = bm25_out.latency_ms + rr_bm25.latency_ms
        m = evaluate_config(rr_bm25.ranked_docs, qrels, lat, k=config.k_docs,
                            ranked_chunks=bm25_out.ranked_chunks)
        rec = _record(domain, config, "bm25", reranker_spec.name, m,
                      embedder.tokens_used, config.embedder)
        rec["rerank_degenerate_rate"] = _degen_rate(rr_bm25)
        records.append(rec)

    # --- Dense ---
    dense_out = retrieve_dense(chunk_corpus, queries, embedder,
                               config.k_chunks, config.k_docs)
    dense_m = evaluate_config(dense_out.ranked_docs, qrels, dense_out.latency_ms,
                              k=config.k_docs, ranked_chunks=dense_out.ranked_chunks)
    tokens = embedder.tokens_used
    records.append(_record(domain, config, "dense", None, dense_m,
                           tokens, config.embedder))

    if reranker_spec is not None:
        rr_dense = rerank(dense_out.ranked_docs, queries, corpus, reranker_spec, k=config.k_docs)
        lat = dense_out.latency_ms + rr_dense.latency_ms
        m = evaluate_config(rr_dense.ranked_docs, qrels, lat, k=config.k_docs,
                            ranked_chunks=dense_out.ranked_chunks)
        rec = _record(domain, config, "dense", reranker_spec.name, m,
                      tokens, config.embedder)
        rec["rerank_degenerate_rate"] = _degen_rate(rr_dense)
        records.append(rec)

    return records


def _degen_rate(rr) -> float:
    if not rr.degenerate:
        return 0.0
    return round(sum(rr.degenerate.values()) / len(rr.degenerate), 4)


def _record(domain, config: ExperimentConfig, retriever: str,
            reranker_name: str | None, metrics: dict, tokens: int,
            embedder_spec) -> dict:
    label = _system_label(retriever, embedder_spec.name, reranker_name)
    cost = tokens * (embedder_spec.cost_per_1m_tokens / 1_000_000)
    rec = {
        "domain": domain,
        "dataset": config.dataset,
        "chunking": config.chunker.name,
        "chunker_kind": config.chunker.kind,
        "embedder": embedder_spec.name,
        "embedder_model": embedder_spec.model,
        "system": label,
        "retriever": retriever,
        "reranker": reranker_name,
        **{k: v for k, v in metrics.items() if not k.startswith("_")},
        "embed_tokens": tokens,
        "embed_cost_usd": round(cost, 6),
    }
    # stash per-query nDCG + ranked lists separately (not in summary record)
    rec["_ndcg_per_query"] = metrics.get("_ndcg_per_query")
    rec["_ranked_chunks"] = metrics.get("_ranked_chunks")
    rec["_ranked_docs"] = metrics.get("_ranked_docs")
    return rec


def save_results(dataset: str, domain: str, records: list[dict],
                 n_docs: int, n_queries: int, total_tokens: int,
                 total_cost: float, position_bias: dict | None = None) -> str:
    """Persist results. Ranked lists go to a side file to keep summary small.

    Stores BOTH ranked_docs (doc-level top-k, for saturation/CSI at doc level)
    and ranked_chunks (chunk-level top-k_chunks, for chunk-level analysis).
    """
    out_dir = results_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Split: summary records (no _ keys) + ranked lists sidecar
    summary_records = []
    lists_payload = {}
    for r in records:
        rc = dict(r)
        key = r["system"] + "/" + r["chunking"]
        lists_payload[key] = {
            "ranked_chunks": rc.pop("_ranked_chunks", None),
            "ranked_docs": rc.pop("_ranked_docs", None),
        }
        rc.pop("_ndcg_per_query", None)
        summary_records.append(rc)

    summary_path = os.path.join(out_dir, "chunking_results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "domain": domain,
            "dataset": dataset,
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_docs": n_docs,
            "n_queries": n_queries,
            "total_embed_tokens": total_tokens,
            "total_embed_cost_usd": round(total_cost, 6),
            "position_bias": position_bias or {},
            "results": summary_records,
        }, f, indent=2)

    lists_path = os.path.join(out_dir, "ranked_lists.json")
    with open(lists_path, "w") as f:
        json.dump(lists_payload, f)  # compact; large file

    return summary_path
