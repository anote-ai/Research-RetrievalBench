"""Chunking + Reranking pipeline on a BEIR dataset.

Tests 4 chunking strategies × 2 retrieval systems × reranking, evaluates with
RetrievalBench metrics, saves per-domain results.

Usage:
    python scripts/run_chunking_pipeline.py --dataset scifact
    python scripts/run_chunking_pipeline.py --dataset nfcorpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# domain label for each supported BEIR dataset, used for cross-domain reporting
DATASET_DOMAINS = {
    "scifact": "scientific",
    "nfcorpus": "medical",
    "fiqa": "finance",
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"  # $0.02/1M tokens


def results_dir(dataset: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "results", dataset)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_beir_dataset(dataset: str, domain: str) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    from datasets import load_dataset

    print(f"[{domain.upper()}] Loading {dataset} from HuggingFace...")
    corpus_ds = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{dataset}-qrels", split="test")

    corpus = [
        {"doc_id": str(r["_id"]), "text": (r["title"] + " " + r["text"]).strip()}
        for r in corpus_ds
    ]
    queries = [{"query_id": str(r["_id"]), "text": r["text"]} for r in queries_ds]

    qrels: dict[str, set[str]] = defaultdict(set)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    qrels = dict(qrels)
    queries = [q for q in queries if q["query_id"] in qrels]

    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)}")
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# 2. Chunking strategies
# ---------------------------------------------------------------------------

def chunk_fixed(doc: dict, max_tokens: int = 512, overlap: int = 50) -> list[dict]:
    """Split by whitespace tokens with overlap."""
    words = doc["text"].split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, max(1, len(words)), step):
        chunk_words = words[i: i + max_tokens]
        if chunk_words:
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_c{i}",
                "doc_id": doc["doc_id"],
                "text": " ".join(chunk_words),
            })
    return chunks


def chunk_sentence(doc: dict, window: int = 3) -> list[dict]:
    """Group sentences into windows of `window` sentences per chunk."""
    import nltk
    sentences = [s.strip() for s in nltk.sent_tokenize(doc["text"]) if s.strip()]
    chunks = []
    for i in range(0, max(1, len(sentences)), window):
        text = " ".join(sentences[i: i + window])
        chunks.append({"chunk_id": f"{doc['doc_id']}_s{i}", "doc_id": doc["doc_id"], "text": text})
    return chunks


def chunk_recursive(doc: dict, max_tokens: int = 512) -> list[dict]:
    """Split on paragraphs first, then sentences if still too long."""
    import nltk
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc["text"]) if p.strip()]
    if not paragraphs:
        paragraphs = [doc["text"]]

    chunks = []
    idx = 0
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_tokens:
            chunks.append({"chunk_id": f"{doc['doc_id']}_r{idx}", "doc_id": doc["doc_id"], "text": para})
            idx += 1
        else:
            for sent in nltk.sent_tokenize(para):
                chunks.append({"chunk_id": f"{doc['doc_id']}_r{idx}", "doc_id": doc["doc_id"], "text": sent.strip()})
                idx += 1
    return chunks


def openai_embed(texts: list[str], batch_size: int = 500) -> list[list[float]]:
    """Embed texts using OpenAI text-embedding-3-small in batches.
    Empty strings are replaced with a single space to avoid API errors.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    # Replace empty strings — OpenAI rejects them
    safe_texts = [t if t.strip() else " " for t in texts]
    all_embeddings = []
    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i: i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
        batch_embs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embs)
    return all_embeddings


def build_chunk_corpus(
    corpus: list[dict],
    strategy: str,
) -> list[dict]:
    import nltk
    import numpy as np

    if strategy != "semantic":
        chunks = []
        for doc in corpus:
            if strategy == "fixed_512":
                chunks.extend(chunk_fixed(doc))
            elif strategy == "sentence":
                chunks.extend(chunk_sentence(doc))
            elif strategy == "recursive":
                chunks.extend(chunk_recursive(doc))
        return chunks

    # Semantic: collect ALL sentences from all docs, embed in one batch, then merge
    print("    Collecting all sentences...")
    doc_sentences: list[tuple[str, list[str]]] = []  # (doc_id, [sentences])
    all_sentences: list[str] = []
    offsets: list[int] = []  # start index of each doc's sentences in all_sentences

    for doc in corpus:
        sents = [s.strip() for s in nltk.sent_tokenize(doc["text"]) if s.strip()]
        if not sents:
            sents = [doc["text"]]
        offsets.append(len(all_sentences))
        all_sentences.extend(sents)
        doc_sentences.append((doc["doc_id"], sents))

    print(f"    Embedding {len(all_sentences)} sentences in one batch via OpenAI...")
    all_embs = np.array(openai_embed(all_sentences), dtype="float32")
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_embs /= np.maximum(norms, 1e-9)

    # Merge adjacent similar sentences per doc
    threshold = 0.75
    chunks = []
    for i, (doc_id, sents) in enumerate(doc_sentences):
        start = offsets[i]
        embs = all_embs[start: start + len(sents)]

        if len(sents) == 1:
            chunks.append({"chunk_id": f"{doc_id}_sem0", "doc_id": doc_id, "text": sents[0]})
            continue

        groups, current = [[sents[0]]], 0
        for j in range(1, len(sents)):
            sim = float(np.dot(embs[j - 1], embs[j]))
            if sim >= threshold:
                groups[current].append(sents[j])
            else:
                groups.append([sents[j]])
                current += 1

        for k, g in enumerate(groups):
            chunks.append({"chunk_id": f"{doc_id}_sem{k}", "doc_id": doc_id, "text": " ".join(g)})

    return chunks


# ---------------------------------------------------------------------------
# 3. BM25 retrieval on chunks → map back to doc level
# ---------------------------------------------------------------------------

def retrieve_bm25_chunks(
    chunk_corpus: list[dict],
    queries: list[dict],
    k_chunks: int = 50,
    k_docs: int = 10,
) -> tuple[dict[str, list[str]], float]:
    from rank_bm25 import BM25Okapi

    tokenized = [c["text"].lower().split() for c in chunk_corpus]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [c["chunk_id"] for c in chunk_corpus]
    chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunk_corpus}

    results: dict[str, list[str]] = {}
    latencies = []

    for query in queries:
        t0 = time.perf_counter()
        scores = bm25.get_scores(query["text"].lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_chunks]

        # Aggregate chunk scores to doc level (max pooling)
        doc_scores: dict[str, float] = {}
        for i in top_idx:
            doc_id = chunk_to_doc[chunk_ids[i]]
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), scores[i])

        top_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:k_docs]
        results[query["query_id"]] = top_docs
        latencies.append((time.perf_counter() - t0) * 1000)

    return results, sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# 4. Dense retrieval on chunks → map back to doc level
# ---------------------------------------------------------------------------

def retrieve_dense_chunks(
    chunk_corpus: list[dict],
    queries: list[dict],
    k_chunks: int = 50,
    k_docs: int = 10,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
    import numpy as np

    chunk_ids = [c["chunk_id"] for c in chunk_corpus]
    chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunk_corpus}

    print(f"    Embedding {len(chunk_corpus)} chunks via OpenAI...")
    t_enc = time.perf_counter()
    chunk_embs = np.array(openai_embed([c["text"] for c in chunk_corpus]), dtype="float32")
    norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    chunk_embs /= np.maximum(norms, 1e-9)
    print(f"    Encoded in {time.perf_counter() - t_enc:.1f}s")

    print(f"    Embedding {len(queries)} queries via OpenAI...")
    query_embs = np.array(openai_embed([q["text"] for q in queries]), dtype="float32")
    q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_embs /= np.maximum(q_norms, 1e-9)

    # Pure numpy dot product — avoids FAISS/PyTorch memory conflicts on macOS
    # shape: (n_queries, n_chunks)
    all_scores = np.dot(query_embs, chunk_embs.T)

    results: dict[str, list[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    latencies = []

    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        top_idx = np.argpartition(all_scores[i], -k_chunks)[-k_chunks:]
        top_idx = top_idx[np.argsort(all_scores[i][top_idx])[::-1]]

        doc_scores: dict[str, float] = {}
        for idx in top_idx:
            doc_id = chunk_to_doc[chunk_ids[idx]]
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), float(all_scores[i][idx]))

        top_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:k_docs]
        results[query["query_id"]] = top_docs
        raw_scores[query["query_id"]] = {d: doc_scores[d] for d in top_docs}
        latencies.append((time.perf_counter() - t0) * 1000)

    return results, raw_scores, sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# 5. Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank(
    retrieved: dict[str, list[str]],
    queries: list[dict],
    corpus: list[dict],
    reranker,
    k: int = 10,
) -> tuple[dict[str, list[str]], float]:
    doc_text = {d["doc_id"]: d["text"] for d in corpus}
    results: dict[str, list[str]] = {}
    latencies = []

    for query in queries:
        qid = query["query_id"]
        doc_ids = retrieved.get(qid, [])
        if not doc_ids:
            results[qid] = []
            continue

        t0 = time.perf_counter()
        pairs = [[query["text"], doc_text.get(d, "")] for d in doc_ids]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [d for d, _ in reranked[:k]]
        latencies.append((time.perf_counter() - t0) * 1000)

    return results, sum(latencies) / len(latencies) if latencies else 0.0


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    retrieved: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict:
    from retrievalbench.evaluate import (
        ndcg_at_k, recall_at_k, mean_reciprocal_rank,
        average_precision, bootstrap_ci,
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
        "ndcg@10": round(mean(ndcgs), 4),
        "ndcg@10_ci_lower": round(ci[0], 4),
        "ndcg@10_ci_upper": round(ci[1], 4),
        "recall@10": round(mean(recalls), 4),
        "mrr": round(mean(mrrs), 4),
        "map": round(mean(aps), 4),
        "n_queries": len(retrieved),
        "_ndcg_per_query": ndcgs,
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    from sentence_transformers import CrossEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact", choices=sorted(DATASET_DOMAINS))
    args = parser.parse_args()
    dataset = args.dataset
    domain = DATASET_DOMAINS[dataset]

    corpus, queries, qrels = load_beir_dataset(dataset, domain)

    print("\nLoading reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Skip semantic for large corpora (>20K docs) — too slow and costly
    if len(corpus) > 20000:
        CHUNKING_STRATEGIES = ["fixed_512", "sentence", "recursive"]
        print(f"  Note: skipping semantic chunking (corpus size {len(corpus)} > 20K)")
    else:
        CHUNKING_STRATEGIES = ["fixed_512", "sentence", "recursive", "semantic"]
    all_records = []

    for strategy in CHUNKING_STRATEGIES:
        print(f"\n{'='*60}")
        print(f"[{domain.upper()} / {dataset}] Chunking: {strategy}")
        print(f"{'='*60}")

        # Build chunk corpus
        print("  Building chunk corpus...")
        t0 = time.perf_counter()
        chunk_corpus = build_chunk_corpus(corpus, strategy)
        print(f"  {len(chunk_corpus)} chunks from {len(corpus)} docs ({time.perf_counter()-t0:.1f}s)")

        # BM25
        print("  Running BM25...")
        bm25_results, bm25_lat = retrieve_bm25_chunks(chunk_corpus, queries)
        bm25_metrics = evaluate(bm25_results, qrels)

        # BM25 + Rerank
        print("  Reranking BM25 results...")
        bm25_rerank_results, rerank_lat_bm25 = rerank(bm25_results, queries, corpus, reranker)
        bm25_rerank_metrics = evaluate(bm25_rerank_results, qrels)

        # Dense
        print("  Running Dense (OpenAI text-embedding-3-small)...")
        dense_results, dense_raw, dense_lat = retrieve_dense_chunks(chunk_corpus, queries)
        dense_metrics = evaluate(dense_results, qrels)

        # Dense + Rerank
        print("  Reranking Dense results...")
        dense_rerank_results, rerank_lat_dense = rerank(dense_results, queries, corpus, reranker)
        dense_rerank_metrics = evaluate(dense_rerank_results, qrels)

        # Print mini table
        print(f"\n  {'System':<25} {'nDCG@10':>8} {'Recall@10':>10} {'MRR':>7} {'Lat(ms)':>9}")
        print(f"  {'-'*60}")
        for name, m, lat in [
            ("BM25", bm25_metrics, bm25_lat),
            ("BM25+Rerank", bm25_rerank_metrics, bm25_lat + rerank_lat_bm25),
            ("Dense-OpenAI", dense_metrics, dense_lat),
            ("Dense-OpenAI+Rerank", dense_rerank_metrics, dense_lat + rerank_lat_dense),
        ]:
            print(f"  {name:<25} {m['ndcg@10']:>8.4f} {m['recall@10']:>10.4f} {m['mrr']:>7.4f} {lat:>9.1f}")

        # Store records
        for name, m, lat in [
            ("BM25", bm25_metrics, bm25_lat),
            ("BM25+Rerank", bm25_rerank_metrics, bm25_lat + rerank_lat_bm25),
            ("Dense-OpenAI", dense_metrics, dense_lat),
            ("Dense-OpenAI+Rerank", dense_rerank_metrics, dense_lat + rerank_lat_dense),
        ]:
            all_records.append({
                "domain": domain,
                "dataset": dataset,
                "chunking": strategy,
                "system": name,
                "ndcg@10": m["ndcg@10"],
                "ndcg@10_ci_lower": m["ndcg@10_ci_lower"],
                "ndcg@10_ci_upper": m["ndcg@10_ci_upper"],
                "recall@10": m["recall@10"],
                "mrr": m["mrr"],
                "map": m["map"],
                "n_queries": m["n_queries"],
                "latency_ms": round(lat, 2),
            })

    # Final summary table
    print(f"\n{'='*70}")
    print(f"FULL RESULTS — domain={domain} dataset={dataset}")
    print(f"{'='*70}")
    print(f"{'Chunking':<12} {'System':<20} {'nDCG@10':>8} {'CI':>20} {'Recall@10':>10} {'Lat(ms)':>9}")
    print(f"{'-'*70}")
    for r in all_records:
        ci = f"[{r['ndcg@10_ci_lower']:.3f},{r['ndcg@10_ci_upper']:.3f}]"
        print(f"{r['chunking']:<12} {r['system']:<20} {r['ndcg@10']:>8.4f} {ci:>20} {r['recall@10']:>10.4f} {r['latency_ms']:>9.1f}")

    # Save
    out_dir = results_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chunking_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "domain": domain,
            "dataset": dataset,
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_docs": len(corpus),
            "n_queries": len(queries),
            "results": all_records,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
