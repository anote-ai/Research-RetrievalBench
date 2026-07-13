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
    "quora": "community",
    "arguana": "argumentation",
    "auslegalqa": "legal",
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"  # $0.02/1M tokens
EMBED_COST_PER_TOKEN = 0.02 / 1_000_000  # USD per token

# Global token counter — reset per experiment run
_total_embed_tokens: int = 0


def results_dir(dataset: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "results", dataset)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------


def load_auslegalqa() -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    """Legal domain: isaacus/open-australian-legal-qa. Corpus = source case-law
    snippets (deduplicated), queries = questions, qrels = 1:1 provenance."""
    import re as _re
    from collections import defaultdict as _dd
    from datasets import load_dataset

    print("[LEGAL] Loading isaacus/open-australian-legal-qa from HuggingFace...")
    ds = load_dataset("isaacus/open-australian-legal-qa", split="train")
    snippet_re = _re.compile(r"<snippet>\n(.*?)</snippet>", _re.S)
    corpus, queries, qrels = [], [], _dd(set)
    seen: dict[str, str] = {}
    for i, row in enumerate(ds):
        m = snippet_re.search(row["prompt"])
        if not m:
            continue
        text = m.group(1).strip()
        key = text[:200]
        if key not in seen:
            doc_id = f"doc_{len(seen):05d}"
            seen[key] = doc_id
            corpus.append({"doc_id": doc_id, "text": text, "corpus_position": 0.0})
        qid = f"q_{i:05d}"
        queries.append({"query_id": qid, "text": row["question"]})
        qrels[qid].add(seen[key])
    qrels = dict(qrels)
    n = len(corpus)
    for i, doc in enumerate(corpus):
        doc["corpus_position"] = i / max(n - 1, 1)
    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)}")
    return corpus, queries, qrels


def load_beir_dataset(dataset: str, domain: str) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    from datasets import load_dataset

    print(f"[{domain.upper()}] Loading {dataset} from HuggingFace...")
    corpus_ds = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{dataset}-qrels", split="test")

    raw_corpus = list(corpus_ds)
    n = len(raw_corpus)
    corpus = [
        {
            "doc_id": str(r["_id"]),
            "text": (r["title"] + " " + r["text"]).strip(),
            "corpus_position": i / max(n - 1, 1),
        }
        for i, r in enumerate(raw_corpus)
    ]
    queries = [{"query_id": str(r["_id"]), "text": r["text"]} for r in queries_ds]

    qrels: dict[str, set[str]] = defaultdict(set)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    qrels = dict(qrels)
    queries = [q for q in queries if q["query_id"] in qrels]
    max_c = int(os.getenv("RB_MAX_CORPUS", "0"))
    if max_c and len(corpus) > max_c:
        import random as _rnd
        relevant = set().union(*qrels.values())
        keep = [d for d in corpus if d["doc_id"] in relevant]
        rest = [d for d in corpus if d["doc_id"] not in relevant]
        _rnd.Random(42).shuffle(rest)
        corpus = keep + rest[: max(0, max_c - len(keep))]
        print(f"  Subsampled corpus to {len(corpus)} docs "
              f"({len(keep)} qrels-relevant kept, seed 42)")
    max_q = int(os.getenv("RB_MAX_QUERIES", "0"))
    if max_q and len(queries) > max_q:
        import random as _rnd
        _rnd.Random(42).shuffle(queries)
        queries = queries[:max_q]
        print(f"  Subsampled to {max_q} queries (seed 42)")

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
    Accumulates token usage into global _total_embed_tokens for cost tracking.
    Retries up to 5 times on timeout/connection errors with exponential backoff.
    """
    global _total_embed_tokens
    import time as _time
    from openai import OpenAI, APITimeoutError, APIConnectionError

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)
    safe_texts = [t if t.strip() else " " for t in texts]
    all_embeddings = []
    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i: i + batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
                _total_embed_tokens += resp.usage.total_tokens
                batch_embs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
                all_embeddings.extend(batch_embs)
                break
            except (APITimeoutError, APIConnectionError) as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/5] {e.__class__.__name__}, waiting {wait}s...")
                _time.sleep(wait)
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
    latency_ms: float = 0.0,
    k: int = 10,
) -> dict:
    from retrievalbench.evaluate import (
        ndcg_at_k, recall_at_k, mean_reciprocal_rank,
        average_precision, bootstrap_ci, latency_adjusted_ndcg,
    )

    ndcgs, recalls, mrrs, aps, la_ndcgs = [], [], [], [], []
    for qid, ret_ids in retrieved.items():
        rel = qrels.get(qid, set())
        ndcg = ndcg_at_k(ret_ids, rel, k)
        ndcgs.append(ndcg)
        recalls.append(recall_at_k(ret_ids, rel, k))
        mrrs.append(mean_reciprocal_rank(ret_ids, rel))
        aps.append(average_precision(ret_ids, rel))
        la_ndcgs.append(latency_adjusted_ndcg(ret_ids, rel, latency_ms, k))

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    ci = bootstrap_ci(ndcgs)

    return {
        "ndcg@10": round(mean(ndcgs), 4),
        "ndcg@10_ci_lower": round(ci[0], 4),
        "ndcg@10_ci_upper": round(ci[1], 4),
        "la_ndcg@10": round(mean(la_ndcgs), 4),
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
    global _total_embed_tokens
    from sentence_transformers import CrossEncoder
    from retrievalbench.evaluate import permutation_test, cohens_d

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact", choices=sorted(DATASET_DOMAINS))
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Run only these strategies (e.g. --strategies recursive semantic)")
    args = parser.parse_args()
    dataset = args.dataset
    domain = DATASET_DOMAINS[dataset]

    if args.dataset == "auslegalqa":


        corpus, queries, qrels = load_auslegalqa()


    else:


        corpus, queries, qrels = load_beir_dataset(dataset, domain)

    print("\nLoading reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Skip semantic for large corpora (>20K docs) — too slow and costly
    if len(corpus) > 20000:
        default_strategies = ["fixed_512", "sentence", "recursive"]
        print(f"  Note: skipping semantic chunking (corpus size {len(corpus)} > 20K)")
    else:
        default_strategies = ["fixed_512", "sentence", "recursive", "semantic"]
    CHUNKING_STRATEGIES = args.strategies if args.strategies else default_strategies
    # Load existing results if we're only running a subset of strategies
    out_path = os.path.join(results_dir(dataset), "chunking_results.json")
    if args.strategies and os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        all_records = [r for r in existing["results"]
                       if r["chunking"] not in args.strategies]
        existing_bias = existing.get("position_bias", {})
        print(f"  Loaded {len(all_records)} existing records, will append {args.strategies}")
    else:
        all_records = []
        existing_bias = {}

    # position_bias_audit expects BenchmarkRun-like objects; we'll collect data manually
    # Format: {(strategy, system): {qid: [ret_ids]}} for post-hoc position bias
    pos_bias_data: dict[str, dict[str, list[str]]] = {}

    for strategy in CHUNKING_STRATEGIES:
        print(f"\n{'='*60}")
        print(f"[{domain.upper()} / {dataset}] Chunking: {strategy}")
        print(f"{'='*60}")

        # Reset token counter per strategy so we can track cost per chunking run
        tokens_before = _total_embed_tokens

        # Build chunk corpus
        print("  Building chunk corpus...")
        t0 = time.perf_counter()
        chunk_corpus = build_chunk_corpus(corpus, strategy)
        print(f"  {len(chunk_corpus)} chunks from {len(corpus)} docs ({time.perf_counter()-t0:.1f}s)")

        # BM25
        print("  Running BM25...")
        bm25_results, bm25_lat = retrieve_bm25_chunks(chunk_corpus, queries)
        bm25_metrics = evaluate(bm25_results, qrels, bm25_lat)

        # BM25 + Rerank
        print("  Reranking BM25 results...")
        bm25_rerank_results, rerank_lat_bm25 = rerank(bm25_results, queries, corpus, reranker)
        bm25_rerank_lat = bm25_lat + rerank_lat_bm25
        bm25_rerank_metrics = evaluate(bm25_rerank_results, qrels, bm25_rerank_lat)

        # Dense
        print("  Running Dense (OpenAI text-embedding-3-small)...")
        dense_results, dense_raw, dense_lat = retrieve_dense_chunks(chunk_corpus, queries)
        dense_metrics = evaluate(dense_results, qrels, dense_lat)

        # Dense + Rerank
        print("  Reranking Dense results...")
        dense_rerank_results, rerank_lat_dense = rerank(dense_results, queries, corpus, reranker)
        dense_rerank_lat = dense_lat + rerank_lat_dense
        dense_rerank_metrics = evaluate(dense_rerank_results, qrels, dense_rerank_lat)

        # Cost for this strategy's Dense runs (BM25 has no embedding cost)
        tokens_used = _total_embed_tokens - tokens_before
        cost_usd = tokens_used * EMBED_COST_PER_TOKEN
        cost_per_query = cost_usd / len(queries) if queries else 0.0

        # Significance tests: each system vs BM25 baseline
        sig_bm25r = permutation_test(bm25_rerank_metrics["_ndcg_per_query"], bm25_metrics["_ndcg_per_query"])
        sig_dense = permutation_test(dense_metrics["_ndcg_per_query"], bm25_metrics["_ndcg_per_query"])
        sig_denser = permutation_test(dense_rerank_metrics["_ndcg_per_query"], bm25_metrics["_ndcg_per_query"])
        cd_dense = cohens_d(dense_metrics["_ndcg_per_query"], bm25_metrics["_ndcg_per_query"])

        print(f"\n  Significance vs BM25: BM25+Rerank p={sig_bm25r['p_value']:.4f} | Dense p={sig_dense['p_value']:.4f} | Dense+Rerank p={sig_denser['p_value']:.4f}")
        print(f"  Cohen's d (Dense vs BM25): {cd_dense:.4f}")
        print(f"  Embedding cost this strategy: ${cost_usd:.4f} ({tokens_used} tokens, ${cost_per_query*1000:.4f}/1K queries)")

        # Print mini table
        print(f"\n  {'System':<25} {'nDCG@10':>8} {'LA-nDCG@10':>12} {'Recall@10':>10} {'MRR':>7} {'Lat(ms)':>9}")
        print(f"  {'-'*70}")
        for name, m, lat in [
            ("BM25", bm25_metrics, bm25_lat),
            ("BM25+Rerank", bm25_rerank_metrics, bm25_rerank_lat),
            ("Dense-OpenAI", dense_metrics, dense_lat),
            ("Dense-OpenAI+Rerank", dense_rerank_metrics, dense_rerank_lat),
        ]:
            print(f"  {name:<25} {m['ndcg@10']:>8.4f} {m['la_ndcg@10']:>12.4f} {m['recall@10']:>10.4f} {m['mrr']:>7.4f} {lat:>9.1f}")

        # Collect for position bias (manual implementation — no BenchmarkRun objects needed)
        pos_bias_data[f"{strategy}/BM25"] = bm25_results
        pos_bias_data[f"{strategy}/BM25+Rerank"] = bm25_rerank_results
        pos_bias_data[f"{strategy}/Dense-OpenAI"] = dense_results
        pos_bias_data[f"{strategy}/Dense-OpenAI+Rerank"] = dense_rerank_results

        # Store records
        for name, m, lat, sig, extra in [
            ("BM25", bm25_metrics, bm25_lat, None, {}),
            ("BM25+Rerank", bm25_rerank_metrics, bm25_rerank_lat, sig_bm25r, {}),
            ("Dense-OpenAI", dense_metrics, dense_lat, sig_dense,
             {"cohens_d_vs_bm25": round(cd_dense, 4),
              "embed_tokens": tokens_used,
              "embed_cost_usd": round(cost_usd, 6),
              "cost_per_1k_queries_usd": round(cost_per_query * 1000, 4)}),
            ("Dense-OpenAI+Rerank", dense_rerank_metrics, dense_rerank_lat, sig_denser, {}),
        ]:
            record = {
                "domain": domain,
                "dataset": dataset,
                "chunking": strategy,
                "system": name,
                "ndcg@10": m["ndcg@10"],
                "ndcg@10_ci_lower": m["ndcg@10_ci_lower"],
                "ndcg@10_ci_upper": m["ndcg@10_ci_upper"],
                "la_ndcg@10": m["la_ndcg@10"],
                "recall@10": m["recall@10"],
                "mrr": m["mrr"],
                "map": m["map"],
                "n_queries": m["n_queries"],
                "latency_ms": round(lat, 2),
            }
            if sig is not None:
                record["p_value_vs_bm25"] = round(sig["p_value"], 4)
                record["significant_vs_bm25"] = sig["p_value"] < 0.05
            record.update(extra)
            all_records.append(record)

    # Position bias audit (manual — compute per-tier recall directly)
    print(f"\n{'='*60}")
    print("POSITION BIAS AUDIT")
    print(f"{'='*60}")
    doc_pos = {d["doc_id"]: d["corpus_position"] for d in corpus}

    def pos_tier(rel_ids: set[str]) -> str:
        if not rel_ids:
            return "mid"
        avg = sum(doc_pos.get(d, 0.5) for d in rel_ids) / len(rel_ids)
        return "early" if avg < 0.33 else ("late" if avg >= 0.67 else "mid")

    from retrievalbench.evaluate import recall_at_k as _recall_at_k
    bias_results: dict[str, dict] = dict(existing_bias)
    for key, retrieved in pos_bias_data.items():
        tier_recalls: dict[str, list[float]] = {"early": [], "mid": [], "late": []}
        for qid, ret_ids in retrieved.items():
            rel = qrels.get(qid, set())
            tier = pos_tier(rel)
            tier_recalls[tier].append(_recall_at_k(ret_ids, rel, 10))
        def _mean(lst): return round(sum(lst) / len(lst), 4) if lst else None
        early = _mean(tier_recalls["early"])
        mid = _mean(tier_recalls["mid"])
        late = _mean(tier_recalls["late"])
        bias_gap = round((early or 0.0) - (late or 0.0), 4)
        bias_results[key] = {"early": early, "mid": mid, "late": late, "bias_gap": bias_gap}
        print(f"  {key:<40} early={early} mid={mid} late={late} gap={bias_gap:+.4f}")

    # Final summary table
    print(f"\n{'='*80}")
    print(f"FULL RESULTS — domain={domain} dataset={dataset}")
    print(f"{'='*80}")
    print(f"{'Chunking':<12} {'System':<22} {'nDCG@10':>8} {'LA-nDCG':>8} {'Recall':>7} {'Lat(ms)':>9} {'p-val':>7}")
    print(f"{'-'*80}")
    for r in all_records:
        pval = f"{r['p_value_vs_bm25']:.4f}" if "p_value_vs_bm25" in r else "  base"
        print(f"{r['chunking']:<12} {r['system']:<22} {r['ndcg@10']:>8.4f} {r['la_ndcg@10']:>8.4f} {r['recall@10']:>7.4f} {r['latency_ms']:>9.1f} {pval:>7}")

    total_cost = _total_embed_tokens * EMBED_COST_PER_TOKEN
    print(f"\nTotal embedding cost: ${total_cost:.4f} ({_total_embed_tokens} tokens)")

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
            "total_embed_tokens": _total_embed_tokens,
            "total_embed_cost_usd": round(total_cost, 6),
            "position_bias": bias_results,
            "results": all_records,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
