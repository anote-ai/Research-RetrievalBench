"""Diagnose why rerank is a no-op on quora.

Hypothesis: cross-encoder/ms-marco-MiniLM-L-6-v2 is trained on (query, passage)
pairs. On quora, both query and doc are short questions, so the reranker outputs
near-constant scores. With Python's stable sort, a constant score leaves the
input order unchanged -> rerank becomes a no-op.

This probe loads quora, runs BM25 to get top-10, then prints the reranker's
raw scores for a few queries. If scores are near-identical, the hypothesis holds.

Usage (run in the env that has the deps installed):
    python scripts/probe_quora_rerank.py
"""
from __future__ import annotations

from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402


def load_quora():
    corpus_ds = load_dataset("BeIR/quora", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/quora", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/quora-qrels", split="test")

    corpus = {str(r["_id"]): (r["title"] + " " + r["text"]).strip()
              for r in corpus_ds}
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels = defaultdict(set)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    return corpus, queries, dict(qrels)


def main():
    print("Loading quora...")
    corpus, queries, qrels = load_quora()

    # Subsample for speed: first 20 queries that have qrels
    qids = [q for q in queries if q in qrels][:20]
    print(f"  {len(corpus)} docs | {len(qids)} sampled queries")

    # Quick BM25 over a subset of corpus (first 20k docs) to get candidate top-10
    # — full 100k BM25 is slow; for a probe we just need *some* top-10 per query.
    print("Building BM25 on first 20k docs...")
    doc_ids = list(corpus.keys())[:20000]
    tokenized = [corpus[d].lower().split() for d in doc_ids]
    bm25 = BM25Okapi(tokenized)

    print("Loading reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("\n" + "=" * 70)
    print("PROBE: reranker raw scores on top-10 BM25 candidates")
    print("=" * 70)
    for qid in qids[:5]:
        qtext = queries[qid]
        scores = bm25.get_scores(qtext.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        cand_ids = [doc_ids[i] for i in top_idx]

        pairs = [[qtext, corpus[d]] for d in cand_ids]
        ce_scores = reranker.predict(pairs)

        print(f"\nQuery {qid}: {qtext[:80]}")
        for rank, (did, s) in enumerate(zip(cand_ids, ce_scores)):
            rel = "REL" if did in qrels.get(qid, set()) else "   "
            print(f"  rank {rank}: score={float(s):.6f} {rel} doc={did} | {corpus[did][:60]}")

        # variance of scores — if ~0, rerank is a stable-sort no-op
        import statistics
        if len(ce_scores) >= 2:
            print(f"  >> score stdev = {statistics.stdev(map(float, ce_scores)):.6f} "
                  f"(if ~0, no-op explained)")


if __name__ == "__main__":
    main()
