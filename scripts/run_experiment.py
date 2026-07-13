"""Drive a RetrievalBench experiment grid from the command line.

Examples:
  # one domain, all default embedders/chunkers/rerankers
  python scripts/run_experiment.py --datasets scifact

  # full 12-domain grid (Part C: add datasets in config.DATASET_DOMAINS first)
  python scripts/run_experiment.py --all-datasets

  # subset: only bm25 + openai dense, no rerank, on 2 domains
  python scripts/run_experiment.py --datasets scifact nfcorpus \
      --embedders openai-3-small --no-rerank

  # subsample for a quick smoke test
  RB_MAX_CORPUS=2000 RB_MAX_QUERIES=50 python scripts/run_experiment.py --datasets scifact
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# allow running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievalbench.pipeline import (  # noqa: E402
    DATASET_DOMAINS, ExperimentGrid, ChunkerSpec,
    EMBEDDER_REGISTRY, RERANKER_REGISTRY,
    STRUCTURE_AWARE_CHUNKERS, default_reranker_for,
    load_dataset, run_cell, save_results,
)
from retrievalbench.pipeline.retrievers.embedders import EMBEDDER_REGISTRY as _ER  # noqa


def build_grid(args) -> ExperimentGrid:
    datasets = (args.datasets if args.datasets
                else sorted(DATASET_DOMAINS) if args.all_datasets
                else ["scifact"])

    embedders = [EMBEDDER_REGISTRY[e] for e in args.embedders]
    chunkers = [ChunkerSpec(name=c, kind="generic") for c in args.chunkers]
    # add domain-matched structure-aware chunkers when running that domain
    if not args.no_structure_aware:
        for ds in datasets:
            dom = DATASET_DOMAINS.get(ds, "")
            if dom in STRUCTURE_AWARE_CHUNKERS:
                chunkers.append(STRUCTURE_AWARE_CHUNKERS[dom])

    rerankers: list = [None]  # no-rerank arm always present
    if not args.no_rerank:
        for name in args.rerankers:
            rerankers.append(RERANKER_REGISTRY[name])

    return ExperimentGrid(
        datasets=datasets, embedders=embedders,
        chunkers=chunkers, rerankers=rerankers,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="BEIR dataset names (default: scifact)")
    p.add_argument("--all-datasets", action="store_true",
                   help="run every dataset in DATASET_DOMAINS")
    p.add_argument("--embedders", nargs="+",
                   default=["openai-3-small", "bge-small", "e5-small"],
                   help="embedder registry keys")
    p.add_argument("--chunkers", nargs="+",
                   default=["fixed_512", "sentence", "recursive"],
                   help="generic chunker names")
    p.add_argument("--rerankers", nargs="+",
                   default=None,
                   help="reranker registry keys (default: per-domain best + bge)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-structure-aware", action="store_true",
                   help="skip domain-matched structure-aware chunkers")
    p.add_argument("--dry-run", action="store_true",
                   help="print the grid size and exit")
    args = p.parse_args()

    # default rerankers: domain-appropriate + general fallback, deduped
    if args.rerankers is None and not args.no_rerank:
        datasets = args.datasets or (sorted(DATASET_DOMAINS) if args.all_datasets else ["scifact"])
        picks = []
        for ds in datasets:
            r = default_reranker_for(DATASET_DOMAINS.get(ds, ""))
            if r not in picks:
                picks.append(r)
        if "bge-reranker" not in picks:
            picks.append("bge-reranker")
        args.rerankers = picks

    grid = build_grid(args)
    print(f"Grid: {len(grid.datasets)} datasets × {len(grid.embedders)} embedders "
          f"× {len(grid.chunkers)} chunkers × {len(grid.rerankers)} reranker-arms "
          f"= {len(grid)} cells ({len(grid)*2} retriever runs)")
    if args.dry_run:
        return

    for ds in grid.datasets:
        domain = DATASET_DOMAINS.get(ds, "unknown")
        print(f"\n{'#'*70}\n# Dataset: {ds} ({domain})\n{'#'*70}")
        try:
            corpus, queries, qrels = load_dataset(ds)
        except Exception as e:
            print(f"  [SKIP] failed to load {ds}: {type(e).__name__}: {e}")
            continue
        all_records = []
        total_tokens = 0
        total_cost = 0.0

        cells = [c for c in grid.cells() if c.dataset == ds]
        for i, cell in enumerate(cells, 1):
            rr_spec = (RERANKER_REGISTRY[cell.reranker.name]
                       if cell.reranker else None)
            print(f"\n[{i}/{len(cells)}] chunker={cell.chunker.name} "
                  f"embedder={cell.embedder.name} reranker="
                  f"{cell.reranker.name if cell.reranker else 'none'}")
            recs = run_cell(cell, corpus, queries, qrels, reranker_spec=rr_spec)
            for r in recs:
                total_tokens += r.get("embed_tokens", 0)
                total_cost += r.get("embed_cost_usd", 0.0)
            all_records.extend(recs)
            for r in recs:
                print(f"    {r['system']:40} nDCG={r['ndcg@10']:.4f} "
                      f"LA={r['la_ndcg@10']:.4f} rec={r['recall@10']:.4f} "
                      f"lat={r['latency_ms']:.1f}"
                      + (f" degen={r['rerank_degenerate_rate']}"
                         if 'rerank_degenerate_rate' in r else ""))

        save_results(ds, domain, all_records, len(corpus), len(queries),
                     total_tokens, total_cost)
        print(f"\nSaved results/{ds}/chunking_results.json "
              f"(cost ${total_cost:.4f}, {total_tokens} tokens)")


if __name__ == "__main__":
    main()
