"""Context-window saturation analysis (DESIGN_DOC §5.4).

Question: as more retrieved passages are fed to the generator (top-k growing),
at what k does downstream quality saturate? Retrieval beyond that point adds
noise, not signal — the core motivation for LA-NDCG and saturation-aware retrieval.

REQUIRES the ranked_lists sidecar produced by pipeline/run.py (B5). For each
(domain, system) we recompute retrieval-side recall@k and nDCG@k at
k in {1,3,5,10,20,50,100} — NO retrieval re-run. Optionally, a generator LLM
can be called to measure end-to-end answer quality at each k (needs OPENAI_API_KEY).

Usage:
    # retrieval-side saturation only (no LLM cost)
    python scripts/analyze_saturation.py --dataset scifact

    # with generator LLM quality scoring
    python scripts/analyze_saturation.py --dataset scifact --with-generator
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv()

from retrievalbench.evaluate import ndcg_at_k, recall_at_k  # noqa: E402

K_GRID = [1, 3, 5, 10, 20, 50, 100]


def load_lists(dataset: str) -> dict:
    path = os.path.join("results", dataset, "ranked_lists.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the new pipeline (run_experiment.py) first — "
            "the legacy run_chunking_pipeline.py did not store ranked lists.")
    with open(path) as f:
        return json.load(f)


def load_qrels(dataset: str) -> dict:
    from retrievalbench.pipeline.loaders import load_dataset
    _, _, qrels = load_dataset(dataset)
    return qrels


def retrieval_saturation(ranked_lists: dict, qrels: dict) -> dict:
    """For each (system/chunking) cell, compute recall@k and nDCG@k across K_GRID.

    Uses the doc-level ranked list (`ranked_docs`), which run.py persists
    alongside ranked_chunks. Saturation is a doc-level phenomenon: how many
    of the top-ranked *documents* need to be fed to the generator.
    """
    out = {}
    for key, payload in ranked_lists.items():
        qid_to_docs = payload.get("ranked_docs") if isinstance(payload, dict) else payload
        if not qid_to_docs:
            continue
        curve = {k: {"recall": [], "ndcg": []} for k in K_GRID}
        for qid, ids in qid_to_docs.items():
            rel = qrels.get(qid, set())
            if not rel:
                continue
            for k in K_GRID:
                curve[k]["recall"].append(recall_at_k(ids, rel, k))
                curve[k]["ndcg"].append(ndcg_at_k(ids, rel, k))
        out[key] = {
            str(k): {
                "recall": round(sum(v["recall"]) / len(v["recall"]), 4) if v["recall"] else 0.0,
                "ndcg": round(sum(v["ndcg"]) / len(v["ndcg"]), 4) if v["ndcg"] else 0.0,
            }
            for k, v in curve.items()
        }
    return out


def find_saturation_point(curve: dict) -> int:
    """Smallest k where nDCG improves <1% over the previous k."""
    ks = sorted(int(k) for k in curve)
    for i in range(1, len(ks)):
        prev = curve[str(ks[i - 1])]["ndcg"]
        cur = curve[str(ks[i])]["ndcg"]
        if prev > 0 and (cur - prev) / prev < 0.01:
            return ks[i - 1]
    return ks[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--with-generator", action="store_true",
                   help="also call a generator LLM to score end-to-end quality at each k")
    args = p.parse_args()

    print(f"Loading ranked lists for {args.dataset}...")
    lists = load_lists(args.dataset)
    qrels = load_qrels(args.dataset)
    print(f"  {len(lists)} cells | {len(qrels)} queries")

    sat = retrieval_saturation(lists, qrels)
    print(f"\n{'Cell':<45} {'sat@k':>6} | " + " ".join(f"k={k}" for k in K_GRID))
    print("-" * 80)
    for key, curve in sat.items():
        sat_k = find_saturation_point(curve)
        ndcgs = " ".join(f"{curve[str(k)]['ndcg']:.3f}" for k in K_GRID)
        print(f"{key:<45} {sat_k:>6} | {ndcgs}")

    out_path = os.path.join("results", args.dataset, "saturation.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "curves": sat}, f, indent=2)
    print(f"\nSaved {out_path}")

    if args.with_generator:
        print("\n[with-generator] generator LLM scoring not yet implemented — "
              "wire up an LLM judge here once ranked_lists are populated.")


if __name__ == "__main__":
    main()
