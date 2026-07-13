"""Build cost-quality Pareto frontier + leaderboard from results/ JSON.

Reads every results/*/chunking_results.json, extracts (system, domain, nDCG@10,
latency_ms, embed_cost_usd), computes the Pareto frontier (maximize nDCG,
minimize cost and latency), and prints a leaderboard + optional matplotlib chart.

No retrieval is re-run — pure post-processing on existing results.

Usage:
    python scripts/build_pareto.py
    python scripts/build_pareto.py --out results/pareto.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_all_records() -> list[dict]:
    records = []
    for path in glob.glob("results/*/chunking_results.json"):
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            r["_source"] = os.path.relpath(path)
            r["_domain"] = data.get("domain", r.get("domain", "?"))
            records.append(r)
    return records


def pareto_front(points: list[dict]) -> list[bool]:
    """Mark each point Pareto-optimal: max nDCG, min cost, min latency."""
    flags = [True] * len(points)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            # q dominates p if q >= p on nDCG and <= on cost & latency, strict on one
            if (q["ndcg@10"] >= p["ndcg@10"]
                    and q["_cost"] <= p["_cost"]
                    and q["latency_ms"] <= p["latency_ms"]
                    and (q["ndcg@10"] > p["ndcg@10"]
                         or q["_cost"] < p["_cost"]
                         or q["latency_ms"] < p["latency_ms"])):
                flags[i] = False
                break
    return flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="write a PNG chart here (needs matplotlib)")
    p.add_argument("--per-domain", action="store_true",
                   help="compute a separate frontier per domain")
    args = p.parse_args()

    records = load_all_records()
    if not records:
        print("No results found under results/*/chunking_results.json")
        return

    # normalize cost
    for r in records:
        r["_cost"] = r.get("embed_cost_usd", 0.0)
        # local embedders have 0 API cost; surface latency as the proxy then
    # total cost-per-1k-queries = embed_cost_usd / n_queries * 1000
    for r in records:
        nq = r.get("n_queries", 1) or 1
        r["_cost_per_1k"] = r["_cost"] / nq * 1000

    groups: dict[str, list[dict]] = {"ALL": records}
    if args.per_domain:
        for r in records:
            groups.setdefault(r["_domain"], []).append(r)

    print(f"{'System':<42} {'Domain':<14} {'nDCG@10':>8} {'Lat(ms)':>8} "
          f"{'$/1Kq':>8} {'Pareto':>7}")
    print("-" * 90)

    for gname, pts in groups.items():
        flags = pareto_front(pts)
        # sort by nDCG desc
        order = sorted(range(len(pts)), key=lambda i: pts[i]["ndcg@10"], reverse=True)
        for i in order:
            r = pts[i]
            star = "✓" if flags[i] else ""
            print(f"{r['system']:<42} {r['_domain']:<14} {r['ndcg@10']:>8.4f} "
                  f"{r['latency_ms']:>8.1f} {r['_cost_per_1k']:>8.5f} {star:>7}")

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping chart")
            return
        fig, ax = plt.subplots(figsize=(9, 6))
        # plot ALL group: x=cost_per_1k (log), y=nDCG
        pts = groups["ALL"]
        flags = pareto_front(pts)
        for r, f in zip(pts, flags):
            color = "red" if f else "gray"
            ax.scatter(r["_cost_per_1k"], r["ndcg@10"], c=color, s=40,
                       marker="*" if f else "o")
        pareto_pts = sorted([r for r, f in zip(pts, flags) if f],
                            key=lambda r: r["_cost_per_1k"])
        if len(pareto_pts) >= 2:
            ax.plot([r["_cost_per_1k"] for r in pareto_pts],
                    [r["ndcg@10"] for r in pareto_pts], "r--", lw=1)
        ax.set_xlabel("Embedding cost per 1K queries (USD, log)")
        ax.set_ylabel("nDCG@10")
        ax.set_xscale("symlog")
        ax.set_title("Cost-quality Pareto frontier (all domains, all systems)")
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"\nChart saved to {args.out}")


if __name__ == "__main__":
    main()
