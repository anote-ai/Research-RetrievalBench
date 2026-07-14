#!/usr/bin/env python3
"""CSI / (proxy-)ESI summary across the full domain matrix.

Reads every results/<domain>/chunking_results.json and, per domain, builds the
nDCG@10 matrix M[chunker][system]. Reports:

  CSI  = chunking sensitivity  = mean_system var(nDCG across chunkers) / total_var
  SSI  = system  sensitivity   = mean_chunker var(nDCG across systems ) / total_var

CSI answers Claim 1/2's spirit ("chunking is a high-variance knob"). SSI uses the
4 retrieval SYSTEMS (BM25, Dense, ±rerank) as the second axis. NOTE: this is NOT
the design's true ESI (embedding-MODEL sensitivity), which needs >=2 embedders;
the matrix currently has one (openai-3-small). Reported as a proxy, labeled.
"""
import glob
import json
import os
import statistics as st

BONUS = {"touche-2020"}  # not one of the 12; report separately


def variance(xs):
    return st.pvariance(xs) if len(xs) >= 2 else 0.0


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "results")
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "chunking_results.json"))):
        dom_dir = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        for r in d.get("results", []):
            rows.append({**r, "_dir": dom_dir})

    # TRUE Claim-2 test: fix system = pure Dense (no rerank), vary EMBEDDER.
    # M[chunker][embedder] = ndcg@10 for dense retrieval.
    by_dom = {}
    for r in rows:
        emb = r.get("embedder", "openai")
        sys = r["system"]
        # keep only pure-dense rows (one per embedder) for the embedder axis
        if not sys.startswith("Dense-") or sys.endswith("+Rerank"):
            continue
        by_dom.setdefault((r["_dir"], r.get("domain", "?")), {})[
            (r["chunking"], emb)] = r["ndcg@10"]

    print(f"{'domain':<15}{'dataset':<15}{'#chnk':>6}{'#emb':>5}{'CSI':>8}{'ESI':>8}  CSI>ESI")
    print("-" * 68)
    core = []
    for (dset, dom), cells in sorted(by_dom.items(), key=lambda kv: kv[0][1]):
        chunkers = sorted({c for c, _ in cells})
        embs = sorted({e for _, e in cells})
        tot = variance(list(cells.values()))
        if tot == 0 or len(chunkers) < 2 or len(embs) < 2:
            note = f"need>=2 chnk&emb (have {len(chunkers)}chnk {len(embs)}emb)"
            print(f"{dom:<15}{dset:<15}{len(chunkers):>6}{len(embs):>5}   {note}")
            continue
        csi_terms = [variance([cells[(c, e)] for c in chunkers if (c, e) in cells]) for e in embs]
        esi_terms = [variance([cells[(c, e)] for e in embs if (c, e) in cells]) for c in chunkers]
        csi = sum(csi_terms) / len(csi_terms) / tot
        esi = sum(esi_terms) / len(esi_terms) / tot
        flag = "YES" if csi > esi else "no"
        star = "*" if dset in BONUS else " "
        print(f"{dom:<15}{dset:<15}{len(chunkers):>6}{len(embs):>5}{csi:>8.3f}{esi:>8.3f}    {flag}{star}")
        if dset not in BONUS:
            core.append((dom, dset, csi, esi))

    n = len(core)
    hold = sum(1 for *_, c, e in core if c > e)
    print("-" * 68)
    print(f"Core domains with computable CSI/ESI: {n} | "
          f"CSI>ESI (Claim 2 holds: chunking > embedding-model): {hold}/{n}")
    print("Pure-Dense rows only; embedders = " +
          ", ".join(sorted({r.get('embedder', 'openai') for r in rows if r['system'].startswith('Dense-') and not r['system'].endswith('+Rerank')})))
    out = os.path.join(root, "csi_esi_summary.json")
    json.dump({"core": [{"domain": d, "dataset": ds, "csi": round(c, 4),
                         "esi": round(e, 4), "claim2_holds": c > e}
                        for d, ds, c, e in core],
               "claim2_holds_count": hold, "n_core": n},
              open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
