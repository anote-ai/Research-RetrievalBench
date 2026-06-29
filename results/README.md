# results/

This directory is the designated home for **real, measured** experiment
output — as opposed to the synthetic demo output currently printed by
`scripts/run_demo.py` and summarized in the root `README.md`.

Nothing has been checked in here yet because no real (non-synthetic)
experiment has been run against real document corpora or real retrieval
systems. This directory exists now so that the convention is established
before the first real run:

## Convention

Each real experiment run should add a subdirectory named
`<experiment-id>_<yyyy-mm-dd>/` containing:

- `config.json` — the exact configuration used (systems, domains, chunking
  variants, hyperparameters, random seeds).
- `raw_metrics.csv` or `raw_metrics.jsonl` — per-query, per-config metric
  values (not just aggregates), so statistical tests in
  `src/retrievalbench/evaluate.py` (bootstrap CIs, permutation tests) can be
  re-run or audited later.
- `summary.md` — a short human-readable summary of what was run and the
  top-line findings, with explicit caveats about what was *not* covered.
- `environment.txt` — pinned dependency versions and hardware description,
  since several of the design doc's claims (latency, cost) are
  environment-sensitive (see DESIGN_DOC.md Section 10, "Open Questions and
  Risks").

## Mapping to DESIGN_DOC.md experiments

| Experiment | Design doc section | Status |
|---|---|---|
| Baseline BEIR replication | 5.1 | Not run |
| Chunking sensitivity analysis (CSI/ESI) | 5.2 | Not run |
| Latency-Adjusted NDCG | 5.3 | Not run (metric exists in `evaluate.py`, no real latency data collected) |
| Cost-Quality Pareto Frontier | 5.4 | Not run |
| Context-Window Saturation | 5.5 | Not run |
| Synthetic 4-domain RAG ablation (current demo) | n/a — infrastructure validation only | Run via `scripts/run_demo.py`; output is synthetic and lives in `README.md`, not here, to avoid being mistaken for a real result |

When the first real experiment runs, update this table and add the
corresponding subdirectory.
