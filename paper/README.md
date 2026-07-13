# RetrievalBench Paper Draft

`main.tex` — LaTeX draft of the RetrievalBench paper, written from real experiment results.

## Compile

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Status (as of 2026-07-13)

This is a **working draft** based on completed experiments. Honest status:

### Claims supported by data now
- **Reranker is not monotonically beneficial** — 3/7 domains show Dense+Rerank *worse* than Dense (arguana −0.026, fiqa −0.022, scidocs −0.005). Section 4.1, Table 2. ✅ real data
- **Context-window saturation** — BM25+Rerank nDCG declines 0.620→0.607 as k grows past 1. Section 4.2, Table 3. ✅ real data (scifact)
- **Dense dominates BM25 uniformly** — Table 2. ✅ real data
- **Cost ranges** — $0.013–$0.19 per run. Section 4.4. ✅ real data

### Claims marked in-progress in the draft
- **CSI > ESI** (chunking matters more than embedder) — Section 4.3 explicitly says "in progress"; multi-chunker data exists for nfcorpus/arguana/fiqa/quora/scidocs but the structure-aware ablation (CUAD legal_clause, fiqa financial_table, nfcorpus medical_section) needs the remaining domains.
- **12 domains** — Table 1 lists 7; 5 more (climate-fever, dbpedia-entity, hotpotqa, msmarco, CUAD) are integrated and running. Draft's Limitations (Section 5) states this explicitly.
- **Local embedder comparison** (BGE/E5/GTE/Nomic) — wired in code, not yet run at scale.
- **Generator LLM judge for saturation** — stubbed, not wired.

### Data source
All numbers in the paper come from `results/*/chunking_results.json` and `results/scifact/saturation.json`. Regenerate with:
```bash
python scripts/run_experiment.py --datasets <...>   # produces chunking_results.json + ranked_lists.json
python scripts/analyze_saturation.py --dataset scifact  # produces saturation.json
```

## File layout
- `main.tex` — paper
- `refs.bib` — bibliography (BEIR, MTEB, DPR, reranking, BGE)
