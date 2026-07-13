"""Dataset loaders. Currently BEIR via HuggingFace; CUAD (non-BEIR) added in Part C.

Returns a uniform (corpus, queries, qrels) tuple:
  corpus:   list[{doc_id, text, corpus_position}]
  queries:  list[{query_id, text}]
  qrels:    dict[query_id -> set[doc_id]]
"""
from __future__ import annotations

import os
from collections import defaultdict

from .config import DATASET_DOMAINS


# BEIR config key -> HuggingFace repo name suffix. Most match the key directly;
# touche-2020 is published as webis-touche2020. Add overrides here as needed.
BEIR_HF_NAME: dict[str, str] = {
    "touche-2020": "webis-touche2020",
}


def load_beir(dataset: str):
    """Load a BEIR dataset from HuggingFace (BeIR/{name} + {name}-qrels).

    Honors RB_MAX_CORPUS / RB_MAX_QUERIES env vars for subsampling (keeps all
    qrels-relevant docs when subsampling corpus, seed 42).
    """
    from datasets import load_dataset

    hf_name = BEIR_HF_NAME.get(dataset, dataset)
    domain = DATASET_DOMAINS.get(dataset, "unknown")
    print(f"[{domain.upper()}] Loading {dataset} (BeIR/{hf_name}) from HuggingFace...")
    corpus_ds = load_dataset(f"BeIR/{hf_name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{hf_name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{hf_name}-qrels", split="test")

    n_total = len(corpus_ds)

    queries = [{"query_id": str(r["_id"]), "text": r["text"]} for r in queries_ds]

    qrels: dict[str, set[str]] = defaultdict(set)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    qrels = dict(qrels)
    queries = [q for q in queries if q["query_id"] in qrels]

    # Subsample at the Arrow level when possible (msmarco=8.8M, hotpotqa=5M):
    # materializing the full corpus via list() would OOM. Read only the _id
    # column (fast), pick the indices to keep (all relevant + random sample),
    # then select() those rows before building dicts.
    max_c = int(os.getenv("RB_MAX_CORPUS", "0"))
    if max_c and n_total > max_c:
        import random as _rnd
        all_ids = corpus_ds["_id"]  # column read, much cheaper than full materialize
        relevant = set().union(*qrels.values()) if qrels else set()
        keep_idx = [i for i, _id in enumerate(all_ids) if str(_id) in relevant]
        rest_idx = [i for i, _id in enumerate(all_ids) if str(_id) not in relevant]
        _rnd.Random(42).shuffle(rest_idx)
        chosen = sorted(keep_idx + rest_idx[: max(0, max_c - len(keep_idx))])
        raw_corpus = corpus_ds.select(chosen)
        n = n_total  # corpus_position normalized against full corpus length
        corpus = [
            {
                "doc_id": str(r["_id"]),
                "text": (r["title"] + " " + r["text"]).strip(),
                "corpus_position": chosen[i] / max(n - 1, 1),
            }
            for i, r in enumerate(raw_corpus)
        ]
        print(f"  Subsampled corpus to {len(corpus)} docs (of {n_total}, seed 42)")
    else:
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

    max_q = int(os.getenv("RB_MAX_QUERIES", "0"))
    if max_q and len(queries) > max_q:
        import random as _rnd
        _rnd.Random(42).shuffle(queries)
        queries = queries[:max_q]
        print(f"  Subsampled to {max_q} queries (seed 42)")

    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)}")
    return corpus, queries, qrels


def load_dataset(dataset: str):
    """Dispatch by dataset name. BEIR datasets go through load_beir; CUAD
    (non-BEIR legal contracts) has a dedicated loader."""
    if dataset == "cuad":
        return load_cuad()
    if dataset in DATASET_DOMAINS:
        return load_beir(dataset)
    raise ValueError(f"unknown dataset: {dataset} (add a loader in loaders.py)")


# ---------------------------------------------------------------------------
# CUAD — legal contracts (non-BEIR). Claim 3 structure-aware chunker main战场.
# ---------------------------------------------------------------------------

# The 41 CUAD clause types double as natural language queries for doc-level
# retrieval: "Identify the agreement's expiration date." etc. A contract is
# relevant to a clause-query if it has a non-empty answer span for that clause.
_CUAD_CLAUSE_QUERIES = [
    "Identify the agreement's expiration date.",
    "What is the effective date of the agreement?",
    "Which party is the one granting the license?",
    "What is the name of the party receiving the license?",
    "Is there a clause restricting the licensee from competing with the licensor?",
    "Is there a non-compete clause restricting the licensee from competing?",
    "Is there a clause prohibiting the licensee from soliciting employees?",
    "Is there a clause prohibiting the licensee from soliciting customers?",
    "Is there a clause prohibiting the licensee from reverse engineering?",
    "Is there an indemnification clause requiring one party to indemnify the other?",
    "Is there a clause requiring the licensor to indemnify the licensee?",
    "Is there a clause requiring the licensee to indemnify the licensor?",
    "Is there an insurance clause requiring one party to maintain insurance?",
    "Is there a clause requiring the licensee to maintain insurance?",
    "Is there a limitation of liability clause capping damages?",
    "What is the cap on damages in the limitation of liability clause?",
    "Is there a warranty disclaimer clause?",
    "Is there a clause disclaiming all warranties?",
    "Is there a clause requiring notice in case of breach?",
    "Is there a confidentiality clause?",
    "Is there a clause requiring confidentiality of the agreement terms?",
    "Is there a clause requiring confidentiality of information shared between parties?",
    "Is there a termination for convenience clause?",
    "Is there a clause allowing termination for convenience?",
    "Is there a termination for breach clause?",
    "Is there a clause allowing termination for material breach?",
    "Is there a clause requiring cure period before termination for breach?",
    "Is there an assignment clause restricting assignment of the agreement?",
    "Is there a clause requiring consent for assignment?",
    "Is there a clause governing which law applies to the agreement?",
    "What is the governing law of the agreement?",
    "Is there a clause specifying the venue for disputes?",
    "What is the venue for disputes under the agreement?",
    "Is there an arbitration clause?",
    "Is there a clause requiring disputes to be resolved by arbitration?",
    "Is there a force majeure clause?",
    "Is there a clause excusing performance due to force majeure events?",
    "Is there a clause requiring notice of force majeure events?",
    "Is there a clause permitting the agreement to be amended only in writing?",
    "Is there a clause requiring amendments to be in writing?",
    "Is there a clause specifying the entire agreement between the parties?",
]


def load_cuad():
    """Load CUAD v1 as a doc-level retrieval task.

    corpus: 510 commercial contracts (each a full text doc).
    queries: the 41 CUAD clause types as natural-language questions.
    qrels: a contract is relevant to a clause-query iff it has a non-empty
           answer span for that clause (CUAD answers are spans; empty span =
           "not present in this contract").

    This is the natural setup for testing structure-aware legal chunking:
    generic fixed-size chunking will split clause boundaries, while
    legal_clause chunking preserves them.
    """
    from datasets import load_dataset

    print("[LEGAL] Loading CUAD from HuggingFace...")
    # CUAD is distributed as a single split with one row per contract
    ds = load_dataset("theatticusproject/cuad", split="train")

    corpus = []
    # CUAD rows: 'title', 'text' (the contract), 'labels' (per-clause answer spans)
    # Field names vary by mirror; handle both 'labels' and the parquet schema.
    for i, row in enumerate(ds):
        text = row.get("text") or row.get("contract") or ""
        doc_id = str(row.get("title") or f"contract_{i}")
        corpus.append({
            "doc_id": doc_id,
            "text": text,
            "corpus_position": i / max(len(ds) - 1, 1),
        })

    # Build qrels from the answer spans.
    # In CUAD's HF schema, each row has a 'labels' list aligned to the 41 clause
    # types; an empty string or empty list means the clause is absent.
    n_clauses = len(_CUAD_CLAUSE_QUERIES)
    queries = [{"query_id": f"clause_{i}", "text": q}
               for i, q in enumerate(_CUAD_CLAUSE_QUERIES)]
    qrels: dict[str, set[str]] = {f"clause_{i}": set() for i in range(n_clauses)}

    for row in ds:
        doc_id = str(row.get("title") or "")
        labels = row.get("labels", [])
        # labels may be a list of {text: [...]} dicts or a list of strings
        for i in range(min(n_clauses, len(labels))):
            entry = labels[i]
            # treat any non-empty answer as relevance
            if isinstance(entry, dict):
                spans = entry.get("text", []) or entry.get("answer", [])
                if spans and any(s.strip() for s in spans):
                    qrels[f"clause_{i}"].add(doc_id)
            elif isinstance(entry, str):
                if entry.strip():
                    qrels[f"clause_{i}"].add(doc_id)
            elif isinstance(entry, list):
                if any(s.strip() for s in entry if isinstance(s, str)):
                    qrels[f"clause_{i}"].add(doc_id)

    # drop clause-queries with no relevant docs (clause absent across all contracts)
    qrels = {qid: docs for qid, docs in qrels.items() if docs}
    queries = [q for q in queries if q["query_id"] in qrels]

    import os as _os
    max_c = int(_os.getenv("RB_MAX_CORPUS", "0"))
    if max_c and len(corpus) > max_c:
        import random as _rnd
        relevant = set().union(*qrels.values())
        keep = [d for d in corpus if d["doc_id"] in relevant]
        rest = [d for d in corpus if d["doc_id"] not in relevant]
        _rnd.Random(42).shuffle(rest)
        corpus = keep + rest[: max(0, max_c - len(keep))]
        print(f"  Subsampled corpus to {len(corpus)} contracts (seed 42)")

    print(f"  Corpus: {len(corpus)} contracts | Queries: {len(queries)} clause-types")
    return corpus, queries, qrels
