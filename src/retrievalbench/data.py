from __future__ import annotations
import random
import uuid
from .core import RetrievalResult, Domain

_DOMAIN_WORDS: dict[str, list[str]] = {
    Domain.FINANCE: [
        "revenue", "earnings", "balance sheet", "equity", "liability",
        "cash flow", "operating income", "dividend", "fiscal year", "amortization",
    ],
    Domain.LEGAL: [
        "defendant", "plaintiff", "jurisdiction", "contract", "tort",
        "statute", "precedent", "arbitration", "indemnity", "fiduciary",
    ],
    Domain.MEDICAL: [
        "diagnosis", "prognosis", "clinical trial", "biomarker", "treatment",
        "etiology", "pathology", "pharmacokinetics", "adverse event", "dosage",
    ],
    Domain.TECHNICAL: [
        "algorithm", "latency", "throughput", "distributed system", "API",
        "embedding", "transformer", "fine-tuning", "inference", "benchmark",
    ],
}


def make_corpus(n_docs: int = 100, seed: int = 42) -> list[dict]:
    """Generate a synthetic corpus of documents.

    Returns:
        List of {"doc_id": str, "text": str, "domain": str}.
    """
    rng = random.Random(seed)
    domains = list(Domain)
    corpus = []
    for i in range(n_docs):
        domain = rng.choice(domains)
        words = _DOMAIN_WORDS[domain]
        sentences = [
            f"This document discusses {rng.choice(words)} and {rng.choice(words)}."
            for _ in range(rng.randint(3, 7))
        ]
        corpus.append(
            {
                "doc_id": f"doc_{i:04d}",
                "text": " ".join(sentences),
                "domain": domain.value,
            }
        )
    return corpus


def make_queries(
    n: int = 20,
    corpus: list[dict] | None = None,
    seed: int = 42,
) -> tuple[list[dict], dict[str, set[str]]]:
    """Generate synthetic queries and ground-truth relevance judgements.

    Args:
        n: Number of queries to generate.
        corpus: Corpus to sample relevant docs from.
        seed: Random seed.

    Returns:
        Tuple of (queries, qrels) where qrels maps query_id -> set[doc_id].
    """
    if corpus is None:
        corpus = make_corpus(seed=seed)
    rng = random.Random(seed)
    queries: list[dict] = []
    qrels: dict[str, set[str]] = {}
    for i in range(n):
        query_id = f"q_{i:04d}"
        domain = rng.choice(list(Domain))
        words = _DOMAIN_WORDS[domain]
        query_text = f"What is the relationship between {rng.choice(words)} and {rng.choice(words)}?"
        # Sample 1-3 relevant docs from the corpus in the same domain
        domain_docs = [d for d in corpus if d["domain"] == domain.value]
        n_relevant = rng.randint(1, min(3, len(domain_docs)))
        relevant = rng.sample(domain_docs, n_relevant)
        qrels[query_id] = {d["doc_id"] for d in relevant}
        queries.append({"query_id": query_id, "text": query_text, "domain": domain.value})
    return queries, qrels


def make_retrieval_result(
    query_id: str,
    corpus: list[dict],
    relevant_ids: set[str],
    recall: float = 0.7,
) -> RetrievalResult:
    """Simulate a retrieval result with the given recall level.

    Relevant docs are inserted into the top-k positions with probability
    proportional to *recall*, and the rest are filled with random docs.
    """
    rng = random.Random(hash(query_id))
    all_ids = [d["doc_id"] for d in corpus]
    non_relevant = [d for d in all_ids if d not in relevant_ids]

    # Determine how many relevant docs to include
    n_rel_include = max(1, round(len(relevant_ids) * recall))
    included_rel = rng.sample(list(relevant_ids), min(n_rel_include, len(relevant_ids)))

    # Fill to 10 results with non-relevant
    filler = rng.sample(non_relevant, min(10 - len(included_rel), len(non_relevant)))
    retrieved = included_rel + filler
    rng.shuffle(retrieved)
    retrieved = retrieved[:10]

    scores = sorted([rng.random() for _ in retrieved], reverse=True)
    return RetrievalResult(
        query_id=query_id,
        retrieved_ids=retrieved,
        scores=scores,
        latency_ms=rng.uniform(10.0, 200.0),
    )
