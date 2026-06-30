from __future__ import annotations
import hashlib
import random
from .core import RetrievalResult, Domain

# ---------------------------------------------------------------------------
# Domain vocabulary — richer, realistic terminology per domain
# ---------------------------------------------------------------------------
_DOMAIN_WORDS: dict[str, list[str]] = {
    Domain.FINANCE: [
        "revenue", "earnings per share", "balance sheet", "shareholders equity",
        "total liabilities", "cash flow from operations", "operating income",
        "dividend yield", "fiscal year", "goodwill amortization",
        "debt-to-equity ratio", "net income margin", "working capital",
        "capital expenditure", "return on equity",
    ],
    Domain.LEGAL: [
        "defendant", "plaintiff", "subject matter jurisdiction", "breach of contract",
        "tortious interference", "statute of limitations", "stare decisis",
        "arbitration clause", "indemnification", "fiduciary duty",
        "force majeure", "liquidated damages", "material adverse change",
        "representations and warranties", "governing law",
    ],
    Domain.MEDICAL: [
        "differential diagnosis", "prognosis", "randomized controlled trial",
        "biomarker expression", "first-line treatment", "etiology",
        "histopathology", "pharmacokinetics", "adverse event reporting",
        "dosage regimen", "comorbidity", "sensitivity and specificity",
        "intention-to-treat analysis", "hazard ratio", "confidence interval",
    ],
    Domain.TECHNICAL: [
        "algorithmic complexity", "end-to-end latency", "throughput",
        "distributed consensus", "REST API", "dense embeddings",
        "transformer architecture", "fine-tuning", "inference throughput",
        "benchmark suite", "vector similarity search", "semantic chunking",
        "retrieval-augmented generation", "cross-encoder reranking",
        "approximate nearest neighbour",
    ],
    Domain.SCIENTIFIC: [
        "hypothesis", "experimental design", "statistical significance",
        "p-value", "effect size", "meta-analysis", "systematic review",
        "peer review", "replication crisis", "control group",
        "double-blind study", "confounding variable", "sample size",
        "measurement error", "cohort study",
    ],
    Domain.NEWS: [
        "breaking news", "press release", "editorial", "investigative journalism",
        "source credibility", "headline", "attribution", "fact-checking",
        "misinformation", "news cycle", "wire service", "op-ed",
        "retraction", "embargo", "on background",
    ],
}

# Difficulty labels and their effect on n_relevant (more relevant = easier)
_DIFFICULTY_LEVELS = {
    "easy": (2, 4),     # (min_relevant, max_relevant)
    "medium": (1, 3),
    "hard": (1, 2),
}

QUERY_TYPES = ["single_hop", "multi_hop", "temporal"]

_QUERY_TEMPLATES: dict[str, list[str]] = {
    "single_hop": [
        "What is the relationship between {w1} and {w2}?",
        "How does {w1} affect {w2} in practice?",
        "Compare and contrast {w1} with {w2}.",
        "What are the implications of {w1} on {w2}?",
        "Explain the role of {w1} when assessing {w2}.",
    ],
    "multi_hop": [
        "How does {w1} relate to {w2}, and what does that imply for {w3}?",
        "Trace the connection from {w1} through {w2} to {w3}.",
        "What is the combined effect of {w1} and {w2} on {w3}?",
        "Given {w1} and {w2}, how should one evaluate {w3}?",
        "Explain how {w1}, {w2}, and {w3} interact in practice.",
    ],
    "temporal": [
        "As of {year}, what was the status of {w1} with respect to {w2}?",
        "How did {w1} change between {year} and the present in terms of {w2}?",
        "What were the {year} guidelines for {w1} given {w2}?",
        "In {year}, how was {w1} defined relative to {w2}?",
        "What developments in {w1} occurred around {year} affecting {w2}?",
    ],
}

_TEMPORAL_YEARS = [2019, 2020, 2021, 2022, 2023]


def _coerce_domain(domain: Domain | str) -> Domain:
    if isinstance(domain, Domain):
        return domain
    return Domain(domain)


def _normalize_domains(
    domains: list[Domain | str] | tuple[Domain | str, ...] | None,
) -> list[Domain]:
    if domains is None:
        return list(Domain)
    normalized = [_coerce_domain(domain) for domain in domains]
    if not normalized:
        raise ValueError("domains must include at least one domain")
    return normalized


def make_corpus(
    n_docs: int = 100,
    seed: int = 42,
    domains: list[Domain | str] | tuple[Domain | str, ...] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    domain_choices = _normalize_domains(domains)
    corpus = []
    for i in range(n_docs):
        domain = rng.choice(domain_choices)
        words = _DOMAIN_WORDS[domain]
        n_sentences = rng.randint(3, 8)
        sentences = [
            f"This document discusses {rng.choice(words)} and {rng.choice(words)}."
            for _ in range(n_sentences)
        ]
        corpus.append(
            {
                "doc_id": f"doc_{i:04d}",
                "text": " ".join(sentences),
                "domain": domain.value,
                "word_count": sum(len(s.split()) for s in sentences),
                "corpus_position": i / max(n_docs - 1, 1),
            }
        )
    return corpus


def make_queries(
    n: int = 20,
    corpus: list[dict] | None = None,
    seed: int = 42,
    difficulty: str = "medium",
    domain: Domain | str | None = None,
    query_id_prefix: str = "q",
    query_type: str = "single_hop",
) -> tuple[list[dict], dict[str, set[str]]]:
    """Generate queries with a specified difficulty level and query type.

    Args:
        n: Number of queries to generate.
        corpus: Document corpus; generated if not supplied.
        seed: Random seed for reproducibility.
        difficulty: One of 'easy', 'medium', 'hard'. Controls the number of
            relevant documents assigned to each query.
        domain: Optional fixed domain for all generated queries.
        query_id_prefix: Prefix used to keep query ids unique across generated sets.
        query_type: One of 'single_hop', 'multi_hop', 'temporal'.
            - single_hop: one passage suffices to answer the query.
            - multi_hop: answer requires synthesising 2-3 relevant passages;
              min_relevant is raised to 2.
            - temporal: query references a specific year; relevant docs are
              sampled with a recency bias toward that year.

    Returns:
        (queries, qrels) where qrels maps query_id -> set of relevant doc_ids.
    """
    if difficulty not in _DIFFICULTY_LEVELS:
        raise ValueError(f"difficulty must be one of {list(_DIFFICULTY_LEVELS)}")
    if query_type not in QUERY_TYPES:
        raise ValueError(f"query_type must be one of {QUERY_TYPES}")

    fixed_domain = _coerce_domain(domain) if domain is not None else None
    if corpus is None:
        corpus_domains = (fixed_domain,) if fixed_domain is not None else None
        corpus = make_corpus(seed=seed, domains=corpus_domains)
    domain_choices = _domains_available_in_corpus(corpus, fixed_domain)
    rng = random.Random(seed)
    min_rel, max_rel = _DIFFICULTY_LEVELS[difficulty]

    # multi_hop needs at least 2 relevant docs to model the evidence chain
    if query_type == "multi_hop":
        min_rel = max(min_rel, 2)
        max_rel = max(max_rel, 3)

    queries: list[dict] = []
    qrels: dict[str, set[str]] = {}
    for i in range(n):
        query_id = f"{query_id_prefix}_{i:04d}"
        query_domain = fixed_domain or rng.choice(domain_choices)
        words = _DOMAIN_WORDS[query_domain]
        template = rng.choice(_QUERY_TEMPLATES[query_type])

        if query_type == "temporal":
            year = rng.choice(_TEMPORAL_YEARS)
            query_text = template.format(w1=rng.choice(words), w2=rng.choice(words), year=year)
        elif query_type == "multi_hop":
            query_text = template.format(
                w1=rng.choice(words), w2=rng.choice(words), w3=rng.choice(words)
            )
        else:
            query_text = template.format(w1=rng.choice(words), w2=rng.choice(words))

        domain_docs = [d for d in corpus if d["domain"] == query_domain.value]
        max_available = min(max_rel, len(domain_docs))
        min_available = min(min_rel, max_available)
        n_relevant = rng.randint(min_available, max_available)
        relevant = rng.sample(domain_docs, max(1, n_relevant))
        qrels[query_id] = {d["doc_id"] for d in relevant}
        queries.append(
            {
                "query_id": query_id,
                "text": query_text,
                "domain": query_domain.value,
                "difficulty": difficulty,
                "query_type": query_type,
            }
        )
    return queries, qrels


def _domains_available_in_corpus(
    corpus: list[dict],
    fixed_domain: Domain | None,
) -> list[Domain]:
    available_values = {doc["domain"] for doc in corpus}
    if fixed_domain is not None:
        if fixed_domain.value not in available_values:
            raise ValueError(f"corpus does not contain documents for domain '{fixed_domain.value}'")
        return [fixed_domain]

    domain_choices = [domain for domain in Domain if domain.value in available_values]
    if not domain_choices:
        raise ValueError("corpus must contain at least one document with a known domain")
    return domain_choices


def make_nuggets(
    qrels: dict[str, set[str]],
    seed: int = 42,
    nugget_fraction: float = 0.5,
) -> dict[str, set[str]]:
    """Designate a subset of each query's relevant docs as nuggets.

    Nuggets represent atomic fact units — the specific documents that contain
    the key claim needed to answer the query. A retriever may score high on
    NDCG (retrieving many peripheral relevant docs) while missing nuggets.

    Args:
        qrels: Standard qrels mapping query_id -> set of relevant doc_ids.
        seed: Random seed for reproducibility.
        nugget_fraction: Fraction of relevant docs to designate as nuggets.
            At least one nugget is always assigned per query.

    Returns:
        nuggets dict mapping query_id -> set of nugget doc_ids (subset of qrels).
    """
    rng = random.Random(seed)
    nuggets: dict[str, set[str]] = {}
    for query_id, relevant_ids in qrels.items():
        rel_list = sorted(relevant_ids)
        n_nuggets = max(1, round(len(rel_list) * nugget_fraction))
        nuggets[query_id] = set(rng.sample(rel_list, min(n_nuggets, len(rel_list))))
    return nuggets


def make_retrieval_result(
    query_id: str,
    corpus: list[dict],
    relevant_ids: set[str],
    recall: float = 0.7,
    seed_salt: str = "",
    relevance_bias: float = 0.0,
    position_penalty_scale: float = 0.0,
) -> RetrievalResult:
    seed_material = f"{query_id}:{seed_salt}:{recall:.4f}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    doc_position = {d["doc_id"]: d.get("corpus_position", 0.0) for d in corpus}
    all_ids = [d["doc_id"] for d in corpus]
    non_relevant = [d for d in all_ids if d not in relevant_ids]
    n_rel_include = max(1, round(len(relevant_ids) * recall))
    included_rel = rng.sample(list(relevant_ids), min(n_rel_include, len(relevant_ids)))
    filler = rng.sample(non_relevant, min(10 - len(included_rel), len(non_relevant)))
    candidates = included_rel + filler
    bias = max(0.0, min(1.0, relevance_bias))
    scale = max(0.0, position_penalty_scale)
    scored = []
    for doc_id in candidates:
        base_score = rng.random() + (bias if doc_id in relevant_ids else 0.0)
        if scale > 0.0 and doc_id in relevant_ids:
            # Primacy bias: late-positioned relevant docs score lower
            penalty = scale * doc_position.get(doc_id, 0.0)
            base_score -= penalty
        scored.append((doc_id, min(1.0, base_score)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    retrieved = [doc_id for doc_id, _ in scored[:10]]
    scores = [score for _, score in scored[:10]]
    return RetrievalResult(
        query_id=query_id,
        retrieved_ids=retrieved,
        scores=scores,
        latency_ms=rng.uniform(10.0, 200.0),
    )
