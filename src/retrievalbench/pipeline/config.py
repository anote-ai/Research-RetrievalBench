"""Configuration for RetrievalBench real-pipeline experiments.

A single ExperimentConfig describes one (dataset, embedder, chunker, reranker)
cell of the full ablation grid. ExperimentGrid holds the cartesian product of
datasets × embedders × chunkers × rerankers and is what run.py iterates over.

Kept deliberately simple (dataclasses + registry dicts) so configs can also be
loaded from YAML in scripts/run_experiment.py without a heavy dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


# BEIR dataset -> human-readable domain label, used for cross-domain reporting.
# Extend here when adding new datasets (Part C).
DATASET_DOMAINS: dict[str, str] = {
    # --- existing 6 (already have results) ---
    "scifact": "scientific",
    "nfcorpus": "medical",
    "fiqa": "finance",
    "quora": "community",
    "arguana": "argumentation",
    "scidocs": "technical",
    # --- Part C: 5 BEIR domains (loader is BEIR-generic, ready to run) ---
    "touche-2020": "argumentation",
    "climate-fever": "scientific",
    "dbpedia-entity": "encyclopedic",
    "hotpotqa": "multi_hop",
    "msmarco": "web",
    # --- Part C: non-BEIR (needs custom loader in loaders.py) ---
    "cuad": "legal",  # structure-aware chunker main战场, Claim 3 core
}


@dataclass
class EmbedderSpec:
    """One embedding backend.

    `backend` is "openai" or "local". `model` is the HF/OpenAI model id.
    `cost_per_1m_tokens` is USD; local models are 0 (no API cost).
    """
    name: str          # short label, e.g. "openai-3-small", "bge-small"
    backend: str       # "openai" | "local"
    model: str         # "text-embedding-3-small" | "BAAI/bge-small-en-v1.5"
    dim: int
    cost_per_1m_tokens: float = 0.0
    normalize: bool = True


@dataclass
class RerankerSpec:
    """One cross-encoder reranker backend."""
    name: str          # "msmarco-minilm", "quora-distilroberta", "bge-reranker"
    model: str         # HF model id
    # domains this reranker is a sensible default for; None = general
    suited_domains: tuple[str, ...] | None = None


@dataclass
class ChunkerSpec:
    """One chunking strategy.

    `kind` is "generic" (fixed/sentence/recursive/semantic) or
    "structure_aware" (legal_clause/financial_table/medical_section).
    """
    name: str          # "fixed_512", "legal_clause", ...
    kind: str          # "generic" | "structure_aware"
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """One cell of the ablation grid."""
    dataset: str
    embedder: EmbedderSpec
    chunker: ChunkerSpec
    reranker: RerankerSpec | None  # None = no rerank
    k_chunks: int = 50
    k_docs: int = 10

    @property
    def domain(self) -> str:
        return DATASET_DOMAINS.get(self.dataset, "unknown")

    def system_label(self) -> str:
        """Label persisted in results JSON, e.g. 'BM25', 'Dense-bge-small+Rerank-quora'."""
        raise NotImplementedError  # set in run.py where retriever type is known


@dataclass
class ExperimentGrid:
    """Cartesian product of datasets × embedders × chunkers × rerankers.

    Reranker dimension includes None (no rerank) plus each spec.
    For each cell run.py executes BOTH bm25 and dense retrievers.
    """
    datasets: list[str]
    embedders: list[EmbedderSpec]
    chunkers: list[ChunkerSpec]
    rerankers: list[RerankerSpec | None]  # None entry = no-rerank arm

    def cells(self) -> list[ExperimentConfig]:
        cells = []
        for ds, emb, chk, rr in product(self.datasets, self.embedders,
                                        self.chunkers, self.rerankers):
            cells.append(ExperimentConfig(
                dataset=ds, embedder=emb, chunker=chk, reranker=rr,
            ))
        return cells

    def __len__(self) -> int:
        n_rr = len(self.rerankers)
        return len(self.datasets) * len(self.embedders) * len(self.chunkers) * n_rr
