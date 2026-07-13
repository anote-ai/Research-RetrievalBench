"""RetrievalBench real-pipeline package.

Configuration-driven experiment engine for BEIR + non-BEIR datasets, supporting
multiple embedders (Claim 2 / CSI-ESI), multiple rerankers (incl. quora fix),
structure-aware chunkers (Claim 3), and per-query ranked-list storage
(context-saturation, B5).

The old synthetic engine in retrievalbench.core / .data is kept for the demo
and unit tests; this package is the production-research path.
"""
from .config import (
    DATASET_DOMAINS, EmbedderSpec, RerankerSpec, ChunkerSpec,
    ExperimentConfig, ExperimentGrid,
)
from .loaders import load_dataset, load_beir
from .chunkers import build_chunk_corpus, GENERIC_CHUNKERS, STRUCTURE_AWARE_CHUNKERS
from .retrievers import (
    Embedder, make_embedder, EMBEDDER_REGISTRY,
    retrieve_bm25, retrieve_dense, RetrievalOutput,
)
from .rerankers import (
    rerank, RERANKER_REGISTRY, default_reranker_for, RerankResult,
)
from .metrics import evaluate_config, csi_esi
from .run import run_cell, save_results, results_dir

__all__ = [
    "DATASET_DOMAINS", "EmbedderSpec", "RerankerSpec", "ChunkerSpec",
    "ExperimentConfig", "ExperimentGrid",
    "load_dataset", "load_beir",
    "build_chunk_corpus", "GENERIC_CHUNKERS", "STRUCTURE_AWARE_CHUNKERS",
    "Embedder", "make_embedder", "EMBEDDER_REGISTRY",
    "retrieve_bm25", "retrieve_dense", "RetrievalOutput",
    "rerank", "RERANKER_REGISTRY", "default_reranker_for", "RerankResult",
    "evaluate_config", "csi_esi",
    "run_cell", "save_results", "results_dir",
]
