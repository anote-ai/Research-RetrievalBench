"""RetrievalBench: Cross-Domain Ablation of RAG Techniques over Structured Documents."""

__version__ = "0.1.0"

from retrievalbench.core import (
    Domain,
    RAGConfig,
    RetrievalResult,
    BenchmarkRun,
    CHUNKING_STRATEGIES,
    RetrievalBench,
)

__all__ = [
    "Domain",
    "RAGConfig",
    "RetrievalResult",
    "BenchmarkRun",
    "CHUNKING_STRATEGIES",
    "RetrievalBench",
]
