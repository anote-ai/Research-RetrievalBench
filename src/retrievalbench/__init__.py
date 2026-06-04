"""RetrievalBench: RAG configuration benchmarking toolkit."""
from .core import (
    Domain,
    CHUNKING_STRATEGIES,
    RAGConfig,
    RetrievalResult,
    BenchmarkRun,
    RetrievalBench,
)
from .evaluate import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    evaluate_run,
    ablation_table,
)

__all__ = [
    "Domain",
    "CHUNKING_STRATEGIES",
    "RAGConfig",
    "RetrievalResult",
    "BenchmarkRun",
    "RetrievalBench",
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "evaluate_run",
    "ablation_table",
]
__version__ = "0.1.0"
