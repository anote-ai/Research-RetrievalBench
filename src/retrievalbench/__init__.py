"""RetrievalBench: RAG configuration benchmarking toolkit."""
from .core import (
    Domain,
    CHUNKING_STRATEGIES,
    RETRIEVAL_STRATEGIES,
    RAGConfig,
    HybridConfig,
    LatencySLA,
    RetrievalResult,
    BenchmarkRun,
    RetrievalBench,
)
from .evaluate import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    r_precision,
    evaluate_run,
    compare_domains,
    ablation_table,
)

__all__ = [
    "Domain",
    "CHUNKING_STRATEGIES",
    "RETRIEVAL_STRATEGIES",
    "RAGConfig",
    "HybridConfig",
    "LatencySLA",
    "RetrievalResult",
    "BenchmarkRun",
    "RetrievalBench",
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "average_precision",
    "mean_average_precision",
    "r_precision",
    "evaluate_run",
    "compare_domains",
    "ablation_table",
]
__version__ = "0.2.0"
