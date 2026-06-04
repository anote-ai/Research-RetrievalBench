"""Core data structures and benchmark runner for RetrievalBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Domain(Enum):
    """Document domains for cross-domain ablation."""

    FINANCE = "finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"


@dataclass
class RAGConfig:
    """Configuration for a single RAG pipeline variant."""

    chunking_strategy: str
    use_reranking: bool
    use_metadata: bool
    use_query_expansion: bool
    embedding_model: str

    def __post_init__(self) -> None:
        if self.chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError(
                f"chunking_strategy must be one of {CHUNKING_STRATEGIES}, "
                f"got '{self.chunking_strategy}'"
            )


@dataclass
class RetrievalResult:
    """Result for a single query retrieval."""

    query_id: str
    retrieved_ids: list[str]
    scores: list[float]
    latency_ms: float

    def __post_init__(self) -> None:
        if len(self.retrieved_ids) != len(self.scores):
            raise ValueError("retrieved_ids and scores must have the same length")


@dataclass
class BenchmarkRun:
    """Container for a full benchmark run under one config/domain pair."""

    config: RAGConfig
    domain: Domain
    results: list[RetrievalResult] = field(default_factory=list)

    @property
    def num_queries(self) -> int:
        return len(self.results)

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)


CHUNKING_STRATEGIES: list[str] = ["fixed_512", "sentence", "recursive", "semantic"]


class RetrievalBench:
    """Orchestrates loading domains and running ablation configs."""

    def load_domain(self, domain: Domain) -> list[dict]:
        """Load documents for a given domain.

        Returns a list of document dicts with keys: doc_id, text, metadata.
        Stub implementation — override or monkeypatch for real data.
        """
        return []

    def run_config(self, config: RAGConfig, domain: Domain) -> BenchmarkRun:
        """Run retrieval for all queries in a domain under a given config.

        Returns a BenchmarkRun with per-query RetrievalResults.
        Stub implementation — override with actual retrieval logic.
        """
        return BenchmarkRun(config=config, domain=domain, results=[])
