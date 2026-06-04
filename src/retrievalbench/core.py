from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import uuid


class Domain(str, Enum):
    FINANCE = "finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"


CHUNKING_STRATEGIES = ["fixed_512", "sentence", "recursive", "semantic"]


@dataclass
class RAGConfig:
    chunking_strategy: str
    use_reranking: bool = False
    use_metadata: bool = False
    use_query_expansion: bool = False
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        if self.chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError(f"chunking_strategy must be one of {CHUNKING_STRATEGIES}")

    def name(self) -> str:
        parts = [self.chunking_strategy]
        if self.use_reranking:
            parts.append("rerank")
        if self.use_metadata:
            parts.append("meta")
        if self.use_query_expansion:
            parts.append("qexp")
        return "+".join(parts)


@dataclass
class RetrievalResult:
    query_id: str
    retrieved_ids: list[str]
    scores: list[float]
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if len(self.retrieved_ids) != len(self.scores):
            raise ValueError("retrieved_ids and scores must have same length")


@dataclass
class BenchmarkRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: RAGConfig = field(default_factory=lambda: RAGConfig("fixed_512"))
    domain: Domain = Domain.FINANCE
    results: list[RetrievalResult] = field(default_factory=list)

    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)


class RetrievalBench:
    def __init__(self) -> None:
        self.runs: list[BenchmarkRun] = []

    def add_run(self, run: BenchmarkRun) -> None:
        self.runs.append(run)

    def filter_by_domain(self, domain: Domain) -> list[BenchmarkRun]:
        return [r for r in self.runs if r.domain == domain]

    def best_config_by_ndcg(self, ndcg_map: dict[str, float]) -> str:
        return max(ndcg_map, key=ndcg_map.__getitem__)
