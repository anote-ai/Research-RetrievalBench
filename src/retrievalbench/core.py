from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import uuid


class Domain(str, Enum):
    FINANCE = "finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    NEWS = "news"


CHUNKING_STRATEGIES = ["fixed_512", "sentence", "recursive", "semantic"]
RETRIEVAL_STRATEGIES = ["bm25", "dense", "sparse", "hybrid"]


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
class HybridConfig:
    """Combine multiple retrieval strategies with per-strategy weights.

    Weights are automatically normalised to sum to 1.0 so callers can pass
    raw importance scores.  At least one strategy must have a positive weight.
    """

    strategies: list[str] = field(
        default_factory=lambda: ["bm25", "dense"]
    )
    weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    fusion: Literal["rrf", "linear"] = "rrf"
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    def __post_init__(self) -> None:
        if len(self.strategies) != len(self.weights):
            raise ValueError("strategies and weights must have the same length")
        if not self.strategies:
            raise ValueError("at least one strategy is required")
        for s in self.strategies:
            if s not in RETRIEVAL_STRATEGIES:
                raise ValueError(f"unknown strategy '{s}'; must be one of {RETRIEVAL_STRATEGIES}")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.weights = [w / total for w in self.weights]

    def name(self) -> str:
        pairs = "+".join(
            f"{s}:{w:.2f}" for s, w in zip(self.strategies, self.weights)
        )
        return f"hybrid({pairs},fusion={self.fusion})"

    def fuse_results(
        self,
        ranked_lists: list[list[str]],
    ) -> list[str]:
        """Fuse *ranked_lists* (one per strategy) into a single ranked list.

        Uses Reciprocal Rank Fusion when ``fusion=='rrf'``, otherwise a
        weighted linear combination of reciprocal ranks.
        """
        if len(ranked_lists) != len(self.strategies):
            raise ValueError("ranked_lists must have one list per strategy")

        scores: dict[str, float] = {}
        for ranked, weight in zip(ranked_lists, self.weights):
            for rank, doc_id in enumerate(ranked):
                if self.fusion == "rrf":
                    contrib = weight / (self.rrf_k + rank + 1)
                else:  # linear
                    contrib = weight / (rank + 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + contrib

        return sorted(scores, key=scores.__getitem__, reverse=True)


@dataclass
class LatencySLA:
    """Latency Service-Level Agreement (SLA) for a retrieval pipeline."""

    p50_ms: float  # median latency target
    p95_ms: float  # 95th-percentile latency target
    p99_ms: float  # 99th-percentile latency target

    def __post_init__(self) -> None:
        if not (self.p50_ms <= self.p95_ms <= self.p99_ms):
            raise ValueError("SLA targets must satisfy p50 <= p95 <= p99")

    def evaluate(self, latencies_ms: list[float]) -> dict[str, object]:
        """Return a dict with percentile measurements and pass/fail flags."""
        if not latencies_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "sla_met": True}

        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)

        def percentile(p: float) -> float:
            idx = min(int(p / 100 * n), n - 1)
            return sorted_lat[idx]

        p50 = percentile(50)
        p95 = percentile(95)
        p99 = percentile(99)
        return {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "p50_ok": p50 <= self.p50_ms,
            "p95_ok": p95 <= self.p95_ms,
            "p99_ok": p99 <= self.p99_ms,
            "sla_met": p50 <= self.p50_ms and p95 <= self.p95_ms and p99 <= self.p99_ms,
        }


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

    def latencies(self) -> list[float]:
        return [r.latency_ms for r in self.results]

    def check_sla(self, sla: LatencySLA) -> dict[str, object]:
        """Evaluate this run's latencies against *sla*."""
        return sla.evaluate(self.latencies())


class RetrievalBench:
    def __init__(self) -> None:
        self.runs: list[BenchmarkRun] = []

    def add_run(self, run: BenchmarkRun) -> None:
        self.runs.append(run)

    def filter_by_domain(self, domain: Domain) -> list[BenchmarkRun]:
        return [r for r in self.runs if r.domain == domain]

    def best_config_by_ndcg(self, ndcg_map: dict[str, float]) -> str:
        return max(ndcg_map, key=ndcg_map.__getitem__)

    def domain_summary(self) -> dict[str, int]:
        """Return a count of runs per domain."""
        summary: dict[str, int] = {}
        for run in self.runs:
            summary[run.domain.value] = summary.get(run.domain.value, 0) + 1
        return summary
