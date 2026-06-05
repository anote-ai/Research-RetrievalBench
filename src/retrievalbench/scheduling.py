from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class PowerMode:
    """A hardware operating point used for generation and retrieval work."""

    name: str
    frequency_scale: float
    generation_power_w: float
    retrieval_power_w: float

    def __post_init__(self) -> None:
        if self.frequency_scale <= 0:
            raise ValueError("frequency_scale must be positive")
        if self.generation_power_w <= 0 or self.retrieval_power_w <= 0:
            raise ValueError("power values must be positive")


@dataclass(frozen=True)
class HardwareProfile:
    """Simple latency and power model for hardware-aware scheduling experiments."""

    name: str
    power_modes: tuple[PowerMode, ...]
    prefill_latency_ms: float
    generation_latency_ms_per_token: float
    retrieval_base_latency_ms: float
    retrieval_latency_ms_per_doc: float

    def __post_init__(self) -> None:
        if not self.power_modes:
            raise ValueError("at least one power mode is required")
        names = [mode.name for mode in self.power_modes]
        if len(names) != len(set(names)):
            raise ValueError("power mode names must be unique")
        if self.prefill_latency_ms < 0:
            raise ValueError("prefill_latency_ms must be non-negative")
        if self.generation_latency_ms_per_token <= 0:
            raise ValueError("generation_latency_ms_per_token must be positive")
        if self.retrieval_base_latency_ms < 0 or self.retrieval_latency_ms_per_doc <= 0:
            raise ValueError("retrieval latency values must be valid")

    def get_mode(self, name: str) -> PowerMode:
        for mode in self.power_modes:
            if mode.name == name:
                return mode
        choices = [mode.name for mode in self.power_modes]
        raise ValueError(f"unknown power mode '{name}'; must be one of {choices}")

    def generation_latency_ms(self, mode_name: str) -> float:
        mode = self.get_mode(mode_name)
        return self.generation_latency_ms_per_token / mode.frequency_scale

    def retrieval_latency_ms(self, top_k: int, mode_name: str) -> float:
        if top_k <= 0:
            raise ValueError("top_k must be positive for retrieval")
        mode = self.get_mode(mode_name)
        raw_latency = self.retrieval_base_latency_ms + top_k * self.retrieval_latency_ms_per_doc
        return raw_latency / mode.frequency_scale


EDGE_GPU = HardwareProfile(
    name="edge_gpu",
    power_modes=(
        PowerMode("eco", frequency_scale=0.70, generation_power_w=32.0, retrieval_power_w=42.0),
        PowerMode(
            "balanced",
            frequency_scale=1.00,
            generation_power_w=52.0,
            retrieval_power_w=68.0,
        ),
        PowerMode("turbo", frequency_scale=1.25, generation_power_w=78.0, retrieval_power_w=105.0),
    ),
    prefill_latency_ms=95.0,
    generation_latency_ms_per_token=24.0,
    retrieval_base_latency_ms=14.0,
    retrieval_latency_ms_per_doc=3.2,
)

SERVER_GPU = HardwareProfile(
    name="server_gpu",
    power_modes=(
        PowerMode("eco", frequency_scale=0.75, generation_power_w=90.0, retrieval_power_w=115.0),
        PowerMode(
            "balanced",
            frequency_scale=1.00,
            generation_power_w=135.0,
            retrieval_power_w=170.0,
        ),
        PowerMode("turbo", frequency_scale=1.35, generation_power_w=220.0, retrieval_power_w=285.0),
    ),
    prefill_latency_ms=55.0,
    generation_latency_ms_per_token=12.0,
    retrieval_base_latency_ms=8.0,
    retrieval_latency_ms_per_doc=1.9,
)

CPU_ONLY = HardwareProfile(
    name="cpu_only",
    power_modes=(
        PowerMode("eco", frequency_scale=0.65, generation_power_w=18.0, retrieval_power_w=22.0),
        PowerMode(
            "balanced",
            frequency_scale=1.00,
            generation_power_w=32.0,
            retrieval_power_w=38.0,
        ),
        PowerMode("turbo", frequency_scale=1.20, generation_power_w=52.0, retrieval_power_w=65.0),
    ),
    prefill_latency_ms=180.0,
    generation_latency_ms_per_token=48.0,
    retrieval_base_latency_ms=28.0,
    retrieval_latency_ms_per_doc=7.5,
)

HARDWARE_PROFILES = {
    EDGE_GPU.name: EDGE_GPU,
    SERVER_GPU.name: SERVER_GPU,
    CPU_ONLY.name: CPU_ONLY,
}


@dataclass(frozen=True)
class RetrievalSignal:
    """Per-token signals used by adaptive retrieval policies."""

    token_index: int
    uncertainty: float
    semantic_drift: float
    retrieval_score: float
    document_overlap: float

    def __post_init__(self) -> None:
        if self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        values = {
            "uncertainty": self.uncertainty,
            "semantic_drift": self.semantic_drift,
            "retrieval_score": self.retrieval_score,
            "document_overlap": self.document_overlap,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RetrievalAction:
    """Scheduler output for one generation step."""

    should_retrieve: bool
    top_k: int = 0
    power_mode: str = "balanced"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.should_retrieve and self.top_k <= 0:
            raise ValueError("retrieval actions require top_k > 0")
        if not self.should_retrieve and self.top_k != 0:
            raise ValueError("non-retrieval actions must use top_k=0")


@dataclass
class FixedStrideScheduler:
    """Retrieve every fixed number of generated tokens."""

    stride: int = 16
    top_k: int = 6
    power_mode: str = "balanced"

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def reset(self) -> None:
        return None

    def name(self) -> str:
        return f"fixed_stride_{self.stride}_k{self.top_k}_{self.power_mode}"

    def decide(self, signal: RetrievalSignal) -> RetrievalAction:
        if signal.token_index == 0 or signal.token_index % self.stride == 0:
            return RetrievalAction(
                should_retrieve=True,
                top_k=self.top_k,
                power_mode=self.power_mode,
                reasons=("fixed_stride",),
            )
        return RetrievalAction(should_retrieve=False)


@dataclass
class AdaptiveRetrievalScheduler:
    """Trigger retrieval and choose depth/power mode from model and retrieval signals."""

    min_interval: int = 4
    max_interval: int = 24
    base_top_k: int = 4
    max_top_k: int = 12
    uncertainty_threshold: float = 0.62
    semantic_drift_threshold: float = 0.55
    retrieval_score_threshold: float = 0.48
    overlap_threshold: float = 0.42
    score_decay_threshold: float = 0.18
    eco_mode: str = "eco"
    balanced_mode: str = "balanced"
    turbo_mode: str = "turbo"
    _last_retrieval_token: int = field(default=-1, init=False)
    _last_retrieval_score: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if self.max_interval <= 0 or self.max_interval < self.min_interval:
            raise ValueError("max_interval must be positive and >= min_interval")
        if self.base_top_k <= 0 or self.max_top_k < self.base_top_k:
            raise ValueError("top-k bounds must satisfy 0 < base_top_k <= max_top_k")
        thresholds = [
            self.uncertainty_threshold,
            self.semantic_drift_threshold,
            self.retrieval_score_threshold,
            self.overlap_threshold,
            self.score_decay_threshold,
        ]
        if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError("thresholds must be in [0, 1]")

    def reset(self) -> None:
        self._last_retrieval_token = -1
        self._last_retrieval_score = None

    def name(self) -> str:
        return f"adaptive_k{self.base_top_k}-{self.max_top_k}"

    def decide(self, signal: RetrievalSignal) -> RetrievalAction:
        reasons: list[str] = []
        first_retrieval = self._last_retrieval_token < 0
        tokens_since = (
            signal.token_index + 1
            if first_retrieval
            else signal.token_index - self._last_retrieval_token
        )

        if first_retrieval:
            reasons.append("initial")
        else:
            if tokens_since < self.min_interval:
                return RetrievalAction(should_retrieve=False)
            if tokens_since >= self.max_interval:
                reasons.append("max_interval")

        if signal.uncertainty >= self.uncertainty_threshold:
            reasons.append("uncertainty")
        if signal.semantic_drift >= self.semantic_drift_threshold:
            reasons.append("semantic_drift")
        if signal.retrieval_score <= self.retrieval_score_threshold:
            reasons.append("low_retrieval_score")
        if signal.document_overlap <= self.overlap_threshold:
            reasons.append("low_document_overlap")
        if self._last_retrieval_score is not None:
            score_decay = self._last_retrieval_score - signal.retrieval_score
            if score_decay >= self.score_decay_threshold:
                reasons.append("score_decay")

        if not reasons:
            return RetrievalAction(should_retrieve=False)

        top_k = self._choose_top_k(signal, tokens_since)
        power_mode = self._choose_power_mode(signal, top_k)
        self._last_retrieval_token = signal.token_index
        self._last_retrieval_score = signal.retrieval_score
        return RetrievalAction(
            should_retrieve=True,
            top_k=top_k,
            power_mode=power_mode,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _choose_top_k(self, signal: RetrievalSignal, tokens_since: int) -> int:
        risk = _signal_risk(signal)
        span = self.max_top_k - self.base_top_k
        top_k = self.base_top_k + round(span * risk)
        if tokens_since >= int(0.8 * self.max_interval):
            top_k += 1
        return max(self.base_top_k, min(self.max_top_k, top_k))

    def _choose_power_mode(self, signal: RetrievalSignal, top_k: int) -> str:
        risk = _signal_risk(signal)
        depth_ratio = top_k / self.max_top_k
        if risk >= 0.85 or depth_ratio >= 0.95:
            return self.turbo_mode
        if risk <= 0.60 and top_k <= self.base_top_k + 2:
            return self.eco_mode
        return self.balanced_mode


@dataclass
class ScheduledRun:
    """Aggregate metrics from a simulated scheduled generation run."""

    scheduler_name: str
    hardware_name: str
    actions: list[RetrievalAction]
    ttft_ms: float
    mean_tbt_ms: float
    total_latency_ms: float
    total_energy_j: float
    generation_energy_j: float
    retrieval_energy_j: float
    quality_score: float
    retrieval_calls: int
    avg_top_k: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler_name,
            "hardware": self.hardware_name,
            "retrieval_calls": self.retrieval_calls,
            "avg_top_k": self.avg_top_k,
            "ttft_ms": self.ttft_ms,
            "mean_tbt_ms": self.mean_tbt_ms,
            "latency_ms": self.total_latency_ms,
            "energy_j": self.total_energy_j,
            "quality": self.quality_score,
        }


def make_generation_signals(
    n_tokens: int = 96,
    seed: int = 42,
    difficulty: str = "medium",
) -> list[RetrievalSignal]:
    """Create deterministic synthetic signals for scheduler experiments."""
    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")

    difficulty_scale = {
        "easy": 0.75,
        "medium": 1.00,
        "hard": 1.25,
    }
    shift_probability = {
        "easy": 0.035,
        "medium": 0.060,
        "hard": 0.090,
    }
    if difficulty not in difficulty_scale:
        raise ValueError(f"difficulty must be one of {list(difficulty_scale)}")

    rng = random.Random(seed)
    scale = difficulty_scale[difficulty]
    drift = rng.uniform(0.05, 0.18) * scale
    signals: list[RetrievalSignal] = []
    for token_index in range(n_tokens):
        if token_index > 0 and rng.random() < shift_probability[difficulty]:
            drift = max(drift, rng.uniform(0.45, 0.85) * scale)
        else:
            drift += rng.uniform(0.004, 0.030) * scale
        drift = _clamp(drift)

        uncertainty = _clamp(0.18 + 0.52 * drift + rng.uniform(-0.08, 0.12))
        retrieval_score = _clamp(0.93 - 0.58 * drift + rng.uniform(-0.10, 0.08))
        document_overlap = _clamp(0.88 - 0.50 * drift + rng.uniform(-0.08, 0.08))
        signals.append(
            RetrievalSignal(
                token_index=token_index,
                uncertainty=uncertainty,
                semantic_drift=drift,
                retrieval_score=retrieval_score,
                document_overlap=document_overlap,
            )
        )
    return signals


def simulate_scheduled_generation(
    signals: list[RetrievalSignal],
    scheduler: FixedStrideScheduler | AdaptiveRetrievalScheduler,
    hardware: HardwareProfile = EDGE_GPU,
    generation_power_mode: str = "balanced",
) -> ScheduledRun:
    """Simulate TTFT, TBT, latency, quality, and energy for a scheduler."""
    if hasattr(scheduler, "reset"):
        scheduler.reset()

    scheduler_name = scheduler.name()
    gen_mode = hardware.get_mode(generation_power_mode)
    prefill_latency_ms = hardware.prefill_latency_ms / gen_mode.frequency_scale
    generation_energy_j = gen_mode.generation_power_w * prefill_latency_ms / 1000.0

    actions: list[RetrievalAction] = []
    step_latencies: list[float] = []
    retrieval_energy_j = 0.0
    latency_ms = prefill_latency_ms
    quality_gain = 0.0
    missed_risk = 0.0
    last_retrieval_token = -1

    for signal in signals:
        token_latency_ms = hardware.generation_latency_ms(generation_power_mode)
        generation_energy_j += gen_mode.generation_power_w * token_latency_ms / 1000.0
        action = scheduler.decide(signal)
        actions.append(action)

        risk = _signal_risk(signal)
        if action.should_retrieve:
            retrieval_latency_ms = hardware.retrieval_latency_ms(action.top_k, action.power_mode)
            retrieval_mode = hardware.get_mode(action.power_mode)
            retrieval_energy_j += retrieval_mode.retrieval_power_w * retrieval_latency_ms / 1000.0
            token_latency_ms += retrieval_latency_ms
            quality_gain += (1.0 - math.exp(-action.top_k / 6.0)) * (0.50 + 0.50 * risk)
            last_retrieval_token = signal.token_index
        else:
            stale_tokens = signal.token_index + 1
            if last_retrieval_token >= 0:
                stale_tokens = signal.token_index - last_retrieval_token
            missed_risk += risk * min(1.0, stale_tokens / 24.0)

        latency_ms += token_latency_ms
        step_latencies.append(token_latency_ms)

    retrieval_actions = [action for action in actions if action.should_retrieve]
    retrieval_calls = len(retrieval_actions)
    avg_top_k = (
        sum(action.top_k for action in retrieval_actions) / retrieval_calls
        if retrieval_calls
        else 0.0
    )
    ttft_ms = prefill_latency_ms + (step_latencies[0] if step_latencies else 0.0)
    mean_tbt_ms = (
        sum(step_latencies[1:]) / (len(step_latencies) - 1)
        if len(step_latencies) > 1
        else (step_latencies[0] if step_latencies else 0.0)
    )
    n_steps = max(1, len(signals))
    quality_score = _clamp(0.48 + quality_gain / (1.8 * n_steps) - missed_risk / (2.2 * n_steps))
    total_energy_j = generation_energy_j + retrieval_energy_j

    return ScheduledRun(
        scheduler_name=scheduler_name,
        hardware_name=hardware.name,
        actions=actions,
        ttft_ms=round(ttft_ms, 2),
        mean_tbt_ms=round(mean_tbt_ms, 2),
        total_latency_ms=round(latency_ms, 2),
        total_energy_j=round(total_energy_j, 4),
        generation_energy_j=round(generation_energy_j, 4),
        retrieval_energy_j=round(retrieval_energy_j, 4),
        quality_score=round(quality_score, 4),
        retrieval_calls=retrieval_calls,
        avg_top_k=round(avg_top_k, 2),
    )


def _signal_risk(signal: RetrievalSignal) -> float:
    return _clamp(
        0.35 * signal.uncertainty
        + 0.30 * signal.semantic_drift
        + 0.20 * (1.0 - signal.retrieval_score)
        + 0.15 * (1.0 - signal.document_overlap)
    )
