from __future__ import annotations

from retrievalbench.scheduling import (
    AdaptiveRetrievalScheduler,
    EDGE_GPU,
    FixedStrideScheduler,
    RetrievalSignal,
    make_generation_signals,
    simulate_scheduled_generation,
)


def _signal(
    token_index: int,
    uncertainty: float = 0.20,
    semantic_drift: float = 0.15,
    retrieval_score: float = 0.90,
    document_overlap: float = 0.85,
) -> RetrievalSignal:
    return RetrievalSignal(
        token_index=token_index,
        uncertainty=uncertainty,
        semantic_drift=semantic_drift,
        retrieval_score=retrieval_score,
        document_overlap=document_overlap,
    )


def test_fixed_stride_scheduler_retrieves_on_stride() -> None:
    scheduler = FixedStrideScheduler(stride=4, top_k=5)
    actions = [scheduler.decide(_signal(i)) for i in range(10)]
    retrieved_at = [i for i, action in enumerate(actions) if action.should_retrieve]
    assert retrieved_at == [0, 4, 8]


def test_adaptive_scheduler_increases_depth_for_high_risk_signal() -> None:
    scheduler = AdaptiveRetrievalScheduler(
        min_interval=2,
        max_interval=20,
        base_top_k=3,
        max_top_k=10,
    )
    assert scheduler.decide(_signal(0)).should_retrieve

    action = scheduler.decide(
        _signal(
            3,
            uncertainty=0.98,
            semantic_drift=0.96,
            retrieval_score=0.05,
            document_overlap=0.05,
        )
    )

    assert action.should_retrieve
    assert action.top_k > scheduler.base_top_k
    assert action.power_mode == "turbo"
    assert "uncertainty" in action.reasons
    assert "semantic_drift" in action.reasons


def test_adaptive_scheduler_enforces_max_interval() -> None:
    scheduler = AdaptiveRetrievalScheduler(
        min_interval=2,
        max_interval=5,
        uncertainty_threshold=1.0,
        semantic_drift_threshold=1.0,
        retrieval_score_threshold=0.0,
        overlap_threshold=0.0,
        score_decay_threshold=1.0,
    )
    calls = []
    for i in range(7):
        action = scheduler.decide(_signal(i))
        if action.should_retrieve:
            calls.append(i)

    assert calls == [0, 5]


def test_generation_signal_factory_is_deterministic() -> None:
    first = make_generation_signals(n_tokens=8, seed=123, difficulty="medium")
    second = make_generation_signals(n_tokens=8, seed=123, difficulty="medium")
    assert first == second


def test_adaptive_scheduler_uses_less_energy_on_low_risk_sequence() -> None:
    signals = [_signal(i) for i in range(40)]
    fixed = simulate_scheduled_generation(
        signals=signals,
        scheduler=FixedStrideScheduler(stride=5, top_k=8, power_mode="balanced"),
        hardware=EDGE_GPU,
    )
    adaptive = simulate_scheduled_generation(
        signals=signals,
        scheduler=AdaptiveRetrievalScheduler(
            min_interval=4,
            max_interval=50,
            base_top_k=3,
            max_top_k=8,
        ),
        hardware=EDGE_GPU,
    )

    assert adaptive.retrieval_calls < fixed.retrieval_calls
    assert adaptive.total_energy_j < fixed.total_energy_j
    assert 0.0 <= adaptive.quality_score <= 1.0
