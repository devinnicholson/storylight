from __future__ import annotations

from bookforge.anticipatory_simulator import SimulationCase, default_cases, simulate


def test_default_simulation_is_explicitly_not_hardware_evidence() -> None:
    report = simulate(default_cases())
    assert report.evidence_kind == "deterministic_simulation_not_hardware_measurement"
    assert report.summary.cases == 10
    assert report.summary.exact_lookahead_hit_rate == 1
    assert report.summary.speculative_hit_rate == 1
    assert report.summary.wasted_render_attempts == 2
    assert report.summary.render_attempts == 13
    assert report.summary.wasted_render_ratio < 0.20
    assert report.gates.passed is True
    assert report.gates.p95_activation_below_250_ms is True


def test_long_enough_lookahead_hides_generation_latency() -> None:
    report = simulate(
        [
            SimulationCase(
                case_id="known_scene",
                mode="exact_lookahead",
                candidates=1,
                lookahead_ms=3000,
                render_ms=1800,
                critic_ms=400,
            )
        ],
        committed_swap_ms=8,
    )
    result = report.results[0]
    assert result.speculative_hit is True
    assert result.ordinary_activation_ms == 2200
    assert result.anticipatory_activation_ms == 8
    assert result.latency_saved_ms == 2192
    assert report.gates.passed is True


def test_short_lookahead_reports_remaining_work_without_negative_savings() -> None:
    report = simulate(
        [
            SimulationCase(
                case_id="short_scene",
                mode="exact_lookahead",
                candidates=1,
                lookahead_ms=500,
                render_ms=1000,
                critic_ms=250,
            )
        ]
    )
    result = report.results[0]
    assert result.speculative_hit is False
    assert result.anticipatory_activation_ms == 760
    assert result.latency_saved_ms == 490
