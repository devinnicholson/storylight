import pytest

from bookforge.planner_benchmark import _require_loopback, _summarize


def test_planner_benchmark_requires_loopback_model_endpoint() -> None:
    assert _require_loopback("http://127.0.0.1:11435/") == "http://127.0.0.1:11435"
    assert _require_loopback("http://localhost:11434") == "http://localhost:11434"
    with pytest.raises(ValueError, match="loopback"):
        _require_loopback("https://models.example.com")


def test_planner_benchmark_summary_preserves_latency_and_token_maxima() -> None:
    summary = _summarize(
        [
            {"planning_ms": 4000.125, "output_tokens": 108},
            {"planning_ms": 5000.375, "output_tokens": 116},
        ]
    )

    assert summary == {
        "cases": 2,
        "mean_planning_ms": 4500.25,
        "median_planning_ms": 4500.25,
        "maximum_planning_ms": 5000.375,
        "mean_output_tokens": 112.0,
        "maximum_output_tokens": 116,
    }
