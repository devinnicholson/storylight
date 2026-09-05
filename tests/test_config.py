from pathlib import Path

import pytest
from pydantic import ValidationError

from bookforge.config import Settings

ROOT = Path(__file__).parents[1]


def test_allowed_origins_accept_comma_separated_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "BOOKFORGE_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    )

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


def test_jetson_environment_example_is_parseable(monkeypatch) -> None:
    for name in (
        "BOOKFORGE_ALLOWED_ORIGINS",
        "BOOKFORGE_DATA_DIR",
        "BOOKFORGE_CACHE_DIR",
        "BOOKFORGE_ASR_BACKEND",
        "BOOKFORGE_ASR_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=ROOT / "deploy/jetson/bookforge.env.example")

    assert settings.environment == "jetson"
    assert settings.data_dir == Path("/var/lib/bookforge")
    assert settings.cache_dir == Path("/var/cache/bookforge")
    assert settings.asr_backend == "disabled"
    assert settings.live_scene_planner == "deterministic"
    assert settings.allowed_origins == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


def test_live_scene_model_planner_has_a_bounded_independent_timeout() -> None:
    settings = Settings(
        _env_file=None,
        live_scene_planner="model",
        live_scene_planner_timeout_seconds=12,
        live_scene_planner_model_revision="sha256:gemma3-fixture",
        model_context_tokens=4_096,
        model_max_output_tokens=320,
    )

    assert settings.live_scene_planner_max_output_tokens == 64
    assert settings.live_scene_planner_timeout_seconds == 12
    assert settings.model_context_tokens == 4_096
    assert settings.model_max_output_tokens == 320
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_planner_timeout_seconds=61)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_planner_max_output_tokens=129)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_planner_fallback_ready_seconds=31)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_context_tokens=1_024)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_max_output_tokens=32)
