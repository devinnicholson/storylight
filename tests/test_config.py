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

    assert settings.live_scene_planner == "model"
    assert settings.live_scene_planner_timeout_seconds == 12
    assert settings.live_scene_planner_model_revision == "sha256:gemma3-fixture"
    assert settings.live_scene_planner_compact_wire is False
    assert settings.live_scene_planner_cache_entries == 32
    assert (
        Settings(
            _env_file=None, live_scene_planner_compact_wire=True
        ).live_scene_planner_compact_wire
        is True
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_planner_cache_entries=257)
    assert settings.model_context_tokens == 4_096
    assert settings.model_max_output_tokens == 320
    assert settings.live_scene_master_width == 896
    assert settings.live_scene_master_height == 512
    assert settings.live_scene_master_steps == 2
    assert settings.live_scene_master_guidance_scale == 4.5
    assert settings.live_scene_auto_prewarm_on_submit is False
    assert (
        Settings(
            _env_file=None, live_scene_auto_prewarm_on_submit=True
        ).live_scene_auto_prewarm_on_submit
        is True
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_planner_timeout_seconds=61)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_context_tokens=1_024)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_max_output_tokens=32)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_master_width=900)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_master_height=480)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_scene_master_steps=1)


def test_general_model_output_default_remains_large_enough_for_story_compilation(
    monkeypatch,
) -> None:
    monkeypatch.delenv("BOOKFORGE_MODEL_MAX_OUTPUT_TOKENS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.model_max_output_tokens == 4_096
