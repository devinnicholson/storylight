from pathlib import Path

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
    assert settings.allowed_origins == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
