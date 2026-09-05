from pathlib import Path

import pytest

from bookforge.config import Settings
from bookforge.edge_preflight import evaluate


def test_jetson_preflight_accepts_private_local_first_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    report = evaluate(
        Settings(
            environment="jetson",
            model_backend="fake",
            model_name="fake",
            asr_backend="disabled",
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
        )
    )

    assert report["ready"] is True
    assert report["checks"]["model_boundary"]["ready"] is True
    assert report["checks"]["edge_asr"]["ready"] is True


def test_jetson_preflight_rejects_relative_paths_remote_model_and_mac_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    report = evaluate(
        Settings(
            environment="jetson",
            model_backend="openai",
            model_base_url="https://models.example.com",
            asr_backend="mlx_whisper",
            asr_model="/missing/model",
            data_dir=Path("relative/data"),
            cache_dir=Path("relative/cache"),
        )
    )

    assert report["ready"] is False
    assert report["checks"]["edge_paths"]["ready"] is False
    assert report["checks"]["model_boundary"]["ready"] is False
    assert report["checks"]["edge_asr"]["ready"] is False


def test_jetson_preflight_rejects_checkout_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("BOOKFORGE_MODEL_API_KEY=secret\n")

    report = evaluate(
        Settings(
            _env_file=None,
            environment="jetson",
            model_backend="fake",
            asr_backend="disabled",
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
        )
    )

    assert report["ready"] is False
    assert report["checks"]["checkout_env"]["ready"] is False
