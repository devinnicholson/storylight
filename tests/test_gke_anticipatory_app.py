from __future__ import annotations

import pytest

from bookforge.anticipatory_app import build_runtime


def test_gke_runtime_refuses_missing_renderer_identity_and_unbounded_settings(
    monkeypatch,
) -> None:
    monkeypatch.delenv("BOOKFORGE_RENDERER_URL", raising=False)
    with pytest.raises(RuntimeError, match="BOOKFORGE_RENDERER_URL is required"):
        build_runtime()

    monkeypatch.setenv("BOOKFORGE_RENDERER_URL", "https://renderer.example.run.app")
    monkeypatch.setenv("BOOKFORGE_RENDERER_GPU", "unbounded-gpu")
    with pytest.raises(RuntimeError, match="must be L4 or RTX_PRO_6000"):
        build_runtime()

    monkeypatch.setenv("BOOKFORGE_RENDERER_GPU", "L4")
    monkeypatch.setenv("BOOKFORGE_ANTICIPATORY_MAX_RENDERS", "3")
    with pytest.raises(RuntimeError, match="must be between 1 and 2"):
        build_runtime()

    monkeypatch.setenv("BOOKFORGE_ANTICIPATORY_MAX_RENDERS", "2")
    monkeypatch.setenv("BOOKFORGE_ANTICIPATORY_ASSET_BYTES", str(8 * 1024 * 1024))
    with pytest.raises(RuntimeError, match="must be between 16777216"):
        build_runtime()
