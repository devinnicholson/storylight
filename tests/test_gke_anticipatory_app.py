from __future__ import annotations

import asyncio

import pytest

from bookforge.anticipatory_app import build_runtime
from bookforge.anticipatory_gcp import CloudRunAnticipatoryRenderer


def test_gke_runtime_assembles_pinned_roles_without_opening_network_connections(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "BOOKFORGE_RENDERER_URL",
        "https://bookforge-scene-rtx.example.run.app",
    )
    monkeypatch.setenv(
        "BOOKFORGE_RENDERER_AUDIENCE",
        "https://bookforge-scene-rtx.example.run.app",
    )
    monkeypatch.setenv("BOOKFORGE_RENDERER_GPU", "RTX_PRO_6000")

    runtime = build_runtime()

    assert isinstance(runtime.orchestrator.renderer, CloudRunAnticipatoryRenderer)
    assert runtime.orchestrator.max_candidates == 64
    assert runtime.orchestrator.cache_entries == 32
    assert runtime.orchestrator.critic.critic.base_url == "http://127.0.0.1:8000"
    assert runtime.asset_store.max_bytes == 64 * 1024 * 1024
    asyncio.run(runtime.orchestrator.close())


def test_gke_runtime_accepts_the_private_nemotron_service_dns(monkeypatch) -> None:
    monkeypatch.setenv("BOOKFORGE_RENDERER_URL", "https://renderer.example.run.app")
    monkeypatch.setenv(
        "BOOKFORGE_NEMOTRON_BASE_URL",
        "http://bookforge-nemotron.bookforge.svc.cluster.local:8000",
    )
    runtime = build_runtime()
    assert runtime.orchestrator.critic.critic.base_url == (
        "http://bookforge-nemotron.bookforge.svc.cluster.local:8000"
    )
    asyncio.run(runtime.orchestrator.close())


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
