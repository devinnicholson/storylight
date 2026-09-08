"""Reviewed text reaches the real job/provider path, with only rendering mocked."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bookforge.api import app
from bookforge.bounded_description import REVISION
from bookforge.config import Settings
from bookforge.finite_modal_provider import (
    FAST_MODEL,
    FAST_MODEL_REVISION,
    FiniteModalLiveSceneProvider,
    WarmModalSceneProvider,
    load_finite_scene_bundle,
)
from bookforge.live_scene import LiveSceneCreateRequest

TEXT = "A quick brown fox jumps over a lazy dog."
INVALID = "A quick brown box. Do not throw a lazy dog."


class Renderer:
    def __init__(self):
        self.calls = []

    async def generate_fast(self, request, *, output_dir):
        self.calls.append(request)
        output_dir.mkdir(parents=True)
        artifacts = {}
        for role, color in (("master", "brown"), ("depth", "grey")):
            path = output_dir / f"{role}.png"
            Image.new("RGB", (1024, 576), color).save(path)
            data = path.read_bytes()
            artifacts[role] = dict(path=path.name, sha256=hashlib.sha256(data).hexdigest(),
                                   bytes=len(data), mime_type="image/png", width=1024, height=576)
        manifest = output_dir / "scene.manifest.json"
        manifest.write_text(json.dumps(dict(
            schema_version="1.0", provider="modal-finite", scene_id=request.scene_id,
            stages={"fast": dict(model=FAST_MODEL, model_revision=FAST_MODEL_REVISION,
                                 gpu="L4", finite_call=True, estimated_gpu_usd=0.01,
                                 remote_seconds=0.1, inference_seconds=0.08,
                                 provider_overhead_seconds=0.02)}, artifacts=artifacts,
        )))
        return load_finite_scene_bundle(manifest)


class ForbiddenPlanner:
    async def plan(self, **kwargs):
        raise AssertionError("reviewed mode must not invoke the model planner")

    async def has_cached_plan(self, **kwargs):
        raise AssertionError("reviewed mode must not consult the model plan cache")


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, model_backend="fake", model_name="fake",
                        asset_backend="fake", live_scene_enable_motion=False,
                        data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    renderer = Renderer()
    state = SimpleNamespace(renderer=renderer)

    def build(*args, **kwargs):
        state.adapter = FiniteModalLiveSceneProvider(
            renderer, cache=kwargs["cache"], output_root=tmp_path / "rendered",
            planner=None, enable_preview=True, fidelity_mode="deferred",
        )
        return state.adapter

    monkeypatch.setattr("bookforge.api.get_settings", lambda: settings)
    monkeypatch.setattr("bookforge.api.build_live_scene_provider", build)
    with TestClient(app) as client:
        state.client = client
        yield state


def create(client, *, text=TEXT, reviewed=True, seed=41):
    response = client.post("/v1/live-scenes", json=dict(
        text=text, visual_style="watercolor", seed=seed,
        session_id="reviewed-test", reviewed_description=reviewed,
    ))
    assert response.status_code == 202, response.text
    job = response.json()["job_id"]
    events = client.get(f"/v1/live-scenes/{job}/events")
    assert events.status_code == 200
    return client.get(f"/v1/live-scenes/{job}").json()


def test_reviewed_api_renders_once_without_planner_and_records_deterministic_provenance(api_client):
    state = api_client
    prepared = state.client.post("/v1/live-scene-planner/prepare", json=dict(
        text=TEXT, visual_style="watercolor", seed=41, reviewed_description=True,
    ))
    assert prepared.status_code == 200, prepared.text
    assert state.renderer.calls == []
    result = create(state.client)
    assert result["complete"] and result["stage"] != "failed", result
    assert len(state.renderer.calls) == 1
    assert result["metrics"]["planning_status"] == "deterministic"
    assert result["metrics"]["models"][0]["model"] == REVISION
    prompt = state.renderer.calls[0].prompt.lower()
    assert all(word in prompt for word in ("brown", "fox", "dog", "lazy"))
    assert "over" in prompt
    for artifact in result["artifacts"]:
        response = state.client.get(artifact["uri"])
        assert hashlib.sha256(response.content).hexdigest() == artifact["checksum_sha256"]
    repeated = create(state.client)
    assert repeated["metrics"]["scene_cache_hit"] is True
    assert repeated["metrics"]["planning_status"] == "deterministic"
    assert len(state.renderer.calls) == 1


def test_reviewed_rejection_precedes_renderer_preview_prewarm_and_model_cache(api_client):
    state = api_client
    state.adapter.planner = ForbiddenPlanner()
    prepared = state.client.post("/v1/live-scene-planner/prepare", json=dict(
        text=INVALID, visual_style="watercolor", seed=41, reviewed_description=True,
    ))
    assert prepared.status_code in {400, 422, 503}
    result = create(state.client, text=INVALID)
    assert result["stage"] == "failed"
    assert state.renderer.calls == []

    async def warm_boundary():
        warm = object.__new__(WarmModalSceneProvider)

        async def forbidden(*args, **kwargs):
            raise AssertionError("invalid reviewed description must not prepare a paid renderer")

        warm.is_renderer_likely_warm = forbidden
        state.adapter.provider = warm
        state.adapter.auto_prewarm_on_submit = True
        request = LiveSceneCreateRequest(text=INVALID, reviewed_description=True)
        assert await state.adapter._should_generate_preview(request) is False
        state.adapter._prepare_warm_renderer = forbidden
        from bookforge.live_scene import build_live_scene_story_pack

        draft = build_live_scene_story_pack(request, job_id="reviewed-invalid", seed=1,
                                            assets=[], compiler_model="fixture")
        with pytest.raises(RuntimeError, match="no cloud image was generated"):
            await state.adapter._resolve_plan_while_preparing_renderer(
                request, job_id="reviewed-invalid", seed=1, draft=draft,
            )

    original = state.adapter.provider
    try:
        asyncio.run(warm_boundary())
    finally:
        state.adapter.provider = original


def test_completed_old_mode_and_model_cache_cannot_shadow_reviewed_mode(api_client, monkeypatch):
    state = api_client
    first = create(state.client)
    assert first["stage"] != "failed", first
    pack = state.client.portal.call(app.state.story_store.latest)
    stale = pack.model_copy(update={"compiler_model": "gemma-old-cached-plan"})

    async def old_completed(**kwargs):
        return stale

    monkeypatch.setattr(app.state.story_store, "find_live_scene", old_completed)
    state.adapter.planner = ForbiddenPlanner()
    assert REVISION != "bounded-description-v1"
    for calls, compiler in enumerate(("gemma-old-cached-plan", "bounded-description-v1"), 2):
        stale = pack.model_copy(update={"compiler_model": compiler})
        regenerated = create(state.client)
        assert regenerated["stage"] != "failed", regenerated
        assert len(state.renderer.calls) == calls
        assert regenerated["metrics"]["scene_cache_hit"] is False
        assert regenerated["metrics"]["models"][0]["model"] == REVISION
    stale = pack
    normal_mode = create(state.client, reviewed=False)
    assert normal_mode["stage"] == "failed"  # ForbiddenPlanner proves cache was not reused.
    assert len(state.renderer.calls) == 3
