"""Reviewed text reaches the real job/provider path, with only rendering mocked."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
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
from bookforge.reviewed_description import PARSER_REVISION
from bookforge.reviewed_description import REVISION as LEARNED_REVISION
from bookforge.scene_facts import SceneFactsV2, SceneObjectFact, SceneSettingFact, SceneSubjectFact

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


def create(client, *, text=TEXT, reviewed=True, seed=41, confirm=False, digest=None, defer=False):
    response = client.post("/v1/live-scenes", json=dict(
        text=text, visual_style="watercolor", seed=seed,
        session_id="reviewed-test", reviewed_description=reviewed,
        confirm_visual_facts=confirm,
        visual_fact_digest=digest,
        display_when_complete=defer, defer_presentation=defer,
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
    assert prepared.json()["requires_fact_review"] is False
    assert prepared.json()["local_omissions"] == []
    assert prepared.json()["visual_fact_digest"] is None
    assert state.renderer.calls == []
    facts = SceneFactsV2.model_validate(prepared.json()["visual_facts"])
    assert facts.subjects[0].label == "fox"
    assert facts.subjects[0].color == "brown"
    assert facts.subjects[0].actions == ("jumps over a lazy dog",)
    assert facts.objects[0].label == "dog"
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


def test_completed_voice_scene_presentation_is_explicit_and_does_not_render(api_client):
    state = api_client
    create(state.client)
    previous_pack = state.client.get("/v1/story-packs/latest").json()
    result = create(state.client, defer=True, seed=42)
    assert result["complete"] and result["presentation_ready"] is False
    assert state.client.get("/v1/story-packs/latest").json() == previous_pack
    pointer = state.client.get("/v1/live-scene-sessions/reviewed-test").json()
    endpoint = f"/v1/live-scenes/{result['job_id']}/present"
    expected = {key: pointer[key] for key in ("server_instance_id", "session_revision")}
    stale = state.client.post(endpoint, json={**expected, "session_revision": 9999})
    assert stale.status_code == 409
    response = state.client.post(endpoint, json=expected)
    assert response.status_code == 200, response.text
    assert response.json()["presentation_ready"] is True
    assert response.json()["artifacts"] == result["artifacts"]
    assert len(state.renderer.calls) == 2
    assert state.client.get("/v1/story-packs/latest").json() == result["story_pack"]
    assert state.client.get("/v1/live-scene-sessions/reviewed-test").json()["job"][
        "presentation_ready"
    ] is True


def test_reviewed_rejection_precedes_renderer_preview_prewarm_and_model_cache(
    api_client, monkeypatch,
):
    state = api_client
    previous = create(state.client)
    assert previous["stage"] != "failed"
    pointer = state.client.get("/v1/live-scene-sessions/reviewed-test").json()
    state.adapter.planner = ForbiddenPlanner()

    async def forbidden_submit(*args, **kwargs):
        raise AssertionError("unsupported description must not create any job")

    monkeypatch.setattr(app.state.live_scenes, "submit", forbidden_submit)
    for text in (INVALID, "A cat she's seeing a mouse in a dark alleyway in London."):
        payload = dict(text=text, visual_style="watercolor", seed=41,
                       session_id="reviewed-test", reviewed_description=True)
        prepared = state.client.post("/v1/live-scene-planner/prepare", json=payload)
        created = state.client.post("/v1/live-scenes", json=payload)
        assert prepared.status_code == created.status_code == 422
        assert prepared.json() == created.json()
        detail = prepared.json()["detail"]
        assert detail["code"] == "reviewed_description_unsupported"
        assert "explicit subjects and actions" in detail["message"]
        assert text not in prepared.text and "London" not in prepared.text
        assert "job_id" not in created.json()
        assert state.client.get("/v1/live-scene-sessions/reviewed-test").json() == pointer
    assert len(state.renderer.calls) == 1  # Only the prior, still-visible scene.

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


def _learned_parser(api_client, monkeypatch):
    state = api_client
    socket = Path("/tmp/bookforge-test-scene-parser.sock")
    app.state.settings.reviewed_scene_parser_socket = socket
    state.adapter.reviewed_scene_parser_socket = socket
    text = "A red bird is eating a green apple in London."
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(SceneSubjectFact(ref="bird", label="bird", color="red",
                                   actions=("eating green apple",)),),
        objects=(SceneObjectFact(ref="apple", label="apple", color="green"),),
    )
    state.parsed = dict(
        revision=PARSER_REVISION, status="omission_review",
        facts=facts.model_dump(mode="json"), reason=None, render_admitted=False,
        requires_fact_review=True,
        local_omissions=[dict(ref=9, local_text="London", reason="proper_name_policy",
                              omission_requires_review=True)],
        syntax_issues=[],
        renderer_prompt_preview=facts.to_renderer_prompt(
            source_text=text, visual_style="watercolor",
        ),
    )
    state.parser_calls = []
    state.parser_status = 200
    state.parser_error = None
    state.change_after_api = False

    def handler(request):
        assert str(request.url) == "http://scene-parser/v1/scene-facts"
        assert json.loads(request.content) == {"text": text, "visual_style": "watercolor"}
        state.parser_calls.append(request)
        if state.change_after_api and len(state.parser_calls) == 3:
            changed = json.loads(json.dumps(state.parsed["facts"]))
            changed["objects"][0]["count"] = 1
            state.parsed["facts"] = changed
            state.parsed["renderer_prompt_preview"] = SceneFactsV2.model_validate(
                changed,
            ).to_renderer_prompt(source_text=text, visual_style="watercolor")
        if state.parser_error:
            raise state.parser_error
        return httpx.Response(state.parser_status, json=state.parsed,
                              headers={"Location": "https://invalid.example/forbidden"})

    def transport(**kwargs):
        assert kwargs == {"uds": str(socket), "retries": 0}
        return httpx.MockTransport(handler)

    monkeypatch.setattr("bookforge.reviewed_description.httpx.AsyncHTTPTransport", transport)
    return text


def test_actual_breed_row_requires_confirmation_then_reaches_fake_renderer(api_client, monkeypatch):
    from bookforge.voice_language import graph_from_row

    state = api_client
    socket = Path("/tmp/bookforge-breed-test.sock")
    app.state.settings.reviewed_scene_parser_socket = socket
    state.adapter.reviewed_scene_parser_socket = socket
    row = json.loads((Path(__file__).resolve().parents[1]
                      / "benchmarks/voice-retriever-2026-09-08/parser.json").read_bytes())["row"]
    parsed = graph_from_row(row, "watercolor")
    state.adapter.planner = ForbiddenPlanner()

    def handler(request):
        assert str(request.url) == "http://scene-parser/v1/scene-facts"
        assert json.loads(request.content) == {"text": row["text"], "visual_style": "watercolor"}
        return httpx.Response(200, json=parsed)

    def transport(**kwargs):
        assert kwargs == {"uds": str(socket), "retries": 0}
        return httpx.MockTransport(handler)

    monkeypatch.setattr("bookforge.reviewed_description.httpx.AsyncHTTPTransport", transport)
    payload = dict(text=row["text"], visual_style="watercolor", seed=41,
                   reviewed_description=True, session_id="reviewed-test")
    prepared = state.client.post("/v1/live-scene-planner/prepare", json=payload)
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["requires_fact_review"] is True
    assert prepared.json()["local_omissions"][0]["local_text"] == "Paris"
    assert state.renderer.calls == []
    unconfirmed = state.client.post("/v1/live-scenes", json=payload)
    assert unconfirmed.status_code == 422
    assert unconfirmed.json()["detail"]["code"] == "visual_fact_confirmation_required"
    assert state.renderer.calls == []
    result = create(state.client, text=row["text"], confirm=True,
                    digest=prepared.json()["visual_fact_digest"])
    assert result["complete"] and result["stage"] != "failed", result
    assert len(state.renderer.calls) == 1
    assert state.renderer.calls[0].prompt == (
        parsed["renderer_prompt_preview"]
        + " Full-bleed luminous storybook projection, strong foreground/background depth,"
        " clean silhouettes, no border, no interface."
    )
    assert "paris" not in state.renderer.calls[0].prompt.lower()
    assert result["metrics"]["models"][0]["model"] == LEARNED_REVISION


def test_learned_draft_requires_confirmation_and_rechecks_before_one_render(
    api_client, monkeypatch,
):
    state = api_client
    text = _learned_parser(state, monkeypatch)
    state.adapter.planner = ForbiddenPlanner()
    payload = dict(text=text, visual_style="watercolor", seed=41,
                   session_id="reviewed-test", reviewed_description=True)
    prepared = state.client.post("/v1/live-scene-planner/prepare", json=payload)
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["requires_fact_review"] is True
    assert prepared.json()["local_omissions"][0]["local_text"] == "London"
    assert prepared.json()["model"] == LEARNED_REVISION
    refused = state.client.post("/v1/live-scenes", json=payload)
    assert refused.status_code == 422
    assert refused.json()["detail"]["code"] == "visual_fact_confirmation_required"
    assert state.client.get("/v1/live-scene-sessions/reviewed-test").status_code == 404
    assert state.renderer.calls == []

    digest = prepared.json()["visual_fact_digest"]
    assert len(digest) == 64
    result = create(state.client, text=text, confirm=True, digest=digest)
    assert result["stage"] != "failed", result
    assert len(state.parser_calls) == 4  # Prepare, refusal, API, independent provider validation.
    assert len(state.renderer.calls) == 1
    assert "London" not in state.renderer.calls[0].prompt
    assert result["metrics"]["planning_status"] == "model"
    assert result["metrics"]["models"][0]["model"] == LEARNED_REVISION
    repeated = create(state.client, text=text, confirm=True, digest=digest)
    assert repeated["metrics"]["scene_cache_hit"] is True
    assert repeated["metrics"]["planning_status"] == "model"
    assert len(state.renderer.calls) == 1
    changed = json.loads(json.dumps(state.parsed["facts"]))
    changed["objects"][0]["count"] = 1
    state.parsed["facts"] = changed
    state.parsed["renderer_prompt_preview"] = SceneFactsV2.model_validate(
        changed,
    ).to_renderer_prompt(source_text=text, visual_style="watercolor")
    stale_confirmation = state.client.post("/v1/live-scenes", json={
        **payload, "confirm_visual_facts": True, "visual_fact_digest": digest,
    })
    assert stale_confirmation.status_code == 422
    prepared_again = state.client.post("/v1/live-scene-planner/prepare", json=payload).json()
    assert prepared_again["visual_fact_digest"] != digest
    regenerated = create(state.client, text=text, confirm=True,
                         digest=prepared_again["visual_fact_digest"])
    assert regenerated["stage"] != "failed", regenerated
    assert regenerated["metrics"]["scene_cache_hit"] is False
    assert len(state.renderer.calls) == 2  # Old completed graph cannot shadow the new review.
    normal = create(state.client, text=text, reviewed=False)
    assert normal["stage"] == "failed"  # The reviewed pack cannot shadow normal planning.
    assert len(state.renderer.calls) == 2


def test_parser_response_cannot_bypass_fact_grounding_or_provider_confirmation(
    api_client, monkeypatch,
):
    state = api_client
    text = _learned_parser(state, monkeypatch)
    original = json.loads(json.dumps(state.parsed))
    payload = dict(text=text, visual_style="watercolor", seed=41,
                   reviewed_description=True, confirm_visual_facts=True)
    for changed in (
        {"revision": "dependency-scene-draft-v1"},
        {"revision": "dependency-scene-draft-old"},
        {"renderer_prompt_preview": "an unrelated picture"},
        {"facts": {**original["facts"], "subjects": [
            {**original["facts"]["subjects"][0], "color": "green"},
        ]}},
    ):
        state.parsed = {**original, **changed}
        response = state.client.post("/v1/live-scenes", json=payload)
        assert response.status_code == 422
        assert "London" not in response.text and "job_id" not in response.json()
    assert state.renderer.calls == []
    state.parsed = original
    for status, error in ((302, None), (200, httpx.ReadTimeout("private upstream message"))):
        state.parser_status, state.parser_error = status, error
        before = len(state.parser_calls)
        response = state.client.post("/v1/live-scenes", json=payload)
        assert response.status_code == 422
        assert len(state.parser_calls) == before + 1  # No redirect or retry.
        assert "private upstream message" not in response.text
    state.parser_status, state.parser_error = 200, None

    async def unconfirmed_provider():
        from bookforge.live_scene import build_live_scene_story_pack

        request = LiveSceneCreateRequest(text=text, visual_style="watercolor",
                                         reviewed_description=True)
        draft = build_live_scene_story_pack(request, job_id="unconfirmed", seed=41,
                                            assets=[], compiler_model="fixture")
        with pytest.raises(RuntimeError, match="no cloud image was generated"):
            await state.adapter._resolve_plan(request, job_id="unconfirmed", seed=41, draft=draft)

    asyncio.run(unconfirmed_provider())
    assert state.renderer.calls == []

    # A changed but independently grounded parse after API acceptance must still
    # stop at the provider boundary before it can use the paid renderer.
    state.parser_calls.clear()
    prepared = state.client.post("/v1/live-scene-planner/prepare", json={
        key: value for key, value in payload.items() if key != "confirm_visual_facts"
    })
    assert prepared.status_code == 200
    state.change_after_api = True
    result = create(state.client, text=text, confirm=True,
                    digest=prepared.json()["visual_fact_digest"])
    assert result["stage"] == "failed"
    assert len(state.parser_calls) == 3
    assert state.renderer.calls == []


def test_configured_nominal_audit_blocks_adverb_labels_and_invalidates_unaudited_cache(
    api_client, monkeypatch,
):
    state = api_client
    initial = create(state.client)
    assert initial["metrics"]["planning_status"] == "deterministic"
    socket = Path("/tmp/bookforge-nominal-test.sock")
    app.state.settings.reviewed_scene_parser_socket = socket
    state.adapter.reviewed_scene_parser_socket = socket
    audits = []

    def handler(request):
        payload = json.loads(request.content)
        assert set(payload) == {"text", "visual_style", "nominal_labels"}
        audits.append(payload)
        issues = [dict(label_index=i, reason="not_nominal_span")
                  for i, label in enumerate(payload["nominal_labels"]) if "slowly" in label]
        return httpx.Response(200, json=dict(revision="dependency-nominal-audit-v1",
                                             accepted=not issues, issues=issues))

    def transport(**kwargs):
        assert kwargs == {"uds": str(socket), "retries": 0}
        return httpx.MockTransport(handler)

    monkeypatch.setattr("bookforge.reviewed_description.httpx.AsyncHTTPTransport", transport)
    audited = create(state.client)
    assert audited["stage"] != "failed", audited
    assert audited["metrics"]["scene_cache_hit"] is False
    assert audited["metrics"]["planning_status"] == "model"
    assert audited["metrics"]["models"][0]["model"] == "reviewed-language-bounded-v1"
    assert len(state.renderer.calls) == 2
    assert create(state.client)["metrics"]["scene_cache_hit"] is True
    assert len(state.renderer.calls) == 2
    pointer = state.client.get("/v1/live-scene-sessions/reviewed-test").json()
    payload = dict(text="A cat slowly chases a mouse.", visual_style="watercolor",
                   seed=41, session_id="reviewed-test", reviewed_description=True)
    for endpoint in ("/v1/live-scene-planner/prepare", "/v1/live-scenes"):
        response = state.client.post(endpoint, json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "reviewed_description_unsupported"
    assert any("slowly" in label for label in audits[-1]["nominal_labels"])
    assert len(state.renderer.calls) == 2
    assert state.client.get("/v1/live-scene-sessions/reviewed-test").json() == pointer

    for text in (
        "The white golden retriever and the Merle Aussie are playing in the field.",
        "A brown fox stands beside a white dog. The fox does not jump over the dog.",
        "A fox stands beside a stream. No dogs.",
    ):
        response = state.client.post(
            "/v1/live-scene-planner/prepare", json={**payload, "text": text},
        )
        assert response.status_code == 200, response.text
        assert response.json()["requires_fact_review"] is False
        assert response.json()["model"] == "reviewed-language-bounded-v1"
        assert response.json()["revision"] == "dependency-nominal-audit-v1"
    assert "dogs" in audits[-1]["nominal_labels"]  # Absent nouns are audited too.
    assert len(state.renderer.calls) == 2
