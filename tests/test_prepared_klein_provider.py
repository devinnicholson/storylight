import asyncio
import base64
import hashlib
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from storylight.finite_modal_provider import FastSceneRequest, FiniteModalProviderError
from storylight.prepared_klein_provider import (
    IDENTITY,
    PreparedKleinProvider,
    load_prepared_klein_bundle,
)


def fixture(tmp_path, mutate=None):
    now = [1000.0]
    calls = []
    stream = io.BytesIO()
    Image.new("RGB", (1024, 576), "blue").save(stream, format="JPEG")
    jpeg = stream.getvalue()
    digest = hashlib.sha256(jpeg).hexdigest()
    lease = None

    async def token(audience):
        assert audience == "https://prepared-test.run.app"
        return "private-token"

    def handler(request):
        nonlocal lease
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/v1/prewarm":
            assert set(body) == {"session_id"}
            lease = {
                "schema_version": 1, "state": "READY", "session_id": body["session_id"],
                "instance_id": "a" * 32, "service": "prepared-test",
                "revision": "prepared-test-00001", "identity": IDENTITY,
                "started_at": 1000, "expires_at": 1300,
                "warmups": [{"sequence_bucket": b, "seed": 42,
                             "master_sha256": digest, "depth_sha256": digest}
                            for b in (128, 256)],
            }
            now[0] += 70
            result = {"lease": lease, "model_load_seconds": 50, "warmup_seconds": 15}
        else:
            assert set(body) == {"session_id", "instance_id", "request_id", "scene_id",
                                 "prompt", "seed"}
            assert body["session_id"] == lease["session_id"]
            assert body["instance_id"] == lease["instance_id"]
            now[0] += 0.5
            result = {"lease": lease, "request_id": body["request_id"],
                      "scene_id": body["scene_id"],
                      "master_b64": base64.b64encode(jpeg).decode(),
                      "depth_b64": base64.b64encode(jpeg).decode(),
                      "metrics": {"seed": body["seed"], "token_count": 70,
                                  "sequence_bucket": 128, "image_seconds": .3,
                                  "depth_seconds": .03, "encoding_seconds": .01,
                                  "total_seconds": .34, "peak_allocated_gib": 16,
                                  "peak_reserved_gib": 18,
                                  "master_sha256": digest, "depth_sha256": digest}}
        if mutate:
            return mutate(request.url.path, json.loads(json.dumps(result)))
        return httpx.Response(200, json=result)

    def client_factory(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    def new():
        return PreparedKleinProvider(
            base_url="https://prepared-test.run.app", service="prepared-test",
            revision="prepared-test-00001", external_owner_id="finite-canary",
            claims_dir=tmp_path / "claims", token_source=token, client_factory=client_factory,
            clock=lambda: now[0], wall_clock=lambda: now[0])

    return new, now, calls


def request(scene="scene-1"):
    return FastSceneRequest(scene, "One red fox walks in a forest.", steps=4, guidance_scale=1)


def test_coalesced_actual_lease_fixed_expiry_claims_and_bundle(tmp_path):
    async def run():
        new, now, calls = fixture(tmp_path)
        provider = new()
        assert not await provider.is_prewarmed()
        assert not (await provider.warm_status()).ready and not calls
        with pytest.raises(FiniteModalProviderError):
            await provider.generate_fast(request(), output_dir=tmp_path / "early")
        first, second = await asyncio.gather(
            provider.prewarm(prewarm_id="one"), provider.prewarm(prewarm_id="one"))
        assert first.expires_in_seconds == second.expires_in_seconds == 230
        assert first.fast_remote_seconds == 70 and len(calls) == 1
        bundle = await provider.generate_fast(request(), output_dir=tmp_path / "scene")
        assert bundle.master.path.is_file() and bundle.depth.path.is_file()
        assert bundle.estimated_gpu_usd > 0
        assert bundle.manifest["stages"]["fast"]["warm_state"] == "warm"
        assert not bundle.manifest["policy"]["cloud_deletion_by_adapter"]
        assert provider.deadline == 1300
        with pytest.raises(FiniteModalProviderError):
            await provider.generate_fast(request(), output_dir=tmp_path / "duplicate")
        assert len(calls) == 2
        # A new adapter cannot forget a prior preparation claim, even with a new UUID.
        with pytest.raises(FiniteModalProviderError):
            await new().prewarm(prewarm_id="two")
        assert len(calls) == 2
        claims = "".join(p.read_text() for p in (tmp_path / "claims").iterdir())
        assert "private-token" not in claims and request().prompt not in claims
        now[0] = 1300
        assert not await provider.is_prewarmed()
        with pytest.raises(FiniteModalProviderError):
            await provider.prewarm(prewarm_id="one")
        with pytest.raises(FiniteModalProviderError):
            await provider.generate_fast(request("late"), output_dir=tmp_path / "late")
        await provider.aclose()
        assert len(calls) == 2  # local close never issues cloud deletion or warmup.

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["bucket", "identity", "expiry", "uuid", "checksum", "redirect"])
def test_bad_receipts_poison_session_without_retry_or_output(tmp_path, fault):
    def mutate(path, result):
        if fault == "bucket":
            result["lease"]["warmups"][1]["sequence_bucket"] = 128
        elif fault == "identity":
            result["lease"]["identity"]["steps"] = 2
        elif path.endswith("generate"):
            if fault == "expiry":
                result["lease"]["started_at"] += 1
                result["lease"]["expires_at"] += 1
            elif fault == "uuid":
                result["lease"]["instance_id"] = "b" * 32
            elif fault == "checksum":
                result["metrics"]["depth_sha256"] = "0" * 64
            else:
                return httpx.Response(307, headers={"Location": "https://example.com"})
        return httpx.Response(200, json=result)

    async def run():
        new, _, calls = fixture(tmp_path, mutate)
        provider = new()
        with pytest.raises(FiniteModalProviderError):
            await provider.prewarm(prewarm_id="one")
            await provider.generate_fast(request(), output_dir=tmp_path / "bad")
        count = len(calls)
        assert not await provider.is_prewarmed() and not (tmp_path / "bad").exists()
        with pytest.raises(FiniteModalProviderError):
            await provider.generate_fast(request("other"), output_dir=tmp_path / "other")
        assert len(calls) == count

    asyncio.run(run())


def test_privacy_profile_and_cancel_are_fail_closed(tmp_path):
    async def run():
        new, _, calls = fixture(tmp_path)
        provider = new()
        await provider.prewarm(prewarm_id="one")
        for bad in [replace(request(), prompt="Contact jane@example.com"),
                    replace(request(), steps=2), replace(request(), fidelity_label="fox")]:
            with pytest.raises(FiniteModalProviderError):
                await provider.generate_fast(bad, output_dir=tmp_path / "bad")
        assert len(calls) == 1
        waiting = asyncio.Event()

        async def stalled_token(_):
            waiting.set()
            await asyncio.Future()

        provider.token_source = stalled_token
        task = asyncio.create_task(
            provider.generate_fast(request(), output_dir=tmp_path / "cancel"))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not await provider.is_prewarmed()
        with pytest.raises(FiniteModalProviderError):
            await provider.generate_fast(request(), output_dir=tmp_path / "retry")
        assert len(calls) == 1

    asyncio.run(run())


def test_explicit_bundle_recovery_checks_request_provenance_and_actual_bytes(tmp_path):
    async def run():
        new, now, _ = fixture(tmp_path)
        provider = new()
        await provider.prewarm(prewarm_id="one")
        bundle = await provider.generate_fast(request(), output_dir=tmp_path / "saved")
        now[0] = 5000  # Historical image reuse grants no renewed GPU lease.
        args = {"request": request(), "service": provider.service,
                "revision": provider.revision, "external_owner_id": provider.external_owner_id}
        recovered = load_prepared_klein_bundle(bundle.manifest_path, **args)
        assert recovered.manifest == bundle.manifest and not await provider.is_prewarmed()
        stage = recovered.manifest["stages"]["fast"]
        assert stage["estimated_gpu_usd"] == (stage["estimated_gpu_only_usd"]
                + stage["estimated_cpu_usd"] + stage["estimated_memory_usd"])
        for key, value in [("request", replace(request(), prompt="A blue boat floats.")),
                           ("service", "other"), ("external_owner_id", "other")]:
            with pytest.raises(FiniteModalProviderError):
                load_prepared_klein_bundle(bundle.manifest_path, **(args | {key: value}))
        original = bundle.manifest_path.read_bytes()
        for mutation in ("model", "traversal", "cost"):
            saved = json.loads(original)
            if mutation == "model":
                saved["lease"]["identity"]["model"] = "unqualified-model"
            elif mutation == "traversal":
                saved["artifacts"]["master"]["path"] = "../master.jpg"
            else:
                saved["stages"]["fast"]["estimated_gpu_usd"] = 0
            bundle.manifest_path.write_text(json.dumps(saved))
            with pytest.raises(FiniteModalProviderError):
                load_prepared_klein_bundle(bundle.manifest_path, **args)
        bundle.manifest_path.write_bytes(original)
        bundle.depth.path.write_bytes(b"bad JPEG")
        with pytest.raises(FiniteModalProviderError):
            load_prepared_klein_bundle(bundle.manifest_path, **args)

    asyncio.run(run())


def test_one_owned_http_pool_reuse_remaining_timeout_and_cancelled_close(tmp_path):
    async def run():
        new, now, calls = fixture(tmp_path)
        provider = new()
        factory, clients, timeouts = provider.client_factory, [], []

        async def observe(request):
            timeouts.append(request.extensions["timeout"])

        def track(**kwargs):
            client = factory(**kwargs)
            client.event_hooks["request"].append(observe)
            clients.append(client)
            return client

        provider.client_factory = track
        assert not clients
        await provider.prewarm(prewarm_id="one")
        await provider.generate_fast(request(), output_dir=tmp_path / "first")
        now[0] = 1299
        await provider.generate_fast(request("second"), output_dir=tmp_path / "second")
        assert len(clients) == 1 and not clients[0].is_closed
        assert timeouts[-1] == dict.fromkeys(("connect", "read", "write", "pool"), 1.0)
        assert all(value == 180 for value in timeouts[0].values())
        await provider.aclose()
        await provider.aclose()
        assert clients[0].is_closed and len(calls) == 3

        # Cancel an in-flight HTTP body while close waits for the operation lock.
        other_dir = tmp_path / "cancelled"
        other_dir.mkdir()
        entered, body_closed = asyncio.Event(), []

        class BlockedBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Future()
                yield b"unreachable"

            async def aclose(self):
                body_closed.append(True)

        def block(path, result):
            if path.endswith("generate"):
                return httpx.Response(200, stream=BlockedBody())
            return httpx.Response(200, json=result)

        second, _, dispatched = fixture(other_dir, block)
        candidate = second()
        await candidate.prewarm(prewarm_id="one")
        client = candidate._client
        generating = asyncio.create_task(
            candidate.generate_fast(request(), output_dir=other_dir / "scene"))
        await entered.wait()
        closing = asyncio.create_task(candidate.aclose())
        await asyncio.sleep(0)
        assert not await candidate.is_prewarmed()
        assert not (await candidate.warm_status()).ready
        closing.cancel()
        await asyncio.sleep(0)
        assert not client.is_closed and not closing.done()
        generating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generating
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert body_closed == [True] and client.is_closed
        assert candidate.closed and not await candidate.is_prewarmed()
        assert len(dispatched) == 2 and not (other_dir / "scene").exists()
        with pytest.raises(FiniteModalProviderError):
            await candidate.generate_fast(request(), output_dir=other_dir / "retry")
        assert len(dispatched) == 2

    asyncio.run(run())
