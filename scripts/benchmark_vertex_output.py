"""Finite synthetic Vertex response comparison; never alters live routing."""

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import io
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from PIL import Image

from bookforge import vertex_scene_provider
from bookforge.finite_modal_provider import FastSceneRequest
from bookforge.vertex_scene_provider import (
    DEFAULT_MODEL,
    _extract_image,
    _image_dimensions,
    _request_payload,
    google_access_token,
)

STYLE = (
    "Rich luminous watercolor storybook illustration, layered depth, detailed natural "
    "scenery, full-bleed 16:9. "
)
SCENES = [
    "Exactly one orange cat chases exactly one gray mouse along a garden path.",
    "Exactly one pink fox jumps over a narrow stream in a lush forest.",
    "Exactly two brown rabbits sit beside exactly three red mushrooms in a meadow.",
    "Exactly one white dog lies on a blue couch inside a cozy cottage.",
    "Exactly one owl perches above exactly one sleeping fox beneath an oak tree.",
    "Exactly one girl in a yellow coat holds one blue umbrella beside a pond; no rain.",
]
FIELD_MASK = ",".join([
    "candidates.content.parts.inlineData", "candidates.content.parts.text",
    "candidates.content.parts.thought", "candidates.finishReason", "candidates.safetyRatings",
    "promptFeedback", "usageMetadata", "modelVersion", "responseId", "createTime",
])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def response_evidence(payload):
    evidence = copy.deepcopy(payload)
    images = []
    for candidate in evidence.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            signature = part.pop("thoughtSignature", None)
            if signature:
                part["thoughtSignatureEvidence"] = {
                    "characters": len(signature), "sha256": sha(signature.encode()),
                }
            inline = part.get("inlineData", {})
            if "data" in inline:
                encoded = inline.pop("data")
                inline["encoded_sha256"] = sha(encoded.encode())
                if not part.get("thought"):
                    images.append(inline)
    return evidence, images


def validate_output(payload):
    _, images = response_evidence(payload)
    candidates = payload.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP" or len(images) != 1:
        raise ValueError("Expected one completed non-thought image")
    image_parts = [p for p in candidates[0]["content"]["parts"]
                   if "inlineData" in p and not p.get("thought")]
    return _extract_image({"candidates": [{"content": {"parts": image_parts}}]})


def schedule(field_mask=False):
    rows = []
    for case, scene in enumerate(SCENES):
        arms = ["text_image", "image", "image", "text_image"]
        if case % 2:
            arms = ["image", "text_image", "text_image", "image"]
        for position, arm in enumerate(arms):
            repeat = position // 2
            request = FastSceneRequest(
                scene_id=f"modality-{case}-{repeat}", prompt=STYLE + scene,
                seed=7400 + case * 2 + repeat,
            )
            payload = _request_payload(request)
            if arm == "image":
                if field_mask:
                    arm = "field_mask"
                else:
                    payload["generationConfig"]["responseModalities"] = ["IMAGE"]
            rows.append({
                "ordinal": len(rows), "case": case, "repeat": repeat, "arm": arm,
                "payload": payload,
            })
    return rows


def summarize(directory):
    records = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
    complete = [r for r in records if r["event"] == "complete"]
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = manifest["operations"]
    candidate = "field_mask" if manifest.get("field_mask") else "image"
    dispatched = [r for r in records if r["event"] == "dispatch"]
    assert [r["ordinal"] for r in dispatched] == list(range(len(dispatched)))
    assert len(dispatched) <= len(expected)
    assert len({r["ordinal"] for r in complete}) == len(complete)
    for r in records:
        row = expected[r["ordinal"]]
        assert all(r[k] == row[k] for k in ("case", "repeat", "arm"))
        assert r["ordinal"] < len(dispatched)
    for r in complete:
        assert sha((directory / r["image_file"]).read_bytes()) == r["image_sha256"]
    arms = {}
    for arm in ("text_image", candidate):
        times = [r["artifact_ready_ms"] for r in complete if r["arm"] == arm]
        arms[arm] = {"n": len(times), "median_ms": statistics.median(times) if times else None,
                     "max_ms": max(times) if times else None}
    result = {"arms": arms, "attempts": sum(r["event"] == "dispatch" for r in records),
              "complete": len(complete), "failures": sum(r["event"] == "failed" for r in records),
              "median_improvement_percent": None, "paired_improvements_percent": []}
    for case in range(6):
        for repeat in range(2):
            pair = {r["arm"]: r for r in complete if (r["case"], r["repeat"]) == (case, repeat)}
            if len(pair) == 2:
                result["paired_improvements_percent"].append(
                    100 * (1 - pair[candidate]["artifact_ready_ms"]
                           / pair["text_image"]["artifact_ready_ms"])
                )
    if all(arms[a]["n"] == 12 for a in arms):
        result["median_improvement_percent"] = 100 * (
            1 - arms[candidate]["median_ms"] / arms["text_image"]["median_ms"]
        )
    if manifest.get("field_mask"):
        pairs = []
        for case in range(6):
            for repeat in range(2):
                pair = {r["arm"]: r for r in complete
                        if (r["case"], r["repeat"]) == (case, repeat)}
                if len(pair) != 2:
                    continue
                baseline, filtered = pair["text_image"], pair[candidate]
                decoded = []
                for r in (baseline, filtered):
                    with Image.open(directory / r["image_file"]) as im:
                        decoded.append((im.size, sha(im.convert("RGB").tobytes())))
                pairs.append({
                    "case": case, "repeat": repeat,
                    "identical_pixels_and_dimensions": decoded[0] == decoded[1],
                    "response_byte_reduction_percent": 100 * (
                        1 - filtered["response_bytes"] / baseline["response_bytes"]),
                    "baseline_response_bytes": baseline["response_bytes"],
                    "candidate_response_bytes": filtered["response_bytes"],
                })
        result["pairs"] = pairs
        result["promotion_qualified"] = (
            len(pairs) == 12 and result["failures"] == 0
            and result["median_improvement_percent"] >= 5
            and arms[candidate]["max_ms"] <= arms["text_image"]["max_ms"]
            and all(p["identical_pixels_and_dimensions"]
                    and p["response_byte_reduction_percent"] >= 60 for p in pairs)
        )
    return result


async def run(directory, *, field_mask=False, interval=0):
    directory.mkdir(parents=True, exist_ok=False)
    rows = schedule(field_mask)
    endpoint = ("https://aiplatform.googleapis.com/v1/projects/your-gcp-project/"
                f"locations/global/publishers/google/models/{DEFAULT_MODEL}:generateContent")
    manifest = {"created_at": datetime.now(UTC).isoformat(), "model": DEFAULT_MODEL,
                "endpoint": endpoint, "script_sha256": sha(Path(__file__).read_bytes()),
                "provider_sha256": sha(Path(vertex_scene_provider.__file__).read_bytes()),
                "python": platform.python_version(), "platform": platform.platform(),
                "httpx": importlib.metadata.version("httpx"), "http2": False,
                "field_mask": FIELD_MASK if field_mask else None,
                "minimum_dispatch_interval_seconds": interval,
                "maximum_requests": 24, "deadline_seconds": 600,
                "image_component_estimate_usd": 24 * 0.0336,
                "total_planning_allowance_usd": 3.0, "automatic_retries": 0,
                "maximum_request_reservation_usd": 4096 * 30 / 1e6 + 4000 * 0.25 / 1e6,
                "client": "Mac; direct Vertex; excludes Jetson and browser",
                "promotion_gate": (
                    "5% median gain; 60% fewer bytes; identical pixels; no worse maximum/failures"
                    if field_mask else "20% median gain; no worse maximum/failures; visual review"
                ),
                "operations": rows}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    token = await asyncio.wait_for(google_access_token(), timeout=30)
    with (directory / "journal.jsonl").open("x") as journal:
        def record(event):
            journal.write(json.dumps(event) + "\n")
            journal.flush()

        async with asyncio.timeout(600), httpx.AsyncClient(
            http2=False, follow_redirects=False, timeout=90,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1,
                                keepalive_expiry=300),
        ) as client:
            previous_dispatch = 0
            for row in rows:
                await asyncio.sleep(max(0, interval - (time.monotonic() - previous_dispatch)))
                previous_dispatch = time.monotonic()
                assert (row["ordinal"] + 1) * manifest["maximum_request_reservation_usd"] <= 3.0
                metadata = {k: v for k, v in row.items() if k != "payload"}
                record({**metadata, "event": "dispatch", "time": datetime.now(UTC).isoformat(),
                        "payload_sha256": sha(json.dumps(row["payload"], sort_keys=True).encode())})
                started = time.perf_counter()
                try:
                    headers = {"Authorization": f"Bearer {token}"}
                    if row["arm"] == "field_mask":
                        headers["X-Goog-FieldMask"] = FIELD_MASK
                    async with asyncio.timeout(90), client.stream(
                        "POST", endpoint, headers=headers,
                        json=row["payload"],
                    ) as response:
                        headers_ms = (time.perf_counter() - started) * 1000
                        chunks = []
                        size = 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > 16 * 1024 * 1024:
                                raise ValueError("response exceeds bound")
                            chunks.append(chunk)
                        received_ms = (time.perf_counter() - started) * 1000
                        payload = json.loads(b"".join(chunks))
                        evidence, _ = response_evidence(payload)
                        (directory / f"{row['ordinal']:02d}-response.json").write_text(
                            json.dumps(evidence, indent=2) + "\n")
                        response.raise_for_status()
                    image, mime = validate_output(payload)
                    width, height = _image_dimensions(image, mime)
                    with Image.open(io.BytesIO(image)) as decoded:
                        pixels_sha256 = sha(decoded.convert("RGB").tobytes())
                    filename = f"{row['ordinal']:02d}.{'jpg' if mime == 'image/jpeg' else 'png'}"
                    (directory / filename).write_bytes(image)
                    result = {**metadata, "event": "complete", "status": response.status_code,
                              "response_headers_ms": headers_ms,
                              "response_received_ms": received_ms,
                              "artifact_ready_ms": (time.perf_counter() - started) * 1000,
                              "response_bytes": size, "image_bytes": len(image),
                              "image_file": filename, "image_sha256": sha(image),
                              "pixels_sha256": pixels_sha256,
                              "width": width, "height": height, "mime": mime,
                              "model_version": payload.get("modelVersion"),
                              "usage": payload.get("usageMetadata"),
                              "text_characters": sum(
                                  len(p.get("text", "")) for c in payload.get("candidates", [])
                                  for p in c.get("content", {}).get("parts", []))}
                    record(result)
                    print(json.dumps({k: result[k] for k in (
                        "ordinal", "arm", "artifact_ready_ms", "text_characters",
                    )}), flush=True)
                except BaseException as error:
                    record({**metadata, "event": "failed", "error_type": type(error).__name__,
                            "status": getattr(getattr(error, "response", None),
                                              "status_code", None)})
                    raise
    (directory / "summary.json").write_text(json.dumps(summarize(directory), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--run", action="store_true", help="Makes up to 24 billable image calls")
    parser.add_argument("--field-mask", action="store_true", help="Compare signature filtering")
    parser.add_argument("--interval", type=float, default=0, help="Seconds between dispatches")
    args = parser.parse_args()
    if args.run:
        if not 0 <= args.interval <= 20:
            parser.error("interval must be between 0 and 20 seconds")
        asyncio.run(run(args.directory, field_mask=args.field_mask, interval=args.interval))
    else:
        print(json.dumps(summarize(args.directory), indent=2))
