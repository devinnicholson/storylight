"""Read-only Monitoring evidence for release of one deleted qualification service.

Instance count is sampled every 60s and can arrive 120s late:
https://docs.cloud.google.com/monitoring/api/metrics_gcp_p_z
Missing data is unknown. This cannot reserve GPU quota or prove provider inventory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PROJECT = "your-gcp-project"
REGION = "us-central1"
METRIC = "run.googleapis.com/container/instance_count"
URL = f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/timeSeries"
MAX_BYTES = 2 * 1024 * 1024
DEADLINE_SECONDS = 900


def utc(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def timestamp(value: str) -> float:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z", value
    ):
        raise ValueError("invalid UTC timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def decode(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON")

    if len(raw) > MAX_BYTES:
        raise ValueError("response too large")
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("invalid response")
    return result


def query(service: str, after: float, now: float) -> dict:
    return {
        "filter": (
            f'metric.type="{METRIC}" AND resource.type="cloud_run_revision" '
            f'AND resource.labels.project_id="{PROJECT}" '
            f'AND resource.labels.location="{REGION}" '
            f'AND resource.labels.service_name="{service}"'
        ),
        "interval.startTime": utc(after - 900),
        "interval.endTime": utc(now),
        "view": "FULL",
        "pageSize": "10000",
    }


def evaluate(data: dict, service: str, after: float, now: float, seen: set[str]) -> dict:
    seen.add(f"{service}-trial")
    result = {"proven_zero": False, "reason": "unknown", "revisions": sorted(seen)}
    try:
        if (
            ("nextPageToken" in data and data["nextPageToken"] != "")
            or data.get("executionErrors")
            or "error" in data
        ):
            raise ValueError("incomplete response")
        series = data["timeSeries"]
        if not isinstance(series, list) or not series:
            raise ValueError("missing series")
        samples = {}
        for item in series:
            labels = item["resource"]["labels"]
            if (
                item["resource"]["type"] != "cloud_run_revision"
                or labels["project_id"] != PROJECT
                or labels["location"] != REGION
                or labels["service_name"] != service
                or item["metric"]["type"] != METRIC
                or item["metricKind"] != "GAUGE"
                or item["valueType"] != "INT64"
            ):
                raise ValueError("wrong series")
            revision = labels["revision_name"]
            state = item["metric"]["labels"]["state"]
            if not isinstance(revision, str) or not re.fullmatch(r"[a-z0-9-]{1,128}", revision):
                raise ValueError("invalid revision")
            if state not in {"active", "idle"} or (revision, state) in samples:
                raise ValueError("ambiguous series")
            seen.add(revision)
            points = item["points"]
            if not isinstance(points, list) or not points:
                raise ValueError("empty series")
            samples[revision, state] = {}
            for point in points:
                end = timestamp(point["interval"]["endTime"])
                value = point["value"]["int64Value"]
                if not isinstance(value, str) or not re.fullmatch(r"0|[1-9]\d{0,18}", value):
                    raise ValueError("invalid count")
                if end > now or end in samples[revision, state]:
                    raise ValueError("invalid sample time")
                samples[revision, state][end] = int(value)
        expected = {(revision, state) for revision in seen for state in ("active", "idle")}
        if set(samples) != expected:
            raise ValueError("missing state or revision")
        times = sorted({t for points in samples.values() for t in points}, reverse=True)
        if len(times) < 2:
            raise ValueError("missing samples")
        latest, previous = times[:2]
        if any(t not in points for points in samples.values() for t in (latest, previous)):
            raise ValueError("incomplete latest samples")
        totals = [sum(points[t] for points in samples.values()) for t in (previous, latest)]
        result.update(
            sample_times=[utc(previous), utc(latest)],
            summed_instance_counts=totals,
            latest_age_seconds=now - latest,
        )
        if previous <= after or latest - previous < 60 or now - latest > 180:
            result["reason"] = "unqualified_sample_times"
        elif any(totals):
            result["reason"] = "nonzero"
        elif now - after < 600:
            result["reason"] = "minimum_release_wait"
        else:
            result.update(proven_zero=True, reason="two_complete_zero_samples")
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    result["revisions"] = sorted(seen)
    return result


def write(path: Path, value: dict | bytes) -> None:
    raw = (
        value
        if isinstance(value, bytes)
        else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


async def monitoring_read(params: dict, timeout: float) -> bytes:
    async with asyncio.timeout(timeout):
        process = await asyncio.create_subprocess_exec(
            "gcloud",
            "auth",
            "print-access-token",
            "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=min(10, timeout))
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        token = stdout.decode().strip()
        if process.returncode or not token or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token):
            raise ValueError("token unavailable")
        async with (
            httpx.AsyncClient(trust_env=False, follow_redirects=False) as client,
            client.stream(
                "GET", URL, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=20
            ) as response,
        ):
            if response.status_code != 200:
                raise ValueError("monitoring read failed")
            chunks, length = [], 0
            async for chunk in response.aiter_bytes():
                length += len(chunk)
                if length > MAX_BYTES:
                    raise ValueError("response too large")
                chunks.append(chunk)
            return b"".join(chunks)


async def wait_for_release(
    service: str,
    after_text: str,
    output: Path,
    *,
    reader=monitoring_read,
    now=time.time,
    monotonic=time.monotonic,
    sleep=asyncio.sleep,
) -> dict:
    if not re.fullmatch(r"bookforge-klein-qualification-20260906-[a-z]", service):
        raise ValueError("invalid qualification service")
    after = timestamp(after_text)
    if not math.isfinite(after) or after > now():
        raise ValueError("invalid deletion audit time")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    deadline = monotonic() + DEADLINE_SECONDS
    write(
        output / "attempt.json",
        {
            "schema_version": 1,
            "service": service,
            "after": after_text,
            "project": PROJECT,
            "region": REGION,
            "started": utc(now()),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "deadline_seconds": DEADLINE_SECONDS,
            "minimum_after_seconds": 600,
            "interpretation": (
                "Monitoring evidence only; not a GPU quota reservation "
                "or provider inventory guarantee."
            ),
        },
    )
    seen, ordinal, last = set(), 0, None
    while monotonic() < deadline:
        poll_started = monotonic()
        params = query(service, after, now())
        prefix = output / f"read-{ordinal:02}"
        write(prefix.with_suffix(".query.json"), {"url": URL, "params": params})
        try:
            raw = await reader(params, min(30, deadline - monotonic()))
            write(prefix.with_suffix(".response.json"), raw)
            last = evaluate(decode(raw), service, after, now(), seen)
            last["response_sha256"] = hashlib.sha256(raw).hexdigest()
        except Exception:
            last = {
                "proven_zero": False,
                "reason": "read_or_validation_failed",
                "revisions": sorted(seen),
            }
        last.update(observed_at=utc(now()), ordinal=ordinal)
        if monotonic() >= deadline:
            last.update(proven_zero=False, reason="deadline")
        write(prefix.with_suffix(".proof.json"), last)
        ordinal += 1
        if last["proven_zero"]:
            break
        delay = min(max(0, 60 - (monotonic() - poll_started)), max(0, deadline - monotonic()))
        if delay:
            await sleep(delay)
    summary = {
        "schema_version": 1,
        "decision": "observed_zero" if last and last["proven_zero"] else "unknown",
        "reads": ordinal,
        "last_proof": last,
        "quota_reserved": False,
    }
    write(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument(
        "--after", required=True, help="Actual deletion-audit timestamp in ISO UTC ending Z"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = asyncio.run(wait_for_release(args.service, args.after, args.output))
        print(json.dumps({"decision": result["decision"], "reads": result["reads"]}))
        return 0 if result["decision"] == "observed_zero" else 1
    except (Exception, KeyboardInterrupt):
        print('{"decision":"unknown","reason":"guard_failed"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
