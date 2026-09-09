"""Measure four synthetic local scene checks without requesting images."""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

CASES = (
    "A brown fox carries a lantern.",
    "A green cat chases a mouse.",
    "A golden retriever eats dinner.",
    "A red balloon.",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "httpx": httpx.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_requests": 0,
        "cases": [],
    }
    with args.output.open("x") as output, httpx.Client(timeout=15, trust_env=False) as client:
        for text in CASES:
            started = time.perf_counter()
            response = client.post(
                "http://127.0.0.1:18767/v1/live-scene-planner/prepare",
                json={
                    "text": text,
                    "visual_style": "rich watercolor",
                    "seed": 0,
                    "session_id": "voice-overlap-check",
                    "reviewed_description": True,
                },
            )
            elapsed = (time.perf_counter() - started) * 1000
            data = response.json()
            report["cases"].append({
                "synthetic_text": text,
                "status": response.status_code,
                "wall_ms": round(elapsed, 3),
                **{
                    key: data.get(key)
                    for key in ("ready", "planning_ms", "cache_hit", "model", "revision")
                },
            })
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
