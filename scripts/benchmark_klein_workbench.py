"""Bounded, explicit paid smoke through the candidate's real HTTP job boundary."""

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18087")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite benchmark evidence")
    report = {"boundary": "HTTP job complete, not browser first draw", "cases": []}
    with httpx.Client(base_url=args.base_url, timeout=330) as client:
        response = client.get("/v1/live-scene-provider/warm-status")
        response.raise_for_status()
        if "Klein" not in response.json().get("detail", ""):
            raise ValueError("Expected the opt-in Klein candidate")
        response = client.post(
            "/v1/live-scene-provider/prewarm",
            json={
                "prewarm_id": "overnight-http-benchmark",
                "scaledown_window_seconds": 90,
            },
        )
        response.raise_for_status()
        report["preparation"] = response.json()
        passages = [
            "A brown bear carries one blue balloon across a grassy meadow.",
            "Two yellow boats float on a violet lake.",
            "A small turtle rests beside a purple watering can in a garden.",
        ]
        for index, text in enumerate(passages):
            started = time.perf_counter()
            response = client.post(
                "/v1/live-scenes",
                json={
                    "text": text,
                    "seed": 202609040 + index,
                    "session_id": "klein-http-benchmark",
                },
            )
            response.raise_for_status()
            job = response.json()
            while not job["complete"] and time.perf_counter() - started < 330:
                time.sleep(0.15)
                response = client.get(f"/v1/live-scenes/{job['job_id']}")
                response.raise_for_status()
                job = response.json()
            row = {
                "text": text,
                "client_job_complete_ms": 1000 * (time.perf_counter() - started),
                "job": job,
            }
            report["cases"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"id": job["job_id"], "stage": job["stage"], "metrics": job["metrics"]}),
                flush=True,
            )
            if not job["complete"] or job.get("error"):
                raise RuntimeError("Smoke failed; no automatic retry or further paid work")


if __name__ == "__main__":
    main()
