"""Generate each demo scene once using the configured API, then save a local catalog."""

import argparse
import json
import time
from pathlib import Path

import httpx

STYLE = "rich luminous watercolor storybook illustration, layered depth, full-bleed 16:9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, default=Path("examples/alice-demo.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new catalog path")
    scenes = json.loads(args.scenes.read_text())["scenes"]
    prepared = []
    with httpx.Client(base_url=args.api.rstrip("/"), timeout=120) as client:
        for scene in scenes:
            response = client.post("/v1/live-scenes", json={
                "text": scene["text"], "visual_style": STYLE, "seed": 42,
                "display_when_complete": True,
            })
            response.raise_for_status()
            job = response.json()
            deadline = time.monotonic() + 300
            while not job["complete"] and job["stage"] != "failed":
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Scene still running: {job['job_id']}")
                time.sleep(1)
                response = client.get(f"/v1/live-scenes/{job['job_id']}")
                response.raise_for_status()
                job = response.json()
            if job["stage"] == "failed" or not job.get("story_pack"):
                raise RuntimeError(f"Scene preparation failed: {job['job_id']}")
            prepared.append({key: scene[key] for key in ("scene_id", "title", "cues")}
                            | {"request": job["request"], "pack": job["story_pack"],
                               "provider": job["provider"]})
            print(f"Prepared {scene['title']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump({"scenes": prepared}, target, indent=2)
    print(f"Catalog: {args.output}. Keep it with the API's existing asset cache.")


if __name__ == "__main__":
    main()
