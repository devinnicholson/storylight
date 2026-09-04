"""Record actual local slots and every contract boundary, without calling a cloud model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

from bookforge.tensorrt_slot_client import _slot_messages, tensor_slot_wire_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18435")
    parser.add_argument("--split", choices=("development", "validation", "privacy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if httpx.URL(args.base_url).host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("planner must be exposed through local loopback")
    corpus_path = Path(__file__).with_name("overnight-corpus.json")
    corpus = json.loads(corpus_path.read_text())
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "split": args.split,
        "model": "actual Jetson Gemma TensorRT slots",
        "cases": [],
    }
    with httpx.Client(base_url=args.base_url, timeout=25) as client:
        for index, case in enumerate(corpus[args.split]):
            started = time.perf_counter()
            row = {**case, "seed": 20260904 + index}
            try:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llm",
                        "messages": _slot_messages(case["text"]),
                        "max_tokens": 64,
                        "temperature": 0,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                row.update(
                    planner_ms=(time.perf_counter() - started) * 1000,
                    slots=payload["choices"][0]["message"]["content"],
                    usage=payload.get("usage"),
                )
                wire = tensor_slot_wire_plan(row["slots"], source_text=case["text"])
                row["wire"] = wire.model_dump(mode="json")
                plan = wire.to_live_scene_plan(context_text=case["text"])
                row["plan"] = plan.model_dump(mode="json")
                page = plan.to_page(
                    source_text=case["text"],
                    visual_style="Luminous watercolor illustration",
                    seed=row["seed"],
                )
                row["full_prompt"] = page.scene_spec.master_prompt
                row["concise_prompt"] = plan.to_page(
                    source_text=case["text"],
                    visual_style="Luminous watercolor illustration",
                    seed=row["seed"],
                    render_contract="concise",
                ).scene_spec.master_prompt
            except Exception as error:
                row["failure"] = {"type": type(error).__name__, "detail": str(error)}
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "planner_ms": row.get("planner_ms"),
                        "failure": row.get("failure"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
