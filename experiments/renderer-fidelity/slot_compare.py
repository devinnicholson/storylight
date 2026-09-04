"""Finite local TensorRT prompt comparison using synthetic passages only."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import httpx

from bookforge.live_scene_planner import validate_live_scene_plan_privacy
from bookforge.planner_benchmark import CASES
from bookforge.tensorrt_slot_client import (
    TENSORRT_SLOT_SYSTEM_PROMPT,
    tensor_slot_wire_plan,
)

CANDIDATE = TENSORRT_SLOT_SYSTEM_PROMPT.replace(
    "MAGIC is one supported result or transformation.",
    "MAGIC is one supported result or transformation. For a non-magical scene, MAGIC is "
    "its most important distinct secondary object or visual detail, not a repetition of the "
    "setting or actor. Keep explicit counts and spatial relationships even when objects "
    "are stationary.",
)
STATIC_CASES = [
    "A single golden paper boat floats on a calm indigo pond beneath a crescent moon.",
    "One silver fox stands to the left of a glowing golden lantern among tall cedar trees.",
    "A single owl perches on a tree branch above a calm sea. "
    "A lighthouse is far to the left of the owl.",
    "One red fox carries a small golden lantern in its mouth while walking through a snowy forest.",
    "Exactly two red paper boats float side by side on a calm blue pond.",
    "One child stands on a wooden bridge holding an open green book in both hands. "
    "No other people.",
]


async def run(endpoint: str, output: Path) -> None:
    report = {
        "baseline_system": TENSORRT_SLOT_SYSTEM_PROMPT,
        "candidate_system": CANDIDATE,
        "automatic_retries": 0,
        "samples": [],
    }
    with output.open("x") as file:
        async with httpx.AsyncClient(base_url=endpoint, timeout=20) as client:
            try:
                for index, passage in enumerate(STATIC_CASES + [case.text for case in CASES]):
                    variants = [("baseline", TENSORRT_SLOT_SYSTEM_PROMPT), ("candidate", CANDIDATE)]
                    if index % 2:
                        variants.reverse()
                    for variant, system in variants:
                        row = {"case": index, "passage": passage, "variant": variant}
                        started = perf_counter()
                        response = await client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "llm",
                                "temperature": 0,
                                "top_p": 1,
                                "max_tokens": 64,
                                "stream": False,
                                "messages": [
                                    {"role": "system", "content": system},
                                    {
                                        "role": "user",
                                    "content": (
                                        f"STORY:\n{passage}\nSelect the single focal event "
                                        "and return the four required lines."
                                    ),
                                    },
                                ],
                            },
                        )
                        response.raise_for_status()
                        data = response.json()
                        choice = data["choices"][0]
                        row.update(
                            wall_ms=(perf_counter() - started) * 1000,
                            finish_reason=choice.get("finish_reason"),
                            raw=choice["message"]["content"],
                            usage=data.get("usage"),
                        )
                        try:
                            wire = tensor_slot_wire_plan(row["raw"], source_text=passage)
                            plan = wire.to_live_scene_plan(context_text=passage)
                            validate_live_scene_plan_privacy(plan, source_text=passage)
                            row.update(privacy_pass=True, wire=wire.model_dump())
                        except ValueError as error:
                            row.update(privacy_pass=False, error=str(error))
                        report["samples"].append(row)
                        print(json.dumps(row), flush=True)
            finally:
                json.dump(report, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11436")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if httpx.URL(args.endpoint).host != "127.0.0.1":
        parser.error("only a loopback endpoint or SSH tunnel is accepted")
    asyncio.run(run(args.endpoint, args.output))
