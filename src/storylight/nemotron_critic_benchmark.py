"""Paired critic-only experiments on a previously generated, inspected illustration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import httpx

from storylight.anticipatory_gcp import _nemotron_review_copy
from storylight.nemotron_critic import (
    NEMOTRON_CRITIC_SYSTEM_PROMPT,
    NEMOTRON_CRITIC_WIRE_SCHEMA,
    NemotronCriticError,
    NemotronCriticRequest,
    NemotronVisionCritic,
)

AUTHORIZATION = "I_AUTHORIZE_BOUNDED_GPU_REVIEW_OF_THIS_GENERATED_IMAGE"
FOX_IMAGE_SHA256 = "fcc298275a6a37473e614af7bab8cdc700edf28035d4be0469e542f03624235a"
OBSERVATION_PROMPT = (
    "Return compact JSON. First r: describe what is ACTUALLY VISIBLE in 8-12 words, "
    "including animal count, physical action/contact, and relative positions. "
    "Describe the image, not the requested brief; never assume the brief is true. "
    "Then compare that observation with ALL requirements in the brief. "
    "Set d=accept only when all requirements match; otherwise d=refine or reject. "
    "Set f=fidelity, c=composition, p=projection legibility (0..1), "
    "i=identity consistent, t=unintended text. For refine/reject, "
    "x gives a concrete correction in at most 15 words. "
    "Never request passage, reader, audio, or camera data."
)
PROMPTS = {"baseline": NEMOTRON_CRITIC_SYSTEM_PROMPT, "observation_first": OBSERVATION_PROMPT}
OBSERVATION_SCHEMA = {
    **NEMOTRON_CRITIC_WIRE_SCHEMA,
    "properties": {
        key: NEMOTRON_CRITIC_WIRE_SCHEMA["properties"][key]
        for key in ("r", "d", "f", "c", "p", "i", "t", "x")
    },
    "required": ["r", "d", "f", "c", "p", "i", "t"],
}


def regression_cases() -> list[tuple[str, bool, NemotronCriticRequest]]:
    descriptions = [
        (
            "beside_positive",
            True,
            "A silver fox stands beside a glowing golden lantern in a forest.",
        ),
        (
            "carrying_negative",
            False,
            "A silver fox carries a glowing golden lantern in its mouth through a forest.",
        ),
        (
            "two_foxes_negative",
            False,
            "Two silver foxes stand beside a glowing golden lantern in a forest.",
        ),
        (
            "dragon_negative",
            False,
            "A blue dragon flies over a snowy castle, with no fox or lantern.",
        ),
        (
            "left_positive",
            True,
            "One silver fox stands to the left of a glowing golden lantern among tall trees.",
        ),
        (
            "right_negative",
            False,
            "One silver fox stands to the right of a glowing golden lantern among tall trees.",
        ),
    ]
    return [
        (
            name,
            accept,
            NemotronCriticRequest(
                visual_brief=brief,
                forbidden_content=["readable text", "interface chrome"],
            ),
        )
        for name, accept, brief in descriptions
    ]


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for variant in PROMPTS:
        selected = [row for row in rows if row["variant"] == variant]
        successful = [row for row in selected if "evidence" in row]
        latencies = sorted(row["evidence"]["latency_ms"] for row in successful)
        summaries[variant] = {
            "calls": len(selected),
            "errors": len(selected) - len(successful),
            "correct": sum(row["correct"] for row in selected),
            "false_accepts": sum(
                row["accepted"] and not row["expected_accept"] for row in successful
            ),
            "false_rejects": sum(
                not row["accepted"] and row["expected_accept"] for row in successful
            ),
            "median_ms": median(latencies) if latencies else None,
            "p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None,
            "median_output_tokens": median(row["evidence"]["output_tokens"] for row in successful)
            if successful
            else None,
        }
    return summaries


def paired_cases(repeats: int):
    for repeat in range(repeats):
        for index, case in enumerate(regression_cases()):
            order = list(PROMPTS) if (repeat + index) % 2 == 0 else list(reversed(PROMPTS))
            for name in order:
                yield repeat + 1, name, case


async def run_benchmark(
    *,
    image_bytes: bytes,
    base_url: str,
    repeats: int = 2,
    critic_factory: Callable[..., NemotronVisionCritic] = NemotronVisionCritic,
) -> dict:
    if not 1 <= repeats <= 3:
        raise ValueError("repeats must be 1-3")
    if hashlib.sha256(image_bytes).hexdigest() != FOX_IMAGE_SHA256:
        raise ValueError("this regression suite requires the inspected generated fox fixture")
    review = _nemotron_review_copy(image_bytes)
    rows = []
    warmups = []
    critics = {}
    transport_interrupted = False
    try:
        for name, prompt in PROMPTS.items():
            critic = critic_factory(base_url=base_url, allow_loopback_http=True, timeout_seconds=90)
            critic.system_prompt = prompt
            if name == "observation_first":
                critic.wire_schema = OBSERVATION_SCHEMA
            critics[name] = critic
            evidence = await critic.evaluate(
                regression_cases()[0][2], image_bytes=review, media_type="image/jpeg"
            )
            warmups.append({"variant": name, "evidence": evidence.model_dump(mode="json")})
        for repeat, name, (case_id, expected_accept, request) in paired_cases(repeats):
            row = {
                "variant": name,
                "case_id": case_id,
                "repeat": repeat,
                "expected_accept": expected_accept,
                "correct": False,
                "request": request.model_dump(mode="json"),
            }
            try:
                evidence = await critics[name].evaluate(
                    request, image_bytes=review, media_type="image/jpeg"
                )
                row.update(
                    evidence=evidence.model_dump(mode="json"),
                    accepted=evidence.verdict.decision == "accept",
                )
                row["correct"] = row["accepted"] == expected_accept
            except NemotronCriticError as error:
                row["error"] = str(error)
                transport_interrupted = isinstance(error.__cause__, httpx.TransportError)
            rows.append(row)
            print(
                json.dumps({key: row[key] for key in ("variant", "case_id", "repeat", "correct")}),
                flush=True,
            )
            if transport_interrupted:
                break
    finally:
        for critic in critics.values():
            await critic.aclose()
    return {
        "schema_version": "1.0",
        "evidence_kind": "measured_nim_critic_paired_regression",
        "created_at": datetime.now(UTC).isoformat(),
        "transport_interrupted": transport_interrupted,
        "endpoint": base_url,
        "source_image_sha256": FOX_IMAGE_SHA256,
        "review_image_sha256": hashlib.sha256(review).hexdigest(),
        "privacy": "previously generated synthetic image; no reader media or passage",
        "limitations": (
            "Six contracts on one inspected image, not a general fidelity or latency benchmark."
        ),
        "prompts": PROMPTS,
        "observation_schema": OBSERVATION_SCHEMA,
        "warmups": warmups,
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--authorization", choices=(AUTHORIZATION,), required=True)
    args = parser.parse_args()
    # Reserve the evidence path before any paid request; never overwrite a prior run.
    with args.output.open("x", encoding="utf-8") as output:
        report = asyncio.run(
            run_benchmark(
                image_bytes=args.image.read_bytes(), base_url=args.base_url, repeats=args.repeats
            )
        )
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
