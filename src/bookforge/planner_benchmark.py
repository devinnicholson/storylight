"""Repeatable, local-only acceptance benchmark for the Jetson scene planner."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from bookforge.config import Settings
from bookforge.live_scene_planner import StructuredLiveScenePlanner
from bookforge.model_client import OllamaClient

Contract = Literal["standard", "compact"]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    text: str
    visual_style: str
    seed: int


CASES = (
    BenchmarkCase(
        case_id="origami_library",
        text=(
            "On the darkest night, a child opens a silent book and every unwritten "
            "letter becomes a luminous origami bird, forming a bridge of constellations "
            "toward a floating school above the clouds."
        ),
        visual_style=(
            "luminous cinematic layered paper theater, projection-bright cobalt and cyan "
            "with warm amber light, tactile paper fibers, full-bleed 16:9"
        ),
        seed=20261026,
    ),
    BenchmarkCase(
        case_id="whale_library",
        text=(
            "A silver whale swims through a flooded library, carrying a tiny lantern while "
            "books open into schools of bright fish."
        ),
        visual_style=(
            "hand-painted layered paper theater, bright indigo water, warm gold windows, "
            "crisp silhouettes, full-bleed 16:9"
        ),
        seed=20261027,
    ),
    BenchmarkCase(
        case_id="clockwork_fox",
        text=(
            "A clockwork fox plants a brass seed in the snow, and a transparent forest of "
            "glass branches rises around it."
        ),
        visual_style=(
            "luminous cut-paper storybook, icy blue midtones, copper light, tactile layers, "
            "projector-bright full-bleed 16:9"
        ),
        seed=20261028,
    ),
    BenchmarkCase(
        case_id="classroom_garden",
        text=(
            "A student lifts one folded butterfly from a desk, and the classroom ceiling "
            "blooms into a floating garden of paper flowers."
        ),
        visual_style=(
            "joyful watercolor paper theater, bright coral and turquoise, clear central "
            "subject, layered depth, full-bleed 16:9"
        ),
        seed=20261029,
    ),
    BenchmarkCase(
        case_id="moon_turtle",
        text=(
            "A moonlit turtle climbs a staircase made of clouds while glowing jellyfish "
            "drift between the stars."
        ),
        visual_style=(
            "dreamlike layered paper theater, lifted violet and teal midtones, warm rim light, "
            "crisp silhouettes, full-bleed 16:9"
        ),
        seed=20261030,
    ),
)


def _require_loopback(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("planner benchmark requires a loopback model endpoint")
    return base_url.rstrip("/")


def _summarize(cases: list[dict[str, object]]) -> dict[str, float | int]:
    planning = [float(case["planning_ms"]) for case in cases]
    tokens = [int(case["output_tokens"]) for case in cases]
    return {
        "cases": len(cases),
        "mean_planning_ms": round(statistics.fmean(planning), 3),
        "median_planning_ms": round(statistics.median(planning), 3),
        "maximum_planning_ms": round(max(planning), 3),
        "mean_output_tokens": round(statistics.fmean(tokens), 3),
        "maximum_output_tokens": max(tokens),
    }


async def _run_contract(
    client: OllamaClient,
    *,
    contract: Contract,
    timeout_seconds: float,
    model_revision: str,
) -> dict[str, object]:
    planner = StructuredLiveScenePlanner(
        client,
        timeout_seconds=timeout_seconds,
        model_revision=model_revision,
        compact_wire=contract == "compact",
    )
    results: list[dict[str, object]] = []
    for case in CASES:
        result = await planner.plan(
            text=case.text,
            visual_style=case.visual_style,
            seed=case.seed,
        )
        results.append(
            {
                "case_id": case.case_id,
                "seed": case.seed,
                "planning_ms": round(result.wall_ms, 3),
                "model_total_ms": round(result.metrics.total_ms, 3),
                "load_ms": round(result.metrics.load_ms, 3),
                "input_tokens": result.metrics.input_tokens,
                "output_tokens": result.metrics.output_tokens,
                "scene_summary": result.plan.scene_summary,
                "background": result.plan.background_prompt,
                "focus": result.plan.focus.prompt,
                "magic": result.plan.accent.prompt,
            }
        )
    return {"contract": contract, "summary": _summarize(results), "cases": results}


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    base_url = _require_loopback(args.base_url)
    settings = Settings(
        _env_file=None,
        model_backend="ollama",
        model_name=args.model,
        model_base_url=base_url,
        model_timeout_seconds=args.model_timeout_seconds,
        model_keep_alive=args.keep_alive,
        model_context_tokens=args.context_tokens,
        model_max_output_tokens=args.max_output_tokens,
    )
    client = OllamaClient(settings)
    contracts: tuple[Contract, ...] = (
        ("standard", "compact") if args.contract == "both" else (args.contract,)
    )
    try:
        ready, detail = await client.probe()
        if not ready:
            raise RuntimeError(detail)
        if not args.skip_warmup:
            warmup = StructuredLiveScenePlanner(
                client,
                timeout_seconds=max(args.planner_timeout_seconds, 30),
                model_revision=args.model_revision,
                compact_wire=contracts[0] == "compact",
            )
            await warmup.plan(
                text=CASES[0].text,
                visual_style=CASES[0].visual_style,
                seed=CASES[0].seed,
            )
        results = [
            await _run_contract(
                client,
                contract=contract,
                timeout_seconds=args.planner_timeout_seconds,
                model_revision=args.model_revision,
            )
            for contract in contracts
        ]
    finally:
        await client.client.aclose()

    technical_pass = all(
        float(result["summary"]["maximum_planning_ms"])  # type: ignore[index]
        <= args.planner_timeout_seconds * 1_000
        and int(result["summary"]["maximum_output_tokens"])  # type: ignore[index]
        <= args.max_output_tokens
        for result in results
    )
    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(UTC).isoformat(),
        "result": (
            "technical_pass_human_semantic_review_required"
            if technical_pass
            else "technical_fail"
        ),
        "runtime": {
            "model": args.model,
            "model_revision": args.model_revision,
            "base_url": base_url,
            "context_tokens": args.context_tokens,
            "maximum_output_tokens": args.max_output_tokens,
            "planner_timeout_seconds": args.planner_timeout_seconds,
            "warmup_excluded": not args.skip_warmup,
        },
        "contracts": results,
        "privacy": {
            "endpoint_loopback_only": True,
            "structured_plan_privacy_gate_exercised": True,
            "modal_or_cloud_called": False,
            "fixtures_are_synthetic": True,
        },
        "acceptance": {
            "technical_pass": technical_pass,
            "human_semantic_review_required": True,
            "enable_compact_wire_only_after_review": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma3:1b-it-q4_K_M")
    parser.add_argument("--model-revision", default="configured-local-model")
    parser.add_argument("--contract", choices=("standard", "compact", "both"), default="both")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=180)
    parser.add_argument("--planner-timeout-seconds", type=float, default=12)
    parser.add_argument("--model-timeout-seconds", type=float, default=35)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = asyncio.run(benchmark(args))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if report["result"] == "technical_fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
