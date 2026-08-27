"""Bounded, privacy-safe benchmark for the private Cloud Run scene renderer."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from bookforge.finite_modal_provider import FastSceneRequest, FiniteSceneBundle
from bookforge.gcp_scene_provider import GcpCloudRunSceneProvider

BENCHMARK_PROMPTS = (
    "Luminous layered-paper theater: one silver fox raises a round amber lantern beneath "
    "monumental cedar trees, crisp silhouette, lifted indigo midtones, full-bleed 16:9, no text.",
    "Hand-painted storybook observatory: one child turns a brass star wheel while "
    "constellations bloom overhead, projector-bright cobalt and gold, tactile paper fibers, "
    "full-bleed 16:9, no text.",
    "Cut-paper ocean library: one small turtle carries a glowing book through coral arches as "
    "pages become gentle fish, clear subject and transformation, bright teal and coral, "
    "full-bleed 16:9, no text.",
    "Layered watercolor mountain workshop: one young inventor opens a wooden music box and "
    "luminous birds spiral upward, strong depth planes, warm foreground, cool distance, "
    "full-bleed 16:9, no text.",
    "Cinematic paper garden at dusk: one rabbit plants a moon-white seed and an enormous "
    "flowering tree rises around it, readable action, projection-bright violet and gold, "
    "full-bleed 16:9, no text.",
)


class SceneBenchmarkProvider(Protocol):
    async def probe(self) -> tuple[bool, str]: ...

    async def prewarm(
        self,
        *,
        prewarm_id: str,
        include_motion: bool,
        scaledown_window_seconds: int,
    ) -> Any: ...

    async def generate_fast(
        self, request: FastSceneRequest, *, output_dir: Path
    ) -> FiniteSceneBundle: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    mode: Literal["probe", "prepared"]
    samples: int
    output_root: Path
    gpu: Literal["L4", "RTX_PRO_6000"]
    prewarm_id: str = "bookforge-renderer-benchmark"
    scaledown_window_seconds: int = 90
    seed: int = 20260826

    def __post_init__(self) -> None:
        if not 1 <= self.samples <= 20:
            raise ValueError("benchmark samples must be between 1 and 20")
        if self.mode == "probe" and self.samples != 1:
            raise ValueError("probe mode requires exactly one sample")
        if not 90 <= self.scaledown_window_seconds <= 900:
            raise ValueError("scaledown window must be 90-900 seconds")


async def collect_benchmark(
    provider: SceneBenchmarkProvider,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Collect one finite experiment with no automatic retry."""

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "captured_at": datetime.now(UTC).isoformat(),
        "result": "failed",
        "mode": config.mode,
        "gpu": config.gpu,
        "requested_samples": config.samples,
        "automatic_retries": 0,
        "cost_policy": {
            "estimate_scope": "completed application responses only",
            "billing_reconciliation_required": True,
            "zero_cost_claim": False,
        },
        "privacy": {
            "synthetic_prompts_only": True,
            "prompt_text_recorded": False,
            "prompt_sha256": [
                hashlib.sha256(prompt.encode()).hexdigest() for prompt in BENCHMARK_PROMPTS
            ]
            if config.mode == "prepared"
            else [],
        },
        "samples": [],
    }
    try:
        ready, detail = await provider.probe()
        report["probe"] = {"ready": ready, "detail": detail}
        if not ready:
            raise RuntimeError("renderer probe did not pass")
        if config.mode == "probe":
            report["result"] = "passed"
            report["aggregate"] = _aggregate([])
            return report

        prewarm = await provider.prewarm(
            prewarm_id=config.prewarm_id,
            include_motion=False,
            scaledown_window_seconds=config.scaledown_window_seconds,
        )
        report["prewarm"] = asdict(prewarm)
        for ordinal in range(config.samples):
            scene_id = f"gcp-bench-{config.seed}-{ordinal + 1:02d}"
            prompt = BENCHMARK_PROMPTS[ordinal % len(BENCHMARK_PROMPTS)]
            started = time.perf_counter()
            bundle = await provider.generate_fast(
                FastSceneRequest(
                    scene_id=scene_id,
                    prompt=prompt,
                    seed=(config.seed + ordinal) % (2**32),
                ),
                output_dir=config.output_root / scene_id,
            )
            wall_ms = (time.perf_counter() - started) * 1000
            report["samples"].append(_sample(bundle, ordinal + 1, wall_ms))

        report["aggregate"] = _aggregate(report["samples"])
        report["result"] = "passed"
    except Exception as error:
        report["failure"] = {
            "type": type(error).__name__,
            "detail": str(error)[:500],
        }
        report["aggregate"] = _aggregate(report["samples"])
    finally:
        await provider.aclose()
    return report


def _sample(bundle: FiniteSceneBundle, ordinal: int, wall_ms: float) -> dict[str, Any]:
    stage = bundle.manifest["stages"]["fast"]
    artifacts = bundle.manifest["artifacts"]
    return {
        "ordinal": ordinal,
        "scene_id": bundle.scene_id,
        "wall_ms": wall_ms,
        "remote_ms": float(stage["remote_seconds"]) * 1000,
        "inference_ms": float(stage["inference_seconds"]) * 1000,
        "image_ms": float(stage["image_seconds"]) * 1000,
        "depth_ms": float(stage["depth_seconds"]) * 1000,
        "image_gpu_ms": stage.get("image_gpu_ms"),
        "depth_gpu_ms": stage.get("depth_gpu_ms"),
        "packaging_ms": float(stage["packaging_seconds"]) * 1000,
        "model_load_ms": float(stage["model_load_seconds"]) * 1000,
        "container_age_seconds": float(stage["container_age_seconds"]),
        "warm_state": stage["warm_state"],
        "estimated_gpu_usd": float(stage["estimated_gpu_usd"]),
        "master": {
            "sha256": artifacts["master"]["sha256"],
            "bytes": artifacts["master"]["bytes"],
        },
        "depth": {
            "sha256": artifacts["depth"]["sha256"],
            "bytes": artifacts["depth"]["bytes"],
        },
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "completed_samples": len(samples),
        "wall_ms": _distribution(samples, "wall_ms"),
        "remote_ms": _distribution(samples, "remote_ms"),
        "inference_ms": _distribution(samples, "inference_ms"),
        "estimated_gpu_usd": sum(float(sample["estimated_gpu_usd"]) for sample in samples),
    }


def _distribution(samples: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(sample[key]) for sample in samples]
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--impersonate-service-account", required=True)
    parser.add_argument("--gpu", choices=("L4", "RTX_PRO_6000"), default="RTX_PRO_6000")
    parser.add_argument("--mode", choices=("probe", "prepared"), default="prepared")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--session-gpu-cap-usd", type=float, default=0.50)
    parser.add_argument("--scaledown-window-seconds", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    provider = GcpCloudRunSceneProvider(
        base_url=args.base_url,
        audience=args.audience,
        impersonate_service_account=args.impersonate_service_account,
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        session_gpu_cap_usd=args.session_gpu_cap_usd,
    )
    config = BenchmarkConfig(
        mode=args.mode,
        samples=args.samples,
        output_root=args.output_root,
        gpu=args.gpu,
        scaledown_window_seconds=args.scaledown_window_seconds,
        seed=args.seed,
    )
    report = await collect_benchmark(provider, config)
    report["endpoint"] = {
        "host": urlsplit(args.base_url).netloc,
        "private_iam": True,
        "impersonation": True,
    }
    return report


def main() -> None:
    args = _parser().parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite benchmark report: {args.report}")
    report = asyncio.run(_run(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    if report["result"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
