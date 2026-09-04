"""Repeatable, local-only acceptance benchmark for the Jetson scene planner."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from bookforge.config import Settings
from bookforge.live_scene_planner import StructuredLiveScenePlanner
from bookforge.model_client import OllamaClient, OpenAICompatibleClient
from bookforge.tensorrt_slot_client import TensorRTSlotModelClient

Contract = Literal["standard", "compact"]
Suite = Literal["five", "contest"]
SEMANTIC_SCREEN_REVISION = "lexical-v3-negation-aware"
PLANNER_INSTRUCTION_REVISION = "semantic-fidelity-v1"


@dataclass(frozen=True, slots=True)
class SemanticExpectation:
    """One required visual idea, expressed through acceptable lexical alternatives."""

    label: str
    alternatives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    text: str
    visual_style: str
    seed: int
    expectations: tuple[SemanticExpectation, ...] = ()
    forbidden_terms: tuple[str, ...] = ()


def _expect(label: str, *alternatives: str) -> SemanticExpectation:
    return SemanticExpectation(label=label, alternatives=alternatives)


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
        expectations=(
            _expect("actor", "child"),
            _expect("source object", "book"),
            _expect("transformation", "origami bird", "paper bird"),
            _expect("destination", "floating school", "school above", "cloud school"),
        ),
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
        expectations=(
            _expect("actor", "whale"),
            _expect("setting", "library"),
            _expect("carried object", "lantern"),
            _expect("transformation", "bright fish", "school of fish", "fish"),
        ),
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
        expectations=(
            _expect("actor", "clockwork fox", "mechanical fox"),
            _expect("action", "plant", "placing", "buries"),
            _expect("object", "brass seed", "metal seed"),
            _expect("result", "glass forest", "glass branches", "transparent forest"),
        ),
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
        expectations=(
            _expect("actor", "student", "child"),
            _expect("object", "butterfly"),
            _expect("setting", "classroom"),
            _expect("transformation", "floating garden", "paper flowers", "flower garden"),
        ),
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
        expectations=(
            _expect("actor", "turtle"),
            _expect("path", "staircase", "stairs", "steps"),
            _expect("path material", "cloud", "clouds"),
            _expect("supporting creatures", "jellyfish"),
            _expect("setting", "stars", "starry", "night sky"),
        ),
    ),
)


CONTEST_CASES = CASES + (
    BenchmarkCase(
        case_id="lighthouse_violin",
        text=(
            "At the foot of a lighthouse, a small crab plays a violin, and the sweeping "
            "beam curls into a golden ribbon above the waves."
        ),
        visual_style="bold cut-paper nocturne, navy sea, warm gold, crisp full-bleed 16:9",
        seed=20261031,
        expectations=(
            _expect("actor", "crab"),
            _expect("instrument", "violin"),
            _expect("setting", "lighthouse"),
            _expect("transformation", "golden ribbon", "light ribbon", "ribbon"),
        ),
    ),
    BenchmarkCase(
        case_id="desert_umbrella",
        text=(
            "A young elephant opens a red umbrella in the empty desert, and bright flowers "
            "burst from every place its shadow touches."
        ),
        visual_style="sunlit watercolor paper theater, coral and turquoise, full-bleed 16:9",
        seed=20261032,
        expectations=(
            _expect("actor", "elephant"),
            _expect("object", "red umbrella", "umbrella"),
            _expect("setting", "desert", "sand"),
            _expect("result", "flowers", "blossoms", "bloom"),
        ),
    ),
    BenchmarkCase(
        case_id="attic_moth_map",
        text=(
            "Inside a dusty attic, a pale moth unfolds an old map, and the wooden roof "
            "dissolves into a deep night sky."
        ),
        visual_style="mysterious layered paper diorama, violet shadows, silver stars, 16:9",
        seed=20261033,
        expectations=(
            _expect("actor", "moth"),
            _expect("object", "map"),
            _expect("setting", "attic"),
            _expect("transformation", "night sky", "starry sky", "stars"),
        ),
    ),
    BenchmarkCase(
        case_id="teacup_boat",
        text=(
            "A field mouse sails a cracked blue teacup across a frozen pond while tiny "
            "snowflakes rise upward like lanterns."
        ),
        visual_style="whimsical winter storybook, ice blue and amber, tactile paper, 16:9",
        seed=20261034,
        expectations=(
            _expect("actor", "mouse"),
            _expect("vehicle", "teacup", "cup boat"),
            _expect("setting", "frozen pond", "ice", "icy pond"),
            _expect(
                "reversed motion",
                "rising snow",
                "snowflakes rise",
                "snowflakes rising",
                "upward snow",
            ),
        ),
    ),
    BenchmarkCase(
        case_id="pencil_river",
        text=(
            "A child draws a blue river across a blank page; the ink spills beyond the paper "
            "and becomes real water carrying little boats."
        ),
        visual_style="bright hand-drawn paper theater, cobalt ink, warm desk light, 16:9",
        seed=20261035,
        expectations=(
            _expect("actor", "child"),
            _expect("action", "draw", "sketch"),
            _expect("transformation", "ink becomes water", "real water", "river"),
            _expect("result", "boats", "little boats"),
        ),
    ),
    BenchmarkCase(
        case_id="bridge_spatial",
        text=(
            "A white rabbit waits beneath a stone bridge holding a red umbrella, while three "
            "paper lanterns float high above the bridge."
        ),
        visual_style="cinematic cut-paper rain scene, slate blue and red, clear depth, 16:9",
        seed=20261036,
        expectations=(
            _expect("actor", "rabbit"),
            _expect("object", "red umbrella", "umbrella"),
            _expect("lower relation", "beneath bridge", "under bridge"),
            _expect(
                "upper relation",
                "lanterns above",
                "lanterns float",
                "floating lanterns",
                "lanterns floating",
            ),
        ),
    ),
    BenchmarkCase(
        case_id="flashlight_birds",
        text=(
            "In a quiet cave, a child raises a flashlight and its enormous shadow breaks apart "
            "into a flock of black birds."
        ),
        visual_style="high-contrast shadow-puppet paper theater, charcoal and amber, 16:9",
        seed=20261037,
        expectations=(
            _expect("actor", "child"),
            _expect("object", "flashlight", "torch"),
            _expect("setting", "cave"),
            _expect("transformation", "flock of birds", "black birds", "birds"),
        ),
        forbidden_terms=("dragon",),
    ),
    BenchmarkCase(
        case_id="bakery_volcano",
        text=(
            "A round robot baker opens the oven, and a mountain of bread dough erupts with "
            "colorful confetti instead of smoke."
        ),
        visual_style="playful clay-and-paper bakery, warm orange, projector-bright, 16:9",
        seed=20261038,
        expectations=(
            _expect("actor", "robot baker", "robot"),
            _expect("setting", "bakery", "oven"),
            _expect("object", "bread dough", "dough mountain", "dough"),
            _expect("surprise", "confetti"),
        ),
        forbidden_terms=("smoke",),
    ),
    BenchmarkCase(
        case_id="underwater_train",
        text=(
            "An octopus conductor guides a tiny train through an underwater station, where "
            "bubbles swell into glowing clocks with no numbers."
        ),
        visual_style="luminous underwater paper theater, teal and gold, full-bleed 16:9",
        seed=20261039,
        expectations=(
            _expect("actor", "octopus"),
            _expect("vehicle", "train"),
            _expect("setting", "underwater station", "underwater"),
            _expect("transformation", "bubble clocks", "glowing clocks", "clocks"),
        ),
        forbidden_terms=("numbers", "digits"),
    ),
    BenchmarkCase(
        case_id="beetle_leaf_tower",
        text=(
            "A green beetle pushes one seed into a rooftop garden, and a twisting tower of "
            "giant leaves grows around the chimneys."
        ),
        visual_style="lush layered paper city, emerald and terracotta, clear silhouettes, 16:9",
        seed=20261040,
        expectations=(
            _expect("actor", "beetle"),
            _expect("object", "seed"),
            _expect("setting", "rooftop", "chimneys", "roof garden"),
            _expect("result", "tower of leaves", "giant leaves", "leaf tower", "tower leaves"),
        ),
    ),
    BenchmarkCase(
        case_id="owl_passive_key",
        text=(
            "A brass key is carried through the rain by a snowy owl and unlocks a round door "
            "in the moon."
        ),
        visual_style="poetic moonlit paper theater, silver blue and brass, full-bleed 16:9",
        seed=20261041,
        expectations=(
            _expect("actor", "owl"),
            _expect("carried object", "brass key", "key"),
            _expect("action", "unlock", "opens"),
            _expect("destination", "moon door", "door in the moon", "lunar door", "door moon"),
        ),
    ),
    BenchmarkCase(
        case_id="boat_to_swan",
        text=(
            "After a folded paper boat tumbles through a waterfall, it emerges as a white swan "
            "on a glowing lake."
        ),
        visual_style="elegant watercolor transformation, luminous cyan and white, 16:9",
        seed=20261042,
        expectations=(
            _expect("initial object", "paper boat", "folded boat"),
            _expect("transition", "waterfall"),
            _expect("final subject", "white swan", "swan"),
            _expect("final setting", "glowing lake", "luminous lake", "lake"),
        ),
    ),
    BenchmarkCase(
        case_id="not_a_dragon",
        text=(
            "The shape beside the candle is not a dragon but a tiny blue moth whose wings cast "
            "the shadow of a vast cathedral."
        ),
        visual_style="surreal candlelit shadow theater, blue and amber, full-bleed 16:9",
        seed=20261043,
        expectations=(
            _expect("actual subject", "blue moth", "moth"),
            _expect("light source", "candle"),
            _expect("projection", "cathedral shadow", "shadow of a cathedral", "cathedral"),
        ),
        forbidden_terms=("dragon",),
    ),
    BenchmarkCase(
        case_id="bottle_city",
        text=(
            "A giant turtle carries a glass bottle on its shell; inside the bottle, a miniature "
            "city shines beneath a storm no larger than a marble."
        ),
        visual_style="fantastical scale-play paper diorama, teal glass and gold, 16:9",
        seed=20261044,
        expectations=(
            _expect("outer subject", "giant turtle", "turtle"),
            _expect("container", "glass bottle", "bottle"),
            _expect("contained subject", "miniature city", "tiny city", "city inside"),
            _expect("contained weather", "small storm", "tiny storm", "storm"),
        ),
    ),
    BenchmarkCase(
        case_id="syllable_fireflies",
        text=(
            "Each syllable a child reads aloud becomes a glowing firefly, and together the "
            "fireflies form a path from the bedroom to a distant library."
        ),
        visual_style="hopeful luminous literacy storybook, indigo and warm gold, 16:9",
        seed=20261045,
        expectations=(
            _expect("actor", "child"),
            _expect("literacy action", "reads aloud", "reading", "spoken syllables"),
            _expect("transformation", "glowing fireflies", "fireflies"),
            _expect("destination", "library"),
        ),
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


_SEMANTIC_TOKEN = re.compile(r"[a-z0-9]+")


def _normalized_semantic_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(_SEMANTIC_TOKEN.findall(normalized))


def _semantic_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _semantic_token_matches(token: str, needle: str) -> bool:
    token_stem = _semantic_stem(token)
    needle_stem = _semantic_stem(needle)
    return (
        token_stem == needle_stem
        or f"{token_stem}e" == needle_stem
        or f"{needle_stem}e" == token_stem
        or (len(needle_stem) >= 5 and token_stem.endswith(needle_stem))
        or (len(token_stem) >= 5 and needle_stem.endswith(token_stem))
    )


def _contains_semantic_alternative(haystack: str, alternative: str) -> bool:
    haystack_tokens = _normalized_semantic_text(haystack).split()
    alternative_tokens = _normalized_semantic_text(alternative).split()
    if not alternative_tokens:
        return False
    if len(alternative_tokens) > 1:
        width = len(alternative_tokens)
        return any(
            all(
                _semantic_token_matches(token, needle)
                for token, needle in zip(
                    haystack_tokens[index : index + width],
                    alternative_tokens,
                    strict=True,
                )
            )
            for index in range(len(haystack_tokens) - width + 1)
        )
    needle = alternative_tokens[0]
    return any(_semantic_token_matches(token, needle) for token in haystack_tokens)


_NEGATION_PREFIXES = frozenset({"no", "not", "without", "excluding", "except"})
_NEGATION_PREFIX_PAIRS = frozenset(
    {
        ("free", "of"),
        ("instead", "of"),
        ("rather", "than"),
    }
)
_NEGATION_SUFFIXES = frozenset({"absent", "excluded", "missing", "omitted"})


def _contains_unnegated_semantic_alternative(haystack: str, alternative: str) -> bool:
    """Return whether an idea is asserted, rather than explicitly excluded."""

    haystack_tokens = _normalized_semantic_text(haystack).split()
    alternative_tokens = _normalized_semantic_text(alternative).split()
    width = len(alternative_tokens)
    if not width:
        return False
    for index in range(len(haystack_tokens) - width + 1):
        if not all(
            _semantic_token_matches(token, needle)
            for token, needle in zip(
                haystack_tokens[index : index + width],
                alternative_tokens,
                strict=True,
            )
        ):
            continue
        prefix = haystack_tokens[max(0, index - 3) : index]
        suffix = haystack_tokens[index + width : index + width + 2]
        explicitly_negated = (
            bool(set(prefix) & _NEGATION_PREFIXES)
            or tuple(prefix[-2:]) in _NEGATION_PREFIX_PAIRS
            or (bool(suffix) and suffix[0] in _NEGATION_SUFFIXES)
        )
        if not explicitly_negated:
            return True
    return False


def _semantic_evidence(
    case: BenchmarkCase,
    *,
    generated_text: str,
) -> dict[str, object]:
    checks = [
        {
            "label": expectation.label,
            "alternatives": list(expectation.alternatives),
            "pass": any(
                _contains_semantic_alternative(generated_text, alternative)
                for alternative in expectation.alternatives
            ),
        }
        for expectation in case.expectations
    ]
    forbidden_checks = [
        {
            "term": term,
            "pass": not _contains_unnegated_semantic_alternative(generated_text, term),
        }
        for term in case.forbidden_terms
    ]
    return {
        "semantic_checks": checks,
        "forbidden_checks": forbidden_checks,
        "automatic_semantic_pass": all(
            bool(check["pass"]) for check in (*checks, *forbidden_checks)
        ),
    }


def _semantic_summary(cases: list[dict[str, object]]) -> dict[str, int | bool]:
    passed = sum(bool(case["automatic_semantic_pass"]) for case in cases)
    return {
        "cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "all_passed": passed == len(cases),
    }


def _contract_order_for_case(
    index: int,
    contracts: tuple[Contract, ...],
) -> tuple[Contract, ...]:
    if len(contracts) < 2 or index % 2 == 0:
        return contracts
    return tuple(reversed(contracts))


def _select_cases(
    suite: str,
    case_ids: list[str] | None = None,
) -> tuple[BenchmarkCase, ...]:
    cases = CASES if suite == "five" else CONTEST_CASES
    if not case_ids:
        return cases
    requested = set(case_ids)
    selected = tuple(case for case in cases if case.case_id in requested)
    unknown = requested - {case.case_id for case in selected}
    if unknown:
        rendered = ", ".join(sorted(unknown))
        raise ValueError(f"unknown benchmark case IDs for {suite!r}: {rendered}")
    return selected


async def _run_case(
    planner: StructuredLiveScenePlanner,
    *,
    case: BenchmarkCase,
) -> dict[str, object]:
    result = await planner.plan(
        text=case.text,
        visual_style=case.visual_style,
        seed=case.seed,
    )
    generated_text = " ".join(
        (
            result.plan.scene_summary,
            result.plan.art_direction,
            result.plan.background_prompt,
            result.plan.focus.prompt,
            result.plan.accent.prompt,
        )
    )
    return {
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
        **_semantic_evidence(case, generated_text=generated_text),
    }


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    base_url = _require_loopback(args.base_url)
    cases = _select_cases(args.suite, args.case_id)
    tensorrt_backends = {"tensorrt_slots", "tensorrt_hybrid"}
    if args.backend in tensorrt_backends and args.contract != "standard":
        raise ValueError("TensorRT slot protocols require the standard wire contract")
    settings = Settings(
        _env_file=None,
        model_backend=("openai" if args.backend in tensorrt_backends else args.backend),
        model_name=args.model,
        model_base_url=base_url,
        model_timeout_seconds=args.model_timeout_seconds,
        model_keep_alive=args.keep_alive,
        model_context_tokens=args.context_tokens,
        model_max_output_tokens=args.max_output_tokens,
    )
    if args.backend == "ollama":
        client = OllamaClient(settings)
    elif args.backend in tensorrt_backends:
        client = TensorRTSlotModelClient(
            base_url=base_url,
            model=args.model,
            timeout_seconds=args.model_timeout_seconds,
            max_output_tokens=min(args.max_output_tokens, 128),
            protocol="hybrid" if args.backend == "tensorrt_hybrid" else "slots",
        )
    else:
        client = OpenAICompatibleClient(settings)
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
                text=cases[0].text,
                visual_style=cases[0].visual_style,
                seed=cases[0].seed,
            )
        planners = {
            contract: StructuredLiveScenePlanner(
                client,
                timeout_seconds=args.planner_timeout_seconds,
                model_revision=args.model_revision,
                compact_wire=contract == "compact",
            )
            for contract in contracts
        }
        case_results: dict[Contract, list[dict[str, object]]] = {
            contract: [] for contract in contracts
        }
        execution_order: list[dict[str, object]] = []
        for index, case in enumerate(cases):
            order = _contract_order_for_case(index, contracts)
            execution_order.append({"case_id": case.case_id, "contracts": list(order)})
            for contract in order:
                case_results[contract].append(
                    await _run_case(
                        planners[contract],
                        case=case,
                    )
                )
        results = [
            {
                "contract": contract,
                "summary": _summarize(case_results[contract]),
                "semantic_summary": _semantic_summary(case_results[contract]),
                "cases": case_results[contract],
            }
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
    automatic_semantic_pass = all(
        bool(result["semantic_summary"]["all_passed"])  # type: ignore[index]
        for result in results
    )
    if not technical_pass:
        result_label = "technical_fail"
    elif not automatic_semantic_pass:
        result_label = "automatic_semantic_fail_human_review_required"
    else:
        result_label = "automatic_acceptance_pass_human_review_required"
    return {
        "schema_version": "1.2",
        "captured_at": datetime.now(UTC).isoformat(),
        "result": result_label,
        "runtime": {
            "backend": args.backend,
            "model": args.model,
            "model_revision": args.model_revision,
            "base_url": base_url,
            "context_tokens": args.context_tokens,
            "maximum_output_tokens": args.max_output_tokens,
            "planner_timeout_seconds": args.planner_timeout_seconds,
            "warmup_excluded": not args.skip_warmup,
            "suite": args.suite,
            "case_count": len(cases),
            "semantic_screen_revision": SEMANTIC_SCREEN_REVISION,
            "planner_instruction_revision": PLANNER_INSTRUCTION_REVISION,
        },
        "contracts": results,
        "execution": {
            "policy": "alternate_first_contract_by_case",
            "order": execution_order,
            "reason": "Counterbalances shared-prefix and immediately-prior-request cache effects.",
        },
        "privacy": {
            "endpoint_loopback_only": True,
            "structured_plan_privacy_gate_exercised": True,
            "modal_or_cloud_called": False,
            "fixtures_are_synthetic": True,
        },
        "acceptance": {
            "technical_pass": technical_pass,
            "automatic_semantic_pass": automatic_semantic_pass,
            "automatic_checks_are_lexical_prescreen_only": True,
            "human_semantic_review_required": True,
            "compact_contract_is_research_only": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("ollama", "openai", "tensorrt_slots", "tensorrt_hybrid"),
        default="ollama",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma3:1b-it-q4_K_M")
    parser.add_argument("--model-revision", default="configured-local-model")
    parser.add_argument("--contract", choices=("standard", "compact", "both"), default="both")
    parser.add_argument("--suite", choices=("five", "contest"), default="five")
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run only this named case; repeat to benchmark a validator-selected subset.",
    )
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
    if report["result"] in {
        "technical_fail",
        "automatic_semantic_fail_human_review_required",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
