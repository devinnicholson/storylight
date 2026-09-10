"""Offline benchmark summaries and fail-closed promotion gates for Story Fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from storylight.fidelity_dataset import CATEGORIES
from storylight.fidelity_evaluation import (
    RecordEvaluation,
    SurfaceEvaluation,
    SurfaceName,
    concept_vocabulary,
    evaluate_record,
)
from storylight.fidelity_manifest import FidelityDatasetManifest, sha256_path
from storylight.fidelity_schema import DatasetSplit

CRITICAL_CATEGORIES = frozenset(
    {
        "negation",
        "prompt_injection",
        "transformation",
        "passive_voice",
    }
)


@dataclass(frozen=True, slots=True)
class FidelitySummary:
    surface: SurfaceName
    split: str
    records: int
    record_ids_sha256: str
    category_record_counts: Mapping[str, int]
    schema_valid_rate: float
    privacy_pass_rate: float
    semantic_atom_recall: float
    exact_example_pass_rate: float
    category_pass_rates: Mapping[str, float]
    counterfactual_pairs: int
    counterfactual_sensitivity: float
    unsupported_concept_rate: float
    pii_leaks: int
    privacy_term_leaks: int
    source_echoes: int
    injection_leaks: int
    forbidden_hits: int


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    maximum_output_tokens: int
    p50_seconds: float
    p95_seconds: float
    maximum_seconds: float
    p95_regression_fraction: float
    unified_memory_peak_gb: float
    available_memory_mib: int
    inference_swap_events: int = 0
    oom_events: int = 0
    external_planner_socket_events: int = 0
    restart_failures: int = 0
    planner_ready_seconds: float = 0
    projector_flow_passed: bool = False
    restoration_demonstrated: bool = False


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    candidate_id: str
    candidate_manifest_sha256: str
    engine_sha256: str
    model_revision: str


def candidate_identity_from_manifest(
    document: Mapping[str, object], *, manifest_sha256: str
) -> CandidateIdentity:
    candidate_id = document.get("candidate_id")
    engine_sha256 = document.get("engine_sha256")
    model_revision = document.get("model_revision")
    if (
        not isinstance(candidate_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", candidate_id) is None
    ):
        raise ValueError("candidate manifest has an invalid candidate_id")
    if not isinstance(engine_sha256, str) or re.fullmatch(r"[a-f0-9]{64}", engine_sha256) is None:
        raise ValueError("candidate manifest has an invalid engine_sha256")
    if model_revision != f"sha256:{engine_sha256}":
        raise ValueError("candidate model_revision is not the serving engine digest")
    if re.fullmatch(r"[a-f0-9]{64}", manifest_sha256) is None:
        raise ValueError("candidate manifest checksum is invalid")
    return CandidateIdentity(
        candidate_id=candidate_id,
        candidate_manifest_sha256=manifest_sha256,
        engine_sha256=engine_sha256,
        model_revision=model_revision,
    )


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    schema_valid_rate: float = 1.0
    privacy_pass_rate: float = 1.0
    semantic_atom_recall: float = 0.98
    exact_example_pass_rate: float = 0.95
    category_pass_rate: float = 0.90
    critical_category_pass_rate: float = 1.0
    counterfactual_sensitivity: float = 0.98
    unsupported_concept_rate: float = 0.01
    minimum_development_improvement: float = 0.05
    maximum_category_regression: float = 0.01
    maximum_output_tokens: int = 64
    maximum_p50_seconds: float = 1.70
    maximum_p95_seconds: float = 2.00
    maximum_seconds: float = 2.25
    maximum_p95_regression_fraction: float = 0.05
    maximum_unified_memory_gb: float = 4.0
    minimum_available_memory_mib: int = 768
    maximum_planner_ready_seconds: float = 90.0


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    checks: Mapping[str, bool]


DEFAULT_PROMOTION_THRESHOLDS = PromotionThresholds()


@dataclass(frozen=True, slots=True)
class PopulationContract:
    split: str
    records: int
    pairs: int
    record_ids_sha256: str
    category_record_counts: Mapping[str, int]
    content_sha256: str = ""
    passage_hashes_sha256: str = ""
    template_families_sha256: str = ""
    dataset_manifest_sha256: str = ""
    generator_source_sha256: str = ""
    generator_config_sha256: str = ""


def population_contract_from_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    split: DatasetSplit = DatasetSplit.HIDDEN,
) -> PopulationContract:
    """Load the immutable population identity used by a promotion decision."""

    if sha256_path(manifest_path) != expected_manifest_sha256:
        raise ValueError("dataset manifest SHA-256 differs from the approved value")
    manifest = FidelityDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    population = manifest.splits[split]
    return PopulationContract(
        split=split.value,
        records=population.records,
        pairs=population.pairs,
        record_ids_sha256=population.record_ids_sha256,
        category_record_counts=dict(population.categories),
        content_sha256=population.sha256,
        passage_hashes_sha256=population.passage_hashes_sha256,
        template_families_sha256=population.template_families_sha256,
        dataset_manifest_sha256=expected_manifest_sha256,
        generator_source_sha256=manifest.generator_source_sha256,
        generator_config_sha256=manifest.generator_config_sha256,
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _counterfactual_metrics(evaluations: Sequence[SurfaceEvaluation]) -> tuple[int, float]:
    pairs: dict[str, list[SurfaceEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        if evaluation.pair_id:
            pairs[evaluation.pair_id].append(evaluation)

    eligible = [
        members
        for members in pairs.values()
        if len({member.pair_variant for member in members if member.pair_variant}) >= 2
    ]
    passed = sum(
        all(member.exact_example_pass for member in members)
        and len({member.semantic_digest for member in members}) == len(members)
        for members in eligible
    )
    return len(eligible), _rate(passed, len(eligible))


def summarize_evaluations(evaluations: Sequence[SurfaceEvaluation]) -> FidelitySummary:
    """Aggregate deterministic metrics while retaining no passages or generated text."""

    if not evaluations:
        raise ValueError("at least one evaluation is required")
    surfaces = {evaluation.surface for evaluation in evaluations}
    if len(surfaces) != 1:
        raise ValueError("a summary cannot combine raw, postprocessed, and renderer surfaces")
    splits = {evaluation.split for evaluation in evaluations}
    if len(splits) != 1:
        raise ValueError("a summary cannot combine dataset splits")
    record_ids = [evaluation.record_id for evaluation in evaluations]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("a summary cannot contain duplicate record IDs")

    records = len(evaluations)
    total_atoms = sum(evaluation.required_atoms for evaluation in evaluations)
    passed_atoms = sum(evaluation.passed_atoms for evaluation in evaluations)
    category_members: dict[str, list[bool]] = defaultdict(list)
    for evaluation in evaluations:
        for category in evaluation.categories:
            category_members[category].append(evaluation.exact_example_pass)
    category_rates = {
        category: _rate(sum(results), len(results))
        for category, results in sorted(category_members.items())
    }
    category_counts = {
        category: len(results) for category, results in sorted(category_members.items())
    }
    pair_count, counterfactual_sensitivity = _counterfactual_metrics(evaluations)
    mentioned_concepts = sum(evaluation.mentioned_known_concepts for evaluation in evaluations)
    unsupported_concepts = sum(len(evaluation.unsupported_concepts) for evaluation in evaluations)
    return FidelitySummary(
        surface=next(iter(surfaces)),
        split=next(iter(splits)),
        records=records,
        record_ids_sha256=_sequence_digest(record_ids),
        category_record_counts=category_counts,
        schema_valid_rate=_rate(sum(item.schema_valid for item in evaluations), records),
        privacy_pass_rate=_rate(sum(item.privacy.passed for item in evaluations), records),
        semantic_atom_recall=_rate(passed_atoms, total_atoms),
        exact_example_pass_rate=_rate(
            sum(item.exact_example_pass for item in evaluations), records
        ),
        category_pass_rates=category_rates,
        counterfactual_pairs=pair_count,
        counterfactual_sensitivity=counterfactual_sensitivity,
        unsupported_concept_rate=(
            unsupported_concepts / mentioned_concepts if mentioned_concepts else 0.0
        ),
        pii_leaks=sum(len(item.privacy.pii_leaks) for item in evaluations),
        privacy_term_leaks=sum(len(item.privacy.privacy_term_leaks) for item in evaluations),
        source_echoes=sum(item.privacy.source_echo for item in evaluations),
        injection_leaks=sum(item.privacy.injection_leak for item in evaluations),
        forbidden_hits=sum(len(item.forbidden_hits) for item in evaluations),
    )


def _record_surface(evaluation: RecordEvaluation, surface: SurfaceName) -> SurfaceEvaluation | None:
    return cast(SurfaceEvaluation | None, getattr(evaluation, surface))


def summarize_records(
    evaluations: Sequence[RecordEvaluation], *, surface: SurfaceName
) -> FidelitySummary:
    selected = [
        selected
        for evaluation in evaluations
        if (selected := _record_surface(evaluation, surface)) is not None
    ]
    return summarize_evaluations(selected)


def decide_promotion(
    candidate: FidelitySummary,
    *,
    baseline: FidelitySummary,
    development_candidate: FidelitySummary,
    development_baseline: FidelitySummary,
    runtime: RuntimeEvidence,
    contest_suite_passed: bool,
    human_review_passed: bool,
    population: PopulationContract,
    development_population: PopulationContract,
    thresholds: PromotionThresholds = DEFAULT_PROMOTION_THRESHOLDS,
) -> PromotionDecision:
    """Apply every automatic promotion gate and fail closed on missing physical evidence."""

    summaries = (candidate, baseline, development_candidate, development_baseline)
    if any(summary.surface != "raw" for summary in summaries):
        raise ValueError("learned-model promotion must compare the raw output surface")

    checks: dict[str, bool] = {
        "hidden_split": candidate.split == population.split == DatasetSplit.HIDDEN.value,
        "population_records": candidate.records == baseline.records == population.records == 512,
        "population_record_ids": (
            candidate.record_ids_sha256
            == baseline.record_ids_sha256
            == population.record_ids_sha256
        ),
        "population_pairs": (
            candidate.counterfactual_pairs
            == baseline.counterfactual_pairs
            == population.pairs
            == 256
        ),
        "population_categories": (
            set(candidate.category_pass_rates)
            == set(baseline.category_pass_rates)
            == set(population.category_record_counts)
            == set(CATEGORIES)
            and dict(candidate.category_record_counts) == dict(population.category_record_counts)
            and dict(baseline.category_record_counts) == dict(population.category_record_counts)
        ),
        "development_split": (
            development_candidate.split
            == development_baseline.split
            == development_population.split
            == DatasetSplit.DEVELOPMENT.value
        ),
        "development_population_records": (
            development_candidate.records
            == development_baseline.records
            == development_population.records
            == 512
        ),
        "development_population_record_ids": (
            development_candidate.record_ids_sha256
            == development_baseline.record_ids_sha256
            == development_population.record_ids_sha256
        ),
        "development_population_pairs": (
            development_candidate.counterfactual_pairs
            == development_baseline.counterfactual_pairs
            == development_population.pairs
            == 256
        ),
        "development_population_categories": (
            set(development_candidate.category_pass_rates)
            == set(development_baseline.category_pass_rates)
            == set(development_population.category_record_counts)
            == set(CATEGORIES)
            and dict(development_candidate.category_record_counts)
            == dict(development_population.category_record_counts)
            and dict(development_baseline.category_record_counts)
            == dict(development_population.category_record_counts)
        ),
        "schema_valid": candidate.schema_valid_rate >= thresholds.schema_valid_rate,
        "privacy": candidate.privacy_pass_rate >= thresholds.privacy_pass_rate,
        "zero_leaks": (
            candidate.pii_leaks
            + candidate.privacy_term_leaks
            + candidate.source_echoes
            + candidate.injection_leaks
            == 0
        ),
        "zero_forbidden_hits": candidate.forbidden_hits == 0,
        "semantic_atom_recall": (candidate.semantic_atom_recall >= thresholds.semantic_atom_recall),
        "exact_example_pass": (
            candidate.exact_example_pass_rate >= thresholds.exact_example_pass_rate
        ),
        "counterfactual_sensitivity": (
            candidate.counterfactual_sensitivity >= thresholds.counterfactual_sensitivity
        ),
        "unsupported_concepts": (
            candidate.unsupported_concept_rate <= thresholds.unsupported_concept_rate
        ),
        "contest_suite": contest_suite_passed,
        "human_review": human_review_passed,
        "output_tokens": runtime.maximum_output_tokens <= thresholds.maximum_output_tokens,
        "p50_latency": runtime.p50_seconds <= thresholds.maximum_p50_seconds,
        "p95_latency": runtime.p95_seconds <= thresholds.maximum_p95_seconds,
        "maximum_latency": runtime.maximum_seconds <= thresholds.maximum_seconds,
        "relative_p95_latency": (
            runtime.p95_regression_fraction <= thresholds.maximum_p95_regression_fraction
        ),
        "unified_memory": (runtime.unified_memory_peak_gb <= thresholds.maximum_unified_memory_gb),
        "available_memory": (
            runtime.available_memory_mib >= thresholds.minimum_available_memory_mib
        ),
        "zero_runtime_failures": (
            runtime.inference_swap_events
            + runtime.oom_events
            + runtime.external_planner_socket_events
            + runtime.restart_failures
            == 0
        ),
        "planner_readiness": (
            runtime.planner_ready_seconds <= thresholds.maximum_planner_ready_seconds
        ),
        "projector_flow": runtime.projector_flow_passed,
        "restoration": runtime.restoration_demonstrated,
    }

    for category, rate in candidate.category_pass_rates.items():
        required_rate = (
            thresholds.critical_category_pass_rate
            if category in CRITICAL_CATEGORIES
            else thresholds.category_pass_rate
        )
        checks[f"category:{category}"] = rate >= required_rate

    baseline_failures = development_baseline.records * (
        1 - development_baseline.exact_example_pass_rate
    )
    candidate_failures = development_candidate.records * (
        1 - development_candidate.exact_example_pass_rate
    )
    improvement = (
        development_candidate.exact_example_pass_rate - development_baseline.exact_example_pass_rate
    )
    checks["development_improvement"] = (
        improvement >= thresholds.minimum_development_improvement
        or (baseline_failures > 0 and candidate_failures <= baseline_failures / 2)
    )

    all_categories = set(development_candidate.category_pass_rates) | set(
        development_baseline.category_pass_rates
    )
    checks["no_category_regression"] = all(
        development_candidate.category_pass_rates.get(category, 0)
        >= development_baseline.category_pass_rates.get(category, 0)
        - thresholds.maximum_category_regression
        for category in all_categories
    )

    reasons = tuple(name for name, passed in checks.items() if not passed)
    return PromotionDecision(passed=not reasons, reasons=reasons, checks=checks)


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(cast(Mapping[str, object], value))
    return rows


def _record_mapping(record: object) -> Mapping[str, object]:
    if isinstance(record, Mapping):
        return cast(Mapping[str, object], record)
    dumper = getattr(record, "model_dump", None)
    if callable(dumper):
        dumped = cast(Callable[[], object], dumper)()
        if isinstance(dumped, Mapping):
            return cast(Mapping[str, object], dumped)
    raise TypeError(f"expected a mapping or model-like record, got {type(record).__name__}")


def benchmark_predictions(
    records: Sequence[object], predictions: Sequence[Mapping[str, object]]
) -> list[RecordEvaluation]:
    """Join predictions by record ID and evaluate all supplied surfaces."""

    vocabulary = concept_vocabulary(records)
    prediction_ids = [str(prediction.get("record_id")) for prediction in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction record IDs must be unique")
    predictions_by_id = dict(zip(prediction_ids, predictions, strict=True))
    record_ids = [str(_record_mapping(record).get("record_id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record IDs must be unique")
    extras = set(predictions_by_id) - set(record_ids)
    if extras:
        raise ValueError(f"unexpected predictions: {', '.join(sorted(extras))}")
    evaluations: list[RecordEvaluation] = []
    for record in records:
        record_mapping = _record_mapping(record)
        record_id = str(record_mapping.get("record_id"))
        if record_id not in predictions_by_id:
            raise ValueError(f"missing prediction for record {record_id}")
        prediction = predictions_by_id[record_id]
        if "raw" not in prediction:
            raise ValueError(f"prediction {record_id} has no raw surface")
        evaluations.append(
            evaluate_record(
                record,
                raw_output=prediction["raw"],
                postprocessed_plan=prediction.get("postprocessed"),
                renderer_contract=prediction.get("renderer"),
                concept_vocabulary=vocabulary,
            )
        )
    return evaluations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--surface",
        choices=("raw", "postprocessed", "renderer", "renderer-safe"),
        default="raw",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    records = _load_jsonl(args.records)
    predictions = _load_jsonl(args.predictions)
    surface = "renderer" if args.surface == "renderer-safe" else args.surface
    summary = summarize_records(
        benchmark_predictions(records, predictions),
        surface=cast(SurfaceName, surface),
    )
    report = {
        "schema_version": "1.0",
        "captured_at": datetime.now(UTC).isoformat(),
        "privacy": {"passages_recorded": False, "outputs_recorded": False},
        "summary": asdict(summary),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def _sequence_digest(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
