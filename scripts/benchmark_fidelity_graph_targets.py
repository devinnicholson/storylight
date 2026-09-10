#!/usr/bin/env python3
"""Measure deterministic SceneFactsV2 target coverage on public corpus splits."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from storylight.fidelity_dataset import DATASET_ID, generate_split
from storylight.fidelity_evaluation import FIDELITY_EVALUATOR_REVISION, evaluate_surface
from storylight.fidelity_graph_targets import (
    FidelityGraphTarget,
    derive_fidelity_graph_target,
    summarize_fidelity_graph_coverage,
)
from storylight.fidelity_schema import DatasetSplit
from storylight.scene_facts import estimate_wire_tokens

PUBLIC_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT)
TOKEN_BUDGETS = (64, 96, 128)


def _counts_for(results: Iterable[tuple[FidelityGraphTarget, bool]]) -> dict[str, Any]:
    materialized = tuple(results)
    coverage = summarize_fidelity_graph_coverage(result for result, _ in materialized)
    refusals: Counter[str] = Counter()
    category_exact: Counter[str] = Counter()
    wire_tokens: list[int] = []

    for result, exact in materialized:
        if result.refusal is not None:
            refusals[result.refusal.value] += 1
        if result.facts is not None:
            wire_tokens.append(estimate_wire_tokens(result.facts.to_wire()))
        for category in result.categories:
            if exact:
                category_exact[category] += 1

    categories = [
        {
            "category": row.category,
            "total": row.total,
            "eligible": row.eligible,
            "exact": category_exact[row.category],
            "refusals": dict(row.refusal_counts),
        }
        for row in coverage.by_category
    ]
    return {
        "total": coverage.total,
        "eligible": coverage.eligible,
        "exact": sum(exact for _, exact in materialized),
        "refusals": dict(sorted(refusals.items())),
        "categories": categories,
        "wire_token_estimates": {
            "count": len(wire_tokens),
            "median": statistics.median(wire_tokens) if wire_tokens else None,
            "maximum": max(wire_tokens, default=None),
        },
    }


def build_report(
    splits: Sequence[DatasetSplit],
    *,
    token_budget: Literal[64, 96, 128],
) -> dict[str, Any]:
    """Build a value-free aggregate report for public splits only."""

    requested = set(splits)
    if not requested or not requested.issubset(PUBLIC_SPLITS):
        raise ValueError("at least one public split is required")
    ordered_splits = tuple(split_ for split_ in PUBLIC_SPLITS if split_ in requested)
    all_results: list[tuple[FidelityGraphTarget, bool]] = []
    split_reports: list[dict[str, Any]] = []

    for split_ in ordered_splits:
        split_results: list[tuple[FidelityGraphTarget, bool]] = []
        for record in generate_split(split_):
            target = derive_fidelity_graph_target(record, token_budget=token_budget)
            exact = False
            if target.facts is not None:
                evaluation = evaluate_surface(
                    record.model_dump(mode="json", by_alias=True),
                    target.facts,
                    surface="postprocessed",
                )
                exact = evaluation.exact_example_pass
            split_results.append((target, exact))

        split_report = {"split": split_.value, **_counts_for(split_results)}
        split_reports.append(split_report)
        all_results.extend(split_results)

    return {
        "schema_version": "1.0",
        "benchmark": "story-fidelity-v2-graph-target-coverage",
        "tool_revision": "scene-facts-public-coverage-v2",
        "evaluator_revision": FIDELITY_EVALUATOR_REVISION,
        "dataset_id": DATASET_ID,
        "token_budget": token_budget,
        "selected_splits": [split_.value for split_ in ordered_splits],
        "target_derivation": "deterministic_public_contract_adapter",
        "evaluation_surface": "postprocessed_typed_graph",
        "hidden_evaluated": False,
        "paid_services_used": False,
        **_counts_for(all_results),
        "splits": split_reports,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        choices=tuple(split_.value for split_ in PUBLIC_SPLITS),
        help="public split to benchmark; repeat to select both (default: both)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        choices=TOKEN_BUDGETS,
        default=64,
        help="maximum deterministic SceneFactsV2 wire-token estimate",
    )
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = tuple(DatasetSplit(value) for value in args.split) if args.split else PUBLIC_SPLITS
    report = build_report(splits, token_budget=args.token_budget)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
