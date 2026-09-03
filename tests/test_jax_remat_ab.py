# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity.remat_ab import RunEvidence, compare_remat_runs

BASELINE_CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"
CANDIDATE_CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v3-remat-none-probe.json"
RUN_EPOCH = datetime(2026, 9, 3, tzinfo=UTC).timestamp()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _make_release(
    tmp_path: Path,
    *,
    label: str,
    config_path: Path,
    completed_at: str,
) -> tuple[RunEvidence, Path]:
    release = tmp_path / label
    release.mkdir()
    event_path = release / "events.tfevents"
    event_path.write_bytes(label.encode())
    config_sha256 = _sha256(config_path)
    run = {
        "run_id": f"training-{label}",
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": "d" * 64,
        "created_at": "2026-09-03T00:00:00Z",
        "metadata": {"inputs": {"prepared_train": {"sha256": "e" * 64}}},
    }
    run_path = _write(release / "training-run.json", run)
    completion = {
        "status": "succeeded",
        "run_id": run["run_id"],
        "run_manifest_sha256": _sha256(run_path),
        "source_training_completion_sha256": "f" * 64,
        "completed_at": completed_at,
        "evidence": {
            "learning": {
                "status": "passed",
                "optimizer_steps": 100,
                "event_stream": {"sha256": _sha256(event_path)},
            },
            "learnability_acceptance": {"status": "passed"},
            "checkpoint_progression": {
                "model_lora_array_count": 410,
                "changed_model_lora_array_count": 410,
                "checkpoint_relative_delta": 0.05,
            },
        },
    }
    completion_path = _write(release / "training-completion.json", completion)
    _write(
        release / "completion.json",
        {
            "status": "succeeded",
            "run_id": f"provider-{label}",
            "training_run_id": run["run_id"],
            "config_sha256": config_sha256,
            "source_training_completion_sha256": completion["source_training_completion_sha256"],
            "portable_package": {
                "training_run_sha256": _sha256(run_path),
                "training_completion_sha256": _sha256(completion_path),
            },
            "files": [
                {
                    "path": "training/completion.json",
                    "sha256": _sha256(completion_path),
                }
            ],
        },
    )
    app_id = f"app-{label}"
    billing_path = _write(
        tmp_path / f"billing-{label}.json",
        [{"Object ID": app_id, "Cost": "0.50"}],
    )
    return RunEvidence(label, release, billing_path, app_id), event_path


def _fixture(tmp_path: Path):
    baseline, baseline_event = _make_release(
        tmp_path,
        label="baseline",
        config_path=BASELINE_CONFIG,
        completed_at="2026-09-03T00:05:31Z",
    )
    first, first_event = _make_release(
        tmp_path,
        label="candidate-first",
        config_path=CANDIDATE_CONFIG,
        completed_at="2026-09-03T00:12:04Z",
    )
    warm, warm_event = _make_release(
        tmp_path,
        label="candidate-warm",
        config_path=CANDIDATE_CONFIG,
        completed_at="2026-09-03T00:09:54Z",
    )
    profiles = {
        baseline_event: (RUN_EPOCH + 200.0, 0.95, 606.0),
        first_event: (RUN_EPOCH + 630.0, 0.58, 995.0),
        warm_event: (RUN_EPOCH + 480.0, 0.84, 684.0),
    }

    def load_events(path: Path):
        first_wall_time, step_time, tokens = profiles[path]
        return {
            "perf/step_time_seconds": [
                (step, first_wall_time + step, step_time) for step in range(100)
            ],
            "perf/per_device_tokens_per_sec": [
                (step, first_wall_time + step, tokens) for step in range(100)
            ],
        }

    return baseline, [first, warm], load_events, profiles


def test_compare_remat_runs_keeps_short_canary_on_full_remat(tmp_path: Path) -> None:
    baseline, candidates, load_events, _ = _fixture(tmp_path)

    receipt = compare_remat_runs(
        baseline_config_path=BASELINE_CONFIG,
        candidate_config_path=CANDIDATE_CONFIG,
        baseline=baseline,
        candidates=candidates,
        event_loader=load_events,
    )

    assert receipt["status"] == "passed"
    assert receipt["decision"]["selected_canary_remat_policy"] == "full"
    assert receipt["decision"]["minimum_optimizer_steps_for_no_remat"] > 100
    assert receipt["candidates"][0]["comparison"]["steady_step_time_reduction_fraction"] > 0.38
    assert receipt["host_variance"]["repetitions"] == 2
    assert receipt["baseline"]["provider_release"] == {
        "volume": "bookforge-jax-fidelity-release",
        "path": "provider-baseline",
    }


def test_compare_remat_runs_rejects_unrepeated_candidate(tmp_path: Path) -> None:
    baseline, candidates, load_events, _ = _fixture(tmp_path)

    with pytest.raises(ValueError, match="at least two"):
        compare_remat_runs(
            baseline_config_path=BASELINE_CONFIG,
            candidate_config_path=CANDIDATE_CONFIG,
            baseline=baseline,
            candidates=candidates[:1],
            event_loader=load_events,
        )


def test_compare_remat_runs_rejects_slower_no_remat(tmp_path: Path) -> None:
    baseline, candidates, load_events, profiles = _fixture(tmp_path)
    event_path = candidates[1].release_directory / "events.tfevents"
    first_wall_time, _, tokens = profiles[event_path]
    profiles[event_path] = (first_wall_time, 1.1, tokens)

    with pytest.raises(ValueError, match="did not improve steady-state speed"):
        compare_remat_runs(
            baseline_config_path=BASELINE_CONFIG,
            candidate_config_path=CANDIDATE_CONFIG,
            baseline=baseline,
            candidates=candidates,
            event_loader=load_events,
        )
