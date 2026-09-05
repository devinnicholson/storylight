from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from bookforge.fidelity_closure_evidence import validate_terminal_evidence

ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "deploy/jetson/record-trained-planner-terminal-evidence.py"


def load_recorder():
    spec = importlib.util.spec_from_file_location("jetson_terminal_evidence", RECORDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder = load_recorder()


def identities(*, status: str = "passed", candidate_id: str = "candidate-20260901"):
    baseline = "a" * 64
    candidate = "b" * 64
    manifest_sha = "c" * 64
    gate_sha = "d" * 64
    gate = {
        "schema_version": "1.0",
        "stage": "gate",
        "producer": "bookforge-fidelity-gate-builder",
        "status": status,
        "run_id": "fidelity-run-20260901",
        "training_run_id": "training-run-20260901",
        "config_sha256": "1" * 64,
        "dataset_manifest_sha256": "2" * 64,
        "candidate_manifest_sha256": manifest_sha,
        "candidate_identity": {
            "candidate_id": candidate_id,
            "candidate_manifest_sha256": manifest_sha,
            "engine_sha256": candidate,
            "model_revision": f"sha256:{candidate}",
        },
        "baseline_identity": {
            "candidate_id": "accepted-baseline-20260901",
            "candidate_manifest_sha256": "e" * 64,
            "engine_sha256": baseline,
            "model_revision": f"sha256:{baseline}",
        },
    }
    manifest = {
        "candidate_id": candidate_id,
        "training_run_id": "training-run-20260901",
        "source_config_sha256": "1" * 64,
        "source_dataset_manifest_sha256": "2" * 64,
        "engine_sha256": candidate,
        "model_revision": f"sha256:{candidate}",
    }
    return gate, manifest, baseline, candidate, manifest_sha, gate_sha


@pytest.mark.parametrize("outcome", ["promoted", "retained"])
def test_terminal_documents_match_the_closure_schema(outcome: str) -> None:
    gate, manifest, baseline, candidate, manifest_sha, gate_sha = identities(
        status="passed" if outcome == "promoted" else "rejected"
    )
    active = candidate if outcome == "promoted" else baseline
    lineage = recorder.lineage(
        gate,
        manifest,
        candidate_manifest_sha256=manifest_sha,
        gate_artifact_sha256=gate_sha,
        baseline_engine_sha256=baseline,
        active_engine_sha256=active,
        outcome=outcome,
    )
    documents = recorder.documents(
        outcome=outcome,
        lineage_values=lineage,
        approval_sha256=hashlib.sha256(b"one-purpose-token").hexdigest(),
        maximum_output_tokens=42,
        backup_config_sha256="f" * 64 if outcome == "promoted" else None,
    )

    validate_terminal_evidence(
        documents["terminal-receipt.json"],
        documents["post-action-health.json"],
        documents.get("rollback-state.json"),
        run_id="fidelity-run-20260901",
        training_run_id="training-run-20260901",
        candidate_id="candidate-20260901",
        candidate_manifest_sha256=manifest_sha,
        gate_artifact_sha256=gate_sha,
        baseline_engine_sha256=baseline,
        candidate_engine_sha256=candidate,
        promoted=outcome == "promoted",
    )


def test_terminal_lineage_rejects_an_active_engine_outside_the_gate() -> None:
    gate, manifest, baseline, _, manifest_sha, gate_sha = identities()
    with pytest.raises(ValueError, match="active engine"):
        recorder.lineage(
            gate,
            manifest,
            candidate_manifest_sha256=manifest_sha,
            gate_artifact_sha256=gate_sha,
            baseline_engine_sha256=baseline,
            active_engine_sha256="9" * 64,
            outcome="promoted",
        )


def test_terminal_lineage_allows_a_96_character_candidate_id() -> None:
    candidate_id = "c" * 96
    gate, manifest, baseline, candidate, manifest_sha, gate_sha = identities(
        candidate_id=candidate_id
    )
    values = recorder.lineage(
        gate,
        manifest,
        candidate_manifest_sha256=manifest_sha,
        gate_artifact_sha256=gate_sha,
        baseline_engine_sha256=baseline,
        active_engine_sha256=candidate,
        outcome="promoted",
    )
    assert values["candidate_id"] == candidate_id


def test_terminal_lineage_rejects_an_overlong_candidate_id() -> None:
    gate, manifest, baseline, candidate, manifest_sha, gate_sha = identities(
        candidate_id="c" * 97
    )
    with pytest.raises(ValueError, match="invalid identifier"):
        recorder.lineage(
            gate,
            manifest,
            candidate_manifest_sha256=manifest_sha,
            gate_artifact_sha256=gate_sha,
            baseline_engine_sha256=baseline,
            active_engine_sha256=candidate,
            outcome="promoted",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gate, manifest: gate.update(schema_version="2.0"), "lineage"),
        (lambda gate, manifest: gate.update(stage="summary"), "lineage"),
        (lambda gate, manifest: gate.update(status="rejected"), "lineage"),
        (lambda gate, manifest: gate.update(config_sha256=None), "lineage"),
        (
            lambda gate, manifest: manifest.update(source_dataset_manifest_sha256="9" * 64),
            "lineage",
        ),
        (
            lambda gate, manifest: gate["candidate_identity"].update(
                model_revision="sha256:" + "9" * 64
            ),
            "engine identities",
        ),
        (
            lambda gate, manifest: manifest.update(engine_sha256="9" * 64),
            "engine identities",
        ),
        (
            lambda gate, manifest: manifest.update(model_revision="sha256:" + "9" * 64),
            "engine identities",
        ),
    ],
)
def test_terminal_lineage_rejects_inconsistent_gate_or_manifest(
    mutation, message: str
) -> None:
    gate, manifest, baseline, candidate, manifest_sha, gate_sha = identities()
    mutation(gate, manifest)
    with pytest.raises(ValueError, match=message):
        recorder.lineage(
            gate,
            manifest,
            candidate_manifest_sha256=manifest_sha,
            gate_artifact_sha256=gate_sha,
            baseline_engine_sha256=baseline,
            active_engine_sha256=candidate,
            outcome="promoted",
        )


def test_terminal_approval_tokens_bind_every_action_field() -> None:
    output = recorder.EVIDENCE_ROOT / "run-20260901"
    promotion = recorder.expected_approval_token(
        outcome="promoted",
        user="operator",
        candidate_id="candidate-20260901",
        candidate_manifest_sha256="c" * 64,
        gate_artifact_sha256="d" * 64,
        baseline_engine_sha256="a" * 64,
        output_directory=output,
    )
    assert promotion == (
        "PROMOTE_BOOKFORGE_TRAINED_PLANNER:operator:candidate-20260901:"
        + "c" * 64
        + ":"
        + "d" * 64
        + f":{output}"
    )
    retention = recorder.expected_approval_token(
        outcome="retained",
        user="operator",
        candidate_id="candidate-20260901",
        candidate_manifest_sha256="c" * 64,
        gate_artifact_sha256="d" * 64,
        baseline_engine_sha256="a" * 64,
        output_directory=output,
    )
    action = hashlib.sha256()
    for value in (
        "operator",
        "d" * 64,
        "c" * 64,
        "a" * 64,
        str(output),
    ):
        action.update(value.encode())
        action.update(b"\0")
    assert retention == f"RETAIN_BOOKFORGE_ACCEPTED_BASELINE:{action.hexdigest()}"


def test_terminal_output_must_be_an_immediate_evidence_child() -> None:
    assert recorder.validated_output_path(recorder.EVIDENCE_ROOT / "one") == (
        recorder.EVIDENCE_ROOT / "one"
    )
    for invalid in (
        Path("relative"),
        recorder.EVIDENCE_ROOT,
        recorder.EVIDENCE_ROOT / "parent" / "child",
    ):
        with pytest.raises(ValueError, match="immediate child"):
            recorder.validated_output_path(invalid)
