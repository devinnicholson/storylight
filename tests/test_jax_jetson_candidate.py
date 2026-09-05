import hashlib
import importlib.util
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bookforge.fidelity_benchmark import population_contract_from_manifest
from bookforge.fidelity_schema import DatasetSplit

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "deploy/jetson/install-trained-planner-candidate.sh"
SHADOW = ROOT / "deploy/jetson/run-trained-planner-shadow.sh"
CANDIDATE_EVALUATION = ROOT / "deploy/jetson/run-trained-planner-candidate-evaluation.sh"
CANDIDATE_DEVELOPMENT_GATE = (
    ROOT / "deploy/jetson/trained-planner-candidate-evaluation.py"
)
PROMOTE = ROOT / "deploy/jetson/promote-trained-planner.sh"
ROLLBACK = ROOT / "deploy/jetson/rollback-trained-planner.sh"
SERVICE = ROOT / "deploy/jetson/systemd/bookforge-trained-planner-candidate@.service"
SHADOW_EVIDENCE = ROOT / "deploy/jetson/trained-planner-shadow-evidence.py"
BASELINE_IDENTITY = ROOT / "deploy/jetson/emit-accepted-planner-identity.sh"
TOOLING_BUILDER = ROOT / "deploy/jetson/build-trained-planner-tooling-bundle.py"
TOOLING_INSTALLER = ROOT / "deploy/jetson/install-trained-planner-tooling.py"
ACCEPTANCE_PREFLIGHT = ROOT / "deploy/jetson/preflight-trained-planner-acceptance.py"


def _load_shadow_evidence():
    spec = importlib.util.spec_from_file_location(
        "trained_planner_shadow_evidence", SHADOW_EVIDENCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow_evidence = _load_shadow_evidence()


def _load_acceptance_preflight():
    spec = importlib.util.spec_from_file_location(
        "trained_planner_acceptance_preflight", ACCEPTANCE_PREFLIGHT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acceptance_preflight = _load_acceptance_preflight()


def _load_candidate_development_gate():
    spec = importlib.util.spec_from_file_location(
        "trained_planner_candidate_evaluation", CANDIDATE_DEVELOPMENT_GATE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate_development_gate = _load_candidate_development_gate()


def _load_tooling_installer():
    spec = importlib.util.spec_from_file_location(
        "trained_planner_tooling_installer", TOOLING_INSTALLER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tooling_installer = _load_tooling_installer()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, str]:
    bundle = tmp_path / "candidate"
    engine = bundle / "engines/llm/llm.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"synthetic-tensorrt-engine")
    engine_sha256 = _sha256(engine)
    manifest = {
        "schema_version": "1.0",
        "result": "complete",
        "candidate_id": "jax-r8-seed-20260901",
        "model_id": "google/gemma-4-E2B-it",
        "base_model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "model_revision": "sha256:" + engine_sha256,
        "engine_sha256": engine_sha256,
        "tensorrt_edge_llm_revision": "71dd1bae032e70771265917ec74d3ff4cad07a10",
        "cache_contract_revision": "semantic-v18-tensorrt-slot-privacy",
        "planner_protocol": "four-slot-v1",
        "engine_path": "engines/llm/llm.engine",
        "file_count": 1,
        "total_bytes": engine.stat().st_size,
        "files": [
            {
                "path": "engines/llm/llm.engine",
                "bytes": engine.stat().st_size,
                "sha256": _sha256(engine),
            }
        ],
    }
    manifest_path = bundle / "candidate.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return bundle, _sha256(manifest_path)


def _hidden_report(
    tmp_path: Path,
    candidate_revision: str,
    *,
    candidate_manifest_sha256: str = "d" * 64,
) -> tuple[Path, str]:
    summary = {
        "surface": "raw",
        "split": "hidden",
        "records": 512,
        "record_ids_sha256": "a" * 64,
        "category_record_counts": {"negation": 512},
        "schema_valid_rate": 1.0,
        "privacy_pass_rate": 1.0,
        "semantic_atom_recall": 0.99,
        "exact_example_pass_rate": 0.96,
        "category_pass_rates": {"negation": 1.0},
        "counterfactual_pairs": 256,
        "counterfactual_sensitivity": 0.99,
        "unsupported_concept_rate": 0.0,
        "pii_leaks": 0,
        "privacy_term_leaks": 0,
        "source_echoes": 0,
        "injection_leaks": 0,
        "forbidden_hits": 0,
    }
    report = tmp_path / "hidden-summary.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "story-fidelity-evaluation-v1",
                "split": "hidden",
                "candidate_identity": {
                    "candidate_id": "jax-r8-seed-20260901",
                    "candidate_manifest_sha256": candidate_manifest_sha256,
                    "engine_sha256": candidate_revision.removeprefix("sha256:"),
                    "model_revision": candidate_revision,
                },
                "dataset_manifest_sha256": "d" * 64,
                "custody_receipt_sha256": "e" * 64,
                "privacy": {"passages_recorded": False, "outputs_recorded": False},
                "summary": summary,
            },
            sort_keys=True,
        )
        + "\n"
    )
    report.chmod(0o600)
    return report, _sha256(report)


def _leg(label: str, index: int, revision: str, engine: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "label": label,
        "order_index": index,
        "model_revision": revision,
        "engine_sha256": engine,
        "benchmark_sha256": str(index) * 64,
        "latency_seconds": [1.35, 1.45, 1.55],
        "maximum_output_tokens": 52,
        "automatic_semantic_pass": True,
        "planner_ready_seconds": 8.0,
        "service_memory_peak_bytes": int(3.5 * 1024**3),
        "minimum_available_memory_kib": 1024 * 1024,
        "inference_swap_events": 0,
        "oom_events": 0,
        "external_planner_socket_events": 0,
        "restart_failures": 0,
        "kiosk_active": True,
        "projector_http_passed": True,
        "live_scene_passed": True,
    }


def _candidate_manifest(candidate_engine: str) -> dict[str, object]:
    return {
        "candidate_id": "jax-r8-seed-20260901",
        "engine_sha256": candidate_engine,
        "model_revision": "sha256:" + candidate_engine,
        "training_run_id": "lora-train-20260901",
        "source_config_sha256": "c" * 64,
        "source_dataset_manifest_sha256": "d" * 64,
    }


def test_installer_rejects_tampering_and_undeclared_files(tmp_path: Path) -> None:
    bundle, digest = _bundle(tmp_path)
    (bundle / "engines/llm/llm.engine").write_bytes(b"tampered")

    tampered = subprocess.run(
        [
            "bash",
            str(INSTALLER),
            "--bundle",
            str(bundle),
            "--expected-manifest-sha256",
            digest,
            "--verify-only",
        ],
        capture_output=True,
        text=True,
    )
    assert tampered.returncode != 0
    assert "size mismatch" in tampered.stderr or "checksum mismatch" in tampered.stderr

    bundle, digest = _bundle(tmp_path / "extra")
    (bundle / "undeclared.txt").write_text("not in manifest")
    undeclared = subprocess.run(
        [
            "bash",
            str(INSTALLER),
            "--bundle",
            str(bundle),
            "--expected-manifest-sha256",
            digest,
            "--verify-only",
        ],
        capture_output=True,
        text=True,
    )
    assert undeclared.returncode != 0
    assert "undeclared or missing files" in undeclared.stderr


def test_privileged_directory_ancestry_rejects_non_root_owner() -> None:
    seen: list[Path] = []

    def fake_lstat(path: Path):
        seen.append(path)
        uid = 1000 if path == Path("/var/lib") else 0
        return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=uid, st_gid=0)

    with pytest.raises(ValueError, match="unsafe privileged directory ancestry: /var/lib"):
        tooling_installer.validate_root_ancestry(Path("/var/lib"), lstat=fake_lstat)

    assert seen == [Path("/"), Path("/var"), Path("/var/lib")]


def test_tooling_installer_rejects_a_symlink_bundle_root(tmp_path: Path) -> None:
    output = tmp_path / "tooling"
    commit = "f" * 40
    built = subprocess.run(
        [
            "python3",
            str(TOOLING_BUILDER),
            "--source-commit",
            commit,
            "--allow-uncommitted-source",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(built.stdout)
    linked = tmp_path / "linked-tooling"
    linked.symlink_to(output, target_is_directory=True)
    rejected = subprocess.run(
        [
            "python3",
            str(TOOLING_INSTALLER),
            "--bundle",
            str(linked),
            "--expected-manifest-sha256",
            receipt["manifest_sha256"],
            "--source-commit",
            commit,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "non-symlink directory" in rejected.stderr


def test_acceptance_preflight_requires_checksum_bound_post_oom_evidence(
    tmp_path: Path,
) -> None:
    engine_sha = "d" * 64
    evidence_path = tmp_path / "cold-start.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "producer": "bookforge-cold-start-recorder",
                "status": "passed",
                "accepted_engine_sha256": engine_sha,
                "oom_events": 0,
                "restart_failures": 0,
                "swap_events": 0,
                "projector_flow_passed": True,
                "restoration_demonstrated": True,
                "engine_sha256_after": engine_sha,
                "action_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "config_sha256_before": "b" * 64,
                "completed_realtime_usec": 200,
                "minimum_available_memory_mib": 1024,
                "planner_ready_seconds": 12.5,
            },
            sort_keys=True,
        )
        + "\n"
    )
    evidence_path.chmod(0o600)
    valid, detail, completed = acceptance_preflight.evidence_is_fresh(
        evidence_path,
        _sha256(evidence_path),
        engine_sha,
        100,
        768,
        trusted_root=tmp_path,
        required_owner_uid=evidence_path.stat().st_uid,
        required_owner_gid=evidence_path.stat().st_gid,
    )
    assert valid is True
    assert completed == 200
    assert "post-OOM" in detail

    valid, _, completed = acceptance_preflight.evidence_is_fresh(
        evidence_path,
        "e" * 64,
        engine_sha,
        100,
        768,
        trusted_root=tmp_path,
        required_owner_uid=evidence_path.stat().st_uid,
        required_owner_gid=evidence_path.stat().st_gid,
    )
    assert valid is False
    assert completed == 0


def test_candidate_development_gate_uses_device_report_before_hidden(
    tmp_path: Path,
) -> None:
    dataset_manifest = ROOT / "datasets/story-fidelity-v1/manifest.json"
    dataset_sha256 = _sha256(dataset_manifest)
    population = population_contract_from_manifest(
        dataset_manifest,
        expected_manifest_sha256=dataset_sha256,
        split=DatasetSplit.DEVELOPMENT,
    )
    engine_sha256 = "b" * 64
    candidate_manifest = tmp_path / "candidate.manifest.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "candidate_id": "fidelity-0123456789abcdefabcd",
                "engine_sha256": engine_sha256,
                "model_revision": f"sha256:{engine_sha256}",
                "training_run_id": "lora-train-20260901",
                "source_dataset_manifest_sha256": dataset_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    candidate_manifest_sha256 = _sha256(candidate_manifest)

    def report(path: Path, *, candidate: bool, perfect: bool) -> str:
        accepted_engine = (
            "95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"
        )
        identity = {
            "candidate_id": (
                "fidelity-0123456789abcdefabcd" if candidate else "accepted-baseline-test"
            ),
            "candidate_manifest_sha256": (
                candidate_manifest_sha256 if candidate else "a" * 64
            ),
            "engine_sha256": engine_sha256 if candidate else accepted_engine,
            "model_revision": f"sha256:{engine_sha256 if candidate else accepted_engine}",
        }
        rate = 1.0 if perfect else 0.0
        summary = {
            "surface": "raw",
            "split": "development",
            "records": population.records,
            "record_ids_sha256": population.record_ids_sha256,
            "category_record_counts": dict(population.category_record_counts),
            "schema_valid_rate": rate,
            "privacy_pass_rate": rate,
            "semantic_atom_recall": rate,
            "exact_example_pass_rate": rate,
            "category_pass_rates": {
                name: rate for name in population.category_record_counts
            },
            "counterfactual_pairs": population.pairs,
            "counterfactual_sensitivity": rate,
            "unsupported_concept_rate": 0.0,
            "pii_leaks": 0,
            "privacy_term_leaks": 0,
            "source_echoes": 0,
            "injection_leaks": 0,
            "forbidden_hits": 0,
        }
        path.write_text(
            json.dumps(
                {
                    "schema_version": "story-fidelity-evaluation-v1",
                    "split": "development",
                    "candidate_identity": identity,
                    "dataset_manifest_sha256": dataset_sha256,
                    "custody_receipt_sha256": None,
                    "privacy": {
                        "passages_recorded": False,
                        "outputs_recorded": False,
                    },
                    "summary": summary,
                },
                sort_keys=True,
            )
            + "\n"
        )
        return _sha256(path)

    candidate_report = tmp_path / "candidate-development.json"
    baseline_report = tmp_path / "baseline-development.json"
    candidate_report_sha256 = report(candidate_report, candidate=True, perfect=True)
    baseline_report_sha256 = report(baseline_report, candidate=False, perfect=False)
    gate = candidate_development_gate.build_development_gate(
        candidate_manifest_path=candidate_manifest,
        candidate_manifest_sha256=candidate_manifest_sha256,
        dataset_manifest_path=dataset_manifest,
        dataset_manifest_sha256=dataset_sha256,
        candidate_report_path=candidate_report,
        candidate_report_sha256=candidate_report_sha256,
        baseline_report_path=baseline_report,
        baseline_report_sha256=baseline_report_sha256,
    )

    assert gate["status"] == "passed"
    assert gate["reasons"] == []
    assert all(gate["checks"].values())
    assert gate["candidate_identity"]["engine_sha256"] == engine_sha256


def test_promotion_is_atomic_one_purpose_and_failure_reversible() -> None:
    promotion = PROMOTE.read_text()

    assert (
        "PROMOTE_BOOKFORGE_TRAINED_PLANNER:${target_user}:${candidate_id}:${manifest_sha256}:"
        "${gate_evidence_sha256}" in promotion
    )
    assert "--gate-evidence PATH" in promotion
    assert "--gate-evidence-sha256 SHA256" in promotion
    assert '[[ ! -f "$gate_evidence" || -L "$gate_evidence" ]]' in promotion
    assert 'sha256sum "$gate_evidence"' in promotion
    assert 'document.get("stage") != "gate"' in promotion
    assert 'document.get("producer") != "bookforge-fidelity-gate-builder"' in promotion
    assert (
        'document.get("training_run_id") != candidate_manifest.get("training_run_id")' in promotion
    )
    assert 'decision.get("passed") is not True' in promotion
    assert 'decision.get("reasons") != []' in promotion
    assert '"candidate_hidden_summary"' in promotion
    assert '"baseline_hidden_summary"' in promotion
    assert '"candidate_development_summary"' in promotion
    assert '"baseline_development_summary"' in promotion
    assert '"candidate_manifest"' in promotion
    assert '"human_review"' in promotion
    assert "expected_candidate_identity" in promotion
    assert "source_dataset_manifest_sha256" in promotion
    assert "source_config_sha256" in promotion
    assert "GATE_EVIDENCE_SHA256" in promotion
    assert "GATE_PRODUCER" in promotion
    assert "95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf" in promotion
    assert "semantic-v18-tensorrt-slot-privacy" in promotion
    assert "BOOKFORGE_EDGELLM_SERVER_PORT=11435" in promotion
    assert 'mv -f "$config_tmp" "$CONFIG_FILE"' in promotion
    assert "BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION" in promotion
    assert "BOOKFORGE_LIVE_SCENE_PLANNER_CACHE_CONTRACT_REVISION" in promotion
    assert "trap restore_exact_runtime EXIT INT TERM" in promotion
    assert 'user_systemctl disable "$ACCEPTED_UNIT"' in promotion
    assert 'user_systemctl enable "$UNIT"' in promotion
    assert "promotion-smoke.json" in promotion
    assert 'response.get("revision") != sys.argv[2]' in promotion
    assert "swapon --show --noheadings" in promotion
    assert "--dry-run" in promotion
    assert "sudo password" not in promotion.casefold()


def test_rollback_restores_checksum_bound_env_and_exact_engine() -> None:
    rollback = ROLLBACK.read_text()

    assert (
        "ROLLBACK_BOOKFORGE_TRAINED_PLANNER:${target_user}:${candidate_id}:${manifest_sha256}"
        in rollback
    )
    assert "BACKUP_SHA256" in rollback
    assert "state_field GATE_EVIDENCE_SHA256" in rollback
    assert "state_field GATE_PRODUCER" in rollback
    assert "state_field MODEL_REVISION" in rollback
    assert "state_field ENGINE_SHA256" in rollback
    assert "95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf" in rollback
    assert 'mv -f "$restore_tmp" "$CONFIG_FILE"' in rollback
    assert 'user_systemctl start "$ACCEPTED_UNIT"' in rollback
    assert 'user_systemctl enable "$ACCEPTED_UNIT"' in rollback
    assert "ROLLED_BACK" in rollback
    assert "swapon --show --noheadings" in rollback
    assert "--dry-run" in rollback
    assert "sudo password" not in rollback.casefold()


def test_shadow_aggregate_cli_publishes_gate_compatible_pair(tmp_path: Path) -> None:
    candidate_engine = "b" * 64
    accepted_engine = "a" * 64
    candidate_revision = "sha256:" + candidate_engine
    manifest_path = tmp_path / "candidate.manifest.json"
    manifest_path.write_text(
        json.dumps(_candidate_manifest(candidate_engine), indent=2, sort_keys=True) + "\n"
    )
    hidden_path, hidden_sha256 = _hidden_report(
        tmp_path,
        candidate_revision,
        candidate_manifest_sha256=_sha256(manifest_path),
    )
    leg_paths: list[Path] = []
    for index, (label, revision, engine) in enumerate(
        (
            ("baseline", "sha256:" + accepted_engine, accepted_engine),
            ("candidate", candidate_revision, candidate_engine),
            ("candidate", candidate_revision, candidate_engine),
            ("baseline", "sha256:" + accepted_engine, accepted_engine),
        )
    ):
        path = tmp_path / f"leg-{index}.json"
        path.write_text(json.dumps(_leg(label, index, revision, engine)) + "\n")
        leg_paths.append(path)
    restoration = tmp_path / "restoration.json"
    restoration.write_text(
        json.dumps(
            {
                "accepted_engine_sha256": accepted_engine,
                "engine_sha256_after": accepted_engine,
                "config_sha256_before": "f" * 64,
                "config_sha256_after": "f" * 64,
                "endpoint_ready": True,
                "unit_active": True,
            }
        )
        + "\n"
    )
    runtime = tmp_path / "runtime.json"
    stage = tmp_path / "stage.json"
    command = ["python3", str(SHADOW_EVIDENCE), "aggregate"]
    for path in leg_paths:
        command.extend(("--leg", str(path)))
    command.extend(
        (
            "--candidate-manifest",
            str(manifest_path),
            "--candidate-manifest-sha256",
            _sha256(manifest_path),
            "--accepted-engine-sha256",
            accepted_engine,
            "--hidden-report",
            str(hidden_path),
            "--hidden-report-sha256",
            hidden_sha256,
            "--restoration",
            str(restoration),
            "--run-id",
            "fidelity-campaign-20260901",
            "--config-sha256",
            "c" * 64,
            "--dataset-manifest-sha256",
            "d" * 64,
            "--int4-export-sha256",
            "e" * 64,
            "--runtime-output",
            str(runtime),
            "--stage-output",
            str(stage),
        )
    )

    subprocess.run(command, check=True)

    assert json.loads(runtime.read_text())["schema_version"] == "story-fidelity-runtime-v1"
    stage_document = json.loads(stage.read_text())
    assert stage_document["evidence_sha256"]["runtime"] == _sha256(runtime)
    assert subprocess.run(command, capture_output=True).returncode != 0
