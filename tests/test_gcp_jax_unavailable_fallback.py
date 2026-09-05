# ruff: noqa: E402
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JAX_ROOT = ROOT / "infra/gcp/jax"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JAX_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fallback = _load(
    "bookforge_gcp_unavailable_fallback", JAX_ROOT / "record_unavailable_fallback.py"
)
modal_worker = _load(
    "bookforge_modal_fidelity_unavailable_test", ROOT / "deploy/modal_jax_fidelity.py"
)


def _image_plan(path: Path) -> dict[str, object]:
    files = [
        {"path": relative, "bytes": 1, "sha256": f"{index:x}" * 64}
        for index, relative in enumerate(
            (
                "deploy/worker.py",
                "experiments/jax-fidelity-lab/config.json",
                "infra/gcp/jax/worker.py",
                "src/bookforge/worker.py",
                "training/jax_fidelity/Dockerfile",
                "training/jax_fidelity/patches/maxtext-native-lora-materialization.patch",
            ),
            1,
        )
    ]
    source_sha = fallback._canonical_sha256(files)
    bookforge_source_sha = fallback.source_manifest_sha256(
        fallback.packaged_bookforge_source_manifest_from_rows(files)
    )
    builder = "gcr.io/cloud-builders/docker@sha256:" + "b" * 64
    command = [
        "gcloud",
        "builds",
        "submit",
        "{MATERIALIZED_CONTEXT}",
        "--project=your-gcp-project",
    ]
    command_sha = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode()
    ).hexdigest()
    plan = {
        "schema_version": "1.0",
        "mode": "plan-only",
        "project": fallback.PROJECT_ID,
        "region": fallback.REGION,
        "source_sha256": source_sha,
        "source_files": files,
        "bookforge_source_manifest_sha256": bookforge_source_sha,
        "builder_image": builder,
        "tagged_image_uri": f"{fallback.IMAGE_REPOSITORY}:{source_sha[:20]}",
        "automatic_retries": 0,
        "provider_build_command": command,
        "approval_token": (
            f"APPROVE_GCP_JAX_IMAGE_BUILD:{source_sha}:"
            f"{hashlib.sha256(builder.encode()).hexdigest()}:{command_sha}"
        ),
        "remote_mutation": False,
    }
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _bindings() -> dict[str, str]:
    return {
        "config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "prepared_train_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
        "base_checkpoint_manifest_sha256": "e" * 64,
        "base_checkpoint_receipt_sha256": "f" * 64,
        "tokenizer_manifest_sha256": "0" * 64,
    }


class _Gcloud:
    def __init__(
        self,
        *,
        project: str = fallback.PROJECT_ID,
        billing_enabled: bool = False,
        audit_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.project = project
        self.billing_enabled = billing_enabled
        self.audit_rows = [] if audit_rows is None else audit_rows
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[1:3] == ["config", "list"]:
            value: object = {
                "core": {"project": self.project, "account": "operator@example.com"}
            }
        elif command[1:5] == ["beta", "billing", "projects", "describe"]:
            value = {
                "billingEnabled": self.billing_enabled,
                "billingAccountName": "" if not self.billing_enabled else "billingAccounts/1",
            }
        elif command[1:3] == ["logging", "read"]:
            value = self.audit_rows
        else:  # pragma: no cover - protects the test double itself
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")


def _collect(tmp_path: Path, *, runner: _Gcloud | None = None) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan_path = tmp_path / "image-plan.json"
    _image_plan(plan_path)
    now = datetime.now(UTC)
    run_id = f"jax-full-{now.strftime('%Y%m%d')}-l4x2-v1"
    return fallback.collect_rejection(
        run_id=run_id,
        input_bindings=_bindings(),
        service_account=(
            "bookforge-jax-worker@your-gcp-project.iam.gserviceaccount.com"
        ),
        scratch_uri="gs://bookforge-jax-scratch",
        release_uri="gs://bookforge-jax-release",
        image_plan_path=plan_path,
        image_build_state_directory=tmp_path / "build-state",
        smoke=False,
        runner=runner or _Gcloud(),
        now=now,
    )


def test_refuses_unverified_project_billing_audit_or_old_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="active gcloud project"):
        _collect(tmp_path / "wrong-project", runner=_Gcloud(project="other-project"))

    billing_dir = tmp_path / "billing"
    billing_dir.mkdir()
    with pytest.raises(RuntimeError, match="billing.*disabled"):
        _collect(billing_dir, runner=_Gcloud(billing_enabled=True))

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    with pytest.raises(RuntimeError, match="audit absence"):
        _collect(audit_dir, runner=_Gcloud(audit_rows=[{"insertId": "existing"}]))

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    plan_path = old_dir / "image-plan.json"
    _image_plan(plan_path)
    with pytest.raises(ValueError, match="last seven days"):
        fallback.collect_rejection(
            run_id="jax-full-20200101-l4x2-v1",
            input_bindings=_bindings(),
            service_account=(
                "bookforge-jax-worker@your-gcp-project.iam.gserviceaccount.com"
            ),
            scratch_uri="gs://bookforge-jax-scratch",
            release_uri="gs://bookforge-jax-release",
            image_plan_path=plan_path,
            image_build_state_directory=old_dir / "state",
            smoke=False,
            runner=_Gcloud(),
            now=datetime.now(UTC),
        )

    binding_dir = tmp_path / "bindings"
    binding_dir.mkdir()
    plan_path = binding_dir / "image-plan.json"
    _image_plan(plan_path)
    with pytest.raises(ValueError, match="exactly seven"):
        fallback.collect_rejection(
            run_id=f"jax-full-{datetime.now(UTC).strftime('%Y%m%d')}-l4x2-v1",
            input_bindings={**_bindings(), "unbound_sha256": "1" * 64},
            service_account=(
                "bookforge-jax-worker@your-gcp-project.iam.gserviceaccount.com"
            ),
            scratch_uri="gs://bookforge-jax-scratch",
            release_uri="gs://bookforge-jax-release",
            image_plan_path=plan_path,
            image_build_state_directory=binding_dir / "state",
            smoke=False,
            runner=_Gcloud(),
        )


def _modal_request(evidence: dict[str, object]) -> dict[str, object]:
    rejection_bytes = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    rejection_sha = hashlib.sha256(rejection_bytes).hexdigest()
    bindings = evidence["input_bindings"]
    assert isinstance(bindings, dict)
    run_id = str(evidence["run_id"])
    return {
        "run_id": run_id,
        **bindings,
        "smoke": False,
        "bookforge_source_manifest_sha256": evidence["container_image"][
            "bookforge_source_manifest_sha256"
        ],
        "gcp_rejection": evidence,
        "gcp_rejection_sha256": rejection_sha,
        "approval_token": modal_worker._approval_token(
            run_id,
            str(bindings["config_sha256"]),
            str(bindings["dataset_manifest_sha256"]),
            str(bindings["prepared_train_sha256"]),
            str(bindings["input_manifest_sha256"]),
            str(bindings["base_checkpoint_manifest_sha256"]),
            str(bindings["base_checkpoint_receipt_sha256"]),
            str(bindings["tokenizer_manifest_sha256"]),
            str(evidence["container_image"]["bookforge_source_manifest_sha256"]),
            rejection_sha,
            smoke=False,
        ),
    }


def test_modal_accepts_only_fresh_exact_unavailability_evidence(tmp_path: Path) -> None:
    evidence = _collect(tmp_path)
    assert modal_worker._validate_request(_modal_request(evidence))[-1] is False

    tampered = json.loads(json.dumps(evidence))
    tampered["container_image"]["runnable_digest_uri"] = (
        "us-east1-docker.pkg.dev/your-gcp-project/bookforge-jax/trainer@sha256:"
        + "9" * 64
    )
    with pytest.raises(ValueError, match="pre-billable"):
        modal_worker._validate_request(_modal_request(tampered))

    invented_spec = json.loads(json.dumps(evidence))
    invented_spec["spec_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="pre-billable"):
        modal_worker._validate_request(_modal_request(invented_spec))

    stale = json.loads(json.dumps(evidence))
    stale["checked_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError, match="pre-billable"):
        modal_worker._validate_request(_modal_request(stale))

    source_mismatch = _modal_request(evidence)
    source_mismatch["bookforge_source_manifest_sha256"] = "9" * 64
    bindings = evidence["input_bindings"]
    assert isinstance(bindings, dict)
    source_mismatch["approval_token"] = modal_worker._approval_token(
        str(evidence["run_id"]),
        str(bindings["config_sha256"]),
        str(bindings["dataset_manifest_sha256"]),
        str(bindings["prepared_train_sha256"]),
        str(bindings["input_manifest_sha256"]),
        str(bindings["base_checkpoint_manifest_sha256"]),
        str(bindings["base_checkpoint_receipt_sha256"]),
        str(bindings["tokenizer_manifest_sha256"]),
        "9" * 64,
        str(source_mismatch["gcp_rejection_sha256"]),
        smoke=False,
    )
    with pytest.raises(ValueError, match="pre-billable"):
        modal_worker._validate_request(source_mismatch)
