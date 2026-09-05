# ruff: noqa: E402
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import artifact_manifest, sha256_file

PLAN = ROOT / "experiments/jax-fidelity-lab/modal-plan-2026-09.json"
CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"
CONFIG_V2 = ROOT / "experiments/jax-fidelity-lab/config-v2.json"
CONFIG_V3 = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


modal_jax_fidelity = _load("modal_jax_fidelity", Path("deploy/modal_jax_fidelity.py"))
fetch_modal_jax_release = _load(
    "fetch_modal_jax_release", Path("scripts/fetch_modal_jax_release.py")
)
modal_reconciliation = _load("modal_reconciliation", Path("infra/gcp/jax/modal_reconciliation.py"))


def _request(**updates: object) -> dict[str, object]:
    run_id = f"bookforge-modal-smoke-{datetime.now(UTC).strftime('%Y%m%d')}"
    config_sha = "a" * 64
    dataset_sha = "b" * 64
    source_manifest_sha = "8" * 64
    image_source_sha = "7" * 64
    image_plan_sha = "6" * 64
    tagged_uri = (
        "us-east1-docker.pkg.dev/your-gcp-project/bookforge-jax/trainer:"
        + image_source_sha[:20]
    )
    bindings = {
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "prepared_train_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
        "base_checkpoint_manifest_sha256": "e" * 64,
        "base_checkpoint_receipt_sha256": "0" * 64,
        "tokenizer_manifest_sha256": "f" * 64,
    }
    resource_container = {
        "image_source_sha256": image_source_sha,
        "bookforge_source_manifest_sha256": source_manifest_sha,
        "image_build_plan_sha256": image_plan_sha,
        "intended_tagged_uri": tagged_uri,
        "runnable_digest_uri": None,
        "digest_resolved": False,
        "build_attempted": False,
    }
    resource = {
        "backend": "vertex-custom-job",
        "project": "your-gcp-project",
        "region": "us-east1",
        "display_name": run_id,
        "create_method": "google.cloud.aiplatform.v1.JobService.CreateCustomJob",
        "create_url": (
            "https://us-east1-aiplatform.googleapis.com/v1/projects/"
            "your-gcp-project/locations/us-east1/customJobs"
        ),
        "machine_type": "ct6e-standard-1t",
        "tpu_chips": 1,
        "replicas": 1,
        "timeout_seconds": 2700,
        "automatic_retries": 0,
        "endpoint_created": False,
        "service_account": (
            "bookforge-jax-worker@your-gcp-project.iam.gserviceaccount.com"
        ),
        "input_prefix": f"gs://bookforge-jax-scratch/inputs/{run_id}",
        "release_prefix": f"gs://bookforge-jax-release/releases/{run_id}",
        "input_bindings": bindings,
        "smoke": True,
        "container": resource_container,
    }
    rejection = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-unavailability-recorder",
        "status": "rejected-pre-billable",
        "rejection_kind": "gcp-unavailable-before-image-build",
        "project": "your-gcp-project",
        "region": "us-east1",
        "run_id": run_id,
        "checked_at": datetime.now(UTC).isoformat(),
        "submission_intent_created": False,
        "custom_job_created": False,
        "job_absence_verified": True,
        "run_id_absence_basis": "billing-disabled-plus-empty-create-audit-log-400d",
        "fallback_allowed": True,
        "reason": "billing disabled before image build",
        "billing_verification": {
            "active_project": "your-gcp-project",
            "active_account": "operator@example.com",
            "billing_enabled": False,
            "billing_account_name": "",
        },
        "audit_absence": {
            "log": "cloudaudit.googleapis.com/activity",
            "service_name": "aiplatform.googleapis.com",
            "method_name": "google.cloud.aiplatform.v1.JobService.CreateCustomJob",
            "display_name": run_id,
            "filter": modal_jax_fidelity._gcp_audit_filter(run_id),
            "freshness": "400d",
            "maximum_run_id_age_days": 7,
            "matching_entries": [],
        },
        "container_image": {
            **resource_container,
            "build_intent_created": False,
            "build_receipt_created": False,
            "build_admission": "blocked-billing-disabled",
        },
        "intended_vertex_resource": resource,
        "intended_vertex_resource_sha256": modal_jax_fidelity._canonical_sha256(
            resource
        ),
        "input_bindings": bindings,
        "input_bindings_sha256": modal_jax_fidelity._canonical_sha256(bindings),
        "queries_read_only": True,
        "remote_mutation": False,
    }
    rejection_sha = hashlib.sha256(
        (json.dumps(rejection, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    request: dict[str, object] = {
        "run_id": run_id,
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "prepared_train_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
        "base_checkpoint_manifest_sha256": "e" * 64,
        "base_checkpoint_receipt_sha256": "0" * 64,
        "tokenizer_manifest_sha256": "f" * 64,
        "bookforge_source_manifest_sha256": source_manifest_sha,
        "smoke": True,
        "gcp_rejection": rejection,
        "gcp_rejection_sha256": rejection_sha,
        "approval_token": (
            f"APPROVE_MODAL_JAX_RUN:{run_id}:{config_sha}:{dataset_sha}:{'c' * 64}:"
            f"{'d' * 64}:{'e' * 64}:{'0' * 64}:{'f' * 64}:"
            f"{source_manifest_sha}:{rejection_sha}:smoke"
        ),
    }
    request.update(updates)
    return request


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n")


def _persist_runtime_provenance_fixture(
    scratch: Path, *, run_id: str, training_run_id: str
) -> str:
    manifest = {
        "schema_version": "bookforge-jax-packaged-source-v1",
        "producer": "bookforge-modal-jax-image",
        "container_root": "/opt/bookforge",
        "ignore_patterns": [],
        "file_count": 0,
        "files_sha256": hashlib.sha256(b"[]").hexdigest(),
        "files": [],
    }
    source = scratch.parent / "fixture-source.manifest.json"
    source.write_bytes(
        (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    sources = [
        {
            "module": module,
            "path": f"/opt/MaxText/{relative}",
            "snapshot_path": snapshot,
            "bytes": 1,
            "sha256": hashlib.sha256(module.encode()).hexdigest(),
        }
        for module, relative, snapshot in modal_jax_fidelity._PATCHED_MAXTEXT_SOURCES
    ]
    provenance = {
        "bookforge_source_manifest": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
            "expected_sha256": source_sha256,
            "file_count": 0,
            "files_sha256": manifest["files_sha256"],
        },
        "runtime_lock": {
            "path": "/opt/bookforge/runtime.lock.json",
            "bytes": 1,
            "sha256": "3" * 64,
        },
        "approved_maxtext_patch": {
            "path": "/opt/bookforge/approved.patch",
            "bytes": 1,
            "sha256": "4" * 64,
            "expected_sha256": "4" * 64,
        },
        "maxtext": {
            "root": "/opt/MaxText",
            "expected_revision": "5" * 40,
            "observed_revision": "5" * 40,
            "patched_sources": sources,
        },
    }
    modal_jax_fidelity._persist_runtime_provenance(
        provenance,
        run_id=run_id,
        training_run_id=training_run_id,
        expected_source_manifest_sha256=source_sha256,
        scratch_directory=scratch,
    )
    return source_sha256


def _v2_prepared_inputs(
    root: Path,
) -> tuple[object, dict[str, object], dict[str, str]]:
    experiment = load_config(CONFIG_V2)
    prepared = root / "prepared/train.jsonl"
    source_train = root / "dataset/train.jsonl"
    prepared.parent.mkdir(parents=True)
    source_train.parent.mkdir(parents=True)
    prepared.write_bytes(b'{"messages":[]}\n')
    source_train.write_bytes(b'{"record_id":"source"}\n')
    hashes = {
        "config": experiment.sha256,
        "prepared": sha256_file(prepared),
        "source_train": sha256_file(source_train),
        "tokenizer_manifest": "a" * 64,
    }
    _write_json(
        root / "dataset/manifest.json",
        {"splits": {"train": {"sha256": hashes["source_train"]}}},
    )
    preparation = {
        "schema_version": "bookforge-jax-training-preparation-v2",
        "policy": experiment.training["preparation_policy"],
        "prepared_records": 320,
        "source_train_sha256": hashes["source_train"],
        "prepared_sha256": hashes["prepared"],
        "prompt_contract_sha256": experiment.production["prompt_contract_sha256"],
        "assistant_turns_per_record": 1,
        "pair_adjacency_preserved": True,
    }
    preparation_path = root / "prepared/preparation.manifest.json"
    _write_json(preparation_path, preparation)
    hashes["preparation_manifest"] = sha256_file(preparation_path)
    validation = {
        "schema_version": "bookforge-jax-prepared-validation-v1",
        "status": "passed",
        "config_sha256": hashes["config"],
        "prepared_sha256": hashes["prepared"],
        "preparation_manifest_sha256": hashes["preparation_manifest"],
        "tokenizer_manifest_sha256": hashes["tokenizer_manifest"],
        "prompt_contract_sha256": experiment.production["prompt_contract_sha256"],
        "records": 320,
        "assistant_turns_per_record": 1,
        "maximum_prompt_tokens": 200,
        "maximum_completion_tokens": 40,
        "maximum_total_tokens": 240,
        "input_budget_tokens": experiment.production["input_budget_tokens"],
        "completion_budget_tokens": experiment.production["completion_budget_tokens"],
        "max_target_length": experiment.training["max_target_length"],
    }
    validation_path = root / "prepared/prepared-validation.json"
    _write_json(validation_path, validation)
    hashes["prepared_validation"] = sha256_file(validation_path)
    binding = {
        "policy": experiment.training["preparation_policy"],
        "records": 320,
        "prepared_sha256": hashes["prepared"],
        "preparation_manifest_sha256": hashes["preparation_manifest"],
        "prepared_validation_sha256": hashes["prepared_validation"],
        "prompt_contract_sha256": experiment.production["prompt_contract_sha256"],
        "source_train_sha256": hashes["source_train"],
        "tokenizer_manifest_sha256": hashes["tokenizer_manifest"],
    }
    return experiment, {"prepared_training": binding}, hashes


def test_modal_jax_compilation_cache_rejects_unclaimed_or_changed_volume(
    tmp_path: Path,
) -> None:
    (tmp_path / "unexpected").write_text("not trusted")
    with pytest.raises(RuntimeError, match="not empty"):
        modal_jax_fidelity._prepare_compilation_cache({}, root=tmp_path)

    (tmp_path / "unexpected").unlink()
    (tmp_path / "owner.json").write_text("{}")
    with pytest.raises(RuntimeError, match="owner marker changed"):
        modal_jax_fidelity._prepare_compilation_cache({}, root=tmp_path)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("config_sha256", "prepared-validation"),
        ("prepared_sha256", "prepared-validation"),
        ("preparation_manifest_sha256", "prepared-validation"),
        ("tokenizer_manifest_sha256", "prepared-validation"),
        ("prompt_contract_sha256", "prepared-validation"),
    ],
)
def test_modal_v2_rejects_drifted_prepared_validation_bindings(
    tmp_path: Path, field: str, error: str
) -> None:
    experiment, input_manifest, hashes = _v2_prepared_inputs(tmp_path)
    path = tmp_path / "prepared/prepared-validation.json"
    validation = json.loads(path.read_text())
    validation[field] = "f" * 64
    _write_json(path, validation)
    input_manifest["prepared_training"] = dict(input_manifest["prepared_training"])
    input_manifest["prepared_training"]["prepared_validation_sha256"] = sha256_file(path)

    with pytest.raises(RuntimeError, match=error):
        modal_jax_fidelity._verify_v2_prepared_evidence(
            tmp_path,
            experiment=experiment,
            input_manifest=input_manifest,
            config_sha256=hashes["config"],
            prepared_sha256=hashes["prepared"],
            tokenizer_manifest_sha256=hashes["tokenizer_manifest"],
        )


def test_release_publication_resumes_without_overwrite_and_commits_completion_last(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    first = staging / "adapter/model.bin"
    first.parent.mkdir()
    first.write_bytes(b"adapter")
    second = staging / "training/completion.json"
    second.parent.mkdir()
    second.write_bytes(b"training")
    rows = [
        {
            "path": path.relative_to(staging).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (first, second)
    ]
    (staging / "completion.json").write_text(
        json.dumps({"status": "succeeded", "files": rows}, indent=2, sort_keys=True) + "\n"
    )
    destination = tmp_path / "release"
    existing = destination / "adapter/model.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"adapter")
    existing_stat = existing.stat()
    commits: list[set[str]] = []

    def commit() -> None:
        commits.append(
            {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file()
            }
        )

    modal_jax_fidelity._publish_staged_release(staging, destination, commit=commit)

    assert commits == [
        {"adapter/model.bin", "training/completion.json"},
        {"adapter/model.bin", "training/completion.json", "completion.json"},
    ]
    assert existing.stat().st_ino == existing_stat.st_ino
    assert existing.stat().st_mtime_ns == existing_stat.st_mtime_ns
    commits.clear()
    modal_jax_fidelity._publish_staged_release(staging, destination, commit=commit)
    assert commits == []


def test_finalize_completed_scratch_never_trains_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "bookforge-modal-full-20260902"
    training_run_id = "lora-train-fixture"
    values = {
        "config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "prepared_train_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
        "checkpoint_manifest_sha256": "e" * 64,
        "checkpoint_receipt_sha256": "f" * 64,
        "tokenizer_manifest_sha256": "1" * 64,
        "gcp_rejection_sha256": "2" * 64,
    }
    scratch = tmp_path / "scratch"
    output = scratch / "output"
    output.mkdir(parents=True)
    (output / "adapter.bin").write_bytes(b"adapter")
    values["bookforge_source_manifest_sha256"] = _persist_runtime_provenance_fixture(
        scratch, run_id=run_id, training_run_id=training_run_id
    )
    attempt = modal_jax_fidelity._attempt_document(
        run_id=run_id,
        training_run_id=training_run_id,
        smoke=False,
        **values,
    )
    (scratch / "attempt.json").write_text(json.dumps(attempt))
    (scratch / "gpu-preflight.json").write_text(
        json.dumps(
            {
                "hardware": "gpu",
                "devices": 2,
                "platform": "gpu",
                "memory_fraction": "0.95",
                "ici_fsdp_parallelism": -1,
                "mesh_shape": {"fsdp": 2},
            }
        )
    )
    training = scratch / "runs" / training_run_id
    training.mkdir(parents=True)
    (training / "completion.json").write_text(
        json.dumps(
            {
                "run_id": training_run_id,
                "status": "succeeded",
                "artifacts": [{"path": str(output / "adapter.bin")}],
                "evidence": {"runtime_lock": {"sha256": "3" * 64}},
            }
        )
    )

    def fake_package(**kwargs):
        destination = Path(kwargs["destination"])
        destination.mkdir()
        (destination / "adapter.bin").write_bytes(b"adapter")
        return {"status": "succeeded", "package_manifest_sha256": "4" * 64}

    monkeypatch.setattr(
        "training.jax_fidelity.remote_release.package_training_release", fake_package
    )
    monkeypatch.setattr(
        modal_jax_fidelity.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("finalization must never run training"),
    )
    release = tmp_path / "release"
    commits = 0
    scratch_commits = 0

    def commit() -> None:
        nonlocal commits
        commits += 1

    def scratch_commit() -> None:
        nonlocal scratch_commits
        scratch_commits += 1

    result = modal_jax_fidelity._finalize_completed_scratch(
        run_id=run_id,
        training_run_id=training_run_id,
        smoke=False,
        scratch_directory=scratch,
        release_directory=release,
        scratch_commit=scratch_commit,
        release_commit=commit,
        **values,
    )
    assert result["status"] == "succeeded"
    assert commits == 2
    assert scratch_commits == 1
    first_completion = (release / "completion.json").read_bytes()
    assert (release / "provider/runtime-provenance.json").is_file()
    assert (release / "provider/bookforge-source.manifest.json").is_file()
    assert result["bookforge_source_manifest_sha256"] == values[
        "bookforge_source_manifest_sha256"
    ]
    assert result["runtime_provenance_sha256"] == hashlib.sha256(
        (release / "provider/runtime-provenance.json").read_bytes()
    ).hexdigest()

    result_again = modal_jax_fidelity._finalize_completed_scratch(
        run_id=run_id,
        training_run_id=training_run_id,
        smoke=False,
        scratch_directory=scratch,
        release_directory=release,
        scratch_commit=scratch_commit,
        release_commit=commit,
        **values,
    )
    assert result_again == result
    assert commits == 2
    assert scratch_commits == 1
    assert (release / "completion.json").read_bytes() == first_completion


def test_modal_request_requires_hashes_exact_approval_and_prebillable_rejection() -> None:
    original = _request()
    validated = modal_jax_fidelity._validate_request(original)
    assert validated[-1] is True
    assert modal_jax_fidelity._finalize_approval_token(
        "bookforge-modal-smoke-20260901", "d" * 64, "8" * 64, "7" * 64
    ) != modal_jax_fidelity._finalize_approval_token(
        "bookforge-modal-smoke-20260901", "d" * 64, "9" * 64, "7" * 64
    )

    with pytest.raises(ValueError, match="pre-billable"):
        modal_jax_fidelity._validate_request(_request(gcp_rejection={}))
    with pytest.raises(ValueError, match="approval"):
        modal_jax_fidelity._validate_request(_request(approval_token="approve"))
    with pytest.raises(ValueError, match="pre-billable"):
        modal_jax_fidelity._validate_request(
            _request(bookforge_source_manifest_sha256="9" * 64)
        )
    blocked = _request()
    blocked_rejection = dict(blocked["gcp_rejection"])
    blocked_rejection["fallback_allowed"] = False
    blocked["gcp_rejection"] = blocked_rejection
    blocked["gcp_rejection_sha256"] = hashlib.sha256(
        (json.dumps(blocked_rejection, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="pre-billable"):
        modal_jax_fidelity._validate_request(blocked)
    with pytest.raises(ValueError, match="SHA-256"):
        modal_jax_fidelity._validate_request(_request(config_sha256="latest"))
    with pytest.raises(ValueError, match="SHA-256"):
        modal_jax_fidelity._validate_request(
            _request(base_checkpoint_receipt_sha256="unverified")
        )
    request = _request()
    rejection = dict(request["gcp_rejection"])
    rejection["input_bindings"] = dict(rejection["input_bindings"])
    rejection["input_bindings"]["prepared_train_sha256"] = "0" * 64
    request["gcp_rejection"] = rejection
    request["gcp_rejection_sha256"] = hashlib.sha256(
        (json.dumps(rejection, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="staged inputs"):
        modal_jax_fidelity._validate_request(request)

    source_mismatch = _request()
    mismatch_rejection = json.loads(json.dumps(source_mismatch["gcp_rejection"]))
    mismatch_rejection["container_image"]["bookforge_source_manifest_sha256"] = (
        "9" * 64
    )
    mismatch_rejection["intended_vertex_resource"]["container"][
        "bookforge_source_manifest_sha256"
    ] = "9" * 64
    mismatch_rejection["intended_vertex_resource_sha256"] = (
        modal_jax_fidelity._canonical_sha256(
            mismatch_rejection["intended_vertex_resource"]
        )
    )
    mismatch_sha = hashlib.sha256(
        (json.dumps(mismatch_rejection, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    source_mismatch["gcp_rejection"] = mismatch_rejection
    source_mismatch["gcp_rejection_sha256"] = mismatch_sha
    source_mismatch["approval_token"] = modal_jax_fidelity._approval_token(
        str(source_mismatch["run_id"]),
        str(source_mismatch["config_sha256"]),
        str(source_mismatch["dataset_manifest_sha256"]),
        str(source_mismatch["prepared_train_sha256"]),
        str(source_mismatch["input_manifest_sha256"]),
        str(source_mismatch["base_checkpoint_manifest_sha256"]),
        str(source_mismatch["base_checkpoint_receipt_sha256"]),
        str(source_mismatch["tokenizer_manifest_sha256"]),
        str(source_mismatch["bookforge_source_manifest_sha256"]),
        mismatch_sha,
        smoke=True,
    )
    with pytest.raises(ValueError, match="pre-billable"):
        modal_jax_fidelity._validate_request(source_mismatch)

    legacy = _request()
    legacy_rejection = dict(legacy["gcp_rejection"])
    legacy_rejection["producer"] = "bookforge-gcp-jax-submitter"
    legacy_rejection["spec_sha256"] = "9" * 64
    legacy_sha = hashlib.sha256(
        (json.dumps(legacy_rejection, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    legacy["gcp_rejection"] = legacy_rejection
    legacy["gcp_rejection_sha256"] = legacy_sha
    legacy["approval_token"] = modal_jax_fidelity._approval_token(
        str(legacy["run_id"]),
        str(legacy["config_sha256"]),
        str(legacy["dataset_manifest_sha256"]),
        str(legacy["prepared_train_sha256"]),
        str(legacy["input_manifest_sha256"]),
        str(legacy["base_checkpoint_manifest_sha256"]),
        str(legacy["base_checkpoint_receipt_sha256"]),
        str(legacy["tokenizer_manifest_sha256"]),
        str(legacy["bookforge_source_manifest_sha256"]),
        legacy_sha,
        smoke=True,
    )
    with pytest.raises(ValueError, match="pre-billable"):
        modal_jax_fidelity._validate_request(legacy)


def test_modal_billing_parser_accepts_current_and_legacy_fields() -> None:
    parse = modal_jax_fidelity._parse_modal_billing_total

    assert parse('[{"cost": "1.25"}, {"Cost": 0.5}]') == pytest.approx(1.75)
    with pytest.raises(RuntimeError, match="no cost"):
        parse('[{"date": "2026-09-02"}]')
    with pytest.raises(RuntimeError, match="conflicting"):
        parse('[{"cost": 1, "Cost": 2}]')


def test_modal_release_fetch_rejects_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    run_id = "bookforge-modal-smoke-20260901"

    def fake_get(remote_path: str, destination: Path) -> None:
        if remote_path.endswith("completion.json"):
            destination.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": "succeeded",
                        "files": [{"path": "model.bin", "bytes": 3, "sha256": "0" * 64}],
                    }
                )
            )
            return
        payload = destination / run_id
        payload.mkdir(parents=True)
        (payload / "model.bin").write_bytes(b"bad")

    monkeypatch.setattr(fetch_modal_jax_release, "_modal_get", fake_get)
    with pytest.raises(ValueError, match="failed verification"):
        completion = {
            "run_id": run_id,
            "status": "succeeded",
            "files": [{"path": "model.bin", "bytes": 3, "sha256": "0" * 64}],
        }
        completion_sha = hashlib.sha256(json.dumps(completion).encode()).hexdigest()
        fetch_modal_jax_release.fetch_release(
            run_id, completion_sha, tmp_path / "release"
        )


def _install_release_fetch_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run_id: str,
    extra_path: str,
) -> str:
    package = tmp_path / "package"
    (package / "adapter").mkdir(parents=True)
    (package / "training").mkdir()
    (package / "adapter/model.bin").write_bytes(b"checkpoint")
    (package / "adapter.manifest.json").write_text("{}\n")
    (package / "runtime.lock.json").write_text("{}\n")
    (package / "training/run.json").write_text("{}\n")
    (package / "training/completion.json").write_text("{}\n")
    package_manifest = artifact_manifest(package)
    (package / "package.manifest.json").write_text(
        json.dumps(package_manifest, sort_keys=True) + "\n"
    )
    extra = package / extra_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unexpected")
    rows = [
        {
            "path": path.relative_to(package).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in package.rglob("*") if item.is_file())
    ]
    outer = {"run_id": run_id, "status": "succeeded", "files": rows}

    def fake_get(remote_path: str, destination: Path) -> None:
        if remote_path.endswith("completion.json"):
            destination.write_text(json.dumps(outer))
            return
        shutil.copytree(package, destination / run_id)

    monkeypatch.setattr(fetch_modal_jax_release, "_modal_get", fake_get)
    return hashlib.sha256(json.dumps(outer).encode()).hexdigest()


def test_modal_release_fetch_rejects_undeclared_portable_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "bookforge-modal-undeclared-adapter"
    completion_sha = _install_release_fetch_fixture(
        monkeypatch,
        tmp_path,
        run_id=run_id,
        extra_path="adapter/undeclared.bin",
    )

    with pytest.raises(ValueError, match="undeclared or missing"):
        fetch_modal_jax_release.fetch_release(
            run_id, completion_sha, tmp_path / "release"
        )


def test_modal_reconciliation_retains_remote_state_and_rejects_duplicate_attempts(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    entry = modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id="jax:fixture",
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=1.25,
        declared_ceiling_usd=3.5,
        status="succeeded",
        result={"status": "succeeded"},
    )

    assert entry["reported_delta_usd"] == pytest.approx(0.25)
    assert entry["declared_ceiling_provider_enforced"] is False
    assert entry["remote_state_retained"] is True
    assert entry["automatic_remote_deletion"] is False
    assert entry["billing"] == {
        "status": "provisional-provider-app-cost",
        "workspace_observation": {
            "status": "provisional",
            "observed_at": entry["recorded_at"],
            "before_usd": 1.0,
            "after_usd": 1.25,
            "delta_usd": pytest.approx(0.25),
            "report_error": None,
        },
        "settlements": [],
    }
    with pytest.raises(ValueError, match="already reconciled"):
        modal_reconciliation.append_reconciliation(
            ledger,
            attempt_id="jax:fixture",
            stage="jax-training",
            workspace_before_usd=1.0,
            workspace_after_usd=1.25,
            declared_ceiling_usd=3.5,
            status="succeeded",
            result={"status": "succeeded"},
        )


def test_modal_provider_app_settlement_is_single_use_and_app_id_is_unique(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id="jax:first",
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=None,
        declared_ceiling_usd=0.25,
        status="succeeded",
        result={"status": "succeeded"},
        postrun_report_error="workspace total unavailable",
    )
    settled = modal_reconciliation.settle_reconciliation(
        ledger,
        attempt_id="jax:first",
        provider_app_id="ap-one0000000000000000000",
        provider_app_description="first-app",
        billing_report=(
            '[{"Object ID":"ap-one0000000000000000000",'
            '"Description":"first-app","Cost":"0.500"}]'
        ),
    )
    assert settled["workspace_after_usd"] is None
    assert settled["postrun_report_error"] == "workspace total unavailable"
    assert settled["billing"]["settlements"][0]["declared_ceiling_exceeded"] is True

    with pytest.raises(ValueError, match="already settled"):
        modal_reconciliation.settle_attempt(
            ledger,
            attempt_id="jax:first",
            provider_app_id="ap-two0000000000000000000",
            provider_app_description="second-app",
            billing_report=(
                '[{"Object ID":"ap-two0000000000000000000",'
                '"Description":"second-app","Cost":"0.5"}]'
            ),
        )

    modal_reconciliation.reserve_attempt(
        ledger, attempt_id="jax:second", stage="jax-training"
    )
    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id="jax:second",
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=1.1,
        declared_ceiling_usd=1.0,
        status="succeeded",
        result={"status": "succeeded"},
    )
    with pytest.raises(ValueError, match="provider app is already settled"):
        modal_reconciliation.settle_attempt(
            ledger,
            attempt_id="jax:second",
            provider_app_id="ap-one0000000000000000000",
            provider_app_description="first-app",
            billing_report=(
                '[{"object_id":"ap-one0000000000000000000",'
                '"description":"first-app","cost":"0.1"}]'
            ),
        )


def test_modal_provider_app_settlement_rejects_mismatch_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id="jax:target",
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=1.1,
        declared_ceiling_usd=1.0,
        status="succeeded",
        result=None,
    )
    mismatched = [{"Object ID": "ap-target0000000000000000", "Description": "wrong", "Cost": "0.1"}]
    with pytest.raises(ValueError, match="exactly one row"):
        modal_reconciliation.settle_attempt(
            ledger,
            attempt_id="jax:target",
            provider_app_id="ap-target0000000000000000",
            provider_app_description="target-app",
            billing_report=json.dumps(mismatched),
        )
    duplicate = [
        {"Object ID": "ap-target0000000000000000", "Description": "target-app", "Cost": "0.1"},
        {"Object ID": "ap-target0000000000000000", "Description": "target-app", "Cost": "0.1"},
    ]
    with pytest.raises(ValueError, match="exactly one row"):
        modal_reconciliation.settle_attempt(
            ledger,
            attempt_id="jax:target",
            provider_app_id="ap-target0000000000000000",
            provider_app_description="target-app",
            billing_report=json.dumps(duplicate),
        )


def test_modal_provider_app_settlement_accepts_equal_aliases_but_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id="jax:target",
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=1.1,
        declared_ceiling_usd=1.0,
        status="succeeded",
        result=None,
    )
    duplicate_key_report = (
        '[{"Object ID":"ap-target0000000000000000","Object ID":"ap-target0000000000000000",'
        '"Description":"target-app","Cost":"0.1"}]'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        modal_reconciliation.settle_attempt(
            ledger,
            attempt_id="jax:target",
            provider_app_id="ap-target0000000000000000",
            provider_app_description="target-app",
            billing_report=duplicate_key_report,
        )

    settled = modal_reconciliation.settle_attempt(
        ledger,
        attempt_id="jax:target",
        provider_app_id="ap-target0000000000000000",
        provider_app_description="target-app",
        billing_report=(
            '[{"object_id":"ap-target0000000000000000","Object ID":"ap-target0000000000000000",'
            '"description":"target-app","Description":"target-app",'
            '"cost":"0.10","Cost":0.1}]'
        ),
    )
    assert settled["billing"]["settlements"][0]["provider_app_cost_usd"] == "0.10"


@pytest.mark.parametrize("tamper", ["contents", "mode", "missing"])
def test_modal_provider_app_raw_report_tampering_never_false_greens(
    tamper: str, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id="jax:target",
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=1.1,
        declared_ceiling_usd=1.0,
        status="succeeded",
        result=None,
    )
    settled = modal_reconciliation.settle_attempt(
        ledger,
        attempt_id="jax:target",
        provider_app_id="ap-target0000000000000000",
        provider_app_description="target-app",
        billing_report=(
            b'[{"object_id":"ap-target0000000000000000",'
            b'"description":"target-app","cost":"0.1"}]'
        ),
    )
    artifact = settled["billing"]["settlements"][0]["billing_report_artifact"]
    report_path = ledger.parent / artifact["path"]
    if tamper == "contents":
        report_path.chmod(0o600)
        report_path.write_bytes(
            b'[{"object_id":"ap-target0000000000000000",'
            b'"description":"target-app","cost":"0.2"}]'
        )
        report_path.chmod(0o400)
    elif tamper == "mode":
        report_path.chmod(0o600)
    else:
        report_path.unlink()

    with pytest.raises(ValueError, match="unresolved or provisional"):
        modal_reconciliation.assert_attempt_available(ledger, attempt_id="jax:next")


def test_modal_attempt_is_atomically_reserved_before_paid_call(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    attempt_id = "jax:bookforge-jax-smoke-20260901"

    modal_reconciliation.reserve_attempt(
        ledger, attempt_id=attempt_id, stage="jax-training"
    )

    with pytest.raises(ValueError, match="already started"):
        modal_reconciliation.reserve_attempt(
            ledger, attempt_id=attempt_id, stage="jax-training"
        )
    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id=attempt_id,
        stage="jax-training",
        workspace_before_usd=1.0,
        workspace_after_usd=1.5,
        declared_ceiling_usd=3.5,
        status="succeeded",
        result={"status": "succeeded"},
    )
    document = json.loads(ledger.read_text())
    assert len(document["entries"]) == 1
    assert document["entries"][0]["status"] == "succeeded"
