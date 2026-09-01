from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
JAX_ROOT = ROOT / "infra/gcp/jax"


def _load_job_plan():
    name = "bookforge_jax_job_plan"
    spec = importlib.util.spec_from_file_location(name, JAX_ROOT / "job_plan.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(module, **updates: object):
    values: dict[str, object] = {
        "run_id": "bookforge-jax-smoke-20260901",
        "image_uri": "us-east1-docker.pkg.dev/your-gcp-project/jax/runner@sha256:" + "a" * 64,
        "config_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "prepared_train_sha256": "d" * 64,
        "input_manifest_sha256": "e" * 64,
        "base_checkpoint_manifest_sha256": "f" * 64,
        "tokenizer_manifest_sha256": "1" * 64,
        "service_account": "bookforge-jax-worker@your-gcp-project.iam.gserviceaccount.com",
        "scratch_uri": "gs://bookforge-jax-scratch",
        "release_uri": "gs://bookforge-jax-release",
        "hf_secret_resource": ("projects/your-gcp-project/secrets/bookforge-hf-read/versions/1"),
        "smoke": True,
    }
    values.update(updates)
    return module.JobInputs(**values)


def test_vertex_plan_is_one_attempt_one_chip_and_has_no_endpoint() -> None:
    module = _load_job_plan()
    plan = module.build_plan(_inputs(module))
    job_spec = plan["custom_job"]["jobSpec"]
    worker = job_spec["workerPoolSpecs"][0]

    assert plan["project"] == "your-gcp-project"
    assert plan["region"] == "us-east1"
    assert plan["mode"] == "plan-only"
    assert plan["resource"] == {
        "machine_type": "ct6e-standard-1t",
        "tpu_chips": 1,
        "replicas": 1,
        "timeout_seconds": 2700,
        "automatic_retries": 0,
        "endpoint_created": False,
    }
    assert worker["replicaCount"] == "1"
    assert worker["machineSpec"] == {"machineType": "ct6e-standard-1t"}
    assert "acceleratorType" not in json.dumps(worker)
    assert job_spec["scheduling"] == {
        "timeout": "2700s",
        "restartJobOnWorkerRestart": False,
        "disableRetries": True,
    }
    assert plan["approval_token"].endswith(plan["spec_sha256"])
    assert plan["gross_ceiling_policy"] == "declared-estimate-not-provider-enforced"
    assert (
        plan["input_bindings_sha256"]
        == hashlib.sha256(
            json.dumps(plan["input_bindings"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_vertex_plan_requires_digest_image_project_account_and_separate_buckets() -> None:
    module = _load_job_plan()
    with pytest.raises(ValueError, match="pinned"):
        module.build_plan(_inputs(module, image_uri="us-docker.pkg.dev/image:latest"))
    with pytest.raises(ValueError, match="Bookforge project"):
        module.build_plan(_inputs(module, service_account="worker@example.com"))
    with pytest.raises(ValueError, match="separate"):
        module.build_plan(
            _inputs(
                module,
                scratch_uri="gs://same-bucket/scratch",
                release_uri="gs://same-bucket/release",
            )
        )
    with pytest.raises(ValueError, match="numeric"):
        module.build_plan(
            _inputs(
                module,
                hf_secret_resource=(
                    "projects/your-gcp-project/secrets/bookforge-hf-read/versions/latest"
                ),
            )
        )


def test_vertex_plan_binds_hashes_secret_and_distinct_lifecycles() -> None:
    module = _load_job_plan()
    plan = module.build_plan(_inputs(module))
    serialized = json.dumps(plan)
    scratch = json.loads((JAX_ROOT / "scratch-lifecycle.json").read_text())
    release = json.loads((JAX_ROOT / "release-lifecycle.json").read_text())

    assert "b" * 64 in serialized
    assert "c" * 64 in serialized
    assert "e" * 64 in serialized
    assert "f" * 64 in serialized
    assert "projects/your-gcp-project/secrets/bookforge-hf-read/versions/1" in serialized
    assert scratch["lifecycle"]["rule"][0]["condition"]["age"] == 7
    assert release == {"lifecycle": {"rule": []}}


def test_worker_publishes_generation_guarded_completion_last() -> None:
    worker = (JAX_ROOT / "vertex_entrypoint.py").read_text()
    submitter = (JAX_ROOT / "submit_vertex_job.py").read_text()

    assert "if_generation_match=0" in worker
    assert worker.index("_upload_release(storage_client") < worker.index(
        "_write_completion(storage_client"
    )
    assert "package_training_release(" in worker
    assert 'runtime_lock="/opt/bookforge/runtime.lock.json"' in worker
    assert '"portable_package": package_evidence' in worker
    assert "disableRetries" in (JAX_ROOT / "job_plan.py").read_text()
    assert submitter.index("_create_intent(state_directory, plan)") < submitter.index(
        "urllib.request.Request("
    )
    assert '"retry_allowed": False' in submitter
    assert '"status": "rejected-pre-billable"' in submitter
    assert '"custom_job_created": False' in submitter


def test_input_staging_manifest_binds_every_checkpoint_and_tokenizer_byte(
    tmp_path: Path,
) -> None:
    stage = _load("bookforge_jax_stage_inputs", JAX_ROOT / "stage_inputs.py")
    files = {}
    for name in ("config.json", "dataset.json", "train.jsonl", "checkpoint.json", "tokenizer.json"):
        path = tmp_path / name
        path.write_text(f"{name}\n")
        files[name] = path
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.mkdir()
    tokenizer.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (tokenizer / "tokenizer.model").write_bytes(b"tokens")

    document, sources = stage.build_input_manifest(
        run_id="bookforge-jax-smoke-20260901",
        config=files["config.json"],
        dataset_manifest=files["dataset.json"],
        prepared_train=files["train.jsonl"],
        checkpoint=checkpoint,
        checkpoint_manifest=files["checkpoint.json"],
        tokenizer=tokenizer,
        tokenizer_manifest=files["tokenizer.json"],
    )

    assert document["status"] == "complete"
    assert set(sources) == {
        "config.json",
        "dataset/manifest.json",
        "prepared/train.jsonl",
        "checkpoint.manifest.json",
        "tokenizer.manifest.json",
        "checkpoint/weights.bin",
        "tokenizer/tokenizer.model",
    }
    assert "inputs.manifest.json" not in sources


def test_modal_staging_uploads_manifest_last_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    stage = _load("bookforge_modal_stage_inputs", JAX_ROOT / "stage_modal_inputs.py")
    run_id = "bookforge-jax-smoke-20260901"
    source = tmp_path / "config.json"
    source.write_text("{}\n")
    manifest = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "run_id": run_id,
        "status": "complete",
        "files": [
            {
                "path": "config.json",
                "bytes": source.stat().st_size,
                "sha256": stage.hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_bytes = stage.canonical_bytes(manifest)
    manifest_sha = stage.hashlib.sha256(manifest_bytes).hexdigest()
    monkeypatch.setenv(stage.APPROVAL_ENVIRONMENT, stage.approval_token(run_id, manifest_sha))
    commands: list[list[str]] = []
    uploaded_manifest = b""

    def runner(command, **_kwargs):
        nonlocal uploaded_manifest
        commands.append(command)
        if command[2:4] == ["ls", stage.VOLUME_NAME]:
            return stage.subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")
        if command[2] == "put" and command[-1].endswith("inputs.manifest.json"):
            uploaded_manifest = Path(command[-2]).read_bytes()
        if command[2] == "get":
            Path(command[-1]).write_bytes(uploaded_manifest)
        return stage.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    receipt = stage.stage_inputs(
        run_id=run_id,
        manifest=manifest,
        sources={"config.json": source},
        runner=runner,
    )

    puts = [command for command in commands if command[2] == "put"]
    assert puts[-1][-1] == f"/{run_id}/inputs.manifest.json"
    assert all("--force" not in command for command in puts)
    assert receipt["manifest_uploaded_last"] is True


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cloud_role_and_documentation_exclude_broad_access() -> None:
    role = (JAX_ROOT / "least-privilege-role.yaml").read_text().casefold()
    launcher = (JAX_ROOT / "launcher-role.yaml").read_text().casefold()
    readme = (JAX_ROOT / "README.md").read_text().casefold()
    storage = json.loads((JAX_ROOT / "storage-contract.json").read_text())

    assert "storage.objects.create" in role
    assert "storage.objects.get" in role
    assert "secretmanager.versions.access" in role
    assert "owner" not in role
    assert "editor" not in role
    assert "aiplatform.customjobs.create" in launcher
    assert "iam.serviceaccounts.actas" in launcher
    assert "owner" not in launcher
    assert storage["buckets"]["scratch"]["public_access_prevention"] == "enforced"
    assert storage["buckets"]["release"]["completion_written_last"] is True
    assert "uniform access" in readme
    assert "public-access prevention" in readme
    assert "ambiguous" in readme


def test_image_build_plan_binds_tracked_context_and_requires_pinned_builder() -> None:
    module = _load("bookforge_jax_image_plan", JAX_ROOT / "build_image_plan.py")
    builder = "gcr.io/cloud-builders/docker@sha256:" + "9" * 64

    plan = module.build_plan(ROOT, builder_image=builder)

    assert plan["mode"] == "plan-only"
    assert plan["builder_image"] == builder
    assert plan["tagged_image_uri"].endswith(":" + plan["source_sha256"][:20])
    assert plan["digest_resolution_command"][-1] == "--format=value(image_summary.digest)"
    assert plan["remote_mutation"] is False
    with pytest.raises(ValueError, match="builder"):
        module.build_plan(ROOT, builder_image="gcr.io/cloud-builders/docker:latest")


def _private_bucket(*, lifecycle: bool) -> dict[str, object]:
    document: dict[str, object] = {
        "public_access_prevention": "enforced",
        "uniform_bucket_level_access": True,
    }
    if lifecycle:
        document["lifecycle_config"] = {"rule": [{"condition": {"age": 7}}]}
    return document


def test_read_only_cloud_preflight_requires_every_admission_fact() -> None:
    sys.path.insert(0, str(JAX_ROOT))
    module = _load("bookforge_jax_cloud_preflight", JAX_ROOT / "cloud_preflight.py")
    run_id = "bookforge-jax-smoke-20260901"
    secret = "projects/your-gcp-project/secrets/bookforge-hf-read/versions/3"
    snapshots = {
        "configuration": {
            "core": {"account": "operator@example.com", "project": "your-gcp-project"}
        },
        "billing": {"billingEnabled": True, "billingAccountName": "billingAccounts/123"},
        "services": [
            {"config": {"name": name}}
            for name in sorted(module.REQUIRED_SERVICES)
        ],
        "jobs": [],
        "scratch_bucket": _private_bucket(lifecycle=True),
        "release_bucket": _private_bucket(lifecycle=False),
        "secret": {"name": secret, "state": "ENABLED"},
        "quota": {
            "dimensionsInfos": [
                {"applicableLocations": ["us-east1"], "details": {"value": 1}}
            ]
        },
    }
    credits = {
        "project": "your-gcp-project",
        "run_id": run_id,
        "billing_account_name": "billingAccounts/123",
        "status": "verified-promotional-credit-balance",
        "verified_at": datetime.now(UTC).isoformat(),
    }

    report = module.evaluate(
        snapshots,
        run_id=run_id,
        secret_version=secret,
        credits_attestation=credits,
        spec_sha256="a" * 64,
        input_bindings_sha256="b" * 64,
    )

    assert report["ready"] is True
    snapshots["billing"] = {"billingEnabled": False}
    with pytest.raises(module.AdmissionRejected, match="billing_enabled"):
        module.evaluate(
            snapshots,
            run_id=run_id,
            secret_version=secret,
            credits_attestation=credits,
            spec_sha256="a" * 64,
            input_bindings_sha256="b" * 64,
        )


def _portable_release(source: Path, run_id: str) -> tuple[dict[str, bytes], str]:
    files = {
        "adapter.manifest.json": b'{"schema_version":"1.0"}\n',
        "runtime.lock.json": b'{"schema_version":"1.0"}\n',
        "training/run.json": b'{"status":"started"}\n',
        "training/completion.json": b'{"status":"succeeded"}\n',
    }
    package_rows = [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(files.items())
    ]
    files["package.manifest.json"] = (
        json.dumps({"schema_version": "1.0", "files": package_rows}, sort_keys=True) + "\n"
    ).encode()
    completion_rows = [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(files.items())
    ]
    completion = (
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "status": "succeeded",
                "backend": "vertex-tpu-v6e",
                "files": completion_rows,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    files["completion.json"] = completion
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return files, hashlib.sha256(completion).hexdigest()


def test_gcs_release_fetch_is_completion_pinned_and_rejects_extra_objects(
    tmp_path: Path,
) -> None:
    module = _load("bookforge_jax_gcs_fetch", JAX_ROOT / "fetch_gcs_release.py")
    run_id = "bookforge-jax-smoke-20260901"
    source = tmp_path / "remote"
    source.mkdir()
    files, completion_sha = _portable_release(source, run_id)

    def download(relative: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)

    destination = tmp_path / "fetched"
    result = module.fetch_release(
        run_id=run_id,
        expected_completion_sha256=completion_sha,
        destination=destination,
        download=download,
        remote_objects=files,
    )

    assert result["status"] == "succeeded"
    assert (destination / "training/completion.json").is_file()
    with pytest.raises(ValueError, match="undeclared"):
        module.fetch_release(
            run_id=run_id,
            expected_completion_sha256=completion_sha,
            destination=tmp_path / "bad",
            download=download,
            remote_objects=[*files, "unexpected.bin"],
        )


def test_jax_dashboard_contains_only_operational_dimensions() -> None:
    dashboard = ROOT / "infra/gcp/monitoring/jax-fidelity-dashboard.json"
    value = json.loads(dashboard.read_text())
    serialized = json.dumps(value).casefold()

    assert value["displayName"] == "Bookforge JAX fidelity finite training"
    assert "aiplatform_custom_job" in serialized
    assert "accelerator/duty_cycle" in serialized
    assert "accelerator/memory_used" in serialized
    for forbidden in ("prompt", "passage", "reader", "audio", "camera", "session_id"):
        assert forbidden not in serialized
