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
sys.path.insert(0, str(ROOT))


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
        "base_checkpoint_receipt_sha256": "2" * 64,
        "tokenizer_manifest_sha256": "1" * 64,
        "service_account": "bookforge-jax-worker@your-gcp-project.iam.gserviceaccount.com",
        "scratch_uri": "gs://bookforge-jax-scratch",
        "release_uri": "gs://bookforge-jax-release",
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


def test_vertex_plan_binds_staged_hashes_and_forces_offline_model_access() -> None:
    module = _load_job_plan()
    plan = module.build_plan(_inputs(module))
    serialized = json.dumps(plan)

    assert "b" * 64 in serialized
    assert "c" * 64 in serialized
    assert "e" * 64 in serialized
    assert "f" * 64 in serialized
    assert "2" * 64 in serialized
    assert "hf-secret" not in serialized.casefold()
    assert "secretmanager" not in serialized.casefold()
    environment = plan["custom_job"]["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]
    environment_map = {row["name"]: row["value"] for row in environment}
    assert {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items() <= environment_map.items()


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
    leaf = checkpoint / "run/checkpoints/0/items"
    leaf.mkdir(parents=True)
    (leaf / "weights.bin").write_bytes(b"weights")
    (tokenizer / "tokenizer.model").write_bytes(b"tokens")
    from training.jax_fidelity.integrity import artifact_manifest, canonical_json_bytes
    from training.jax_fidelity.orbax_receipt import orbax_leaf_receipt

    files["checkpoint.json"].write_bytes(canonical_json_bytes(artifact_manifest(leaf)))
    checkpoint_receipt = tmp_path / "checkpoint.receipt.json"
    checkpoint_receipt.write_bytes(
        canonical_json_bytes(
            orbax_leaf_receipt(
                checkpoint,
                leaf,
                expected_step=0,
                role="base-maxtext",
            )
        )
    )

    document, sources = stage.build_input_manifest(
        run_id="bookforge-jax-smoke-20260901",
        config=files["config.json"],
        dataset_manifest=files["dataset.json"],
        prepared_train=files["train.jsonl"],
        checkpoint=checkpoint,
        checkpoint_manifest=files["checkpoint.json"],
        checkpoint_receipt=checkpoint_receipt,
        tokenizer=tokenizer,
        tokenizer_manifest=files["tokenizer.json"],
    )

    assert document["status"] == "complete"
    assert set(sources) == {
        "config.json",
        "dataset/manifest.json",
        "prepared/train.jsonl",
        "checkpoint.manifest.json",
        "checkpoint.receipt.json",
        "tokenizer.manifest.json",
        "checkpoint/run/checkpoints/0/items/weights.bin",
        "tokenizer/tokenizer.model",
    }
    assert "inputs.manifest.json" not in sources
    assert document["base_orbax"] == {
        "role": "base-maxtext",
        "expected_step": 0,
        "relative_path": "run/checkpoints/0/items",
        "receipt_sha256": stage.sha256_file(checkpoint_receipt),
        "manifest_sha256": stage.sha256_file(files["checkpoint.json"]),
        "content_sha256": artifact_manifest(leaf)["content_sha256"],
    }


def test_full_training_staging_rejects_missing_or_wrong_base_orbax_receipt(
    tmp_path: Path,
) -> None:
    stage = _load("bookforge_jax_stage_inputs_orbax_gate", JAX_ROOT / "stage_inputs.py")
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.json"
    prepared = tmp_path / "train.jsonl"
    tokenizer_manifest = tmp_path / "tokenizer.manifest.json"
    for path in (config, dataset, prepared, tokenizer_manifest):
        path.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    leaf = checkpoint / "run/checkpoints/0/items"
    leaf.mkdir(parents=True)
    (leaf / "weights").write_bytes(b"base")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    from training.jax_fidelity.integrity import artifact_manifest, canonical_json_bytes
    from training.jax_fidelity.orbax_receipt import orbax_leaf_receipt

    checkpoint_manifest = tmp_path / "checkpoint.manifest.json"
    checkpoint_manifest.write_bytes(canonical_json_bytes(artifact_manifest(leaf)))
    arguments = {
        "run_id": "bookforge-jax-train-20260901",
        "config": config,
        "dataset_manifest": dataset,
        "prepared_train": prepared,
        "checkpoint": checkpoint,
        "checkpoint_manifest": checkpoint_manifest,
        "tokenizer": tokenizer,
        "tokenizer_manifest": tokenizer_manifest,
    }
    with pytest.raises(ValueError, match="requires a base Orbax receipt"):
        stage.build_input_manifest(**arguments, checkpoint_receipt=None)

    receipt = tmp_path / "checkpoint.receipt.json"
    receipt.write_bytes(
        canonical_json_bytes(
            orbax_leaf_receipt(
                checkpoint,
                leaf,
                expected_step=0,
                role="wrong-base",
            )
        )
    )
    with pytest.raises(ValueError, match="identity changed"):
        stage.build_input_manifest(**arguments, checkpoint_receipt=receipt)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_image_build_plan_binds_tracked_context_and_requires_pinned_builder() -> None:
    module = _load("bookforge_jax_image_plan", JAX_ROOT / "build_image_plan.py")
    source_manifest = _load(
        "bookforge_packaged_source_manifest",
        JAX_ROOT / "packaged_source_manifest.py",
    )
    builder = "gcr.io/cloud-builders/docker@sha256:" + "9" * 64

    plan = module.build_plan(ROOT, builder_image=builder)

    assert plan["mode"] == "plan-only"
    assert plan["builder_image"] == builder
    assert plan["tagged_image_uri"].endswith(":" + plan["source_sha256"][:20])
    assert plan["bookforge_source_manifest_sha256"] == source_manifest.source_manifest_sha256(
        source_manifest.packaged_bookforge_source_manifest(ROOT)
    )
    assert any(
        row["path"] == "infra/gcp/jax/packaged_source_manifest.py"
        for row in plan["source_files"]
    )
    assert plan["digest_resolution_command"][-1] == "--format=value(image_summary.digest)"
    assert (
        "--config={MATERIALIZED_CONTEXT}/infra/gcp/jax/image-cloudbuild.yaml"
        in plan["provider_build_command"]
    )
    assert (
        "--ignore-file={MATERIALIZED_CONTEXT}/infra/gcp/jax/image.gcloudignore"
        in plan["provider_build_command"]
    )
    assert plan["execution_command"][-1] == "--execute"
    assert plan["remote_mutation"] is False
    with pytest.raises(ValueError, match="builder"):
        module.build_plan(ROOT, builder_image="gcr.io/cloud-builders/docker:latest")


def test_image_build_is_reserved_before_the_only_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    planner = _load("bookforge_jax_image_approval", JAX_ROOT / "build_image_plan.py")
    submitter = _load("bookforge_jax_image_submitter", JAX_ROOT / "submit_image_build.py")
    context = tmp_path / "external-context"
    context.mkdir()
    source = context / "source.txt"
    source.write_text("exact\n")
    builder = "gcr.io/cloud-builders/docker@sha256:" + "b" * 64
    rows = [
            {
                "path": "source.txt",
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
    source_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    command = [
            "gcloud",
            "builds",
            "submit",
            "{MATERIALIZED_CONTEXT}",
            "--format=json",
        ]
    token = planner.approval_token(source_sha, builder, command)
    plan = {
        "schema_version": "1.0",
        "mode": "plan-only",
        "source_sha256": source_sha,
        "builder_image": builder,
        "approval_token": token,
        "source_files": rows,
        "provider_build_command": command,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setenv(submitter.APPROVAL_ENVIRONMENT, token)
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object):
        calls.append(command)
        intent = tmp_path / "state" / f"image-{source_sha}.submission-intent.json"
        assert intent.is_file()
        return submitter.subprocess.CompletedProcess(
            command, 0, stdout='{"status":"SUCCESS"}', stderr=""
        )

    result = submitter.submit(
        plan_path=plan_path,
        materialized_context=context,
        state_directory=tmp_path / "state",
        runner=runner,
    )

    assert result["status"] == "SUCCESS"
    assert calls[0][3] == str(context.resolve())
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        submitter.submit(
            plan_path=plan_path,
            materialized_context=context,
            state_directory=tmp_path / "state",
            runner=runner,
        )


def test_read_only_cloud_preflight_never_queries_secret_manager() -> None:
    sys.path.insert(0, str(JAX_ROOT))
    module = _load("bookforge_jax_cloud_preflight_commands", JAX_ROOT / "cloud_preflight.py")
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object):
        calls.append(command)
        return module.subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    snapshots = module.collect_snapshots(
        run_id="bookforge-jax-smoke-20260901",
        scratch_bucket="bookforge-jax-scratch",
        release_bucket="bookforge-jax-release",
        quota_id="tpu-v6e",
        runner=runner,
    )

    assert "secret" not in snapshots
    assert "job_audit_log" in snapshots
    assert "secretmanager.googleapis.com" not in module.REQUIRED_SERVICES
    assert all("secrets" not in command and "secretmanager" not in command for command in calls)
    audit_command = next(command for command in calls if command[1:3] == ["logging", "read"])
    assert "--freshness=400d" in audit_command
    assert 'protoPayload.serviceName="aiplatform.googleapis.com"' in audit_command[3]
    assert "JobService.CreateCustomJob" in audit_command[3]
    assert (
        'protoPayload.request.customJob.displayName="bookforge-jax-smoke-20260901"'
        in audit_command[3]
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


def test_vertex_reconciliation_blocks_new_spend_until_terminal_cost_evidence(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    submitter = _load("bookforge_jax_submitter_state", JAX_ROOT / "submit_vertex_job.py")
    reconciler = _load(
        "bookforge_jax_vertex_reconciler", JAX_ROOT / "reconcile_vertex_attempt.py"
    )
    run_id = "bookforge-jax-smoke-20260901"
    intent = tmp_path / f"{run_id}.submission-intent.json"
    intent.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "spec_sha256": "a" * 64,
                "status": "submission-intent-recorded",
                "retry_allowed": False,
            }
        )
    )
    with pytest.raises(RuntimeError, match="unreconciled"):
        submitter._assert_no_unreconciled_paid_attempt(tmp_path)
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "custom_job_found": True,
                "displayName": run_id,
                "state": "JOB_STATE_SUCCEEDED",
            }
        )
    )
    billing = tmp_path / "billing.json"
    billing.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "verified-final-provider-cost",
                "source": "authenticated billing export",
                "observed_at": datetime.now(UTC).isoformat(),
                "gross_cost_usd": 2.0,
                "credits_applied_usd": 2.0,
                "net_cost_usd": 0.0,
            }
        )
    )

    evidence = reconciler.reconcile(
        run_id=run_id,
        state_directory=tmp_path,
        job_evidence_path=job,
        billing_evidence_path=billing,
    )

    assert evidence["status"] == "reconciled"
    assert evidence["producer"] == "bookforge-gcp-jax-reconciler"
    assert evidence["submission_intent_sha256"] == hashlib.sha256(intent.read_bytes()).hexdigest()
    assert evidence["automatic_remote_deletion"] is False
    submitter._assert_no_unreconciled_paid_attempt(tmp_path)
    original_intent = intent.read_bytes()
    changed_intent = json.loads(original_intent)
    changed_intent["spec_sha256"] = "b" * 64
    intent.write_text(json.dumps(changed_intent))
    with pytest.raises(RuntimeError, match="stale Vertex reconciliation"):
        submitter._assert_no_unreconciled_paid_attempt(tmp_path)
    intent.write_bytes(original_intent)
    submitter._assert_no_unreconciled_paid_attempt(tmp_path)
    reconciliation = tmp_path / f"{run_id}.billing-reconciliation.json"
    original = reconciliation.read_text()
    forged = json.loads(original)
    forged["producer"] = "hand-authored"
    reconciliation.write_text(json.dumps(forged))
    with pytest.raises(RuntimeError, match="invalid Vertex reconciliation"):
        submitter._assert_no_unreconciled_paid_attempt(tmp_path)
    reconciliation.write_text(original)
    canonical_billing = tmp_path / f"{run_id}.final-billing-evidence.json"
    canonical_billing.chmod(0o600)
    canonical_billing.write_text('{"forged":true}\n')
    with pytest.raises(RuntimeError, match="changed Vertex billing evidence"):
        submitter._assert_no_unreconciled_paid_attempt(tmp_path)
