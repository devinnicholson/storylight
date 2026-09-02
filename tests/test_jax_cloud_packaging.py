from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import types
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
    scratch = json.loads((JAX_ROOT / "scratch-lifecycle.json").read_text())
    release = json.loads((JAX_ROOT / "release-lifecycle.json").read_text())

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
    assert "secretmanager" not in worker
    assert "access_secret_version" not in worker
    assert '("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")' in worker
    assert "environment.pop(name, None)" in worker
    assert 'environment["HF_HUB_OFFLINE"] = "1"' in worker
    assert "verify_base_orbax(" in worker
    assert "expected_step=0" in worker
    assert 'role="base-maxtext"' in worker
    assert "disableRetries" in (JAX_ROOT / "job_plan.py").read_text()
    assert submitter.index("_create_intent(state_directory, plan)") < submitter.index(
        "urllib.request.Request("
    )
    assert '"retry_allowed": False' in submitter
    assert '"status": "rejected-pre-billable"' in submitter
    assert '"custom_job_created": False' in submitter
    assert "rejection.fallback_allowed and rejection.job_absence_verified" in submitter


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


def test_full_training_manifest_includes_verified_public_dataset_splits(
    tmp_path: Path,
) -> None:
    stage = _load("bookforge_jax_full_stage_inputs", JAX_ROOT / "stage_inputs.py")
    config = tmp_path / "config.json"
    prepared = tmp_path / "prepared.jsonl"
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    tokenizer_manifest = tmp_path / "tokenizer.manifest.json"
    for path, contents in (
        (config, "{}\n"),
        (prepared, '{"prepared":true}\n'),
        (train, '{"split":"train"}\n'),
        (development, '{"split":"development"}\n'),
        (tokenizer_manifest, "{}\n"),
    ):
        path.write_text(contents, encoding="utf-8")
    dataset_manifest = tmp_path / "manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"path": train.name, "sha256": stage.sha256_file(train)},
                    "development": {
                        "path": development.name,
                        "sha256": stage.sha256_file(development),
                    },
                    "hidden": {"records": 512},
                }
            }
        ),
        encoding="utf-8",
    )
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
    checkpoint_receipt = tmp_path / "checkpoint.receipt.json"
    checkpoint_receipt.write_bytes(
        canonical_json_bytes(
            orbax_leaf_receipt(checkpoint, leaf, expected_step=0, role="base-maxtext")
        )
    )

    document, sources = stage.build_full_training_input_manifest(
        run_id="bookforge-jax-train-20260902",
        config=config,
        dataset_manifest=dataset_manifest,
        prepared_train=prepared,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_receipt=checkpoint_receipt,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
    )

    assert sources["dataset/train.jsonl"] == train.resolve()
    assert sources["dataset/development.jsonl"] == development.resolve()
    assert not any("hidden" in path for path in sources)
    assert {row["path"] for row in document["files"]} == set(sources)


def test_v2_full_training_manifest_binds_preparation_evidence(tmp_path: Path) -> None:
    stage = _load("bookforge_jax_v2_stage_inputs", JAX_ROOT / "stage_inputs.py")
    prompt_sha = "a" * 64
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "experiment_id": "bookforge-gemma4-e2b-lora-r16-v2",
                "production_contract": {
                    "prompt_contract_sha256": prompt_sha,
                    "input_budget_tokens": 512,
                    "completion_budget_tokens": 64,
                },
                "training": {
                    "preparation_policy": "balanced-counterfactual-pairs-v1",
                    "max_target_length": 576,
                },
            }
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    prepared = tmp_path / "prepared.jsonl"
    train.write_text('{"split":"train"}\n', encoding="utf-8")
    development.write_text('{"split":"development"}\n', encoding="utf-8")
    prepared.write_text('{"messages":[]}\n', encoding="utf-8")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "path": train.name,
                        "sha256": stage.sha256_file(train),
                    },
                    "development": {
                        "path": development.name,
                        "sha256": stage.sha256_file(development),
                    },
                    "hidden": {"path": None, "public": False},
                }
            }
        ),
        encoding="utf-8",
    )
    preparation = tmp_path / "preparation.manifest.json"
    preparation.write_text(
        json.dumps(
            {
                "schema_version": "bookforge-jax-training-preparation-v2",
                "policy": "balanced-counterfactual-pairs-v1",
                "source_train_sha256": stage.sha256_file(train),
                "prepared_sha256": stage.sha256_file(prepared),
                "prompt_contract_sha256": prompt_sha,
                "prepared_records": 1,
                "assistant_turns_per_record": 1,
                "pair_adjacency_preserved": True,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    tokenizer_manifest = tmp_path / "tokenizer.manifest.json"
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    validation = tmp_path / "prepared-validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": "bookforge-jax-prepared-validation-v1",
                "status": "passed",
                "config_sha256": stage.sha256_file(config),
                "prepared_sha256": stage.sha256_file(prepared),
                "preparation_manifest_sha256": stage.sha256_file(preparation),
                "tokenizer_manifest_sha256": stage.sha256_file(tokenizer_manifest),
                "prompt_contract_sha256": prompt_sha,
                "records": 1,
                "assistant_turns_per_record": 1,
                "input_budget_tokens": 512,
                "completion_budget_tokens": 64,
                "max_target_length": 576,
                "maximum_prompt_tokens": 20,
                "maximum_completion_tokens": 10,
                "maximum_total_tokens": 30,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    leaf = checkpoint / "0/items"
    leaf.mkdir(parents=True)
    (leaf / "weights").write_bytes(b"base")
    from training.jax_fidelity.integrity import artifact_manifest, canonical_json_bytes
    from training.jax_fidelity.orbax_receipt import orbax_leaf_receipt

    checkpoint_manifest = tmp_path / "checkpoint.manifest.json"
    checkpoint_manifest.write_bytes(canonical_json_bytes(artifact_manifest(leaf)))
    checkpoint_receipt = tmp_path / "checkpoint.receipt.json"
    checkpoint_receipt.write_bytes(
        canonical_json_bytes(
            orbax_leaf_receipt(checkpoint, leaf, expected_step=0, role="base-maxtext")
        )
    )

    document, sources = stage.build_full_training_input_manifest(
        run_id="bookforge-jax-v2-train-20260902",
        config=config,
        dataset_manifest=dataset_manifest,
        prepared_train=prepared,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_receipt=checkpoint_receipt,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        preparation_manifest=preparation,
        prepared_validation=validation,
    )

    assert set(sources) >= {
        "prepared/train.jsonl",
        "prepared/preparation.manifest.json",
        "prepared/prepared-validation.json",
    }
    assert document["prepared_training"] == {
        "policy": "balanced-counterfactual-pairs-v1",
        "records": 1,
        "prepared_sha256": stage.sha256_file(prepared),
        "preparation_manifest_sha256": stage.sha256_file(preparation),
        "prepared_validation_sha256": stage.sha256_file(validation),
        "prompt_contract_sha256": prompt_sha,
        "source_train_sha256": stage.sha256_file(train),
        "tokenizer_manifest_sha256": stage.sha256_file(tokenizer_manifest),
    }
    stage.verify_full_training_sources(document, sources)
    document["prepared_training"] = dict(document["prepared_training"])
    document["prepared_training"]["records"] = 2
    with pytest.raises(ValueError, match="prepared-training binding changed"):
        stage.verify_full_training_sources(document, sources)

    with pytest.raises(ValueError, match="both preparation evidence"):
        stage.build_full_training_input_manifest(
            run_id="bookforge-jax-v2-missing-20260902",
            config=config,
            dataset_manifest=dataset_manifest,
            prepared_train=prepared,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_receipt=checkpoint_receipt,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
        )


@pytest.mark.parametrize(
    "prefix",
    (
        "inputs/bookforge-jax-train-20260902",
        "staging/inputs/bookforge-jax-train-20260902",
    ),
)
def test_full_training_staging_accepts_bucket_root_or_nested_input_prefix(
    prefix: str,
) -> None:
    stage = _load("bookforge_jax_stage_input_prefix", JAX_ROOT / "stage_inputs.py")

    assert stage._is_run_input_prefix(prefix, "bookforge-jax-train-20260902")
    assert not stage._is_run_input_prefix(
        f"{prefix}-other", "bookforge-jax-train-20260902"
    )


def test_vertex_base_orbax_gate_rejects_manifest_tampering(tmp_path: Path) -> None:
    worker = _load("bookforge_vertex_orbax_gate", JAX_ROOT / "vertex_entrypoint.py")
    from training.jax_fidelity.integrity import artifact_manifest, canonical_json_bytes
    from training.jax_fidelity.orbax_receipt import orbax_leaf_receipt

    checkpoint = tmp_path / "checkpoint"
    leaf = checkpoint / "run/checkpoints/0/items"
    leaf.mkdir(parents=True)
    (leaf / "weights").write_bytes(b"base")
    manifest = artifact_manifest(leaf)
    receipt = orbax_leaf_receipt(
        checkpoint,
        leaf,
        expected_step=0,
        role="base-maxtext",
    )
    manifest_path = tmp_path / "checkpoint.manifest.json"
    receipt_path = tmp_path / "checkpoint.receipt.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    input_manifest = {
        "base_orbax": {
            "role": "base-maxtext",
            "expected_step": 0,
            "relative_path": receipt["relative_path"],
            "receipt_sha256": worker._sha256(receipt_path),
            "manifest_sha256": worker._sha256(manifest_path),
            "content_sha256": manifest["content_sha256"],
        }
    }
    selected = worker.verify_base_orbax(
        tmp_path,
        input_manifest=input_manifest,
        expected_manifest_sha256=worker._sha256(manifest_path),
        expected_receipt_sha256=worker._sha256(receipt_path),
    )
    assert selected == leaf.resolve()
    (leaf / "weights").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        worker.verify_base_orbax(
            tmp_path,
            input_manifest=input_manifest,
            expected_manifest_sha256=worker._sha256(manifest_path),
            expected_receipt_sha256=worker._sha256(receipt_path),
        )


def test_modal_staging_uploads_manifest_last_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    stage = _load("bookforge_modal_stage_inputs", JAX_ROOT / "stage_modal_inputs.py")
    stage_inputs = _load(
        "bookforge_modal_stage_inputs_core", JAX_ROOT / "stage_inputs.py"
    )
    run_id = "bookforge-jax-smoke-20260901"
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
    manifest, sources = stage_inputs.build_input_manifest(
        run_id=run_id,
        config=config,
        dataset_manifest=dataset,
        prepared_train=prepared,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_receipt=checkpoint_receipt,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
    )
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
        sources=sources,
        runner=runner,
    )

    puts = [command for command in commands if command[2] == "put"]
    assert puts[-1][-1] == f"/{run_id}/inputs.manifest.json"
    assert all("--force" not in command for command in puts)
    assert receipt["manifest_uploaded_last"] is True

    tampered = dict(manifest)
    tampered["base_orbax"] = {**manifest["base_orbax"], "role": "unverified"}
    with pytest.raises(ValueError, match="base Orbax binding changed"):
        stage.stage_inputs(
            run_id=run_id,
            manifest=tampered,
            sources=sources,
            runner=runner,
        )


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
    assert "secretmanager" not in role
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


def test_image_context_contains_only_planned_checksum_bound_files(tmp_path: Path) -> None:
    module = _load("bookforge_jax_image_context", JAX_ROOT / "build_image_plan.py")
    source = tmp_path / "source"
    source.mkdir()
    (source / "tracked.txt").write_text("tracked\n")
    (source / "untracked.txt").write_text("must not ship\n")
    tracked = source / "tracked.txt"
    plan = {
        "source_files": [
            {
                "path": "tracked.txt",
                "bytes": tracked.stat().st_size,
                "sha256": hashlib.sha256(tracked.read_bytes()).hexdigest(),
            }
        ]
    }

    destination = tmp_path / "context"
    module.materialize_context(source, plan, destination)

    assert (destination / "tracked.txt").read_text() == "tracked\n"
    assert not (destination / "untracked.txt").exists()
    tracked.write_text("drift\n")
    with pytest.raises(ValueError, match="changed after planning"):
        module.materialize_context(source, plan, tmp_path / "drifted")


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


def _private_bucket(*, lifecycle: bool) -> dict[str, object]:
    document: dict[str, object] = {
        "public_access_prevention": "enforced",
        "uniform_bucket_level_access": True,
    }
    if lifecycle:
        document["lifecycle_config"] = {"rule": [{"condition": {"age": 7}}]}
    return document


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


def test_read_only_cloud_preflight_requires_every_admission_fact() -> None:
    sys.path.insert(0, str(JAX_ROOT))
    module = _load("bookforge_jax_cloud_preflight", JAX_ROOT / "cloud_preflight.py")
    run_id = "bookforge-jax-smoke-20260901"
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
        credits_attestation=credits,
        spec_sha256="a" * 64,
        input_bindings_sha256="b" * 64,
    )

    assert report["ready"] is True
    assert report["run_id_absence_basis"] == "vertex-list"
    snapshots["billing"] = {"billingEnabled": False}
    rejected = module.evaluate(
        snapshots,
        run_id=run_id,
        credits_attestation=credits,
        spec_sha256="a" * 64,
        input_bindings_sha256="b" * 64,
    )
    assert rejected["ready"] is False
    assert "billing_enabled" in rejected["failed_checks"]


def test_billing_disabled_fallback_requires_recent_exact_create_audit_absence() -> None:
    sys.path.insert(0, str(JAX_ROOT))
    module = _load(
        "bookforge_jax_billing_disabled_preflight", JAX_ROOT / "cloud_preflight.py"
    )
    date = datetime.now(UTC).strftime("%Y%m%d")
    run_id = f"bookforge-jax-train-{date}-a1b2c3d4"
    snapshots = {
        "configuration": {
            "core": {"account": "operator@example.com", "project": module.PROJECT_ID}
        },
        "billing": {"billingEnabled": False, "billingAccountName": ""},
        "services": [],
        "jobs": {"collection_error": {"returncode": 1}},
        "job_audit_log": [],
        "scratch_bucket": {"collection_error": {"returncode": 1}},
        "release_bucket": {"collection_error": {"returncode": 1}},
        "quota": {"collection_error": {"returncode": 1}},
    }

    report = module.evaluate(
        snapshots,
        run_id=run_id,
        credits_attestation={},
        spec_sha256="a" * 64,
        input_bindings_sha256="b" * 64,
    )

    assert report["ready"] is False
    assert report["checks"]["run_id_absent"] is True
    assert report["run_id_absence_basis"] == (
        "billing-disabled-plus-empty-create-audit-log"
    )
    assert report["run_id_absence_evidence"] == {
        "audit_filter": module._create_job_audit_filter(run_id),
        "freshness": "400d",
        "maximum_run_id_age_days": 7,
    }

    snapshots["job_audit_log"] = [{"insertId": "existing-create"}]
    present = module.evaluate(
        snapshots,
        run_id=run_id,
        credits_attestation={},
        spec_sha256="a" * 64,
        input_bindings_sha256="b" * 64,
    )
    assert present["checks"]["run_id_absent"] is False
    assert present["run_id_absence_basis"] is None


def test_failed_read_only_admission_creates_modal_fallback_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    planner = _load("job_plan", JAX_ROOT / "job_plan.py")
    submitter = _load("bookforge_jax_rejected_submitter", JAX_ROOT / "submit_vertex_job.py")
    plan = planner.build_plan(_inputs(planner))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    evidence = {
        "schema_version": "1.0",
        "mode": "read-only-preflight",
        "project": planner.PROJECT_ID,
        "region": planner.REGION,
        "run_id": plan["run_id"],
        "spec_sha256": plan["spec_sha256"],
        "input_bindings_sha256": plan["input_bindings_sha256"],
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": {"billing_enabled": False, "run_id_absent": True},
        "ready": False,
        "failed_checks": ["billing_enabled"],
        "run_id_absence_basis": "vertex-list",
        "remote_mutation": False,
    }
    evidence_path = tmp_path / "admission.json"
    evidence_path.write_text(json.dumps(evidence))
    state = tmp_path / "state"
    monkeypatch.setattr(submitter, "_preflight", lambda _plan, _admission: None)

    with pytest.raises(submitter.PreflightRejected, match="billing_enabled"):
        submitter.submit(plan_path, state, evidence_path)

    rejection = json.loads(
        (state / f"{plan['run_id']}.preflight-rejection.json").read_text()
    )
    assert rejection["status"] == "rejected-pre-billable"
    assert rejection["fallback_allowed"] is True
    assert rejection["submission_intent_created"] is False
    assert rejection["custom_job_created"] is False
    assert rejection["job_absence_verified"] is True
    assert rejection["run_id_absence_basis"] == "vertex-list"


def test_stale_or_job_present_admission_never_authorizes_fallback(tmp_path: Path) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    planner = _load("job_plan", JAX_ROOT / "job_plan.py")
    submitter = _load("bookforge_jax_ineligible_fallback", JAX_ROOT / "submit_vertex_job.py")
    plan = planner.build_plan(_inputs(planner))
    evidence = {
        "schema_version": "1.0",
        "mode": "read-only-preflight",
        "project": planner.PROJECT_ID,
        "region": planner.REGION,
        "run_id": plan["run_id"],
        "spec_sha256": plan["spec_sha256"],
        "input_bindings_sha256": plan["input_bindings_sha256"],
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": {"run_id_absent": False},
        "ready": False,
        "failed_checks": ["run_id_absent"],
        "run_id_absence_basis": None,
        "remote_mutation": False,
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(evidence))

    with pytest.raises(submitter.PreflightRejected) as job_present:
        submitter._validated_admission_evidence(path, plan)
    assert job_present.value.fallback_allowed is False
    assert job_present.value.job_absence_verified is False

    evidence["checks"] = {"run_id_absent": True}
    evidence["ready"] = True
    evidence["failed_checks"] = []
    evidence["checked_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(evidence))
    with pytest.raises(submitter.PreflightRejected) as stale:
        submitter._validated_admission_evidence(path, plan)
    assert stale.value.fallback_allowed is False
    assert stale.value.job_absence_verified is False

    malformed = {"ready": False, "failed_checks": "billing_enabled"}
    with pytest.raises(submitter.PreflightRejected) as malformed_rejection:
        submitter._require_ready_admission(malformed)
    assert malformed_rejection.value.fallback_allowed is False
    assert malformed_rejection.value.job_absence_verified is False
    assert malformed_rejection.value.run_id_absence_basis is None


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


def test_gcs_release_fetch_rejects_prefix_change_during_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("bookforge_jax_gcs_fetch_race", JAX_ROOT / "fetch_gcs_release.py")
    run_id = "bookforge-jax-smoke-20260901"
    source = tmp_path / "remote-race"
    source.mkdir()
    files, completion_sha = _portable_release(source, run_id)
    prefix = f"releases/{run_id}/"

    class Blob:
        def __init__(self, relative: str, generation: int = 1) -> None:
            self.name = prefix + relative
            self.generation = generation

        def reload(self) -> None:
            return None

        def download_to_filename(self, filename: str, *, if_generation_match: int) -> None:
            assert if_generation_match == self.generation
            shutil.copy2(source / self.name.removeprefix(prefix), filename)

    initial = [Blob(name) for name in files]

    class Bucket:
        iam_configuration = types.SimpleNamespace(
            public_access_prevention="enforced",
            uniform_bucket_level_access_enabled=True,
        )

        def reload(self) -> None:
            return None

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            self.calls = 0

        def bucket(self, _name: str) -> Bucket:
            return Bucket()

        def list_blobs(self, _bucket: Bucket, *, prefix: str):
            self.calls += 1
            assert prefix == f"releases/{run_id}/"
            return initial if self.calls == 1 else [*initial, Blob("late-object.bin")]

    cloud = types.ModuleType("google.cloud")
    cloud.storage = types.SimpleNamespace(Client=Client)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)

    destination = tmp_path / "raced-fetch"
    with pytest.raises(RuntimeError, match="changed during"):
        module.fetch_from_gcs(
            bucket_name="bookforge-jax-release",
            run_id=run_id,
            expected_completion_sha256=completion_sha,
            destination=destination,
        )
    assert not destination.exists()


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


def test_vertex_reconciliation_is_bound_to_current_intent_bytes(tmp_path: Path) -> None:
    sys.path.insert(0, str(JAX_ROOT))
    submitter = _load("bookforge_jax_submitter_intent", JAX_ROOT / "submit_vertex_job.py")
    run_id = "bookforge-jax-smoke-20260902"
    intent = tmp_path / f"{run_id}.submission-intent.json"
    intent.write_text('{"run_id":"bookforge-jax-smoke-20260902"}\n')
    reconciliation = tmp_path / f"{run_id}.billing-reconciliation.json"
    reconciliation.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "producer": "bookforge-gcp-jax-reconciler",
                "status": "reconciled",
                "run_id": run_id,
                "retry_allowed": False,
                "job_evidence_sha256": "a" * 64,
                "billing_evidence_sha256": "b" * 64,
                "submission_intent_sha256": "c" * 64,
            }
        )
    )

    with pytest.raises(RuntimeError, match="stale Vertex reconciliation"):
        submitter._assert_no_unreconciled_paid_attempt(tmp_path)


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
