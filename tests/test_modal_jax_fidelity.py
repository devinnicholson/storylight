# ruff: noqa: E402
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import sha256_file
from training.jax_fidelity.manifests import complete_run, start_run
from training.jax_fidelity.remote_release import package_training_release

PLAN = ROOT / "experiments/jax-fidelity-lab/modal-plan-2026-09.json"
CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"
CONFIG_V2 = ROOT / "experiments/jax-fidelity-lab/config-v2.json"


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
    run_id = "bookforge-modal-smoke-20260901"
    config_sha = "a" * 64
    dataset_sha = "b" * 64
    rejection = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-submitter",
        "run_id": run_id,
        "spec_sha256": "9" * 64,
        "status": "rejected-pre-billable",
        "submission_intent_created": False,
        "custom_job_created": False,
        "job_absence_verified": True,
        "run_id_absence_basis": "vertex-list",
        "fallback_allowed": True,
        "reason": "quota unavailable",
    }
    rejection["input_bindings"] = {
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "prepared_train_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
        "base_checkpoint_manifest_sha256": "e" * 64,
        "base_checkpoint_receipt_sha256": "0" * 64,
        "tokenizer_manifest_sha256": "f" * 64,
    }
    rejection["input_bindings_sha256"] = hashlib.sha256(
        json.dumps(rejection["input_bindings"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
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
        "smoke": True,
        "gcp_rejection": rejection,
        "gcp_rejection_sha256": rejection_sha,
        "approval_token": (
            f"APPROVE_MODAL_JAX_RUN:{run_id}:{config_sha}:{dataset_sha}:{'c' * 64}:"
            f"{'d' * 64}:{'e' * 64}:{'0' * 64}:{'f' * 64}:{rejection_sha}:smoke"
        ),
    }
    request.update(updates)
    return request


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n")


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


def test_modal_fallback_is_finite_pinned_and_has_no_endpoint() -> None:
    plan = json.loads(PLAN.read_text())
    source = Path("deploy/modal_jax_fidelity.py").read_text()

    assert plan["gpu"] == "L4:2"
    assert plan["container_count"] == 1
    assert plan["function_calls"] == 1
    assert plan["timeout_seconds"] == 3600
    assert plan["automatic_retries"] == 0
    assert plan["minimum_containers"] == 0
    assert plan["web_endpoint"] is False
    assert plan["allowed_gcp_terminal_state"] == "rejected-pre-billable"
    assert plan["gross_ceiling_policy"] == "declared-estimate-not-provider-enforced"
    assert "@sha256:" in modal_jax_fidelity._pinned_image_uri()
    assert "gpu=GPU" in source
    assert 'BACKEND = "modal-l4x2"' in source
    assert "retries=0" in source
    assert "max_containers=MAX_CONTAINERS" in source
    assert "input_volume.reload()" in source
    assert "scratch_volume.commit()" in source
    assert source.index("scratch_volume.commit()") < source.index("subprocess.run(command")
    training_call = source.index("subprocess.run(command")
    durable_success = source.index("# This is the durability boundary", training_call)
    inline_finalize = source.index("return _finalize_completed_scratch(", durable_success)
    assert training_call < durable_success < inline_finalize
    assert "timeout=training_timeout" in source
    assert source.rindex("scratch_volume.commit()") < source.rindex(
        "release_commit=release_volume.commit"
    )
    assert "run_two_gpu_fsdp_preflight(" in source
    assert plan["publication_reserve_seconds"] == 600
    assert plan["finalize_recovery_timeout_seconds"] == 1800
    assert plan["finalize_recovery_gpu"] is None
    assert plan["finalize_recovery_function_calls_max"] == 1
    assert plan["finalize_recovery_automatic_retries"] == 0
    assert plan["finalize_recovery_web_endpoint"] is False
    assert "def finalize_finite(" in source
    assert "exact Modal JAX finalize-only approval token" in source
    finalize_definition = source.index("def finalize_finite(")
    finalize_decorator = source.rfind("@app.function(", 0, finalize_definition)
    assert "gpu=" not in source[finalize_decorator:finalize_definition]
    assert "provider/gpu-preflight.json" not in source
    assert "@modal.web_endpoint" not in source
    assert "smoke: bool = False" in source


def test_modal_training_timeout_preserves_publication_reserve() -> None:
    assert modal_jax_fidelity._bounded_training_timeout(0) == 3000
    assert modal_jax_fidelity._bounded_training_timeout(240.25) == 2759
    with pytest.raises(RuntimeError, match="no safe training window"):
        modal_jax_fidelity._bounded_training_timeout(3000)


def test_modal_v2_verifies_preparation_and_validation_before_gpu_training(
    tmp_path: Path,
) -> None:
    experiment, input_manifest, hashes = _v2_prepared_inputs(tmp_path)

    evidence = modal_jax_fidelity._verify_v2_prepared_evidence(
        tmp_path,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=hashes["config"],
        prepared_sha256=hashes["prepared"],
        tokenizer_manifest_sha256=hashes["tokenizer_manifest"],
    )

    assert evidence == input_manifest["prepared_training"]
    source = Path("deploy/modal_jax_fidelity.py").read_text()
    run = source.index("def run_finite(")
    verification = source.index("_verify_v2_prepared_evidence(", run)
    gpu_preflight = source.index("run_two_gpu_fsdp_preflight(", run)
    training = source.index("subprocess.run(command", run)
    assert verification < gpu_preflight < training


def test_modal_v1_remains_compatible_without_prepared_receipts(tmp_path: Path) -> None:
    experiment = load_config(CONFIG)

    assert (
        modal_jax_fidelity._verify_v2_prepared_evidence(
            tmp_path,
            experiment=experiment,
            input_manifest={},
            config_sha256=experiment.sha256,
            prepared_sha256="a" * 64,
            tokenizer_manifest_sha256="b" * 64,
        )
        is None
    )


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


def test_modal_v2_rejects_drifted_preparation_or_population_binding(
    tmp_path: Path,
) -> None:
    experiment, input_manifest, hashes = _v2_prepared_inputs(tmp_path)
    preparation_path = tmp_path / "prepared/preparation.manifest.json"
    preparation = json.loads(preparation_path.read_text())
    preparation["prompt_contract_sha256"] = "f" * 64
    _write_json(preparation_path, preparation)

    with pytest.raises(RuntimeError, match="preparation manifest"):
        modal_jax_fidelity._verify_v2_prepared_evidence(
            tmp_path,
            experiment=experiment,
            input_manifest=input_manifest,
            config_sha256=hashes["config"],
            prepared_sha256=hashes["prepared"],
            tokenizer_manifest_sha256=hashes["tokenizer_manifest"],
        )

    experiment, input_manifest, hashes = _v2_prepared_inputs(tmp_path / "binding")
    input_manifest["prepared_training"] = dict(input_manifest["prepared_training"])
    input_manifest["prepared_training"]["prepared_validation_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="preparation binding"):
        modal_jax_fidelity._verify_v2_prepared_evidence(
            tmp_path / "binding",
            experiment=experiment,
            input_manifest=input_manifest,
            config_sha256=hashes["config"],
            prepared_sha256=hashes["prepared"],
            tokenizer_manifest_sha256=hashes["tokenizer_manifest"],
        )


def test_modal_v2_rejects_changed_source_training_bytes(tmp_path: Path) -> None:
    experiment, input_manifest, hashes = _v2_prepared_inputs(tmp_path)
    (tmp_path / "dataset/train.jsonl").write_bytes(b"changed\n")

    with pytest.raises(RuntimeError, match="source training bytes"):
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


def test_release_publication_recovers_after_first_commit_interruption(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    artifact = staging / "adapter.bin"
    artifact.write_bytes(b"adapter")
    row = {
        "path": "adapter.bin",
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    (staging / "completion.json").write_text(
        json.dumps({"status": "succeeded", "files": [row]}, indent=2, sort_keys=True) + "\n"
    )
    destination = tmp_path / "release"

    def interrupted_commit() -> None:
        raise RuntimeError("simulated commit interruption")

    with pytest.raises(RuntimeError, match="simulated commit"):
        modal_jax_fidelity._publish_staged_release(
            staging, destination, commit=interrupted_commit
        )
    artifact_stat = (destination / "adapter.bin").stat()
    assert not (destination / "completion.json").exists()
    commits = 0

    def commit() -> None:
        nonlocal commits
        commits += 1

    modal_jax_fidelity._publish_staged_release(staging, destination, commit=commit)
    assert commits == 2
    assert (destination / "adapter.bin").stat().st_ino == artifact_stat.st_ino
    assert (destination / "completion.json").is_file()


def test_failed_release_copy_removes_uncommitted_partial_destination(tmp_path: Path) -> None:
    destination = tmp_path / "release/model.bin"
    with pytest.raises(FileNotFoundError):
        modal_jax_fidelity._copy_release_file_once(
            tmp_path / "missing.bin",
            destination,
            {"path": "model.bin", "bytes": 1, "sha256": "0" * 64},
            trusted_root=tmp_path / "release",
        )
    assert not destination.exists()


def test_release_copy_allows_managed_mount_symlink_but_rejects_internal_symlink(
    tmp_path: Path,
) -> None:
    mount_target = tmp_path / "volume"
    mount_target.mkdir()
    mount = tmp_path / "release-mount"
    mount.symlink_to(mount_target, target_is_directory=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    artifact = staging / "adapter.bin"
    artifact.write_bytes(b"adapter")
    row = {
        "path": "adapter.bin",
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    release = mount / "run"
    release.mkdir()

    modal_jax_fidelity._copy_release_file_once(
        artifact,
        release / "nested/adapter.bin",
        row,
        trusted_root=release,
    )
    assert (release / "nested/adapter.bin").read_bytes() == b"adapter"

    outside = tmp_path / "outside"
    outside.mkdir()
    (release / "unsafe").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic-link parent"):
        modal_jax_fidelity._copy_release_file_once(
            artifact,
            release / "unsafe/adapter.bin",
            row,
            trusted_root=release,
        )


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


def test_finalize_discards_only_completionless_derivative_staging(
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
    scratch.mkdir()
    (scratch / "attempt.json").write_text(
        json.dumps(
            modal_jax_fidelity._attempt_document(
                run_id=run_id,
                training_run_id=training_run_id,
                smoke=False,
                **values,
            )
        )
    )
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
    output = scratch / "output"
    output.mkdir()
    (output / "adapter.bin").write_bytes(b"adapter")
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
    incomplete = scratch / "finalized-release"
    incomplete.mkdir()
    (incomplete / "partial.bin").write_bytes(b"partial")
    package_calls = 0

    def fake_package(**kwargs):
        nonlocal package_calls
        package_calls += 1
        destination = Path(kwargs["destination"])
        assert not destination.exists()
        destination.mkdir()
        (destination / "adapter.bin").write_bytes(b"adapter")
        return {"status": "succeeded", "package_manifest_sha256": "4" * 64}

    monkeypatch.setattr(
        "training.jax_fidelity.remote_release.package_training_release", fake_package
    )
    scratch_commits = 0

    def scratch_commit() -> None:
        nonlocal scratch_commits
        scratch_commits += 1

    result = modal_jax_fidelity._finalize_completed_scratch(
        run_id=run_id,
        training_run_id=training_run_id,
        smoke=False,
        scratch_directory=scratch,
        release_directory=tmp_path / "release",
        scratch_commit=scratch_commit,
        release_commit=lambda: None,
        **values,
    )
    assert result["status"] == "succeeded"
    assert package_calls == 1
    assert scratch_commits == 2
    assert not (scratch / "finalized-release/partial.bin").exists()


def test_modal_request_requires_hashes_exact_approval_and_prebillable_rejection() -> None:
    validated = modal_jax_fidelity._validate_request(_request())
    assert validated[-1] is True

    with pytest.raises(ValueError, match="pre-billable"):
        modal_jax_fidelity._validate_request(_request(gcp_rejection={}))
    with pytest.raises(ValueError, match="approval"):
        modal_jax_fidelity._validate_request(_request(approval_token="approve"))
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


def test_modal_budget_gate_is_present_and_maxtext_checkout_is_exact() -> None:
    source = Path("deploy/modal_jax_fidelity.py").read_text()
    image_source = Path("deploy/modal_jax_image.py").read_text()
    assert 'BUDGET_MONTH = "2026-09"' in source
    assert "workspace_total + float(ceiling) > WORKSPACE_HARD_STOP_USD" in source
    assert "git clone https://github.com/AI-Hypercomputer/maxtext.git /opt/MaxText" in image_source
    assert 'apt_install("git", "ca-certificates", "build-essential")' in image_source
    assert '"nvidia-curand-cu12==10.3.10.19"' in image_source
    assert '"nvidia-nvtx-cu12==12.9.79"' in image_source
    assert (
        'NCCL_LIBRARY_DIR = "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib"'
        in image_source
    )
    assert '"LIBRARY_PATH": NCCL_LIBRARY_DIR' in image_source
    assert '"/usr/local/lib/python3.12/site-packages/nvidia/curand/lib"' in image_source
    assert '"LD_LIBRARY_PATH": CUDA_WHEEL_LIBRARY_PATH' in image_source
    assert '"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"' in image_source
    assert "ln -sfnT cuda_runtime" in image_source
    assert "cudart/lib/lib*.so.*[0-9]" in image_source
    assert 'extra_options="--no-build-isolation"' in image_source
    assert "maxtext_revision" in image_source
    assert "git -C /opt/MaxText rev-parse HEAD" in image_source
    assert 'REPOSITORY_ROOT / "deploy", "/opt/bookforge/deploy"' in image_source
    assert '"/opt/bookforge/experiments/jax-fidelity-lab/config.json"' in image_source
    assert "--write-lock /opt/bookforge/runtime.lock.json" in image_source
    assert "--lock /opt/bookforge/runtime.lock.json" in image_source
    assert '"HF_HUB_OFFLINE": "1"' in image_source
    assert '"HF_DATASETS_OFFLINE": "1"' in image_source
    assert '"TRANSFORMERS_OFFLINE": "1"' in image_source
    assert 'for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")' in image_source
    assert "verify_base_orbax(" in source
    assert "base_checkpoint_receipt_sha256" in source


def test_modal_billing_parser_accepts_current_and_legacy_fields() -> None:
    parse = modal_jax_fidelity._parse_modal_billing_total

    assert parse('[{"cost": "1.25"}, {"Cost": 0.5}]') == pytest.approx(1.75)
    with pytest.raises(RuntimeError, match="no cost"):
        parse('[{"date": "2026-09-02"}]')
    with pytest.raises(RuntimeError, match="conflicting"):
        parse('[{"cost": 1, "Cost": 2}]')


def test_modal_release_fetch_verifies_every_file_before_copy(monkeypatch, tmp_path: Path) -> None:
    run_id = "bookforge-modal-smoke-20260901"
    output = tmp_path / "output"
    output.mkdir()
    (output / "model.bin").write_bytes(b"checkpoint")
    runtime_lock = tmp_path / "runtime.lock.json"
    runtime_lock.write_text('{"schema_version":"1.0"}\n')
    runs = tmp_path / "runs"
    run_manifest = start_run(
        runs,
        run_id=run_id,
        stage="lora-smoke",
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        command=["maxtext"],
    )
    complete_run(
        runs,
        run_id=run_id,
        status="succeeded",
        artifacts=[output / "model.bin"],
        evidence={
            "inputs": {"prepared_train": {"sha256": "c" * 64}},
            "runtime_lock": {
                "path": str(runtime_lock),
                "sha256": sha256_file(runtime_lock),
            },
        },
    )
    assert run_manifest.is_file()
    package = tmp_path / "package"
    package_training_release(
        output_directory=output,
        run_directory=runs,
        training_run_id=run_id,
        runtime_lock=runtime_lock,
        destination=package,
    )
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
    destination = tmp_path / "release"
    completion_sha = hashlib.sha256(json.dumps(outer).encode()).hexdigest()
    manifest = fetch_modal_jax_release.fetch_release(run_id, completion_sha, destination)

    assert manifest["status"] == "succeeded"
    assert (destination / "adapter/model.bin").read_bytes() == b"checkpoint"
    assert (destination / "completion.json").is_file()
    assert (destination / "training/run.json").is_file()
    assert (destination / "training/completion.json").is_file()


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


def test_modal_export_fetch_builds_checksummed_jetson_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate_id = "fidelity-" + "a" * 20
    release_sha = "b" * 64
    files = {
        "llm/config.json": b"{}\n",
        "llm/model.onnx": b"onnx",
        "llm/rank0.safetensors": b"int4",
    }
    document = {
        "schema_version": "1.0",
        "status": "succeeded",
        "candidate_id": candidate_id,
        "base_model": {
            "id": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        },
        "source_release_manifest_sha256": release_sha,
        "source_files_content_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "training_run_id": "lora-train-fidelity-001",
        "private_output_prefix": (
            "modal-private://bookforge-tensorrt-edge-llm-fidelity/"
            f"{candidate_id}/{release_sha[:20]}"
        ),
        "quantization": "int4_awq",
        "calibration_dataset": "wikitext",
        "calibration_samples": 128,
        "components": ["thinker"],
        "skip_visual": True,
        "skip_audio": True,
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": "v0.10.0",
        "tensorrt_edge_llm_revision": "71dd1bae032e70771265917ec74d3ff4cad07a10",
        "engine_built_in_cloud": False,
        "files": [
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in sorted(files.items())
        ],
    }
    document["file_count"] = len(document["files"])
    document["total_bytes"] = sum(row["bytes"] for row in document["files"])
    manifest_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    def fake_get(remote_path: str, destination: Path) -> None:
        if remote_path.endswith("export.manifest.json"):
            destination.write_bytes(manifest_bytes)
            return
        marker = "/onnx/"
        destination.write_bytes(files[remote_path.split(marker, 1)[1]])

    monkeypatch.setattr(fetch_modal_jax_release, "_modal_export_get", fake_get)
    destination = tmp_path / "export"
    result = fetch_modal_jax_release.fetch_export(
        candidate_id=candidate_id,
        source_release_manifest_sha256=release_sha,
        expected_export_manifest_sha256=manifest_sha,
        destination=destination,
    )

    assert result["jetson_builder_compatible"] is True
    assert (destination / "export.manifest.json").read_bytes() == manifest_bytes
    assert (destination / "onnx/llm/model.onnx").read_bytes() == b"onnx"


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


def test_modal_export_bridge_invokes_builder_then_installer_verification(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    candidate = tmp_path / "candidate"
    commands: list[list[str]] = []

    def runner(command, **_kwargs):
        commands.append(command)
        if command[0].endswith("build-trained-planner-candidate.sh"):
            candidate.mkdir()
            (candidate / "candidate.manifest.json").write_text('{"result":"fixture"}\n')
        return None

    result = fetch_modal_jax_release.build_jetson_candidate(
        export_bundle=export,
        export_manifest_sha256="a" * 64,
        candidate_output=candidate,
        runner=runner,
    )

    assert commands[0][0].endswith("build-trained-planner-candidate.sh")
    assert commands[1][0].endswith("install-trained-planner-candidate.sh")
    assert "--verify-only" in commands[1]
    assert result["status"] == "built-and-installer-verified"
