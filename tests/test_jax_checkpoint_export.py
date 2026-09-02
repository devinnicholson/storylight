# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The repository's pytest config adds only `src` to sys.path; this package is
# intentionally kept outside the production wheel under `training`.
sys.path.insert(0, str(ROOT))

import training.jax_fidelity.convert as convert_module
from scripts.validate_fidelity_release import (
    canonical_sha256,
    expected_candidate_id,
    validate_release,
)
from training.jax_fidelity.artifact_contract import create_artifact_contract
from training.jax_fidelity.commands import (
    build_hf_to_maxtext_command,
    build_logit_check_command,
    build_maxtext_to_hf_command,
)
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import (
    DatasetIntegrityError,
    artifact_manifest,
    canonical_json_bytes,
    sha256_file,
    verify_conversion_manifest,
)
from training.jax_fidelity.integrity import (
    canonical_sha256 as canonical_lineage_sha256,
)
from training.jax_fidelity.manifests import stable_run_id
from training.jax_fidelity.merged_candidate import build_merged_candidate_manifest
from training.jax_fidelity.orbax_receipt import orbax_leaf_receipt, terminal_checkpoint_step
from training.jax_fidelity.release import (
    ReleaseError,
    produce_release,
    verified_artifact_binding,
    verify_development_evaluation,
    verify_training_lineage,
)
from training.jax_fidelity.roundtrip_smoke import (
    RoundtripError,
    contract_document,
    validate_roundtrip_contract,
    validate_roundtrip_evidence,
)

CONFIG_PATH = ROOT / "experiments/jax-fidelity-lab/config.json"


def _evidence(config, checkpoint: Path) -> dict:
    return {
        "schema_version": "1.0",
        "contract": contract_document(config),
        "checks": {
            "tensor_names": True,
            "tensor_shapes": True,
            "tokenizer": True,
            "special_tokens": True,
            "ple_weights": True,
            "kv_sharing": True,
            "gemma4_metadata": True,
            "eos_token_ids": [1, 106, 50],
            "forward_kl_divergence": 0.012,
            "logit_comparison": "adapted-maxtext-vs-merged-hf",
        },
        "exported_checkpoint_manifest": artifact_manifest(checkpoint),
        "lineage": {
            "hf_to_maxtext_completion_sha256": "a" * 64,
            "smoke_completion_sha256": "b" * 64,
            "smoke_run_id": "lora-smoke-test",
            "maxtext_to_hf_completion_sha256": "c" * 64,
            "logit_completion_sha256": "d" * 64,
            "hf_to_maxtext_run_id": "hf-to-maxtext-test",
            "maxtext_to_hf_run_id": "maxtext-to-hf-test",
            "logit_run_id": "logit-check-test",
        },
    }


def _write_json(path: Path, document: dict) -> str:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _passing_development_evaluation(
    *, candidate_id: str, dataset_manifest_sha256: str, training_run_id: str
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "stage": "development",
        "config_sha256": sha256_file(CONFIG_PATH),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "eligibility_decision": {"passed": True, "hidden_evaluated": False},
        "checks": {"schema_valid": True, "development_improvement": True},
        "reasons": [],
        "baseline_summary_sha256": "a" * 64,
        "candidate_summary_sha256": "b" * 64,
        "development_records_sha256": "d" * 64,
        "predictions_sha256": "e" * 64,
        "prediction_completion_sha256": "f" * 64,
        "evaluation_completion_sha256": "1" * 64,
        "evaluation_input_sha256": canonical_lineage_sha256(
            {
                "development_records_sha256": "d" * 64,
                "predictions_sha256": "e" * 64,
                "prediction_completion_sha256": "f" * 64,
            }
        ),
        "candidate_manifest_sha256": "2" * 64,
        "checkpoint_manifest_sha256": "3" * 64,
        "checkpoint_content_sha256": "4" * 64,
        "summary": {
            "surface": "raw",
            "split": "development",
            "records": 512,
            "record_ids_sha256": "c" * 64,
            "category_record_counts": {"action_binding": 512},
            "schema_valid_rate": 1.0,
            "privacy_pass_rate": 1.0,
            "semantic_atom_recall": 0.99,
            "exact_example_pass_rate": 0.97,
            "category_pass_rates": {"action_binding": 0.99},
            "counterfactual_pairs": 0,
            "counterfactual_sensitivity": 1.0,
            "unsupported_concept_rate": 0.0,
            "pii_leaks": 0,
            "privacy_term_leaks": 0,
            "source_echoes": 0,
            "injection_leaks": 0,
            "forbidden_hits": 0,
        },
    }


def _checkpoint(path: Path, files: dict[str, bytes]) -> tuple[Path, str]:
    path.mkdir()
    for relative, content in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    manifest = path.with_suffix(".manifest.json")
    return manifest, _write_json(manifest, artifact_manifest(path))


def test_roundtrip_contract_accepts_only_complete_checksummed_evidence(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    checkpoint = tmp_path / "hf-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"gemma4_text"}\n')
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"safe-tensor-placeholder")
    evidence = _evidence(config, checkpoint)

    validate_roundtrip_evidence(config, evidence, exported_checkpoint=checkpoint)

    (checkpoint / "config.json").write_text('{"model_type":"changed"}\n')
    with pytest.raises(RoundtripError, match="manifest"):
        validate_roundtrip_evidence(config, evidence, exported_checkpoint=checkpoint)


def test_roundtrip_contract_can_authorize_a_distinct_trained_checkpoint(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    canary = tmp_path / "canary"
    canary.mkdir()
    (canary / "config.json").write_text("{}\n")
    evidence = _evidence(config, canary)

    validate_roundtrip_contract(config, evidence)

    trained = tmp_path / "trained"
    trained.mkdir()
    (trained / "config.json").write_text('{"trained":true}\n')
    with pytest.raises(RoundtripError, match="manifest"):
        validate_roundtrip_evidence(config, evidence, exported_checkpoint=trained)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("tensor_shapes", False, "did not pass"),
        ("eos_token_ids", [1, 50], "EOS"),
        ("forward_kl_divergence", 0.031, "exceeds"),
    ],
)
def test_roundtrip_gate_rejects_architecture_or_numerical_drift(
    tmp_path: Path, key: str, value, match: str
) -> None:
    config = load_config(CONFIG_PATH)
    checkpoint = tmp_path / "hf-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    evidence = _evidence(config, checkpoint)
    evidence["checks"][key] = value

    with pytest.raises(RoundtripError, match=match):
        validate_roundtrip_evidence(config, evidence, exported_checkpoint=checkpoint)


def test_conversion_commands_pin_text_only_unscanned_gemma4() -> None:
    config = load_config(CONFIG_PATH)
    to_maxtext = build_hf_to_maxtext_command(
        config,
        hf_checkpoint="/hf/base",
        output_directory="/orbax/base",
    )
    to_hf = build_maxtext_to_hf_command(
        config,
        base_checkpoint="/orbax/base/0/items",
        lora_checkpoint="/orbax/lora/5/items",
        hf_tokenizer_checkpoint="/hf/base",
        output_directory="/hf/merged",
    )
    logit = build_logit_check_command(
        config,
        maxtext_checkpoint="/orbax/base/0/items",
        adapter_checkpoint="/orbax/lora/5/items",
        hf_checkpoint="/hf/merged",
    )

    for command in (to_maxtext, to_hf, logit):
        assert "model_name=gemma4-e2b" in command
        assert "hardware=gpu" in command
        assert "skip_jax_distributed_system=true" in command
        assert "scan_layers=false" in command
        assert "use_multimodal=false" in command
    assert "lora.lora_restore_path=/orbax/lora/5/items" in to_hf
    assert "--max_kl_div=0.03" in logit
    assert "tokenizer_path=/hf/merged" in logit
    assert "lora.lora_restore_path=/orbax/lora/5/items" in logit


def test_roundtrip_contract_is_bound_to_config_revision() -> None:
    config = load_config(CONFIG_PATH)
    contract = contract_document(config)

    assert contract["base_model"]["revision"] == "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    assert contract["maxtext"]["revision"] == "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
    assert contract["maxtext"]["scan_layers"] is False
    assert contract["maxtext"]["use_multimodal"] is False
    assert contract["requirements"]["eos_token_ids"] == [1, 106, 50]


def test_conversion_manifest_binds_each_named_checkpoint_byte(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer"
    base.mkdir()
    tokenizer.mkdir()
    (base / "checkpoint").write_bytes(b"orbax")
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    manifest = tmp_path / "conversion.json"
    document = {
        "schema_version": "1.0",
        "artifacts": {
            "base_checkpoint": artifact_manifest(base),
            "hf_checkpoint": artifact_manifest(tokenizer),
        },
    }
    manifest_sha = _write_json(manifest, document)

    verify_conversion_manifest(
        manifest,
        expected_manifest_sha256=manifest_sha,
        artifact_roots={"base_checkpoint": base, "hf_checkpoint": tokenizer},
    )
    (base / "checkpoint").write_bytes(b"changed")
    with pytest.raises(DatasetIntegrityError, match="on-disk bytes"):
        verify_conversion_manifest(
            manifest,
            expected_manifest_sha256=manifest_sha,
            artifact_roots={"base_checkpoint": base, "hf_checkpoint": tokenizer},
        )


def test_release_producer_emits_exact_checksum_bound_consumer_schema(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_manifest = ROOT / config.dataset["manifest_path"]
    dataset_sha = sha256_file(dataset_manifest)
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text('{"messages":[]}\n')

    base_root = tmp_path / "base-root"
    base = base_root / "run/checkpoints/0/items"
    base.mkdir(parents=True)
    (base / "checkpoint").write_bytes(b"base-orbax")
    base_manifest = tmp_path / "base.manifest.json"
    base_manifest_sha = _write_json(base_manifest, artifact_manifest(base))
    base_receipt = tmp_path / "base.receipt.json"
    base_receipt_sha = _write_json(
        base_receipt,
        orbax_leaf_receipt(
            base_root,
            base,
            expected_step=0,
            role="base-maxtext",
        ),
    )
    tokenizer_manifest, tokenizer_manifest_sha = _checkpoint(
        tmp_path / "tokenizer", {"tokenizer.json": b"local-tokenizer"}
    )
    original_hf_manifest, original_hf_manifest_sha = _checkpoint(
        tmp_path / "original-hf", {"model.safetensors": b"original-hf"}
    )
    adapter = tmp_path / "adapter"
    adapter_step = terminal_checkpoint_step(config.training["steps"])
    adapter_leaf = adapter / f"run/checkpoints/{adapter_step}/items"
    adapter_leaf.mkdir(parents=True)
    (adapter_leaf / "checkpoint").write_bytes(b"trained-lora")
    adapter_manifest = tmp_path / "adapter.manifest.json"
    adapter_manifest_sha = _write_json(adapter_manifest, artifact_manifest(adapter))
    merge_root = tmp_path / "merge-release"
    merged = merge_root / "merged-hf"
    merged.mkdir(parents=True)
    for relative, content in {
        "config.json": b'{"model_type":"gemma4_text"}\n',
        "model.safetensors": b"merged-weights",
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"eos_token_id":[1,106,50]}\n',
    }.items():
        (merged / relative).write_bytes(content)

    training_run_id = "lora-train-fidelity-001"
    release_inputs = {
        "adapter_checkpoint": verified_artifact_binding(
            adapter, adapter_manifest, adapter_manifest_sha
        ),
        "base_checkpoint": verified_artifact_binding(
            base, base_manifest, base_manifest_sha
        ),
        "prepared_train": {
            "sha256": sha256_file(prepared),
            "bytes": prepared.stat().st_size,
        },
        "tokenizer_checkpoint": verified_artifact_binding(
            tmp_path / "tokenizer", tokenizer_manifest, tokenizer_manifest_sha
        ),
    }
    training_inputs = {
        "base_checkpoint": {
            key: release_inputs["base_checkpoint"][key]
            for key in ("content_sha256", "files", "bytes")
        },
        "prepared_train": release_inputs["prepared_train"],
        "tokenizer_checkpoint": {
            key: release_inputs["tokenizer_checkpoint"][key]
            for key in ("content_sha256", "files", "bytes")
        },
    }
    run = tmp_path / "run.json"
    run_sha = _write_json(
        run,
        {
            "schema_version": "1.0",
            "run_id": training_run_id,
            "stage": "lora-train",
            "status": "planned",
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": dataset_sha,
            "metadata": {"inputs": training_inputs, "smoke": False},
        },
    )
    adapter_file = adapter_leaf / "checkpoint"
    completion = tmp_path / "completion.json"
    completion_sha = _write_json(
        completion,
        {
            "schema_version": "1.0",
            "run_id": training_run_id,
            "status": "succeeded",
            "run_manifest_sha256": run_sha,
            "artifacts": [
                {
                    "path": f"run/checkpoints/{adapter_step}/items/checkpoint",
                    "sha256": sha256_file(adapter_file),
                    "bytes": adapter_file.stat().st_size,
                }
            ],
            "evidence": {"inputs": training_inputs},
        },
    )
    roundtrip = tmp_path / "roundtrip.json"
    roundtrip_sha = _write_json(roundtrip, _evidence(config, merged))
    evaluation = tmp_path / "evaluation.json"
    source_files = artifact_manifest(merged)["files"]
    candidate_document = {
        "base_model": {
            "id": config.production["model_id"],
            "revision": config.production["model_revision"],
        },
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "training_run_id": training_run_id,
        "files_content_sha256": canonical_sha256(source_files),
    }
    candidate_id = expected_candidate_id(candidate_document)
    evaluation_sha = _write_json(
        evaluation,
        _passing_development_evaluation(
            candidate_id=candidate_id,
            dataset_manifest_sha256=dataset_sha,
            training_run_id=training_run_id,
        ),
    )
    evidence_root = merge_root / "evidence"
    evidence_root.mkdir()
    adapter_receipt = evidence_root / "adapter-orbax.receipt.json"
    adapter_receipt_sha = _write_json(
        adapter_receipt,
        orbax_leaf_receipt(
            adapter,
            adapter_leaf,
            expected_step=adapter_step,
            role="full-lora",
        ),
    )
    conversion_input = evidence_root / "maxtext-to-hf.inputs.json"
    conversion_input_sha = _write_json(
        conversion_input,
        create_artifact_contract(
            [
                f"base_checkpoint={base}",
                f"adapter_checkpoint={adapter_leaf}",
                f"hf_checkpoint={tmp_path / 'original-hf'}",
            ]
        ),
    )
    conversion_id = stable_run_id(
        stage="maxtext-to-hf",
        config_sha256=config.sha256,
        dataset_manifest_sha256=conversion_input_sha,
    )
    conversion_run = evidence_root / "maxtext-to-hf.run.json"
    conversion_run_sha = _write_json(
        conversion_run,
        {
            "schema_version": "1.0",
            "run_id": conversion_id,
            "stage": "maxtext-to-hf",
            "status": "planned",
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": conversion_input_sha,
            "metadata": {
                "conversion_input_manifest_sha256": conversion_input_sha,
                "direction": "maxtext-to-hf",
            },
        },
    )
    conversion_completion = evidence_root / "maxtext-to-hf.completion.json"
    conversion_completion_sha = _write_json(
        conversion_completion,
        {
            "schema_version": "1.0",
            "run_id": conversion_id,
            "status": "succeeded",
            "run_manifest_sha256": conversion_run_sha,
            "artifacts": [{"path": "merged", "sha256": "a" * 64, "bytes": 1}],
            "evidence": {
                "direction": "maxtext-to-hf",
                "input_manifest_sha256": conversion_input_sha,
                "output_manifest": artifact_manifest(merged),
            },
        },
    )
    candidate_manifest = merge_root / "candidate.manifest.json"
    candidate_manifest.write_bytes(
        canonical_json_bytes(
            build_merged_candidate_manifest(
                config_path=CONFIG_PATH,
                dataset_manifest_sha256=dataset_sha,
                training_run_id=training_run_id,
                merged_hf_checkpoint=merged,
            )
        )
    )
    candidate_manifest_sha = sha256_file(candidate_manifest)
    merged_manifest = merge_root / "merged-hf.manifest.json"
    merged_manifest.write_bytes(canonical_json_bytes(artifact_manifest(merged)))
    source_bindings = evidence_root / "source-bindings.json"
    source_bindings_sha = _write_json(
        source_bindings,
        {
            "input_manifest_sha256": "0" * 64,
            "hf_snapshot_manifest_sha256": original_hf_manifest_sha,
            "base_orbax_receipt_sha256": base_receipt_sha,
            "base_orbax_manifest_sha256": base_manifest_sha,
            "adapter_manifest_sha256": adapter_manifest_sha,
            "adapter_orbax_receipt_sha256": adapter_receipt_sha,
            "training_run_sha256": run_sha,
            "training_completion_sha256": completion_sha,
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": dataset_sha,
            "training_run_id": training_run_id,
            "conversion_input_manifest_sha256": conversion_input_sha,
            "conversion_run_id": conversion_id,
            "conversion_run_sha256": conversion_run_sha,
            "conversion_completion_sha256": conversion_completion_sha,
            "roundtrip_completion_sha256": "1" * 64,
            "training_release_completion_sha256": "2" * 64,
        },
    )
    merge_files = [
        {
            "path": path.relative_to(merge_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in merge_root.rglob("*") if item.is_file())
    ]
    merge_completion = merge_root / "completion.json"
    merge_completion_sha = _write_json(
        merge_completion,
        {
            "schema_version": "1.0",
            "status": "succeeded",
            "backend": "modal-l40s",
            "release_type": "provisional-merged-hf-development-candidate",
            "merge_run_id": "bookforge-full-merge-20260901",
            "candidate_id": candidate_id,
            "training_run_id": training_run_id,
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": dataset_sha,
            "candidate_manifest_sha256": candidate_manifest_sha,
            "merged_hf_manifest_sha256": sha256_file(merged_manifest),
            "checkpoint_manifest_sha256": canonical_sha256(artifact_manifest(merged)),
            "checkpoint_content_sha256": artifact_manifest(merged)["content_sha256"],
            "input_manifest_sha256": "0" * 64,
            "roundtrip_completion_sha256": "1" * 64,
            "training_release_completion_sha256": "2" * 64,
            "conversion_input_manifest_sha256": conversion_input_sha,
            "conversion_run_id": conversion_id,
            "conversion_run_sha256": conversion_run_sha,
            "conversion_completion_sha256": conversion_completion_sha,
            "development_evaluated": False,
            "release_authorized": False,
            "files": merge_files,
        },
    )
    release_directory = tmp_path / "release"

    result = produce_release(
        config_path=CONFIG_PATH,
        dataset_manifest_path=dataset_manifest,
        dataset_manifest_sha256=dataset_sha,
        prepared_train_path=prepared,
        prepared_train_sha256=sha256_file(prepared),
        base_checkpoint=base,
        base_manifest_path=base_manifest,
        base_manifest_sha256=base_manifest_sha,
        tokenizer_checkpoint=tmp_path / "tokenizer",
        tokenizer_manifest_path=tokenizer_manifest,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
        adapter_checkpoint=adapter,
        adapter_manifest_path=adapter_manifest,
        adapter_manifest_sha256=adapter_manifest_sha,
        merged_hf_checkpoint=merged,
        original_hf_checkpoint=tmp_path / "original-hf",
        original_hf_manifest_path=original_hf_manifest,
        original_hf_manifest_sha256=original_hf_manifest_sha,
        base_checkpoint_root=base_root,
        base_receipt_path=base_receipt,
        base_receipt_sha256=base_receipt_sha,
        adapter_receipt_path=adapter_receipt,
        adapter_receipt_sha256=adapter_receipt_sha,
        merge_release_root=merge_root,
        merge_completion_path=merge_completion,
        merge_completion_sha256=merge_completion_sha,
        candidate_manifest_path=candidate_manifest,
        candidate_manifest_sha256=candidate_manifest_sha,
        source_bindings_path=source_bindings,
        source_bindings_sha256=source_bindings_sha,
        conversion_input_path=conversion_input,
        conversion_input_sha256=conversion_input_sha,
        conversion_run_path=conversion_run,
        conversion_run_sha256=conversion_run_sha,
        conversion_completion_path=conversion_completion,
        conversion_completion_sha256=conversion_completion_sha,
        training_run_id=training_run_id,
        training_run_path=run,
        training_run_sha256=run_sha,
        training_completion_path=completion,
        training_completion_sha256=completion_sha,
        roundtrip_evidence_path=roundtrip,
        roundtrip_evidence_sha256=roundtrip_sha,
        evaluation_evidence_path=evaluation,
        evaluation_evidence_sha256=evaluation_sha,
        release_directory=release_directory,
    )

    validated = validate_release(
        result["manifest_path"],
        result["release_root"],
        expected_manifest_sha256=result["manifest_sha256"],
        expected_config_sha256=config.sha256,
        expected_dataset_manifest_sha256=dataset_sha,
        expected_candidate=result["candidate_id"],
    )
    assert validated.document["inputs"]["base_checkpoint"]["content_sha256"]
    assert set(validated.document["terminal_evidence"]) == {
        "adapter_orbax_receipt",
        "base_orbax_manifest",
        "base_orbax_receipt",
        "candidate_manifest",
        "conversion_completion",
        "conversion_input",
        "conversion_run",
        "evaluation",
        "full_adapter_manifest",
        "merge_completion",
        "merge_source_bindings",
        "merged_hf_manifest",
        "original_hf_manifest",
        "roundtrip",
        "roundtrip_completion",
        "staged_input_manifest",
        "training_completion",
        "training_release_completion",
        "training_run",
    }


def test_conversion_execution_is_write_once_and_has_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    source = tmp_path / "hf"
    source.mkdir()
    (source / "model.bin").write_bytes(b"base")
    input_manifest = tmp_path / "conversion-input.json"
    input_manifest_sha = _write_json(
        input_manifest,
        {"schema_version": "1.0", "artifacts": {"hf_checkpoint": artifact_manifest(source)}},
    )
    output = tmp_path / "orbax"
    runs = tmp_path / "runs"
    run_id = stable_run_id(
        stage="hf-to-maxtext",
        config_sha256=config.sha256,
        dataset_manifest_sha256=input_manifest_sha,
    )
    monkeypatch.setenv(
        "BOOKFORGE_JAX_EXECUTION_APPROVAL",
        f"HF-TO-MAXTEXT:{run_id}:{config.sha256}:{input_manifest_sha}",
    )
    monkeypatch.setattr(convert_module, "validate_maxtext_checkout", lambda *_: tmp_path)

    def fake_run(_command, *, cwd):
        assert cwd == tmp_path
        output.mkdir()
        (output / "checkpoint").write_bytes(b"orbax")

    monkeypatch.setattr(convert_module, "run_checked", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "hf-to-maxtext",
            "--config",
            str(CONFIG_PATH),
            "--input-manifest",
            str(input_manifest),
            "--input-manifest-sha256",
            input_manifest_sha,
            "--hf-checkpoint",
            str(source),
            "--output-directory",
            str(output),
            "--run-directory",
            str(runs),
            "--maxtext-root",
            str(tmp_path),
            "--execute",
        ],
    )

    convert_module.main()

    completion = json.loads((runs / run_id / "completion.json").read_text())
    assert completion["status"] == "succeeded"
    assert completion["evidence"]["output_manifest"]["files"][0]["path"] == "checkpoint"
    with pytest.raises(SystemExit, match="write-once"):
        convert_module.main()


def test_maxtext_export_restores_generation_metadata_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    original = tmp_path / "hf"
    for directory in (base, adapter, original):
        directory.mkdir()
    (base / "checkpoint").write_bytes(b"base")
    (adapter / "checkpoint").write_bytes(b"adapter")
    (original / "tokenizer.json").write_bytes(b"tokenizer")
    _write_json(original / "generation_config.json", {"eos_token_id": [1, 106, 50]})
    input_manifest = tmp_path / "conversion-input.json"
    input_sha = _write_json(
        input_manifest,
        {
            "schema_version": "1.0",
            "artifacts": {
                "adapter_checkpoint": artifact_manifest(adapter),
                "base_checkpoint": artifact_manifest(base),
                "hf_checkpoint": artifact_manifest(original),
            },
        },
    )
    output = tmp_path / "merged"
    runs = tmp_path / "runs"
    run_id = stable_run_id(
        stage="maxtext-to-hf",
        config_sha256=config.sha256,
        dataset_manifest_sha256=input_sha,
    )
    monkeypatch.setenv(
        "BOOKFORGE_JAX_EXECUTION_APPROVAL",
        f"MAXTEXT-TO-HF:{run_id}:{config.sha256}:{input_sha}",
    )
    monkeypatch.setattr(convert_module, "validate_maxtext_checkout", lambda *_: tmp_path)

    def fake_run(_command, *, cwd):
        assert cwd == tmp_path
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"trained")

    monkeypatch.setattr(convert_module, "run_checked", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "maxtext-to-hf",
            "--config",
            str(CONFIG_PATH),
            "--input-manifest",
            str(input_manifest),
            "--input-manifest-sha256",
            input_sha,
            "--base-checkpoint",
            str(base),
            "--adapter-checkpoint",
            str(adapter),
            "--hf-checkpoint",
            str(original),
            "--output-directory",
            str(output),
            "--run-directory",
            str(runs),
            "--maxtext-root",
            str(tmp_path),
            "--execute",
        ],
    )

    convert_module.main()

    completion = json.loads((runs / run_id / "completion.json").read_text())
    receipt = runs / run_id / "generation-normalization.json"
    assert completion["status"] == "succeeded"
    assert completion["evidence"]["generation_normalization_receipt_sha256"] == sha256_file(
        receipt
    )
    assert (output / "generation_config.json").read_bytes() == (
        original / "generation_config.json"
    ).read_bytes()
    assert not output.with_name(f".{output.name}.{run_id}.raw").exists()


def test_release_producer_rejects_tampered_adapter_bytes(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    manifest, manifest_sha = _checkpoint(adapter, {"checkpoint": b"trained-lora"})
    (adapter / "checkpoint").write_bytes(b"tampered")

    with pytest.raises(ReleaseError, match="on-disk bytes"):
        verified_artifact_binding(adapter, manifest, manifest_sha)


def test_release_rejects_adapter_not_emitted_by_training_completion(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_file = adapter / "checkpoint"
    adapter_file.write_bytes(b"unrelated-successful-adapter")
    adapter_declaration = artifact_manifest(adapter)
    training_inputs = {
        "base_checkpoint": {"content_sha256": "a" * 64, "files": 1, "bytes": 10},
        "prepared_train": {"sha256": "b" * 64, "bytes": 10},
        "tokenizer_checkpoint": {"content_sha256": "c" * 64, "files": 1, "bytes": 10},
    }
    run = {
        "schema_version": "1.0",
        "stage": "lora-train",
        "status": "planned",
        "config_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "metadata": {"inputs": training_inputs},
    }
    completion = {
        "run_manifest_sha256": "f" * 64,
        "artifacts": [{"path": "/other/adapter", "sha256": sha256_file(adapter_file), "bytes": 28}],
        "evidence": {"inputs": training_inputs},
    }

    with pytest.raises(ReleaseError, match="not artifacts"):
        verify_training_lineage(
            run=run,
            run_sha256="f" * 64,
            completion=completion,
            config_sha256="d" * 64,
            dataset_manifest_sha256="e" * 64,
            release_inputs={"adapter_checkpoint": {}, **training_inputs},
            adapter_root=adapter,
            adapter_manifest=adapter_declaration,
        )


@pytest.mark.parametrize(
    "decision",
    [
        {"passed": False, "hidden_evaluated": False},
        {"passed": True, "hidden_evaluated": True},
    ],
)
def test_release_rejects_failed_or_premature_hidden_evaluation(decision: dict) -> None:
    with pytest.raises(ReleaseError, match="development-only"):
        verify_development_evaluation(
            {
                "schema_version": "1.0",
                "candidate_id": "fidelity-00000000000000000000",
                "stage": "development",
                "eligibility_decision": decision,
                "summary": {"records": 512},
            },
            candidate_id="fidelity-00000000000000000000",
            config_sha256=sha256_file(CONFIG_PATH),
            dataset_manifest_sha256="e" * 64,
            training_run_id="lora-train-test",
        )


def test_release_rejects_hand_authored_generic_development_success() -> None:
    with pytest.raises(ReleaseError, match="complete passing development-only"):
        verify_development_evaluation(
            {
                "schema_version": "1.0",
                "candidate_id": "fidelity-00000000000000000000",
                "stage": "development",
                "dataset_manifest_sha256": "e" * 64,
                "training_run_id": "lora-train-test",
                "eligibility_decision": {"passed": True, "hidden_evaluated": False},
                "summary": {"records": 512},
            },
            candidate_id="fidelity-00000000000000000000",
            config_sha256=sha256_file(CONFIG_PATH),
            dataset_manifest_sha256="e" * 64,
            training_run_id="lora-train-test",
        )
