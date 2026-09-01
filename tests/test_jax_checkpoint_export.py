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
from training.jax_fidelity.commands import (
    build_hf_to_maxtext_command,
    build_logit_check_command,
    build_maxtext_to_hf_command,
)
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import (
    DatasetIntegrityError,
    artifact_manifest,
    sha256_file,
    verify_conversion_manifest,
)
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
        },
        "exported_checkpoint_manifest": artifact_manifest(checkpoint),
    }


def _write_json(path: Path, document: dict) -> str:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


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
        hf_checkpoint="/hf/base",
    )

    for command in (to_maxtext, to_hf, logit):
        assert "model_name=gemma4-e2b" in command
        assert "scan_layers=false" in command
        assert "use_multimodal=false" in command
    assert "lora.lora_restore_path=/orbax/lora/5/items" in to_hf
    assert "--max_kl_div=0.03" in logit
    assert "tokenizer_path=/hf/base" in logit


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

    base_manifest, base_manifest_sha = _checkpoint(tmp_path / "base", {"checkpoint": b"base-orbax"})
    tokenizer_manifest, tokenizer_manifest_sha = _checkpoint(
        tmp_path / "tokenizer", {"tokenizer.json": b"local-tokenizer"}
    )
    adapter_manifest, adapter_manifest_sha = _checkpoint(
        tmp_path / "adapter", {"checkpoint": b"trained-lora"}
    )
    merged = tmp_path / "merged"
    merged.mkdir()
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
            tmp_path / "adapter", adapter_manifest, adapter_manifest_sha
        ),
        "base_checkpoint": verified_artifact_binding(
            tmp_path / "base", base_manifest, base_manifest_sha
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
    adapter_file = tmp_path / "adapter" / "checkpoint"
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
                    "path": "checkpoint",
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
        {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "stage": "development",
            "eligibility_decision": {"passed": True, "hidden_evaluated": False},
            "summary": {"records": 512, "exact_match": 0.99},
        },
    )
    release_directory = tmp_path / "release"

    result = produce_release(
        config_path=CONFIG_PATH,
        dataset_manifest_path=dataset_manifest,
        dataset_manifest_sha256=dataset_sha,
        prepared_train_path=prepared,
        prepared_train_sha256=sha256_file(prepared),
        base_checkpoint=tmp_path / "base",
        base_manifest_path=base_manifest,
        base_manifest_sha256=base_manifest_sha,
        tokenizer_checkpoint=tmp_path / "tokenizer",
        tokenizer_manifest_path=tokenizer_manifest,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
        adapter_checkpoint=tmp_path / "adapter",
        adapter_manifest_path=adapter_manifest,
        adapter_manifest_sha256=adapter_manifest_sha,
        merged_hf_checkpoint=merged,
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
        "evaluation",
        "roundtrip",
        "training_completion",
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
    run_id = f"hf-to-maxtext-{config.sha256[:12]}-{input_manifest_sha[:12]}"
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
        )
