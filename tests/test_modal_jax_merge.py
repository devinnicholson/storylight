# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infra/gcp/jax"))

from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import (
    artifact_manifest,
    canonical_json_bytes,
    sha256_file,
)
from training.jax_fidelity.manifests import stable_run_id
from training.jax_fidelity.orbax_receipt import terminal_checkpoint_step


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


merge = _load("bookforge_modal_jax_merge", ROOT / "deploy/modal_jax_merge.py")
fetcher = _load("bookforge_modal_jax_merge_fetch", ROOT / "scripts/fetch_modal_jax_merge.py")
CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document))


def _request(**updates: object) -> dict[str, object]:
    identifiers = {
        "merge_run_id": "bookforge-full-merge-20260901",
        "roundtrip_run_id": "bookforge-roundtrip-smoke-20260901",
        "training_release_run_id": "bookforge-full-training-20260901",
        "training_run_id": "lora-train-aaaaaaaaaaaaaaaaaaaa",
    }
    bindings = {name: f"{index:x}" * 64 for index, name in enumerate(merge._BINDING_NAMES, 1)}
    request: dict[str, object] = {**identifiers, **bindings}
    request["approval_token"] = merge._approval_token(**identifiers, bindings=bindings)
    request.update(updates)
    return request


def _training_release(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = load_config(CONFIG)
    dataset_sha = "d" * 64
    run_id = stable_run_id(
        stage="lora-train",
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset_sha,
    )
    release_id = "bookforge-full-training-20260901"
    root = tmp_path / release_id
    terminal_step = terminal_checkpoint_step(config.training["steps"])
    leaf = root / f"adapter/run/checkpoints/{terminal_step}/items"
    leaf.mkdir(parents=True)
    (leaf / "checkpoint").write_bytes(b"full adapter")
    adapter_manifest = artifact_manifest(root / "adapter")
    _write(root / "adapter.manifest.json", adapter_manifest)
    base_binding = {"content_sha256": "b" * 64, "files": 2, "bytes": 19}
    inputs = {
        "base_checkpoint": base_binding,
        "prepared_train": {"sha256": "e" * 64, "bytes": 123},
        "tokenizer_checkpoint": {"content_sha256": "f" * 64, "files": 3, "bytes": 42},
    }
    run = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage": "lora-train",
        "status": "planned",
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "metadata": {"inputs": inputs, "smoke": False},
    }
    _write(root / "training/run.json", run)
    completion = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "succeeded",
        "run_manifest_sha256": sha256_file(root / "training/run.json"),
        "source_training_completion_sha256": "9" * 64,
        "artifacts": adapter_manifest["files"],
        "evidence": {"inputs": inputs, "runtime_lock": {"path": "runtime.lock.json"}},
    }
    _write(root / "training/completion.json", completion)
    (root / "runtime.lock.json").write_text("{}\n", encoding="utf-8")
    package_manifest = artifact_manifest(root)
    _write(root / "package.manifest.json", package_manifest)
    _write(root / "provider/attempt.json", {"status": "admitted"})
    _write(root / "provider/gpu-preflight.json", {"devices": 2})
    portable = {
        "adapter_manifest_sha256": sha256_file(root / "adapter.manifest.json"),
        "package_manifest_sha256": sha256_file(root / "package.manifest.json"),
        "training_run_sha256": sha256_file(root / "training/run.json"),
        "training_completion_sha256": sha256_file(root / "training/completion.json"),
    }
    rows = artifact_manifest(root)["files"]
    outer = {
        "schema_version": "1.0",
        "run_id": release_id,
        "training_run_id": run_id,
        "status": "succeeded",
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "source_training_completion_sha256": "9" * 64,
        "portable_package": portable,
        "files": rows,
    }
    _write(root / "completion.json", outer)
    values: dict[str, object] = {
        "release_run_id": release_id,
        "release_completion_sha256": sha256_file(root / "completion.json"),
        "adapter_manifest_sha256": portable["adapter_manifest_sha256"],
        "training_run_id": run_id,
        "training_run_sha256": portable["training_run_sha256"],
        "training_completion_sha256": portable["training_completion_sha256"],
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "base_binding": base_binding,
    }
    return root, values


def test_merge_request_requires_every_hash_and_exact_approval() -> None:
    validated = merge._validate_request(_request())
    assert validated[0] == "bookforge-full-merge-20260901"
    assert set(validated[-1]) == set(merge._BINDING_NAMES)
    with pytest.raises(ValueError, match="approval"):
        merge._validate_request(_request(approval_token="approve"))
    with pytest.raises(ValueError, match="SHA-256"):
        merge._validate_request(_request(training_run_sha256="latest"))


def test_merge_billing_parser_accepts_current_and_legacy_fields() -> None:
    assert merge._parse_modal_billing_total(
        '[{"cost": "1.25"}, {"Cost": 0.5}]'
    ) == pytest.approx(1.75)
    with pytest.raises(RuntimeError, match="no cost"):
        merge._parse_modal_billing_total('[{"description": "missing"}]')
    with pytest.raises(RuntimeError, match="conflicting"):
        merge._parse_modal_billing_total('[{"cost": 1, "Cost": 2}]')


def test_full_training_release_selects_exact_terminal_adapter_leaf(tmp_path: Path) -> None:
    root, arguments = _training_release(tmp_path)
    adapter, evidence = merge._verify_training_release(root, **arguments)
    terminal_step = terminal_checkpoint_step(load_config(CONFIG).training["steps"])
    assert (
        adapter / f"run/checkpoints/{terminal_step}/items/checkpoint"
    ).read_bytes() == b"full adapter"
    assert evidence["run"]["stage"] == "lora-train"

    arguments["base_binding"] = {"content_sha256": "0" * 64, "files": 2, "bytes": 19}
    with pytest.raises(ValueError, match="verified base"):
        merge._verify_training_release(root, **arguments)


def test_merge_worker_is_one_shot_l4_and_completion_is_last() -> None:
    source = (ROOT / "deploy/modal_jax_merge.py").read_text(encoding="utf-8")
    plan = json.loads(
        (ROOT / "experiments/jax-fidelity-lab/modal-merge-plan-2026-09.json").read_text()
    )
    assert plan["gpu"] == "L4"
    assert 'GPU = "L4"' in source
    assert 'plan.get("gpu") != GPU' in source
    assert 'roundtrip.get("backend") != "modal-l4x2"' in source
    assert plan["function_calls"] == 1
    assert plan["automatic_retries"] == 0
    assert plan["web_endpoint"] is False
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "@modal.web_endpoint" not in source
    assert source.count('"training.jax_fidelity.convert",') == 1
    assert source.index("subprocess.run(", source.index("def merge_finite")) < source.index(
        "build_merged_candidate_manifest(", source.index("def merge_finite")
    )
    assert source.index("files = _release_files(release)") < source.index(
        "_write_once(completion_path, payload)"
    )


def test_merge_fetch_completion_rejects_wrong_hash_and_nonprovisional_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "completion.json"
    document = {
        "schema_version": "1.0",
        "status": "succeeded",
        "backend": "modal-l4",
        "release_type": "provisional-merged-hf-development-candidate",
        "merge_run_id": "bookforge-full-merge-20260901",
        "development_evaluated": False,
        "release_authorized": False,
    }
    _write(path, document)
    fetched = fetcher._completion(
        path,
        merge_run_id="bookforge-full-merge-20260901",
        expected_sha256=sha256_file(path),
    )
    assert fetched["release_authorized"] is False
    document["backend"] = "modal-l40s"
    _write(path, document)
    with pytest.raises(ValueError, match="identity"):
        fetcher._completion(
            path,
            merge_run_id="bookforge-full-merge-20260901",
            expected_sha256=sha256_file(path),
        )
    with pytest.raises(ValueError, match="checksum"):
        fetcher._completion(
            path,
            merge_run_id="bookforge-full-merge-20260901",
            expected_sha256="0" * 64,
        )


def test_merge_release_file_verifier_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.manifest.json"
    artifact.write_bytes(b"candidate")
    rows = [
        {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}
    ]
    fetcher._verify_files(tmp_path, rows)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="failed verification"):
        fetcher._verify_files(tmp_path, rows)
