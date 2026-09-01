from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import stat
from pathlib import Path

import pytest

from bookforge.fidelity_dataset import HIDDEN_KEY_PREFIX
from bookforge.fidelity_manifest import sha256_path

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts/build_fidelity_dataset.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_fidelity_dataset", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _key(label: str) -> str:
    return (
        HIDDEN_KEY_PREFIX
        + base64.urlsafe_b64encode(hashlib.sha384(label.encode()).digest()).decode()
    )


def _write_key(path: Path, label: str, *, legacy: bool = False) -> None:
    value = _key(label)
    if legacy:
        value = value.removeprefix(HIDDEN_KEY_PREFIX)
    path.write_text(value + "\n")
    path.chmod(0o600)


def _arguments(tmp_path: Path, **updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "output_dir": tmp_path / "public",
        "private_hidden_output": tmp_path / "private/hidden.jsonl",
        "hidden_key_file": tmp_path / "private/hidden.key",
        "custody_receipt": tmp_path / "private/custody.json",
        "initialize_hidden_key": False,
        "migrate_legacy_key": False,
        "rotation_token": None,
        "smoke_output": tmp_path / "smoke.jsonl",
        "verify_only": False,
        "public_only": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_build_creates_private_custody_and_public_identity_without_key_oracle(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, initialize_hidden_key=True)
    builder.build(arguments)

    for path in (
        arguments.hidden_key_file,
        arguments.private_hidden_output,
        arguments.custody_receipt,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.is_file() and not path.is_symlink()
    manifest = json.loads((arguments.output_dir / "manifest.json").read_text())
    receipt = json.loads(arguments.custody_receipt.read_text())
    assert "key_fingerprint_sha256" not in manifest
    assert len(receipt["key_fingerprint_sha256"]) == 64
    assert manifest["generator_source_sha256"] == receipt["generator_source_sha256"]
    assert manifest["generator_config_sha256"] == receipt["generator_config_sha256"]
    assert manifest["generator_runtime"] == receipt["generator_runtime"]


def test_wrong_key_cannot_replace_custody_without_exact_rotation_token(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.hidden_key_file.parent.mkdir(parents=True)
    _write_key(arguments.hidden_key_file, "first")
    builder.build(arguments)
    protected = {
        path: sha256_path(path)
        for path in (
            arguments.private_hidden_output,
            arguments.custody_receipt,
            arguments.output_dir / "manifest.json",
        )
    }
    _write_key(arguments.hidden_key_file, "second")

    with pytest.raises(ValueError, match="rotation-token") as failure:
        builder.build(arguments)
    assert all(sha256_path(path) == digest for path, digest in protected.items())
    token_match = re.search(r"--rotation-token (\S+)", str(failure.value))
    assert token_match is not None

    arguments.rotation_token = token_match.group(1)
    builder.build(arguments)
    assert (
        sha256_path(arguments.private_hidden_output) != protected[arguments.private_hidden_output]
    )


def test_lost_local_key_and_hidden_file_recover_from_custodied_key_copy(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.hidden_key_file.parent.mkdir(parents=True)
    _write_key(arguments.hidden_key_file, "recoverable")
    builder.build(arguments)
    hidden_sha256 = sha256_path(arguments.private_hidden_output)
    backup = tmp_path / "offline/hidden.key"
    backup.parent.mkdir()
    backup.write_bytes(arguments.hidden_key_file.read_bytes())
    backup.chmod(0o600)

    arguments.hidden_key_file.unlink()
    arguments.private_hidden_output.unlink()
    arguments.hidden_key_file.write_bytes(backup.read_bytes())
    arguments.hidden_key_file.chmod(0o600)
    builder.build(arguments)

    assert sha256_path(arguments.private_hidden_output) == hidden_sha256
    builder.verify(
        _arguments(
            tmp_path,
            verify_only=True,
            private_hidden_output=arguments.private_hidden_output,
            custody_receipt=arguments.custody_receipt,
        )
    )


def test_private_verification_never_silently_falls_back_and_rejects_bad_mode(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, initialize_hidden_key=True)
    builder.build(arguments)
    missing = tmp_path / "private/missing.jsonl"
    with pytest.raises(ValueError, match="missing or unsafe"):
        builder.verify(
            _arguments(
                tmp_path,
                verify_only=True,
                private_hidden_output=missing,
                custody_receipt=arguments.custody_receipt,
            )
        )

    arguments.private_hidden_output.chmod(0o644)
    with pytest.raises(ValueError, match="group or other"):
        builder.verify(
            _arguments(
                tmp_path,
                verify_only=True,
                private_hidden_output=arguments.private_hidden_output,
                custody_receipt=arguments.custody_receipt,
            )
        )
    arguments.private_hidden_output.chmod(0o600)
    builder.verify(
        _arguments(
            tmp_path,
            verify_only=True,
            public_only=True,
            private_hidden_output=None,
            custody_receipt=None,
            hidden_key_file=None,
        )
    )


def test_legacy_high_entropy_key_requires_explicit_safe_migration(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.hidden_key_file.parent.mkdir(parents=True)
    _write_key(arguments.hidden_key_file, "legacy", legacy=True)
    with pytest.raises(ValueError, match="migrate-legacy-key"):
        builder.build(arguments)

    arguments.migrate_legacy_key = True
    builder.build(arguments)
    assert arguments.hidden_key_file.read_text().startswith(HIDDEN_KEY_PREFIX)
    assert stat.S_IMODE(arguments.hidden_key_file.stat().st_mode) == 0o600
