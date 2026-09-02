from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/backup_fidelity_custody.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("backup_fidelity_custody", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backup = _load_helper()


def test_backup_contract_pins_the_exact_private_and_public_inputs() -> None:
    assert backup.PRIVATE_SHA256 == {
        "hidden.jsonl": "ef32a7ca378d8239352936b08665c3b45fe12dab47b8d5a1bf2356859f8db61c",
        "hidden.key": "0597ce504ae95b47d44e5454d7fce644bff92cc1976130222eb32bf6cd6cf4cd",
        "custody.json": "c0ecfb48ce0135b23a1eaf9a37ee6db57f4a1e854d394a46983a67bc4563a9f4",
    }
    assert (
        backup.DATASET_MANIFEST_SHA256
        == "e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de"
    )
    assert (
        backup.CONFIG_SHA256 == "a009feaaaef4907f1ed41e82d7c0c5ef906b00adc574cd0e00fca8a986097f9c"
    )


def test_private_source_validation_rejects_links_modes_and_changed_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hidden.key"
    source.write_bytes(b"private")
    source.chmod(0o600)
    digest = backup._sha256(source)
    artifact = backup.SourceArtifact("hidden.key", source, digest, True)
    backup._validate_artifact(artifact)

    source.chmod(0o644)
    with pytest.raises(backup.BackupError, match="mode 0600"):
        backup._validate_artifact(artifact)
    source.chmod(0o600)
    source.write_bytes(b"changed")
    with pytest.raises(backup.BackupError, match="changed"):
        backup._validate_artifact(artifact)

    link = tmp_path / "hidden-link.key"
    link.symlink_to(source)
    linked = backup.SourceArtifact("hidden.key", link, backup._sha256(source), True)
    with pytest.raises(backup.BackupError, match="regular file"):
        backup._validate_artifact(linked)


def test_receipt_is_exclusive_private_and_rejects_secret_fields(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    document = {
        "schema_version": backup.SCHEMA_VERSION,
        "encryption": {"cipher": "AES-256", "credential_store": "macOS Keychain"},
        "jetson_archive": {"ciphertext_only": True},
    }
    backup._write_receipt(receipt, document)

    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert json.loads(receipt.read_text()) == document
    with pytest.raises(FileExistsError):
        backup._write_receipt(receipt, document)
    with pytest.raises(backup.BackupError, match="secret material"):
        backup._write_receipt(
            tmp_path / "unsafe.json", {"schema_version": "v1", "passphrase": "no"}
        )


def test_helper_uses_aes_stdinpass_keychain_and_ciphertext_only_transfer() -> None:
    source = HELPER.read_text()
    assert '"AES-256"' in source
    assert '"-stdinpass"' in source
    assert '"find-generic-password"' in source
    assert "SecKeychainAddGenericPassword" in source
    assert '"add-generic-password"' not in source
    assert "_confirm_offline_recovery(passphrase)" in source
    assert 'open("/dev/tty"' in source
    assert "_copy_ciphertext_to_remote(config, archive, remote_partial)" in source
    assert "_copy_ciphertext_from_remote(config, remote_final, restored)" in source
    assert '"ciphertext_only": True' in source
    assert "print(passphrase" not in source
    assert "print(value" not in source
    assert "write_text(passphrase" not in source
    assert "write_bytes(passphrase" not in source
    assert "192.0.2.10" not in source
    assert 'DEFAULT_JETSON_USER = "operator"' not in source


def test_help_is_non_mutating_and_documents_check_only() -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--check-only" in result.stdout
    assert "--recovery-key-ceremony" in result.stdout
    assert "without Keychain, image, network, or writes" in " ".join(result.stdout.split())
