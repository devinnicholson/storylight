from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.jax_fidelity.runtime import (
    ExecutionRefused,
    approved_maxtext_patch_sha256,
    validate_maxtext_checkout,
)


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def _checkout(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    checkout = tmp_path / "MaxText"
    source = checkout / "src/maxtext/utils/train_utils.py"
    train = checkout / "src/maxtext/trainers/pre_train/train.py"
    source.parent.mkdir(parents=True)
    train.parent.mkdir(parents=True)
    source.write_text("before\n", encoding="utf-8")
    train.write_text("before\n", encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "bookforge@example.invalid")
    _git(checkout, "config", "user.name", "Bookforge Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "base")
    revision = str(_git(checkout, "rev-parse", "HEAD")).strip()
    return checkout, SimpleNamespace(versions={"maxtext_revision": revision})


def test_exact_approved_maxtext_patch_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, config = _checkout(tmp_path)
    paths = (
        "src/maxtext/trainers/pre_train/train.py",
        "src/maxtext/utils/train_utils.py",
    )
    for relative in paths:
        (checkout / relative).write_text("after\n", encoding="utf-8")
    patch = tmp_path / "approved.patch"
    patch.write_bytes(
        bytes(
            _git(
                checkout,
                "diff",
                "--no-ext-diff",
                "--binary",
                "--abbrev=8",
                "--unified=1",
                "--",
                *paths,
                text=False,
            )
        )
    )
    monkeypatch.setenv("BOOKFORGE_MAXTEXT_APPROVED_PATCH", str(patch))

    assert validate_maxtext_checkout(checkout, config) == checkout.resolve()


def test_approved_patch_comparison_uses_stable_object_id_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, config = _checkout(tmp_path)
    _git(checkout, "config", "core.abbrev", "12")
    paths = (
        "src/maxtext/trainers/pre_train/train.py",
        "src/maxtext/utils/train_utils.py",
    )
    for relative in paths:
        (checkout / relative).write_text("after\n", encoding="utf-8")
    patch = tmp_path / "approved.patch"
    patch.write_bytes(
        bytes(
            _git(
                checkout,
                "diff",
                "--no-ext-diff",
                "--binary",
                "--abbrev=8",
                "--unified=1",
                "--",
                *paths,
                text=False,
            )
        )
    )
    monkeypatch.setenv("BOOKFORGE_MAXTEXT_APPROVED_PATCH", str(patch))

    assert validate_maxtext_checkout(checkout, config) == checkout.resolve()


def test_changed_or_additional_maxtext_patch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, config = _checkout(tmp_path)
    for relative in (
        "src/maxtext/trainers/pre_train/train.py",
        "src/maxtext/utils/train_utils.py",
    ):
        (checkout / relative).write_text("after\n", encoding="utf-8")
    patch = tmp_path / "approved.patch"
    patch.write_bytes(b"not the checkout diff\n")
    monkeypatch.setenv("BOOKFORGE_MAXTEXT_APPROVED_PATCH", str(patch))

    with pytest.raises(ExecutionRefused, match="differs from the approved patch"):
        validate_maxtext_checkout(checkout, config)

    (checkout / "README.md").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ExecutionRefused, match="unapproved changes"):
        validate_maxtext_checkout(checkout, config)


def test_config_required_patch_is_checksum_bound_and_cannot_be_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, base_config = _checkout(tmp_path)
    patch = tmp_path / "approved.patch"
    patch.write_bytes(b"approved patch bytes\n")
    from training.jax_fidelity.integrity import sha256_file

    config = SimpleNamespace(
        versions=base_config.versions,
        training={"approved_maxtext_patch_sha256": sha256_file(patch)},
    )
    with pytest.raises(ExecutionRefused, match="requires the approved"):
        validate_maxtext_checkout(checkout, config)

    monkeypatch.setenv("BOOKFORGE_MAXTEXT_APPROVED_PATCH", str(patch))
    assert approved_maxtext_patch_sha256(config) == sha256_file(patch)
    patch.write_bytes(b"changed\n")
    with pytest.raises(ExecutionRefused, match="checksum"):
        approved_maxtext_patch_sha256(config)


def test_exact_two_file_config_approved_patch_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, base_config = _checkout(tmp_path)
    (checkout / "src/maxtext/utils/train_utils.py").write_text("after\n", encoding="utf-8")
    (checkout / "src/maxtext/trainers/pre_train/train.py").write_text("after\n", encoding="utf-8")
    patch = tmp_path / "approved.patch"
    patch.write_bytes(
        bytes(
            _git(
                checkout,
                "diff",
                "--no-ext-diff",
                "--binary",
                "--abbrev=8",
                "--unified=1",
                "--",
                "src/maxtext/trainers/pre_train/train.py",
                "src/maxtext/utils/train_utils.py",
                text=False,
            )
        )
    )
    from training.jax_fidelity.integrity import sha256_file

    config = SimpleNamespace(
        versions=base_config.versions,
        training={"approved_maxtext_patch_sha256": sha256_file(patch)},
    )
    monkeypatch.setenv("BOOKFORGE_MAXTEXT_APPROVED_PATCH", str(patch))

    assert validate_maxtext_checkout(checkout, config) == checkout.resolve()


def test_partial_one_file_patch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, config = _checkout(tmp_path)
    source = checkout / "src/maxtext/utils/train_utils.py"
    source.write_text("after\n", encoding="utf-8")
    patch = tmp_path / "approved.patch"
    patch.write_bytes(
        bytes(
            _git(
                checkout,
                "diff",
                "--no-ext-diff",
                "--binary",
                "--abbrev=8",
                "--unified=1",
                "--",
                "src/maxtext/utils/train_utils.py",
                text=False,
            )
        )
    )
    monkeypatch.setenv("BOOKFORGE_MAXTEXT_APPROVED_PATCH", str(patch))

    with pytest.raises(ExecutionRefused, match="unapproved changes"):
        validate_maxtext_checkout(checkout, config)
