from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.jax_fidelity.runtime import (
    ExecutionRefused,
    approved_maxtext_patch_sha256,
    validate_maxtext_checkout,
    validate_maxtext_import_provenance,
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
    checkpointing = checkout / "src/maxtext/common/checkpointing.py"
    source = checkout / "src/maxtext/utils/train_utils.py"
    train = checkout / "src/maxtext/trainers/pre_train/train.py"
    checkpointing.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    train.parent.mkdir(parents=True)
    checkpointing.write_text("before\n", encoding="utf-8")
    source.write_text("before\n", encoding="utf-8")
    train.write_text("before\n", encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "bookforge@example.invalid")
    _git(checkout, "config", "user.name", "Bookforge Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "base")
    revision = str(_git(checkout, "rev-parse", "HEAD")).strip()
    return checkout, SimpleNamespace(versions={"maxtext_revision": revision})


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


def test_exact_three_file_config_approved_patch_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, base_config = _checkout(tmp_path)
    (checkout / "src/maxtext/common/checkpointing.py").write_text(
        "after\n", encoding="utf-8"
    )
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
                "--unified=0",
                "--",
                "src/maxtext/common/checkpointing.py",
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


def test_maxtext_import_provenance_rejects_installed_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "MaxText"
    sources = {
        "maxtext.common.checkpointing": (
            "src/maxtext/common/checkpointing.py",
            "if isinstance(val, (list, tuple)):\n",
        ),
        "maxtext.utils.train_utils": (
            "src/maxtext/utils/train_utils.py",
            "model = lora_utils.apply_lora_to_model(model, None, config)\n",
        ),
        "maxtext.trainers.pre_train.train": (
            "src/maxtext/trainers/pre_train/train.py",
            "nnx.state(new_state.model, train_param_type)\n",
        ),
    }
    for relative, source in sources.values():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    wheel_module = tmp_path / "site-packages/maxtext/utils/train_utils.py"
    wheel_module.parent.mkdir(parents=True)
    wheel_module.write_text("unpatched\n", encoding="utf-8")
    monkeypatch.setattr(
        "training.jax_fidelity.runtime.importlib.util.find_spec",
        lambda _name: SimpleNamespace(origin=str(wheel_module)),
    )

    with pytest.raises(ExecutionRefused, match="outside the approved MaxText checkout"):
        validate_maxtext_import_provenance(checkout)
