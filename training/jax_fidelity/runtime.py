"""Shared no-shell execution boundary for model-heavy entrypoints."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .configuration import ExperimentConfig
from .integrity import sha256_file


class ExecutionRefused(RuntimeError):
    """A heavy command lacked an exact, one-purpose approval boundary."""


_PATCHED_MAXTEXT_MODULES = {
    "maxtext.utils.train_utils": (
        "src/maxtext/utils/train_utils.py",
        "model = lora_utils.apply_lora_to_model(model, mesh, config)",
    ),
    "maxtext.trainers.pre_train.train": (
        "src/maxtext/trainers/pre_train/train.py",
        "nnx.state(new_state.model, train_param_type)",
    ),
}


def approved_maxtext_patch_sha256(config: ExperimentConfig) -> str | None:
    """Verify and return the config-approved MaxText patch digest when required."""

    training = getattr(config, "training", {})
    expected = training.get("approved_maxtext_patch_sha256")
    if expected is None:
        return None
    approved_patch = os.environ.get("BOOKFORGE_MAXTEXT_APPROVED_PATCH", "")
    patch_path = Path(approved_patch)
    if not approved_patch or not patch_path.is_file() or patch_path.is_symlink():
        raise ExecutionRefused("config-approved MaxText patch is missing or unsafe")
    actual = sha256_file(patch_path)
    if actual != expected:
        raise ExecutionRefused("MaxText patch checksum differs from the approved config")
    return actual


def approval_token(
    *,
    stage: str,
    run_id: str,
    config_sha256: str,
    input_sha256: str,
) -> str:
    return f"{stage.upper()}:{run_id}:{config_sha256}:{input_sha256}"


def require_approval(expected: str) -> None:
    actual = os.environ.get("BOOKFORGE_JAX_EXECUTION_APPROVAL", "")
    if actual != expected:
        raise ExecutionRefused(
            "execution refused; set BOOKFORGE_JAX_EXECUTION_APPROVAL to the exact token "
            "printed by the dry run"
        )


def validate_maxtext_checkout(root: Path | str, config: ExperimentConfig) -> Path:
    checkout = Path(root).resolve()
    if not (checkout / ".git").exists():
        raise ExecutionRefused("MaxText must be an explicit Git checkout")
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != config.versions["maxtext_revision"]:
        raise ExecutionRefused("MaxText checkout does not match the pinned revision")
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not dirty:
        training = getattr(config, "training", {})
        if training.get("approved_maxtext_patch_sha256") is not None:
            raise ExecutionRefused("the config requires the approved MaxText patch")
        return checkout

    approved_dirty_state = (
        " M src/maxtext/trainers/pre_train/train.py\n"
        " M src/maxtext/utils/train_utils.py\n"
    )
    if dirty != approved_dirty_state:
        raise ExecutionRefused("MaxText checkout has unapproved changes")
    approved_patch = os.environ.get("BOOKFORGE_MAXTEXT_APPROVED_PATCH", "")
    patch_path = Path(approved_patch)
    approved_maxtext_patch_sha256(config)
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "diff",
            "--no-ext-diff",
            "--binary",
            "--abbrev=8",
            "--unified=0",
            "--",
            "src/maxtext/trainers/pre_train/train.py",
            "src/maxtext/utils/train_utils.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if diff != patch_path.read_bytes():
        raise ExecutionRefused("MaxText checkout differs from the approved patch")
    return checkout


def validate_maxtext_import_provenance(root: Path | str) -> dict[str, str]:
    """Prove this interpreter resolves MaxText from the patched source checkout."""

    checkout = Path(root).resolve()
    resolved: dict[str, str] = {}
    for module_name, (relative_path, required_source) in _PATCHED_MAXTEXT_MODULES.items():
        expected = (checkout / relative_path).resolve()
        if not expected.is_file() or required_source not in expected.read_text(encoding="utf-8"):
            raise ExecutionRefused(f"patched MaxText source is missing for {module_name}")
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError) as error:
            raise ExecutionRefused(
                f"cannot resolve patched MaxText module {module_name}"
            ) from error
        if spec is None or spec.origin is None:
            raise ExecutionRefused(f"cannot resolve patched MaxText module {module_name}")
        actual = Path(spec.origin).resolve()
        if actual != expected:
            raise ExecutionRefused(
                f"{module_name} resolves outside the approved MaxText checkout: {actual}"
            )
        resolved[module_name] = str(actual)
    return resolved


def run_checked(
    command: Sequence[str], *, cwd: Path, environment: dict[str, str] | None = None
) -> None:
    subprocess.run(list(command), cwd=cwd, check=True, env=environment)


def run_checked_capture(command: Sequence[str], *, cwd: Path) -> str:
    """Run a bounded verifier and return its combined machine-parseable log."""

    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return f"{completed.stdout}\n{completed.stderr}"
