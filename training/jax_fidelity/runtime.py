"""Shared no-shell execution boundary for model-heavy entrypoints."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .configuration import ExperimentConfig


class ExecutionRefused(RuntimeError):
    """A heavy command lacked an exact, one-purpose approval boundary."""


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
    if dirty:
        raise ExecutionRefused("MaxText checkout has uncommitted changes")
    return checkout


def run_checked(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)
