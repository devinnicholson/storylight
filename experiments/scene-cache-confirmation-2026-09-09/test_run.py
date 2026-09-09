"""Confirm the pinned answer-free loader and preregistered dispatch schedule."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_and_schedule_match_independent_protocol():
    runner = load(HERE / "run.py", "confirmation_run")
    trainer = load(HERE.parent / "scene-adapter-v5-2026-09-09/train.py", "confirmation_train")
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    assert digest(HERE / "protocol.json") == runner.CONFIRMATION_SHA
    rows = trainer.read_rows(
        HERE,
        {"files": {"inputs.jsonl": {"sha256": runner.INPUT_SHA, "rows": 130}}},
        "inputs.jsonl",
        130,
        SimpleNamespace(digest=digest, require=runner.require),
        ["system", "user"],
    )
    frozen = json.loads((HERE / "protocol.json").read_text())
    assert {mode: runner.plan(rows, mode) for mode in runner.MODES} == frozen["schedule"]
    with pytest.raises(ValueError, match="confirmation_inputs"):
        runner.plan(rows, "bridge-eager")
    with pytest.raises(ValueError, match="confirmation_inputs"):
        runner.plan(rows[:-1], "dynamic")
