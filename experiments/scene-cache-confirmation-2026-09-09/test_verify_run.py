"""Ensure the audit independently covers every measured input and both warmups."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("confirmation_audit", HERE / "verify-run.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_audit_schedule_preserves_all_inputs_and_rejects_missing_or_duplicate():
    examples = [json.loads(s) for s in (HERE / "inputs.jsonl").read_text().splitlines()]
    expected = json.loads((HERE / "protocol.json").read_text())["schedule"]
    planned = {mode: audit.schedule(examples, None, mode) for mode in audit.MODES}
    assert planned == expected
    calls = [r for plan in planned.values() for r in plan]
    assert len(calls) == 260
    assert [r["global_ordinal"] for r in calls] == list(range(260))
    assert sum(r["warmup"] for r in calls) == 4
    for mode in audit.MODES:
        measured = [r for r in calls if r["mode"] == mode and not r["warmup"]]
        assert len(measured) == len({r["id"] for r in measured}) == 128
        assert {r["id"] for r in measured} == {e["id"] for e in examples[2:]}
    for altered in (examples[:-1], [examples[1], *examples[1:]]):
        with pytest.raises(ValueError, match="confirmation scope"):
            audit.schedule(altered, None, "dynamic")
