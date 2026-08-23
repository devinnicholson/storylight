import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_global_modal_entrypoint_bootstraps_repository_source_before_budget_import() -> None:
    source = (ROOT / "deploy/modal_visual_lab.py").read_text()
    helper = source.split("def _budget_types():", 1)[1].split("def _open_budget", 1)[0]

    assert 'Path(__file__).resolve().parents[1] / "src"' in helper
    assert "sys.path.insert(0, str(repository_source))" in helper
    assert helper.index("sys.path.insert") < helper.index("from bookforge.visual_lab import")


def test_budget_gate_uses_authoritative_reconciled_baseline() -> None:
    plan = json.loads((ROOT / "experiments/visual-lab/plan.json").read_text())

    assert plan["budget"]["monthly_credit_usd"] == 30
    assert plan["ledger_baseline_usd"] == 0.18517822
    assert any("28.50" in rule for rule in plan["stop_rules"])
