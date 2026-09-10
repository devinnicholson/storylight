from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_fidelity_closure_evidence.py"


def _write(path: Path, document: object) -> str:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_closure_builder_cli_produces_checksum_bound_cost_and_inventory(
    tmp_path: Path,
) -> None:
    cost_arguments: list[str] = []
    for provider, amount in (("gcp", 2.0), ("modal", 1.5)):
        source = tmp_path / f"{provider}-cost.json"
        source_sha256 = _write(
            source,
            {
                "schema_version": "story-fidelity-provider-cost-v1",
                "producer": f"storylight-{provider}-cost-source",
                "status": "final",
                "run_id": "campaign-v1",
                "provider": provider,
                "currency": "USD",
                "gross_cost_usd": amount,
            },
        )
        cost_arguments.extend(
            [f"--{provider}-source", str(source), f"--{provider}-source-sha256", source_sha256]
        )
    cost_output = tmp_path / "cost.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "cost",
            "--run-id",
            "campaign-v1",
            *cost_arguments,
            "--output",
            str(cost_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(cost_output.read_text())["gross_cost_usd"] == 3.5

    producers = {
        "vertex-jobs": "storylight-vertex-jobs-snapshot",
        "cloud-run-services": "storylight-cloud-run-services-snapshot",
        "modal-tasks": "storylight-modal-tasks-snapshot",
        "modal-functions": "storylight-modal-functions-snapshot",
    }
    inventory_arguments: list[str] = []
    for cli_name, producer in producers.items():
        resource_class = cli_name.replace("-", "_")
        source = tmp_path / f"{cli_name}.json"
        source_sha256 = _write(
            source,
            {
                "schema_version": "story-fidelity-resource-snapshot-v1",
                "producer": producer,
                "status": "observed",
                "run_id": "campaign-v1",
                "resource_class": resource_class,
                "active_paid_resources": 0,
            },
        )
        inventory_arguments.extend(
            [f"--{cli_name}-source", str(source), f"--{cli_name}-source-sha256", source_sha256]
        )
    inventory_output = tmp_path / "inventory.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "inventory",
            "--run-id",
            "campaign-v1",
            *inventory_arguments,
            "--output",
            str(inventory_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(inventory_output.read_text())["active_paid_resources"] == 0
