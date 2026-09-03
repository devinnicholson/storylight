import shutil
import subprocess
from pathlib import Path

import pytest


def test_next_page_workbench_state_machine():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the workbench state-machine check")
    subprocess.run(
        [node, "tests/anticipatory_workbench.test.js"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=15,
    )
