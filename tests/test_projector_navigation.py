import shutil
import subprocess
from pathlib import Path

import pytest


def test_projector_page_activation_preserves_visible_state_until_media_commits():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the projector navigation check")
    subprocess.run(
        [node, "tests/projector_navigation.test.js"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=15,
    )
