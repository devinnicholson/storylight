import importlib.util
import shutil
import subprocess

import pytest


def test_hand_interaction_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for browser state-machine checks")
    subprocess.run([node, "tests/hand_interaction.test.js"], check=True, timeout=15)


def test_asset_install_rejects_untrusted_bytes():
    spec = importlib.util.spec_from_file_location(
        "hand_assets", "scripts/install_hand_tracking_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="integrity"):
        module.unpack(b"not the pinned package", b"not the pinned model")
