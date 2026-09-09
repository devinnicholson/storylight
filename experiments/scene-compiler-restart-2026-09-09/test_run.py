"""Exercise the inventory distinction between empty and reused compiler directories."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compiler_restart", Path(__file__).with_name("run.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_cache_inventory_detects_content_change_and_rejects_redirects(tmp_path):
    first = runner.inventory(tmp_path)
    assert first["file_count"] == first["total_bytes"] == 0
    (tmp_path / "inductor").mkdir()
    artifact = tmp_path / "inductor/graph.py"
    artifact.write_bytes(b"one")
    before = runner.inventory(tmp_path)
    artifact.write_bytes(b"two")
    after = runner.inventory(tmp_path)
    assert before["file_count"] == after["file_count"] == 1
    assert before["total_bytes"] == after["total_bytes"] == 3
    assert before["inventory_sha256"] != after["inventory_sha256"]
    (tmp_path / "redirect").symlink_to(artifact)
    with pytest.raises(ValueError, match="compiler symlink"):
        runner.inventory(tmp_path)
