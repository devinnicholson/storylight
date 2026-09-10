import importlib.util
import os
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "configure_anticipatory", ROOT / "deploy/jetson/configure-anticipatory.py"
)
assert SPEC and SPEC.loader
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def test_only_bridge_settings_change_and_duplicate_keys_are_removed():
    source = (
        "# private configuration\nMODAL_TOKEN_SECRET=keep-this-exactly\n"
        "STORYLIGHT_LIVE_SCENE_PLANNER=model\n"
        "STORYLIGHT_ANTICIPATORY_BACKEND=disabled\n"
        " STORYLIGHT_ANTICIPATORY_BACKEND =disabled\n"
        "STORYLIGHT_ANTICIPATORY_AUDIENCE=old-audience\n"
    )
    values = setup.replacement_values(enabled=True, port=18082)
    changed = setup.rewrite_environment(source, values)
    assert "MODAL_TOKEN_SECRET=keep-this-exactly\n" in changed
    assert "STORYLIGHT_LIVE_SCENE_PLANNER=model\n" in changed
    assert changed.count("STORYLIGHT_ANTICIPATORY_BACKEND=") == 1
    assert "old-audience" not in changed
    assert "STORYLIGHT_ANTICIPATORY_URL=http://127.0.0.1:18082\n" in changed
    assert setup.environment_value(changed, "STORYLIGHT_ANTICIPATORY_BACKEND") == "gke"
    assert setup.rewrite_environment(changed, values) == changed
    disabled = setup.rewrite_environment(
        changed, setup.replacement_values(enabled=False, port=18082)
    )
    assert setup.environment_value(disabled, "STORYLIGHT_ANTICIPATORY_BACKEND") == "disabled"
    assert "http://127.0.0.1:18082" in disabled


@pytest.mark.parametrize("port", [1023])
def test_bridge_rejects_unsafe_ports(port):
    with pytest.raises(ValueError, match="port"):
        setup.replacement_values(enabled=True, port=port)


def test_secure_config_rejects_symlinks_hardlinks_and_group_writes(tmp_path):
    path = tmp_path / "storylight.env"
    path.write_text("SECRET=private\n")
    path.chmod(0o600)
    assert setup.read_config(path, owner_uid=os.getuid()) == "SECRET=private\n"
    link = tmp_path / "link.env"
    link.symlink_to(path)
    with pytest.raises(OSError):
        setup.read_config(link, owner_uid=os.getuid())
    hardlink = tmp_path / "hardlink.env"
    os.link(path, hardlink)
    with pytest.raises(ValueError, match="single-link"):
        setup.read_config(path, owner_uid=os.getuid())
    hardlink.unlink()
    path.chmod(0o660)
    with pytest.raises(ValueError, match="mode-600"):
        setup.read_config(path, owner_uid=os.getuid())
    path.chmod(0o600)
    tmp_path.chmod(0o770)
    with pytest.raises(ValueError, match="directory"):
        setup.read_config(path, owner_uid=os.getuid())


def test_success_preserves_private_backup_and_only_restarts_api(tmp_path):
    path = tmp_path / "storylight.env"
    path.write_text("SECRET=private\n")
    path.chmod(0o600)
    events = []
    backup = setup.apply_configuration(
        path,
        "SECRET=private\n",
        "SECRET=private\nBRIDGE=enabled\n",
        restart=lambda: events.append("restart"),
        verify=lambda: events.append("verify"),
    )
    assert events == ["restart", "verify"]
    assert backup.read_text() == "SECRET=private\n"
    assert path.read_text().endswith("BRIDGE=enabled\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


@pytest.mark.parametrize("failure", ["restart", "verify"])
def test_failure_restores_original_and_restarts_old_configuration(tmp_path, failure, capsys):
    path = tmp_path / "storylight.env"
    original = "TOKEN=must-not-be-printed\n"
    path.write_text(original)
    events = []

    def restart():
        events.append("restart")
        if failure == "restart" and len(events) == 1:
            raise RuntimeError("restart failed")

    def verify():
        raise RuntimeError("verification failed")

    with pytest.raises(RuntimeError):
        setup.apply_configuration(
            path, original, "BRIDGE=enabled\n", restart=restart, verify=verify
        )
    assert path.read_text() == original
    assert events == ["restart", "restart"]
    assert "must-not-be-printed" not in capsys.readouterr().out


def test_rollback_does_not_overwrite_a_concurrent_admin_change(tmp_path):
    path = tmp_path / "storylight.env"
    path.write_text("OLD=yes\n")

    def verify():
        path.write_text("OTHER_ADMIN_CHANGE=yes\n")
        raise RuntimeError("verification failed")

    with pytest.raises(RuntimeError, match="concurrently"):
        setup.apply_configuration(
            path, "OLD=yes\n", "BRIDGE=enabled\n", restart=lambda: None, verify=verify
        )
    assert path.read_text() == "OTHER_ADMIN_CHANGE=yes\n"


def test_bridge_identity_check_never_calls_ready_or_prewarm(monkeypatch):
    seen = []

    def health(url):
        seen.append(url)
        return {
            "ready": True,
            "service": "storylight-anticipatory",
            "privacy_boundary": "sanitized_scene_spec_v1",
        }

    monkeypatch.setattr(setup, "read_json", health)
    setup.verify_bridge(18082)
    assert seen == ["http://127.0.0.1:18082/health"]
    monkeypatch.setattr(setup, "read_json", lambda url: {"ready": True})
    with pytest.raises(ValueError, match="expected Storylight"):
        setup.verify_bridge(18082)
