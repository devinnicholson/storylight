import os
from pathlib import Path

import pytest

from storylight.privacy_audit import audit_process


def make_process(proc_root: Path, remote: str) -> None:
    process = proc_root / "321"
    (process / "fd").mkdir(parents=True)
    (process / "net").mkdir()
    os.symlink("socket:[42]", process / "fd" / "3")
    row = f"0: 0100007F:1F90 {remote}:01BB 01 00000000:00000000 00:00000000 00000000 1000 0 42"
    (process / "net" / "tcp").write_text(f"header\n{row}\n", encoding="ascii")
    (process / "net" / "tcp6").write_text("header\n", encoding="ascii")
    (process / "net" / "udp").write_text("header\n", encoding="ascii")
    (process / "net" / "udp6").write_text("header\n", encoding="ascii")


def test_privacy_audit_accepts_loopback_connection(tmp_path: Path) -> None:
    make_process(tmp_path, "0100007F")

    report = audit_process(321, tmp_path)

    assert report["ready"] is True
    assert report["connections"][0]["remote_host"] == "127.0.0.1"


def test_privacy_audit_rejects_external_connection(tmp_path: Path) -> None:
    make_process(tmp_path, "08080808")

    report = audit_process(321, tmp_path)

    assert report["ready"] is False
    assert report["connections"][0]["remote_host"] == "8.8.8.8"


def test_privacy_audit_fails_closed_without_socket_visibility(tmp_path: Path) -> None:
    (tmp_path / "321").mkdir()

    report = audit_process(321, tmp_path)

    assert report["ready"] is False
    assert report["visibility_errors"]


def test_privacy_audit_fails_closed_when_fd_targets_are_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_process(tmp_path, "0100007F")

    def deny_readlink(_: os.PathLike[str]) -> str:
        raise PermissionError("fd targets are protected")

    monkeypatch.setattr("storylight.privacy_audit.os.readlink", deny_readlink)

    report = audit_process(321, tmp_path)

    assert report["ready"] is False
    assert "fd targets are protected" in report["visibility_errors"][0]


def test_privacy_audit_rejects_non_loopback_listener(tmp_path: Path) -> None:
    process = tmp_path / "321"
    (process / "fd").mkdir(parents=True)
    (process / "net").mkdir()
    os.symlink("socket:[42]", process / "fd" / "3")
    row = "0: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 42"
    (process / "net" / "tcp").write_text(f"header\n{row}\n", encoding="ascii")
    for name in ("tcp6", "udp", "udp6"):
        (process / "net" / name).write_text("header\n", encoding="ascii")

    report = audit_process(321, tmp_path)

    assert report["ready"] is False
    assert report["connections"][0]["state"] == "0A"
