import importlib.util
import socket
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failing_window", [False, True])
def test_nsight_failure_restores_planner_and_kiosk(tmp_path, monkeypatch, failing_window):
    spec = importlib.util.spec_from_file_location(
        "nsight_check", "deploy/jetson/profile-planner-nsight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profiler = tmp_path / "nsys"
    profiler.touch()
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "llm.engine").touch()
    monkeypatch.setattr(module, "NSYS", str(profiler))
    monkeypatch.setattr(module, "ENGINE", engine)
    monkeypatch.setattr(module, "PORT", 0)
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        socket, "socket", lambda: nullcontext(SimpleNamespace(bind=lambda addr: None))
    )
    monkeypatch.setattr(sys, "argv", ["profile", "--output", str(tmp_path / "report")])
    calls = []

    def systemctl(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    def window(output, profile):
        if profile == failing_window:
            raise RuntimeError("synthetic capture failure")
        return {"requests": []}

    monkeypatch.setattr(module, "systemctl", systemctl)
    monkeypatch.setattr(module, "run_window", window)
    monkeypatch.setattr(module, "ready", lambda port: calls.append(("ready", port)))
    with pytest.raises(RuntimeError, match="capture failure"):
        module.main()
    assert calls[-3:] == [
        ("start", module.PLANNER),
        ("ready", 11435),
        ("start", module.KIOSK),
    ]
    assert (tmp_path / "report/measurements.json").is_file()
