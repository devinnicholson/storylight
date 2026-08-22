import pytest

import bookforge.hardware_acceptance as hardware_acceptance
from bookforge.hardware_acceptance import _require_loopback_url, required_failures


def test_collect_marks_missing_service_pid_as_failed_privacy_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware_acceptance,
        "_command",
        lambda *args, **kwargs: {"ok": True, "detail": "nvme"},
    )
    monkeypatch.setattr(
        hardware_acceptance,
        "_file",
        lambda *args, **kwargs: {"ok": True, "detail": "R39"},
    )
    monkeypatch.setattr(
        hardware_acceptance,
        "_display_check",
        lambda: {"ok": True, "detail": ":0"},
    )
    monkeypatch.setattr(
        hardware_acceptance,
        "_executable_check",
        lambda *args: {"ok": True, "detail": "chromium"},
    )
    monkeypatch.setattr(
        hardware_acceptance,
        "_thermal_check",
        lambda: {"ok": True, "detail": "nominal"},
    )

    def fake_http_json(url: str) -> dict[str, object]:
        detail: object = {"ready": True, "data_dir": "/var/lib/bookforge"}
        return {"ok": True, "detail": detail}

    monkeypatch.setattr(hardware_acceptance, "_http_json", fake_http_json)

    report = hardware_acceptance.collect(
        base_url="http://127.0.0.1:8080",
        exercise_io=False,
        camera_device="/dev/video0",
        audio_device="default",
        service_pid=None,
    )

    assert report["checks"]["privacy"]["ok"] is False
    assert "service PID is required" in report["checks"]["privacy"]["detail"]


def test_required_failures_include_unavailable_asr() -> None:
    checks = {
        name: {"ok": True, "detail": {}}
        for name in (
            "platform",
            "l4t",
            "cuda",
            "tensorrt",
            "power",
            "storage",
            "camera_inventory",
            "microphone_inventory",
            "display",
            "chromium",
            "thermals",
            "health",
            "readiness",
            "runtime",
            "latest_story_pack",
            "bookforge_storage",
        )
    }
    checks["runtime"]["detail"] = {
        "storage": {"ready": True},
        "asr": {"ready": False},
    }
    checks["privacy"] = {"ok": True, "detail": "loopback only"}
    report = {"exercise_io": False, "checks": checks}

    assert required_failures(report, require_asr=True) == ["runtime.asr"]


def test_required_failures_include_privacy_boundary() -> None:
    checks = {
        name: {"ok": True, "detail": {"storage": {"ready": True}}}
        for name in (
            "platform",
            "l4t",
            "cuda",
            "tensorrt",
            "power",
            "storage",
            "camera_inventory",
            "microphone_inventory",
            "display",
            "chromium",
            "thermals",
            "health",
            "readiness",
            "runtime",
            "latest_story_pack",
            "bookforge_storage",
        )
    }
    checks["privacy"] = {"ok": False, "detail": "external connection"}

    failures = required_failures({"exercise_io": False, "checks": checks}, require_asr=False)

    assert failures == ["privacy"]


def test_required_failures_reject_missing_privacy_evidence() -> None:
    checks = {
        name: {"ok": True, "detail": {"storage": {"ready": True}}}
        for name in (
            "platform",
            "l4t",
            "cuda",
            "tensorrt",
            "power",
            "storage",
            "camera_inventory",
            "microphone_inventory",
            "display",
            "chromium",
            "thermals",
            "health",
            "readiness",
            "runtime",
            "latest_story_pack",
            "bookforge_storage",
        )
    }

    failures = required_failures({"exercise_io": False, "checks": checks}, require_asr=False)

    assert failures == ["privacy"]


@pytest.mark.parametrize(
    "url", ["https://example.com", "http://192.168.1.10:8080", "http://localhost:8080"]
)
def test_hardware_evidence_rejects_non_loopback_base_url(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        _require_loopback_url(url)


def test_hardware_evidence_enforces_asr_latency_budget() -> None:
    checks = {
        name: {"ok": True, "detail": {"storage": {"ready": True}}}
        for name in (
            "platform",
            "l4t",
            "cuda",
            "tensorrt",
            "power",
            "storage",
            "camera_inventory",
            "microphone_inventory",
            "display",
            "chromium",
            "thermals",
            "health",
            "readiness",
            "runtime",
            "latest_story_pack",
            "bookforge_storage",
            "camera_capture",
            "microphone_capture",
        )
    }
    checks["asr_capture"] = {"ok": True, "detail": {"total_ms": 4_001}}
    checks["privacy"] = {"ok": True, "detail": "loopback only"}
    report = {"exercise_io": True, "checks": checks}

    assert "asr_latency" in required_failures(report, require_asr=True)
