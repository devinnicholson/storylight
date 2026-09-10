from __future__ import annotations

import argparse
import ipaddress
import json
import platform
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from storylight.privacy_audit import audit_process


def collect(
    *,
    base_url: str,
    exercise_io: bool,
    camera_device: str,
    audio_device: str,
    service_pid: int | None,
) -> dict[str, Any]:
    _require_loopback_url(base_url)
    checks: dict[str, dict[str, Any]] = {}
    checks["platform"] = _command(["uname", "-a"])
    checks["l4t"] = _file("/etc/nv_tegra_release", required="R39 (release), REVISION: 2.1")
    checks["cuda"] = _command(["nvcc", "--version"], required="release 13.2")
    checks["tensorrt"] = _command(
        ["python3", "-c", "import tensorrt; print(tensorrt.__version__)"],
        required="10.16.2",
    )
    checks["power"] = _command(["nvpmodel", "-q"])
    checks["storage"] = _command(["lsblk", "-o", "NAME,TRAN,SIZE,FSAVAIL,MOUNTPOINTS,MODEL"])
    if checks["storage"]["ok"] and "nvme" not in str(checks["storage"]["detail"]).lower():
        checks["storage"]["ok"] = False
        checks["storage"]["detail"] = "No NVMe device appeared in lsblk output"
    checks["camera_inventory"] = _command(["v4l2-ctl", "--list-devices"])
    checks["microphone_inventory"] = _command(["arecord", "-l"])
    checks["display"] = _display_check()
    checks["chromium"] = _executable_check("chromium", "chromium-browser")
    checks["thermals"] = _thermal_check()
    checks["health"] = _http_json(f"{base_url.rstrip('/')}/healthz")
    checks["readiness"] = _http_json(f"{base_url.rstrip('/')}/readyz")
    checks["runtime"] = _http_json(f"{base_url.rstrip('/')}/v1/runtime:status")
    checks["latest_story_pack"] = _http_json(f"{base_url.rstrip('/')}/v1/story-packs/latest")
    runtime_detail = checks["runtime"].get("detail")
    if checks["runtime"]["ok"] and (
        not isinstance(runtime_detail, dict) or not runtime_detail.get("ready", False)
    ):
        checks["runtime"]["ok"] = False
    if isinstance(runtime_detail, dict) and runtime_detail.get("data_dir"):
        checks["storylight_storage"] = _command(
            ["findmnt", "-T", str(runtime_detail["data_dir"]), "-no", "SOURCE,TARGET"],
            required="nvme",
        )
    else:
        checks["storylight_storage"] = {
            "ok": False,
            "detail": "runtime did not report a data directory",
        }

    if service_pid is None:
        checks["privacy"] = {
            "ok": False,
            "detail": "A running Storylight service PID is required for privacy evidence",
            "value": None,
        }
    else:
        privacy = audit_process(service_pid)
        checks["privacy"] = {
            "ok": bool(privacy["ready"]),
            "detail": privacy["detail"],
            "value": privacy,
        }

    if exercise_io:
        with tempfile.TemporaryDirectory(prefix="storylight-hardware-") as directory:
            root = Path(directory)
            camera_path = root / "camera.jpg"
            audio_path = root / "microphone.wav"
            checks["camera_capture"] = _command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "v4l2",
                    "-i",
                    camera_device,
                    "-frames:v",
                    "1",
                    "-y",
                    str(camera_path),
                ]
            )
            checks["microphone_capture"] = _command(
                [
                    "arecord",
                    "-D",
                    audio_device,
                    "-d",
                    "3",
                    "-f",
                    "S16_LE",
                    "-r",
                    "16000",
                    "-c",
                    "1",
                    str(audio_path),
                ],
                timeout=10,
            )
            if checks["microphone_capture"]["ok"] and audio_path.exists():
                checks["asr_capture"] = _post_audio(
                    f"{base_url.rstrip('/')}/v1/audio:transcribe", audio_path
                )
            else:
                checks["asr_capture"] = {"ok": False, "detail": "microphone capture failed"}

    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "exercise_io": exercise_io,
        "checks": checks,
    }


def required_failures(
    report: dict[str, Any], *, require_asr: bool, max_asr_ms: float = 4_000
) -> list[str]:
    required = [
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
        "storylight_storage",
        "privacy",
    ]
    if report["exercise_io"]:
        required.extend(["camera_capture", "microphone_capture"])
    failures = [name for name in required if not report["checks"].get(name, {}).get("ok", False)]
    runtime = report["checks"].get("runtime", {}).get("detail", {})
    if isinstance(runtime, dict) and not runtime.get("storage", {}).get("ready", False):
        failures.append("runtime.storage")
    if require_asr:
        if report["exercise_io"]:
            asr_capture = report["checks"].get("asr_capture", {})
            if not asr_capture.get("ok", False):
                failures.append("asr_capture")
            detail = asr_capture.get("detail", {})
            if (
                not isinstance(detail, dict)
                or float(detail.get("total_ms", float("inf"))) > max_asr_ms
            ):
                failures.append("asr_latency")
        elif not isinstance(runtime, dict) or not runtime.get("asr", {}).get("ready", False):
            failures.append("runtime.asr")
    return list(dict.fromkeys(failures))


def _require_loopback_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("base_url must be an HTTP(S) loopback URL")
    try:
        if ipaddress.ip_address(parsed.hostname).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError("base_url must resolve explicitly to a loopback address")


def _command(command: list[str], timeout: int = 5, required: str | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "detail": str(error), "command": command}
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    ok = result.returncode == 0 and (required is None or required in output)
    return {
        "ok": ok,
        "detail": output[-4000:] or f"exit {result.returncode}",
        "command": command,
        "returncode": result.returncode,
    }


def _file(filename: str, required: str | None = None) -> dict[str, Any]:
    path = Path(filename)
    try:
        detail = path.read_text(encoding="utf-8")[:4000]
        return {"ok": required is None or required in detail, "detail": detail}
    except OSError as error:
        return {"ok": False, "detail": str(error)}


def _display_check() -> dict[str, Any]:
    connected = []
    for status in Path("/sys/class/drm").glob("card*-*/status"):
        try:
            if status.read_text(encoding="ascii").strip() == "connected":
                connected.append(str(status.parent))
        except OSError:
            continue
    return {"ok": bool(connected), "detail": connected or "no connected DRM output"}


def _executable_check(*names: str) -> dict[str, Any]:
    for name in names:
        path = shutil.which(name)
        if path:
            return {"ok": True, "detail": path}
    return {"ok": False, "detail": f"none of {', '.join(names)} were found"}


def _thermal_check() -> dict[str, Any]:
    observed: dict[str, str] = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            observed[(zone / "type").read_text(encoding="ascii").strip()] = (
                (zone / "temp").read_text(encoding="ascii").strip()
            )
        except OSError:
            continue
    return {"ok": bool(observed), "detail": observed or "no readable thermal zones"}


def _http_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=5) as response:
            payload = json.load(response)
            return {"ok": 200 <= response.status < 300, "detail": payload}
    except (OSError, URLError, ValueError) as error:
        return {"ok": False, "detail": str(error)}


def _post_audio(url: str, audio_path: Path) -> dict[str, Any]:
    request = Request(
        url,
        data=audio_path.read_bytes(),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return {"ok": 200 <= response.status < 300, "detail": json.load(response)}
    except (OSError, URLError, ValueError) as error:
        return {"ok": False, "detail": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Storylight Jetson hardware evidence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--exercise-io", action="store_true")
    parser.add_argument("--require-asr", action="store_true")
    parser.add_argument("--max-asr-ms", type=float, default=4_000)
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument("--audio-device", default="default")
    parser.add_argument(
        "--service-pid",
        type=int,
        help="running Storylight service PID; readiness requires its privacy audit",
    )
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()

    try:
        report = collect(
            base_url=arguments.base_url,
            exercise_io=arguments.exercise_io,
            camera_device=arguments.camera_device,
            audio_device=arguments.audio_device,
            service_pid=arguments.service_pid,
        )
    except ValueError as error:
        parser.error(str(error))
    failures = required_failures(
        report,
        require_asr=arguments.require_asr,
        max_asr_ms=arguments.max_asr_ms,
    )
    report["required_failures"] = failures
    report["ready"] = not failures
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    arguments.output.write_text(f"{payload}\n", encoding="utf-8")
    print(
        json.dumps(
            {"ready": report["ready"], "output": str(arguments.output), "failures": failures}
        )
    )
    return 1 if arguments.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
