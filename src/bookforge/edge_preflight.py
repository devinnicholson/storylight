from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from bookforge.asr import build_asr_backend
from bookforge.config import Settings


def evaluate(settings: Settings) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    edge_environment = settings.environment.lower() == "jetson"

    for name, path in (("data_dir", settings.data_dir), ("cache_dir", settings.cache_dir)):
        if edge_environment and not path.is_absolute():
            ready, detail = False, "Jetson paths must be absolute"
        else:
            ready, detail = _writable_directory(path)
        checks[name] = {"ready": ready, "detail": detail}

    backend = build_asr_backend(settings)
    asr_required = settings.asr_backend != "disabled"
    checks["asr"] = {
        "ready": backend.available or not asr_required,
        "detail": f"{backend.name}: {'available' if backend.available else 'unavailable'}",
    }

    edge_paths_ready = not edge_environment or (
        settings.data_dir.is_absolute() and settings.cache_dir.is_absolute()
    )
    checks["edge_paths"] = {
        "ready": edge_paths_ready,
        "detail": (
            "Jetson paths are absolute" if edge_paths_ready else "Jetson paths must be absolute"
        ),
    }

    checkout_env_ready = not edge_environment or not Path(".env").exists()
    checks["checkout_env"] = {
        "ready": checkout_env_ready,
        "detail": (
            "no checkout .env can influence the service"
            if checkout_env_ready
            else "remove /opt/bookforge/.env and use /etc/bookforge/bookforge.env"
        ),
    }

    edge_asr_ready = not edge_environment or settings.asr_backend != "mlx_whisper"
    checks["edge_asr"] = {
        "ready": edge_asr_ready,
        "detail": "edge-compatible ASR selected"
        if edge_asr_ready
        else "mlx_whisper is a Mac-only backend",
    }

    engine_required = edge_environment and settings.asr_backend == "whisper_trt"
    engine = Path(settings.asr_engine_path) if settings.asr_engine_path else None
    engine_ready = not engine_required or bool(
        engine and engine.is_file() and engine.stat().st_size
    )
    checks["asr_engine"] = {
        "ready": engine_ready,
        "detail": (
            str(engine)
            if engine_ready and engine is not None
            else "run deploy/jetson/warm-asr.sh before enabling WhisperTRT"
            if engine_required
            else "engine warmup is not required for the selected backend"
        ),
    }

    model_local = settings.model_backend == "fake" or _url_is_loopback(settings.model_base_url)
    checks["model_boundary"] = {
        "ready": not edge_environment or model_local,
        "detail": "model endpoint is local or disabled" if model_local else "remote model endpoint",
    }

    ready = all(bool(check["ready"]) for check in checks.values())
    return {
        "ready": ready,
        "environment": settings.environment,
        "pid": os.getpid(),
        "checks": checks,
    }


def _writable_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        probe = path / f".preflight-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return False, str(error)
    return True, str(path)


def _url_is_loopback(value: str) -> bool:
    host = urlparse(value).hostname
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> int:
    report = evaluate(Settings())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
