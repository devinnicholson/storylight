#!/usr/bin/env python3
"""Verify an installed display pack through an isolated, generation-disabled Jetson API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from bookforge.domain import StoryPack
from bookforge.finite_modal_provider import _jpeg_dimensions

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_story_fidelity_smoke import private_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DISABLED = {
    "MODEL_BACKEND": "fake",
    "MODEL_REQUIRE_GPU": "false",
    "ASSET_BACKEND": "disabled",
    "ASR_BACKEND": "disabled",
    "LIVE_SCENE_BACKEND": "disabled",
    "LIVE_SCENE_PLANNER": "deterministic",
    "LIVE_SCENE_PLANNER_BACKEND": "configured",
    "LIVE_SCENE_PLANNER_SCOPE": "focal",
    "LIVE_SCENE_PLANNER_AUTO_WARMUP": "false",
    "LIVE_SCENE_AUTO_PREWARM_ON_SUBMIT": "false",
    "LIVE_SCENE_ENABLE_MOTION": "false",
    "LIVE_SCENE_ENABLE_PREVIEW": "false",
    "LIVE_SCENE_CRITIC_BACKEND": "disabled",
    "ANTICIPATORY_BACKEND": "disabled",
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def private_read(path: Path, maximum: int) -> bytes:
    private_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > maximum
        ):
            raise ValueError("private input contract differs")
        return stream.read(maximum + 1)


def installed_pack(data: Path, receipt: dict) -> tuple[StoryPack, Path]:
    if data.resolve() != data.absolute() or data.resolve().is_relative_to(ROOT):
        raise ValueError("cache rehearsal requires an external directory")
    if any((parent / ".git").exists() for parent in (data, *data.parents)):
        raise ValueError("cache rehearsal data cannot be in a checkout")
    for folder in (data, data / "cache", data / "cache/assets", data / "story-packs"):
        info = folder.stat()
        mode = stat.S_IMODE(info.st_mode)
        private_mode = not mode & 0o022 if folder == data / "cache" else mode == 0o700
        if folder.is_symlink() or info.st_uid != os.getuid() or not private_mode:
            raise ValueError("installed directories must remain private")
    pointer = private_read(data / "story-packs/latest", 256).decode().strip()
    if not re.fullmatch(r"[a-z0-9_-]+-[a-f0-9]{12}\.story-pack\.json", pointer):
        raise ValueError("installed pointer differs")
    path = data / "story-packs" / pointer
    raw = private_read(path, 2_000_000)
    if digest(raw) != receipt["stored_pack_sha256"]:
        raise ValueError("installed pack proof differs")
    pack = StoryPack.model_validate_json(raw)
    verify_pack(pack, receipt)
    for asset in pack.assets:
        content = private_read(
            data / "cache/assets" / asset.local_uri.removeprefix("/v1/assets/"), 16_000_000
        )
        verify_asset(content, asset)
    return pack, path


def verify_pack(pack: StoryPack, receipt: dict) -> None:
    assets = [
        {
            "asset_id": asset.asset_id,
            "page_id": asset.page_id,
            "role": asset.role.value,
            "checksum_sha256": asset.checksum_sha256,
            "width": asset.width,
            "height": asset.height,
        }
        for asset in pack.assets
    ]
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "fidelity-display-installation"
        or receipt["asset_count"] != 16
        or receipt["completed_render_requests"] != 11
        or receipt["cached_replay_checksums_verified"] is not True
        or pack.schema_version != "2.0"
        or pack.planning_scope != "scene"
        or len(pack.pages) != 8
        or len(pack.assets) != 16
        or len({asset.asset_id for asset in pack.assets}) != 16
        or pack.story_id != receipt["story_id"]
        or [page.page_id for page in pack.pages] != receipt["page_ids"]
        or assets != receipt["assets"]
        or digest(pack.model_dump_json(indent=2).encode()) != receipt["stored_pack_sha256"]
    ):
        raise ValueError("served pack differs from installation")
    for page in pack.pages:
        if sorted(asset.role.value for asset in pack.assets if asset.page_id == page.page_id) != [
            "depth",
            "master",
        ]:
            raise ValueError("page asset roles differ")
    for asset in pack.assets:
        if asset.state.value != "ready" or not re.fullmatch(
            rf"/v1/assets/{asset.checksum_sha256}/[a-z0-9_-]+\.jpg", asset.local_uri
        ):
            raise ValueError("asset cache identity differs")


def verify_asset(content: bytes, asset) -> None:
    if digest(content) != asset.checksum_sha256 or _jpeg_dimensions(content) != (
        asset.width,
        asset.height,
    ):
        raise ValueError("cached image differs")


def get_bytes(client: httpx.Client, uri: str, maximum: int) -> tuple[bytes, str | None]:
    with client.stream("GET", uri) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > maximum:
                raise ValueError("API response exceeds bound")
        return bytes(content), response.headers.get("etag")


def cached_pass(client: httpx.Client, receipt: dict) -> dict:
    begun = time.perf_counter()
    content, _ = get_bytes(client, "/v1/story-packs/latest", 2_000_000)
    pack = StoryPack.model_validate_json(content)
    verify_pack(pack, receipt)
    for asset in pack.assets:
        content, etag = get_bytes(client, asset.local_uri, 16_000_000)
        verify_asset(content, asset)
        if etag != f'"{asset.checksum_sha256}"':
            raise ValueError("cached asset ETag differs")
    return {
        "status": "verified",
        "pages": 8,
        "assets": 16,
        "elapsed_ms": round((time.perf_counter() - begun) * 1000, 3),
    }


@contextmanager
def isolated_api(data: Path, port: int):
    environment = {
        "PATH": os.defpath,
        "PYTHONPATH": str(ROOT / "src"),
        **{f"BOOKFORGE_{key}": value for key, value in DISABLED.items()},
        "BOOKFORGE_DATA_DIR": str(data),
        "BOOKFORGE_CACHE_DIR": str(data / "cache"),
        "BOOKFORGE_LIVE_SCENE_OUTPUT_DIR": str(data / "no-generation"),
    }
    with (
        tempfile.TemporaryDirectory(prefix="bookforge-cache-rehearsal-") as temporary,
        tempfile.TemporaryFile(dir=temporary) as errors,
        socket.socket() as listener,
    ):
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(16)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "bookforge.api:app",
                "--fd",
                str(listener.fileno()),
                "--no-access-log",
                "--log-level",
                "error",
            ],
            cwd=temporary,
            env=environment,
            pass_fds=(listener.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=errors,
            stderr=errors,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=5,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise ValueError("isolated API did not become ready")
                    try:
                        content, _ = get_bytes(client, "/readyz", 65536)
                        if json.loads(content).get("ready") is True:
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    time.sleep(0.1)
                yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def on_jetson() -> bool:
    return (
        sys.platform == "linux"
        and platform.machine() == "aarch64"
        and Path("/etc/nv_tegra_release").is_file()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--installation-receipt", type=Path, required=True)
    parser.add_argument("--proof-installation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18089)
    args = parser.parse_args(argv)
    try:
        if not on_jetson() or not 1024 <= args.port <= 65535 or args.port == 8080:
            raise ValueError("rehearsal requires Jetson and an isolated unprivileged port")
        if (
            args.output.exists()
            or args.output.is_symlink()
            or not args.output.parent.is_dir()
            or args.output.resolve().is_relative_to(args.data_dir.resolve())
        ):
            raise ValueError("receipt requires a fresh separate destination")
        raw = args.installation_receipt.read_bytes()
        if len(raw) > 65536 or digest(raw) != args.proof_installation_sha256:
            raise ValueError("installation receipt proof differs")
        receipt = json.loads(raw)
        _, pack_path = installed_pack(args.data_dir, receipt)
        passes = []
        startups = []
        for _ in range(2):
            begun = time.perf_counter()
            with isolated_api(args.data_dir, args.port) as client:
                startups.append(round((time.perf_counter() - begun) * 1000, 3))
                for _ in range(3):
                    passes.append(cached_pass(client, receipt))
        if digest(private_read(pack_path, 2_000_000)) != receipt["stored_pack_sha256"]:
            raise ValueError("stored pack changed during rehearsal")
        result = {
            "schema_version": 1,
            "kind": "fidelity-cache-rehearsal",
            "status": "verified",
            "harness_sha256": digest(Path(__file__).read_bytes()),
            "installation_receipt_sha256": args.proof_installation_sha256,
            "stored_pack_sha256": receipt["stored_pack_sha256"],
            "page_ids": receipt["page_ids"],
            "asset_count": 16,
            "api_starts": 2,
            "cached_passes_before_restart": 3,
            "cached_passes_after_restart": 3,
            "passes": passes,
            "startup_ms": startups,
            "generation_disabled": True,
            "visual_acceptance_measured": False,
            "physical_playback_verified": False,
        }
        with os.fdopen(
            os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w"
        ) as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception:
        print("cache rehearsal failed; verify local installation and isolated API configuration")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
