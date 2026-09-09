"""Capture deployed source and noncredential voice overrides; run on the Jetson."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

DESTINATION = Path("/tmp/bookforge-best-demo-20260908.tar.gz")
PACKAGE = Path("/opt/bookforge/.venv/lib/python3.12/site-packages/bookforge")
PARSER = Path("/opt/bookforge/voice-language-current").resolve(strict=True)
CONFIG = {
    "voice-language.conf": Path(
        "/etc/systemd/system/bookforge@operator.service.d/50-voice-language.conf"
    ),
    "voice-demo.conf": Path(
        "/run/user/1000/systemd/user/bookforge-kiosk.service.d/50-voice-demo.conf"
    ),
}
KIOSK_KEYS = {"BOOKFORGE_KIOSK_URL", "BOOKFORGE_KIOSK_READY_URL"}


def main():
    files = {}
    for prefix, root in [("package", PACKAGE), ("parser", PARSER)]:
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix in {".py", ".js", ".html", ".css"}
            ):
                files[f"{prefix}/{path.relative_to(root)}"] = path.read_bytes()
    for name, path in CONFIG.items():
        files[f"config/{name}"] = path.read_bytes()
    environment = Path("/run/user/1000/bookforge-voice-demo.env").read_text().splitlines()
    files["config/voice-demo.env"] = (
        "\n".join(line for line in environment if line.partition("=")[0] in KIOSK_KEYS) + "\n"
    ).encode()
    versions = subprocess.check_output(
        [
            "/opt/bookforge/.venv/bin/python",
            "-c",
            'import importlib.metadata as m,json;'
            'print(json.dumps(sorted((d.metadata["Name"],d.version) for d in m.distributions())))',
        ],
        text=True,
    )
    manifest = {
        "git_checkpoint": "59d13bf46ba93b97e66cdacdf964b1b3f01aa7c0",
        "package_path": str(PACKAGE),
        "parser_path": str(PARSER),
        "dependencies": json.loads(versions),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "excludes": [
            "credentials",
            "model weights",
            "runtime jobs",
            "audio",
            "non-source package data",
        ],
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with DESTINATION.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    print(
        json.dumps(
            {
                "path": str(DESTINATION),
                "sha256": hashlib.sha256(DESTINATION.read_bytes()).hexdigest(),
                "files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
