"""Install pinned MediaPipe assets locally; the projector never downloads from a CDN."""

import argparse
import base64
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.10.32"
PACKAGE_URL = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{VERSION}.tgz"
PACKAGE_SHA512 = (
    "3tiAZnmKloYnRXYoO3dKltTUGnqeCwzC4lV03uY0vCsE+aveJTyEVQyZHOlQGQNsjK+gRHzkf9q08C99Qm2K0Q=="
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MEMBERS = {
    "vision_bundle.cjs": "vision_bundle.js",
    "wasm/vision_wasm_internal.js": "wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm": "wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js": "wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm": "wasm/vision_wasm_nosimd_internal.wasm",
    "README.md": "README.md",
}


def unpack(package: bytes, model: bytes) -> dict[str, bytes]:
    if base64.b64encode(hashlib.sha512(package).digest()).decode() != PACKAGE_SHA512:
        raise ValueError("MediaPipe package integrity mismatch")
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise ValueError("MediaPipe model integrity mismatch")
    assets = {"hand_landmarker.task": model}
    with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as archive:
        for source, target in MEMBERS.items():
            member = archive.getmember(f"package/{source}")
            if not member.isfile() or member.size > 30_000_000:
                raise ValueError(f"Invalid package member: {source}")
            assets[target] = archive.extractfile(member).read()
    return assets


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(40_000_001)
    if len(data) > 40_000_000:
        raise ValueError("Asset download exceeded size limit")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("src/bookforge/static/mediapipe"))
    parser.add_argument("--package", type=Path, help="Use an already downloaded pinned package")
    parser.add_argument("--model", type=Path, help="Use an already downloaded pinned model")
    args = parser.parse_args()
    assets = unpack(
        args.package.read_bytes() if args.package else download(PACKAGE_URL),
        args.model.read_bytes() if args.model else download(MODEL_URL),
    )
    for name, data in assets.items():
        path = args.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError(f"Refusing asset symlink: {path}")
        path.write_bytes(data)
    manifest = {
        "version": VERSION,
        "package_url": PACKAGE_URL,
        "model_url": MODEL_URL,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()},
    }
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Installed {len(assets)} verified local assets in {args.destination}")


if __name__ == "__main__":
    main()
