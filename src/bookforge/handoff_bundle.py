from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from bookforge.domain import AssetKind, AssetState, StoryPack


class HandoffBundleError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(local_uri: str, asset_root: Path) -> Path:
    parsed = urlparse(local_uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise HandoffBundleError(f"unsupported asset URI: {local_uri}")
    else:
        path = Path(local_uri)
    root = asset_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise HandoffBundleError(f"asset path escapes its source root: {local_uri}")
    return resolved


def _safe_asset_name(asset_id: str, suffix: str) -> str:
    stem = re.sub(r"[^a-z0-9_-]+", "-", asset_id.lower()).strip("-") or "asset"
    return f"{stem[:64]}{suffix.lower()}"


def build_handoff_bundle(
    story_pack_path: Path,
    *,
    asset_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create a self-contained, checksum-verifiable Story Pack directory."""
    source_pack = StoryPack.model_validate_json(story_pack_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        raise HandoffBundleError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        assets_dir = temporary / "assets"
        assets_dir.mkdir(mode=0o700)
        rewritten = []
        checksum_records: list[dict[str, object]] = []
        seen_names: set[str] = set()
        for asset in source_pack.assets:
            if asset.state is not AssetState.READY or asset.kind is AssetKind.PROCEDURAL:
                rewritten.append(asset)
                continue
            source = _source_path(asset.local_uri, asset_root)
            if not source.is_file():
                raise HandoffBundleError(f"asset source is missing: {source}")
            source_checksum = _sha256(source)
            if source_checksum != asset.checksum_sha256:
                raise HandoffBundleError(f"asset checksum mismatch: {asset.asset_id}")
            filename = _safe_asset_name(asset.asset_id, source.suffix)
            if filename in seen_names:
                raise HandoffBundleError(f"asset filename collision: {filename}")
            seen_names.add(filename)
            destination = assets_dir / filename
            shutil.copy2(source, destination)
            if _sha256(destination) != source_checksum:
                raise HandoffBundleError(f"copied asset checksum mismatch: {asset.asset_id}")
            relative_uri = f"assets/{filename}"
            rewritten.append(asset.model_copy(update={"local_uri": relative_uri}))
            checksum_records.append(
                {
                    "asset_id": asset.asset_id,
                    "bytes": destination.stat().st_size,
                    "path": relative_uri,
                    "sha256": source_checksum,
                }
            )

        bundled_pack = source_pack.model_copy(update={"assets": rewritten})
        pack_name = _safe_asset_name(source_pack.story_id, ".story-pack.json")
        pack_path = temporary / pack_name
        pack_path.write_text(
            bundled_pack.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        pack_checksum = _sha256(pack_path)
        manifest = {
            "schema_version": "1.0",
            "story_id": source_pack.story_id,
            "story_pack": {"path": pack_name, "sha256": pack_checksum},
            "assets": checksum_records,
            "total_asset_bytes": sum(int(record["bytes"]) for record in checksum_records),
        }
        (temporary / "checksums.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# Bookforge offline handoff\n\n"
            "This directory is self-contained. Install it without Modal, GCP, or "
            "network access:\n\n"
            "```bash\n"
            f"python -m bookforge.pack_installer {pack_name} --asset-root .\n"
            "```\n\n"
            "The installer validates every SHA-256 checksum before promoting the pack.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return {
            "story_id": source_pack.story_id,
            "story_pack": str(output_dir / pack_name),
            "story_pack_sha256": pack_checksum,
            "assets": len(checksum_records),
            "asset_bytes": manifest["total_asset_bytes"],
            "output_dir": str(output_dir),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a portable, checksum-verifiable Bookforge handoff directory"
    )
    parser.add_argument("story_pack", type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    result = build_handoff_bundle(
        arguments.story_pack,
        asset_root=arguments.asset_root,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
