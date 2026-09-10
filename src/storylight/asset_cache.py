from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from storylight.domain import AssetKind, AssetRecord, AssetState, StoryPack

CHECKSUM_PATTERN = re.compile(r"^[a-f0-9]{64}$")
ALLOWED_SUFFIXES = {
    AssetKind.IMAGE: {".avif", ".jpeg", ".jpg", ".png", ".webp"},
    AssetKind.SPRITE: {".avif", ".jpeg", ".jpg", ".png", ".webp"},
    AssetKind.DEPTH_MAP: {".jpeg", ".jpg", ".png", ".webp"},
    AssetKind.MASK: {".png", ".webp"},
    AssetKind.VIDEO_LOOP: {".mp4", ".webm"},
}


class AssetCacheError(RuntimeError):
    pass


class AssetCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    async def install_pack(self, pack: StoryPack, source_root: Path) -> StoryPack:
        self._initialize_sync()
        assets = [
            await asyncio.to_thread(self._install_asset, asset, source_root)
            for asset in pack.assets
        ]
        return pack.model_copy(update={"assets": assets})

    def resolve(self, checksum: str, filename: str) -> Path:
        if not CHECKSUM_PATTERN.fullmatch(checksum) or Path(filename).name != filename:
            raise AssetCacheError("Invalid cached asset path")
        path = self.root / checksum / filename
        if not path.is_file():
            raise AssetCacheError("Cached asset was not found")
        return path

    async def store_generated(
        self,
        *,
        asset_id: str,
        kind: AssetKind,
        content: bytes,
        suffix: str,
    ) -> tuple[str, str]:
        return await asyncio.to_thread(
            self._store_generated_sync,
            asset_id=asset_id,
            kind=kind,
            content=content,
            suffix=suffix,
        )

    def _store_generated_sync(
        self,
        *,
        asset_id: str,
        kind: AssetKind,
        content: bytes,
        suffix: str,
    ) -> tuple[str, str]:
        self._initialize_sync()
        suffix = suffix.lower()
        if not content:
            raise AssetCacheError("Generated asset is empty")
        if suffix not in ALLOWED_SUFFIXES.get(kind, set()):
            raise AssetCacheError(f"Unsupported {kind.value} generated asset format: {suffix}")
        digest = hashlib.sha256(content).hexdigest()
        safe_id = re.sub(r"[^a-z0-9_-]+", "-", asset_id.lower()).strip("-") or "asset"
        filename = f"{safe_id[:48]}{suffix}"
        directory = self.root / digest
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        destination = directory / filename
        if destination.exists():
            if _sha256(destination) != digest:
                raise AssetCacheError("Generated asset cache collision")
        else:
            _atomic_write_bytes(destination, content)
        return digest, f"/v1/assets/{digest}/{filename}"

    def _install_asset(self, asset: AssetRecord, source_root: Path) -> AssetRecord:
        if asset.state is not AssetState.READY or asset.kind is AssetKind.PROCEDURAL:
            return asset
        if asset.local_uri.startswith("/v1/assets/"):
            return self._verify_cached_asset(asset)

        source = _source_path(asset, source_root)
        if not source.is_file():
            raise AssetCacheError(f"Asset source is missing: {source}")
        digest = _sha256(source)
        if digest != asset.checksum_sha256:
            raise AssetCacheError(f"Asset checksum mismatch: {asset.asset_id}")

        suffix = source.suffix.lower() or ".bin"
        if suffix not in ALLOWED_SUFFIXES.get(asset.kind, set()):
            raise AssetCacheError(f"Unsupported {asset.kind.value} asset format: {suffix}")
        safe_id = re.sub(r"[^a-z0-9_-]+", "-", asset.asset_id.lower()).strip("-") or "asset"
        filename = f"{safe_id[:48]}{suffix}"
        directory = self.root / digest
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        destination = directory / filename
        _atomic_copy(source, destination)
        return asset.model_copy(update={"local_uri": f"/v1/assets/{digest}/{filename}"})

    def _verify_cached_asset(self, asset: AssetRecord) -> AssetRecord:
        parts = urlparse(asset.local_uri).path.split("/")
        if len(parts) != 5 or parts[1:3] != ["v1", "assets"]:
            raise AssetCacheError(f"Invalid cached asset URI: {asset.local_uri}")
        checksum, filename = parts[3], parts[4]
        if checksum != asset.checksum_sha256:
            raise AssetCacheError(f"Cached asset checksum mismatch: {asset.asset_id}")
        path = self.resolve(checksum, filename)
        if _sha256(path) != checksum:
            raise AssetCacheError(f"Cached asset content mismatch: {asset.asset_id}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES.get(asset.kind, set()):
            raise AssetCacheError(f"Unsupported {asset.kind.value} asset format: {path.suffix}")
        return asset


def _source_path(asset: AssetRecord, source_root: Path) -> Path:
    if not asset.local_uri:
        raise AssetCacheError(f"Ready asset has no local source: {asset.asset_id}")
    parsed = urlparse(asset.local_uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise AssetCacheError(f"Unsupported local asset URI: {asset.local_uri}")
    else:
        path = Path(asset.local_uri)
    root = source_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise AssetCacheError(f"Asset path escapes its package: {asset.local_uri}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output_stream:
            output_stream.write(content)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
