from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from storylight.domain import StoryPack


class StoryPackNotFoundError(LookupError):
    pass


class StoryPackCorruptError(RuntimeError):
    pass


class StoryPackStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = asyncio.Lock()
        self._live_scene_index: dict[str, Path] = {}
        self._live_scene_index_ready = False

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_and_index_sync)

    def _initialize_and_index_sync(self) -> None:
        self._initialize_sync()
        self._rebuild_live_scene_index_sync()

    def _initialize_sync(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    async def save(self, pack: StoryPack) -> Path:
        async with self._lock:
            return await asyncio.to_thread(self._save_sync, pack)

    def _save_sync(self, pack: StoryPack) -> Path:
        self._initialize_sync()
        payload = pack.model_dump_json(indent=2).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        slug = re.sub(r"[^a-z0-9_-]+", "-", pack.story_id.lower()).strip("-") or "story"
        filename = f"{slug[:48]}-{digest[:12]}.story-pack.json"
        destination = self.root / filename
        self._atomic_write(destination, payload)
        self._atomic_write(self.root / "latest", f"{filename}\n".encode())
        index_key = self._stored_live_scene_key(pack)
        if index_key is not None:
            self._live_scene_index[index_key] = destination
        return destination

    async def latest(self) -> StoryPack:
        async with self._lock:
            return await asyncio.to_thread(self._latest_sync)

    async def find_live_scene(
        self,
        *,
        text: str,
        visual_style: str,
        seed: int,
        session_id: str | None,
        planning_scope: str = "focal",
    ) -> StoryPack | None:
        """Return the newest exact completed live scene, if one is stored.

        Asset bytes are deliberately verified by ``AssetCache`` at the caller's
        trust boundary. This method validates only the immutable Story Pack file
        and exact request identity.
        """

        async with self._lock:
            return await asyncio.to_thread(
                self._find_live_scene_sync,
                text=text,
                visual_style=visual_style,
                seed=seed,
                session_id=session_id,
                planning_scope=planning_scope,
            )

    def _latest_sync(self) -> StoryPack:
        pointer = self.root / "latest"
        try:
            filename = pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise StoryPackNotFoundError("No compiled Story Pack is stored") from error
        if not filename or Path(filename).name != filename:
            raise StoryPackNotFoundError("The latest Story Pack pointer is invalid")
        try:
            payload = (self.root / filename).read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise StoryPackNotFoundError("The latest Story Pack file is missing") from error
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        if not filename.endswith(f"-{digest}.story-pack.json"):
            raise StoryPackCorruptError("The latest Story Pack checksum does not match its name")
        try:
            return StoryPack.model_validate_json(payload)
        except ValueError as error:
            raise StoryPackCorruptError("The latest Story Pack is invalid") from error

    def _find_live_scene_sync(
        self,
        *,
        text: str,
        visual_style: str,
        seed: int,
        session_id: str | None,
        planning_scope: str = "focal",
    ) -> StoryPack | None:
        del session_id  # Transport identity does not change exact visual identity.
        self._initialize_sync()
        if not self._live_scene_index_ready:
            self._rebuild_live_scene_index_sync()
        cache_key = self._live_scene_request_key(
            text=text,
            visual_style=visual_style,
            seed=seed,
            planning_scope=planning_scope,
        )
        candidate = self._live_scene_index.get(cache_key)
        if candidate is None:
            return None
        pack = self._read_verified_pack(candidate)
        if pack is None or self._stored_live_scene_key(pack) != cache_key:
            self._live_scene_index.pop(cache_key, None)
            return None
        return pack

    def _rebuild_live_scene_index_sync(self) -> None:
        index: dict[str, Path] = {}
        try:
            candidates = sorted(
                self.root.glob("*.story-pack.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            candidates = []
        for candidate in candidates:
            pack = self._read_verified_pack(candidate)
            if pack is None:
                continue
            cache_key = self._stored_live_scene_key(pack)
            if cache_key is not None:
                index.setdefault(cache_key, candidate)
        self._live_scene_index = index
        self._live_scene_index_ready = True

    @staticmethod
    def _read_verified_pack(candidate: Path) -> StoryPack | None:
        try:
            payload = candidate.read_text(encoding="utf-8")
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
            if not candidate.name.endswith(f"-{digest}.story-pack.json"):
                return None
            return StoryPack.model_validate_json(payload)
        except (OSError, ValueError):
            return None

    @classmethod
    def _stored_live_scene_key(cls, pack: StoryPack) -> str | None:
        story_identity = re.fullmatch(r".+-[a-f0-9]{12}", pack.story_id)
        if story_identity is None or len(pack.pages) != 1:
            return None
        ready_assets = [asset for asset in pack.assets if asset.state.value == "ready"]
        if len(ready_assets) != len(pack.assets):
            return None
        ready_roles = {asset.role.value for asset in ready_assets}
        if not {"master", "depth"} <= ready_roles:
            return None
        master_assets = [asset for asset in ready_assets if asset.role.value == "master"]
        if len(master_assets) != 1:
            return None
        return cls._live_scene_request_key(
            text=pack.pages[0].source_text,
            visual_style=pack.visual_style,
            seed=master_assets[0].seed,
            planning_scope=pack.planning_scope,
        )

    @staticmethod
    def _live_scene_request_key(
        *,
        text: str,
        visual_style: str,
        seed: int,
        planning_scope: str = "focal",
    ) -> str:
        payload = json.dumps(
            {
                "text": text,
                "visual_style": visual_style,
                "seed": seed,
                "planning_scope": planning_scope,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def ready(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK | os.X_OK)

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
