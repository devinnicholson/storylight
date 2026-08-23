from __future__ import annotations

import asyncio
import os
import shutil
import signal
import struct
import tempfile
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from bookforge.config import Settings


class AssetGenerationError(RuntimeError):
    pass


class AssetGeneratorUnavailableError(AssetGenerationError):
    pass


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    content: bytes
    mime_type: str
    width: int
    height: int
    provider: str
    model: str
    seed: int
    generation_ms: float
    embedded_depth: bytes | None = None
    embedded_depth_provider: str | None = None
    embedded_depth_model: str | None = None
    embedded_depth_ms: float | None = None


class AssetGenerator(ABC):
    @abstractmethod
    async def generate(self, request: ImageGenerationRequest) -> GeneratedImage:
        raise NotImplementedError

    @abstractmethod
    async def probe(self) -> tuple[bool, str]:
        raise NotImplementedError


class DepthEstimator(ABC):
    @abstractmethod
    async def estimate(self, image: GeneratedImage) -> GeneratedImage:
        raise NotImplementedError

    @abstractmethod
    async def probe(self) -> tuple[bool, str]:
        raise NotImplementedError


class DisabledAssetGenerator(AssetGenerator):
    async def generate(self, request: ImageGenerationRequest) -> GeneratedImage:
        del request
        raise AssetGeneratorUnavailableError("Image generation is disabled")

    async def probe(self) -> tuple[bool, str]:
        return False, "Image generation is disabled"


class DisabledDepthEstimator(DepthEstimator):
    async def estimate(self, image: GeneratedImage) -> GeneratedImage:
        del image
        raise AssetGeneratorUnavailableError("Depth estimation is disabled")

    async def probe(self) -> tuple[bool, str]:
        return False, "Depth estimation is disabled"


class MfluxAssetGenerator(AssetGenerator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def probe(self) -> tuple[bool, str]:
        executable = shutil.which(self.settings.asset_command)
        if executable is None:
            return False, f"MFLUX command was not found: {self.settings.asset_command}"
        return True, f"MFLUX is available at {executable}"

    async def generate(self, request: ImageGenerationRequest) -> GeneratedImage:
        ready, detail = await self.probe()
        if not ready:
            raise AssetGeneratorUnavailableError(detail)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="bookforge-mflux-") as directory_name:
            output = Path(directory_name) / "master.png"
            command = [
                self.settings.asset_command,
                "--model",
                self.settings.asset_model,
                "--quantize",
                str(self.settings.asset_quantize),
                "--steps",
                str(self.settings.asset_modal_steps),
                "--seed",
                str(request.seed),
                "--width",
                str(request.width),
                "--height",
                str(request.height),
                "--prompt",
                request.prompt,
                "--negative-prompt",
                request.negative_prompt,
                "--output",
                str(output),
                "--no-metadata",
            ]
            if self.settings.asset_low_ram:
                command.extend(["--low-ram", "--vae-tiling"])
            await _run_command(command, self.settings.asset_timeout_seconds)
            try:
                content = output.read_bytes()
            except FileNotFoundError as error:
                raise AssetGenerationError("MFLUX completed without an output image") from error
        width, height = _png_dimensions(content)
        if (width, height) != (request.width, request.height):
            raise AssetGenerationError(
                f"MFLUX returned {width}x{height}; expected {request.width}x{request.height}"
            )
        return GeneratedImage(
            content=content,
            mime_type="image/png",
            width=width,
            height=height,
            provider="mflux",
            model=self.settings.asset_model,
            seed=request.seed,
            generation_ms=(time.perf_counter() - started) * 1000,
        )


class MfluxDepthEstimator(DepthEstimator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def probe(self) -> tuple[bool, str]:
        executable = shutil.which(self.settings.asset_depth_command)
        if executable is None:
            return False, f"MFLUX depth command was not found: {self.settings.asset_depth_command}"
        return True, f"MFLUX Depth Pro is available at {executable}"

    async def estimate(self, image: GeneratedImage) -> GeneratedImage:
        ready, detail = await self.probe()
        if not ready:
            raise AssetGeneratorUnavailableError(detail)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="bookforge-depth-") as directory_name:
            source = Path(directory_name) / "master.png"
            depth = Path(directory_name) / "master_depth.png"
            source.write_bytes(image.content)
            command = [
                self.settings.asset_depth_command,
                "--image-path",
                str(source),
                "--quantize",
                str(self.settings.asset_quantize),
            ]
            await _run_command(command, self.settings.asset_timeout_seconds)
            try:
                content = depth.read_bytes()
            except FileNotFoundError as error:
                raise AssetGenerationError("Depth Pro completed without a depth map") from error
        width, height = _png_dimensions(content)
        return GeneratedImage(
            content=content,
            mime_type="image/png",
            width=width,
            height=height,
            provider="mflux-depth-pro",
            model="apple-depth-pro",
            seed=image.seed,
            generation_ms=(time.perf_counter() - started) * 1000,
        )


class ModalAssetGenerator(AssetGenerator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def probe(self) -> tuple[bool, str]:
        executable = shutil.which(self.settings.asset_modal_command)
        if executable is None:
            return False, f"Modal command was not found: {self.settings.asset_modal_command}"
        if not self.settings.asset_modal_app.is_file():
            return False, f"Modal Scene Foundry app was not found: {self.settings.asset_modal_app}"
        return True, f"Modal Scene Foundry is available through {executable}"

    async def generate(self, request: ImageGenerationRequest) -> GeneratedImage:
        ready, detail = await self.probe()
        if not ready:
            raise AssetGeneratorUnavailableError(detail)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="bookforge-modal-") as directory_name:
            master_path = Path(directory_name) / "master.png"
            depth_path = Path(directory_name) / "depth.png"
            command = [
                self.settings.asset_modal_command,
                "run",
                "--quiet",
                f"{self.settings.asset_modal_app}::generate_cli",
                "--prompt",
                request.prompt,
                "--negative-prompt",
                request.negative_prompt,
                "--seed",
                str(request.seed),
                "--width",
                str(request.width),
                "--height",
                str(request.height),
                "--steps",
                str(self.settings.asset_steps),
                "--output-path",
                str(master_path),
                "--depth-output-path",
                str(depth_path),
            ]
            await _run_command(command, self.settings.asset_timeout_seconds)
            try:
                content = master_path.read_bytes()
                depth_content = depth_path.read_bytes()
            except FileNotFoundError as error:
                raise AssetGenerationError(
                    "Modal completed without both master and depth images"
                ) from error
        width, height = _png_dimensions(content)
        depth_width, depth_height = _png_dimensions(depth_content)
        if (width, height) != (request.width, request.height):
            raise AssetGenerationError(
                f"Modal returned {width}x{height}; expected {request.width}x{request.height}"
            )
        if (depth_width, depth_height) != (request.width, request.height):
            raise AssetGenerationError(
                "Modal depth dimensions do not match the generated master image"
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return GeneratedImage(
            content=content,
            mime_type="image/png",
            width=width,
            height=height,
            provider="modal",
            model="sdxl-turbo",
            seed=request.seed,
            generation_ms=elapsed_ms,
            embedded_depth=depth_content,
            embedded_depth_provider="modal",
            embedded_depth_model="depth-anything-v2-small",
            embedded_depth_ms=0,
        )


class ModalDepthEstimator(DepthEstimator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def probe(self) -> tuple[bool, str]:
        generator = ModalAssetGenerator(self.settings)
        return await generator.probe()

    async def estimate(self, image: GeneratedImage) -> GeneratedImage:
        if image.embedded_depth is None:
            raise AssetGenerationError("Modal image result did not include its depth sidecar")
        width, height = _png_dimensions(image.embedded_depth)
        return GeneratedImage(
            content=image.embedded_depth,
            mime_type="image/png",
            width=width,
            height=height,
            provider=image.embedded_depth_provider or "modal",
            model=image.embedded_depth_model or "depth-anything-v2-small",
            seed=image.seed,
            generation_ms=image.embedded_depth_ms or 0,
        )


class FakeAssetGenerator(AssetGenerator):
    async def probe(self) -> tuple[bool, str]:
        return True, "Fake image generator is ready"

    async def generate(self, request: ImageGenerationRequest) -> GeneratedImage:
        return GeneratedImage(
            content=_solid_png(request.width, request.height, (13, 31, 48)),
            mime_type="image/png",
            width=request.width,
            height=request.height,
            provider="fake",
            model="fake-image",
            seed=request.seed,
            generation_ms=1,
        )


class FakeDepthEstimator(DepthEstimator):
    async def probe(self) -> tuple[bool, str]:
        return True, "Fake depth estimator is ready"

    async def estimate(self, image: GeneratedImage) -> GeneratedImage:
        return GeneratedImage(
            content=_solid_png(image.width, image.height, (128, 128, 128)),
            mime_type="image/png",
            width=image.width,
            height=image.height,
            provider="fake-depth",
            model="fake-depth",
            seed=image.seed,
            generation_ms=1,
        )


def build_asset_generator(settings: Settings) -> AssetGenerator:
    if settings.asset_backend == "modal":
        return ModalAssetGenerator(settings)
    if settings.asset_backend == "mflux":
        return MfluxAssetGenerator(settings)
    if settings.asset_backend == "fake":
        return FakeAssetGenerator()
    return DisabledAssetGenerator()


def build_depth_estimator(settings: Settings) -> DepthEstimator:
    if settings.asset_backend == "modal":
        return ModalDepthEstimator(settings)
    if settings.asset_backend == "mflux":
        return MfluxDepthEstimator(settings)
    if settings.asset_backend == "fake":
        return FakeDepthEstimator()
    return DisabledDepthEstimator()


async def _run_command(command: list[str], timeout_seconds: float) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except (TimeoutError, asyncio.CancelledError):
        _terminate_process_group(process)
        await process.wait()
        raise
    if process.returncode != 0:
        detail = output.decode("utf-8", errors="replace")[-4_000:]
        raise AssetGenerationError(f"Image command failed ({process.returncode}): {detail}")


def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssetGenerationError("Image backend did not return a valid PNG")
    width, height = struct.unpack(">II", content[16:24])
    if width <= 0 or height <= 0:
        raise AssetGenerationError("Image backend returned invalid PNG dimensions")
    return width, height


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    pixels = zlib.compress(row * height, level=9)
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
