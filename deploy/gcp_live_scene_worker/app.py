from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

FAST_MODEL = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
FAST_MODEL_REVISION = "19683c58b7ea290e55cedd8950ae1d86ada7ef96"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "b4769fd619394250528294b658587285526fab1c"
DEPTH_DTYPE = "float16"
PROVIDER_NAME = "gcp-cloud-run"
MASTER_JPEG_QUALITY = 95
DEPTH_JPEG_QUALITY = 85
MODEL_CACHE = os.environ.get("BOOKFORGE_MODEL_CACHE", "/models/huggingface")
EXPECTED_GPU = os.environ.get("BOOKFORGE_EXPECTED_GPU", "L4")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrewarmRequest(StrictModel):
    prewarm_id: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,96}$"),
    ]


class GenerateRequest(StrictModel):
    scene_id: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,96}$"),
    ]
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    negative_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    ]
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    width: Annotated[int, Field(ge=512, le=1536, multiple_of=32)]
    height: Annotated[int, Field(ge=512, le=1536, multiple_of=32)]
    # Sana Sprint's SCM scheduler only supports its native two-step path.
    steps: Annotated[int, Field(ge=2, le=2)]
    guidance_scale: Annotated[float, Field(ge=0, le=12)]


class SceneRuntime:
    def __init__(self) -> None:
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.image_pipe: Any | None = None
        self.depth_pipe: Any | None = None
        self.model_load_seconds = 0.0
        self.inference_warmup_seconds = 0.0
        self.inference_warmed = False
        self.loaded_at = 0.0
        self.gpu = "unloaded"
        self.gpu_compute_capability = "unloaded"
        self.torch_version = "unloaded"
        self.torch_cuda_version = "unloaded"

    async def ensure_loaded(self) -> None:
        if self.image_pipe is not None:
            return
        async with self._load_lock:
            if self.image_pipe is not None:
                return
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        import diffusers
        import torch
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        started = time.perf_counter()
        gpu_name = torch.cuda.get_device_name(0).upper()
        compute_capability = torch.cuda.get_device_capability(0)
        expected_arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
        supported_arches = torch.cuda.get_arch_list()
        matches = {
            "L4": "L4" in gpu_name,
            "RTX_PRO_6000": "RTX PRO 6000" in gpu_name,
        }
        if EXPECTED_GPU not in matches or not matches[EXPECTED_GPU]:
            raise RuntimeError(
                f"Bookforge Cloud Run worker requires {EXPECTED_GPU}, got {gpu_name}"
            )
        if expected_arch not in supported_arches:
            raise RuntimeError(
                "PyTorch CUDA build does not support the attached GPU architecture: "
                f"requires {expected_arch}, supports {supported_arches}"
            )
        self.gpu = EXPECTED_GPU
        self.gpu_compute_capability = f"{compute_capability[0]}.{compute_capability[1]}"
        self.torch_version = torch.__version__
        self.torch_cuda_version = torch.version.cuda or "unknown"
        self.image_pipe = diffusers.SanaSprintPipeline.from_pretrained(
            FAST_MODEL,
            revision=FAST_MODEL_REVISION,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.image_pipe.set_progress_bar_config(disable=True)
        depth_model_path = snapshot_download(
            repo_id=DEPTH_MODEL,
            revision=DEPTH_MODEL_REVISION,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=depth_model_path,
            image_processor=depth_model_path,
            dtype=torch.float16,
            device=0,
        )
        self.model_load_seconds = time.perf_counter() - started
        self.loaded_at = time.monotonic()

    async def prewarm(self) -> dict[str, Any]:
        await self.ensure_loaded()
        async with self._inference_lock:
            if not self.inference_warmed:
                started = time.perf_counter()
                await asyncio.to_thread(
                    self._infer,
                    "bright layered paper theater with one simple lantern",
                    1,
                    1024,
                    576,
                    2,
                    4.5,
                    False,
                )
                self.inference_warmup_seconds = time.perf_counter() - started
                self.inference_warmed = True
        return self.identity() | {
            "ready": True,
            "model_load_seconds": self.model_load_seconds,
            "inference_warmup_seconds": self.inference_warmup_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
            "warm_state": "prewarmed",
        }

    async def generate(self, request: GenerateRequest) -> dict[str, Any]:
        await self.ensure_loaded()
        async with self._inference_lock:
            result = await asyncio.to_thread(
                self._infer,
                request.prompt,
                request.seed,
                request.width,
                request.height,
                request.steps,
                request.guidance_scale,
                True,
            )
            self.inference_warmed = True
        return self.identity() | {
            "scene_id": request.scene_id,
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
            "warm_state": "warm" if self.inference_warmup_seconds else "cold",
            **result,
        }

    def _infer(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        encode: bool,
    ) -> dict[str, Any]:
        import torch

        assert self.image_pipe is not None
        assert self.depth_pipe is not None
        image_started = time.perf_counter()
        with torch.inference_mode():
            master = self.image_pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
        torch.cuda.synchronize()
        image_seconds = time.perf_counter() - image_started
        depth_started = time.perf_counter()
        with torch.inference_mode():
            depth = self.depth_pipe(master)["depth"].resize((width, height))
        torch.cuda.synchronize()
        depth_seconds = time.perf_counter() - depth_started
        if not encode:
            return {}
        master_bytes, depth_bytes, packaging_seconds = _encode_scene_assets(master, depth)
        return {
            "image_seconds": image_seconds,
            "depth_seconds": depth_seconds,
            "packaging_seconds": packaging_seconds,
            "negative_prompt_supported": False,
            "master_b64": base64.b64encode(master_bytes).decode("ascii"),
            "master_sha256": hashlib.sha256(master_bytes).hexdigest(),
            "master_media_type": "image/jpeg",
            "master_width": width,
            "master_height": height,
            "master_jpeg_quality": MASTER_JPEG_QUALITY,
            "depth_b64": base64.b64encode(depth_bytes).decode("ascii"),
            "depth_sha256": hashlib.sha256(depth_bytes).hexdigest(),
            "depth_media_type": "image/jpeg",
            "depth_width": width,
            "depth_height": height,
            "depth_jpeg_quality": DEPTH_JPEG_QUALITY,
        }

    def identity(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER_NAME,
            "gpu": self.gpu,
            "gpu_compute_capability": self.gpu_compute_capability,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "fast_model": FAST_MODEL,
            "fast_model_revision": FAST_MODEL_REVISION,
            "depth_model": DEPTH_MODEL,
            "depth_model_revision": DEPTH_MODEL_REVISION,
            "depth_dtype": DEPTH_DTYPE,
        }


def _encode_scene_assets(master: Any, depth: Any) -> tuple[bytes, bytes, float]:
    started = time.perf_counter()

    def encode_master() -> bytes:
        buffer = io.BytesIO()
        master.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=MASTER_JPEG_QUALITY,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        return buffer.getvalue()

    def encode_depth() -> bytes:
        buffer = io.BytesIO()
        depth.convert("L").save(
            buffer,
            format="JPEG",
            quality=DEPTH_JPEG_QUALITY,
            optimize=False,
            progressive=False,
        )
        return buffer.getvalue()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bookforge-pack") as pool:
        master_future = pool.submit(encode_master)
        depth_future = pool.submit(encode_depth)
        master_bytes = master_future.result()
        depth_bytes = depth_future.result()
    return master_bytes, depth_bytes, time.perf_counter() - started


runtime = SceneRuntime()
app = FastAPI(title="Bookforge GCP Live Scene Worker", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "provider": PROVIDER_NAME,
        "fast_model_revision": FAST_MODEL_REVISION,
        "depth_model_revision": DEPTH_MODEL_REVISION,
        "loaded": runtime.image_pipe is not None,
    }


@app.post("/v1/prewarm")
async def prewarm(payload: PrewarmRequest) -> dict[str, Any]:
    result = await runtime.prewarm()
    return {"prewarm_id": payload.prewarm_id, **result}


@app.post("/v1/generate")
async def generate(payload: GenerateRequest) -> dict[str, Any]:
    return await runtime.generate(payload)
